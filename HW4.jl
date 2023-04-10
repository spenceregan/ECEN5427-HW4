
using CSV, LinearAlgebra, DataFrames, PrettyTables, Dates
using JuMP, Gurobi

function microgrid_model()
    root = dirname(@__FILE__)
    tspath = joinpath(root, "timeseries.csv")
    timeseries = CSV.read(tspath, DataFrame)

    G = ["solar", "storage", "inverter"]
    n = length(G)
    B = ["solar_gen", "solar_curt", "batt_discharge", "batt_charge"]
    m = length(B)
    s = nrow(timeseries)
    h = 1.0
    χ = h*(s/8760.0)*[190.0, 150.0, 14.0]
    η = [0.96, 0.0, 0.96, -1.0/0.96]
    e = sqrt(0.85/(0.96^2))
    υ = [0.0, 0.0, -1.0/e, e]
    d = 4.0
    c = 0.2
    δ = timeseries.load_kW
    ν = timeseries.solar_cf
    Γ = [1*(i≥j) for i in 1:s, j in 1:s]
    σ = [1,1,0,0]

    params = (G, n, B, m, s, h, χ, η, e, υ, d, c, δ, ν, Γ, σ, timeseries)

    microgrid = Model(Gurobi.Optimizer)

    @variable(microgrid, γ[1:n] ≥ 0)
    @variable(microgrid, P[1:s, 1:m] ≥ 0)
    @variable(microgrid, ξ[1:s] ≥ 0)

    @constraint(microgrid, P .≤ ones(s)*[γ[1] γ[1] γ[2] γ[2]])
    @constraint(microgrid, P*η + ξ .== δ)
    @constraint(microgrid, zeros(s) .≤ h*Γ*P*υ)
    @constraint(microgrid, h*Γ*P*υ .≤ d*γ[2]*ones(s))
    @constraint(microgrid, P*η .≤ γ[3]*ones(s))
    @constraint(microgrid, P*η .≥ -γ[3]*ones(s))
    @constraint(microgrid, P*σ .== γ[1]*ν)
    return microgrid, params
end

function minimize_costs()
    microgrid, params = microgrid_model()
    G, n, B, m, s, h, χ, η, e, υ, d, c, δ, ν, Γ, σ, timeseries = params
    γ = microgrid[:γ]
    P = microgrid[:P]
    ξ = microgrid[:ξ]

    set_optimizer_attribute(microgrid, "Method", 0)
    set_optimizer_attribute(microgrid, "Presolve", 0)

    @objective(microgrid, Min, χ'*γ + c*h*ones(s)'*ξ)

    optimize!(microgrid)

    optimal_system = DataFrame(tech = G, capacity_kW = value.(γ))

    dc_power = value.(P)
    ac_power = dc_power*Diagonal(η)
    solar_ac = ac_power[:, 1]
    solar_curt = dc_power[:, 2]
    battery_ac = ac_power[:, 3] + ac_power[:, 4]
    grid = value.(ξ)
    SoC = h*Γ*dc_power*υ / (value(γ[2])*d)
    optimal_dispatch = DataFrame(
        t = timeseries.t,
        solar_cf = ν,
        demand = δ,
        solar = solar_ac, 
        battery = battery_ac, 
        grid = grid,
        solar_curt = solar_curt, 
        SoC = SoC
        )
    start = DateTime(2023, 1, 1)
    transform!(optimal_dispatch, :t => ByRow(t -> start + Dates.Hour(t)) => :datetime)

    optimal_cost = objective_value(microgrid)

    return optimal_system, optimal_dispatch, optimal_cost
end

optimal_system, optimal_dispatch, optimal_cost = minimize_costs()


pretty_table(optimal_system, nosubheader = true)


maxdemand = maximum(optimal_dispatch.demand)
maxhour = optimal_dispatch[optimal_dispatch.demand .== maxdemand, :t]
peakdispatch = subset(optimal_dispatch, :t => t -> abs.(t .- maxhour).≤24)


using Plots, PlotThemes
theme(:dark)
plot(
    peakdispatch.datetime,
    [peakdispatch.solar peakdispatch.battery peakdispatch.grid peakdispatch.solar_curt peakdispatch.demand],
    label = ["AC Solar" "AC Battery" "Grid" "Curtailed Solar" "Demand"],
    xlabel = "datetime",
    ylabel = "power (kW)"
)


plot(
    peakdispatch.datetime,
    peakdispatch.SoC,
    label = "State of Charge",
    xlabel = "datetime"
)


function minimize_gridenergy()
    microgrid, params = microgrid_model()
    G, n, B, m, s, h, χ, η, e, υ, d, c, δ, ν, Γ, σ, timeseries = params
    γ = microgrid[:γ]
    P = microgrid[:P]
    ξ = microgrid[:ξ]

    @constraint(microgrid, γ[1] ≤ 1000)
    @objective(microgrid, Min, h*ones(s)'*ξ)

    set_optimizer_attribute(microgrid, "Method", 3)
    set_optimizer_attribute(microgrid, "Presolve", -1)

    optimize!(microgrid)

    γ_opt = value.(γ)
    ξ_opt = value.(ξ)
    P_opt = value.(P)
    optimal_system = DataFrame(tech = G, capacity_kW = γ_opt)
    total_cost = χ'*γ_opt + c*h*ones(s)'*ξ_opt

    return optimal_system, total_cost
end

min_gridenergy_system, min_gridenergy_cost = minimize_gridenergy()


pretty_table(min_gridenergy_system, nosubheader = true)

println("Annual cost minimizing grid imports: \$", round(min_gridenergy_cost, digits = 2))
println("Minimum annual costs for optimal system: \$", round(optimal_cost, digits = 2))

