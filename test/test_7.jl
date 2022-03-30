#######################################################
# 1 Original BoundaryCondition problem from example
# https://diffeq.sciml.ai/stable/tutorials/bvp_example/
#######################################################
using OrdinaryDiffEq: Tsit5, Vern7
import Logging
using Logging: LogLevel, with_logger
using MechGlueDiffEqBase # exports ArrayPartition
using MechGluePlots
using MechanicalUnits: @import_expand, ∙
using Test, Plots, OrdinaryDiffEq
using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4, Shooting
@import_expand(s, m, N, kg)
Plots.PlotlyJSBackend()
Plots.theme(:wong2)
Plots.plotly(ticks=:native, framestyle = :zerolines)
function strround(x)
    if oneunit(x) == 1.0
        string(round(x, digits = 3))
    else
        string(round(typeof(x), x, digits = 3))
    end
end
function plotsol(sol)
    pl = plot(sol)
    mid = searchsortedfirst(sol.t, sol.t[end] / 2)
    x = sol.t[mid]
    y = sol[mid][1]
    plot!(pl, [x], [y], marker = true)
    str = "θ $(strround(y)) @ t $(strround(x))"
    annotation = (1.1x / oneunit(x), 1.2y / oneunit(y), Plots.text(str,  11, :left))
    plot!(pl[1]; annotation)
    x = sol.t[end]
    y = sol[end][1]
    plot!(pl, [x], [y], marker = true)
    str = "θ $(strround(y)) @ t $(strround(x))"
    annotation = (0.8x / oneunit(x), 1.2y / oneunit(y), Plots.text(str,  11, :left))
    plot!(pl[1]; annotation)
    # 
    x = sol.t[mid]
    y = sol[mid][2]
    str = "θ´"
    annotation = (1.1x / oneunit(x), 1.2y / oneunit(y), Plots.text(str,  11, :left))
    plot!(pl[1], marker = true; annotation)
end
const p1 = (9.81, 1.0) # g, L
const tspan1 = (0.0, π/2)
function simplependulum´!(u´, u , p, t)
    g, L = p
    θ  = u[1]
    θ´ = u[2]
    u´[1] = θ´
    u´[2] = -(g/L) * sin(θ) # θ´´
    @info typeof(u´) maxlog=20
    u´
end
function bc1!(residual, u, p, t)
    # the solution at the middle of the time span should be -π/2
    mid = searchsortedfirst(t, t[end] / 2)
    # residual is a DE solution.
    residual[1] = u[mid][1] + π/2 
    # the solution at the end of the time span should be π/2
    residual[2] = u[end][1] - π/2 
end
#=
# 1 a) Vector domain and range
let
    "Template for mutable local tuple at t = 0 - will be adapted to fit boundary conditions `bc1!`"
    u₀ = [0.0, π/2]
    # The boundary conditions are specified by a function that calculates the 
    # residual in-place from the problem solution, such that the 
    # residual is \vec{0} when the boundary condition is satisfied.
    bvp1 = BVProblem{true}(simplependulum´!, bc1!, u₀, tspan1, p1)
    @time sol1 = solve(bvp1, Shooting(Tsit5()), dtmax = 0.05) # 0.007s
    plotsol(sol1)
end
=#


#################################################
# 2 Modified to use ArrayPartition, dimensionless
#################################################

#2 a)
#function packin!(u´, θ´, θ´´)
#    u´.x[1][1] = θ´
#    u´.x[2][1] = θ´´
#    return u´
#end
#packout(u) = u.x[1][1], u.x[2][1]

let
    "Template for mutable local tuple at t = 0 - will be adapted to fit boundary conditions `bc1!`"
    u₀ = ArrayPartition([0.0], [π/2])
    @inferred simplependulum´!(u₀ ./ 0.01, u₀, p1, 0.01)
    bvp1 = BVProblem(simplependulum´!, bc1!, u₀, tspan1, p1)
    @time sol1 = solve(bvp1, Shooting(Tsit5()), dtmax = 0.05) # 0.012s
    plotsol(sol1)
end


    #function simplependulum´!(u´, u , p, t)
    #    @debug "u = "   maxlog = 2
    #    @debug "u´ = " u´  maxlog = 2
    #    θ, θ´ = packout(u)
    #    @debug "θ = " θ  maxlog = 2
    #    @debug "θ´ = " θ´  maxlog = 2
    #    θ´´ = -(g/L) * sin(θ)
    #    @debug "θ´´ = " θ´´  maxlog = 2
    #    packin!(u´, θ´, θ´´)
    #end
    #function simplependulum2´!(u´, u , p, t)
    #    θ  = u[1]
    #    θ´ = u[2]
    #    u´[1] = θ´
    #    u´[2] = -(g/L) * sin(θ) # θ´´
    #    u´
    #end
    #function solve_guarded(Γ´!;  Γᵢₙ = u₀, alg = Tsit5(), debug=false)
        # Test the functions
    #    @inferred packout(Γᵢₙ)
    #    @inferred packout(-0.5Γᵢₙ)
    #    @inferred Γ´!(-0.5 .* Γᵢₙ, Γᵢₙ, nothing, nothing)
    #    # Define and solve
    #    prob = ODEProblem{true}(simplependulum´!, Γᵢₙ, tspan)
    #    sol = if debug
    #        with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
    #            solve(prob, Tsit5())
    #        end
    #    else
    #        solve(prob, Tsit5())
    #    end
    #    sol
    #end
    #@time sol1 = solve_guarded(simplependulum´!; debug = false)
    #@time sol1 = solve(ODEProblem(simplependulum2´!, u₀, tspan), Tsit5())
#end
#= 
    function bc1!(residual, u, p, t)
        # the solution at the middle of the time span should be -π/2
        residual[1] = u[end÷2][1] + π/2 
        # the solution at the end of the time span should be π/2
        residual[2] = u[end][1] - π/2
        residual
    end
    "Template for mutable residual value"
    residual = 0.0 .* u₀
    bc1!(residual, sol1, nothing, nothing)
    @info "Residual with u₀ = $u₀ is " residual
    bvp1 = BVProblem(simplependulum´!, bc1!, u₀, tspan)
    @time sol1 = solve(bvp1, Shooting(Tsit5()))
    @time sol1 = solve(bvp1, GeneralMIRK4(), dt = 0.05)
    plotsol(sol1)
end
2 b)
let 
    
end
sol2 = let 
    g = 9.81;  L = 1.0; tspan = (0.0, π/2)

    function simplependulum!(u´, u, p, t)
        @show u´, u, p, t
        θ, θ´ = packout(u)
        @show θ, θ´
        packin!(u´, θ´, -(g/L)*sin(θ))
    end
    u₀ = ArrayPartition([0.5], [0.0])


    #prob = ODEProblem{true}(simplependulum!, u₀, tspan)
    #solve(prob, Tsit5())
end
    #=
    function solve_guarded₃(Γ´!, par;  Γᵢₙ = Γ₀₃(par), alg = BS5(), debug = false)
        # Test
        if debug
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
                @inferred packout₃(Γᵢₙ)
                @inferred packout₃(0.01Γᵢₙ / cm)
                @inferred packout_params₃(par)
                @inferred Γ´!(Γᵢₙ/cm, Γᵢₙ, par, 10.0cm)
            end
        else
            @inferred packout₃(Γᵢₙ)
            @inferred packout₃(0.01Γᵢₙ / cm)
            @inferred packout_params₃(par)
            @inferred Γ´!(Γᵢₙ/cm, Γᵢₙ, par, 10.0cm)
        end
        # Do it, and add exact solution at some important points.
        prob = ODEProblem(Γ´!, Γᵢₙ, (-χ_limb, χ_limb), par; tstops = (-χ_bridge, 0.0cm, χ_bridge),  dtmax = 1.5cm, rel_tol = 1.0e-11) 
        if debug
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
                solve(prob, alg)
            end
        else
            solve(prob, alg)
        end
    end
    #sol1 = solve(prob, Tsit5());
    =#
    function bc1!(residual, u, p, t)
        residual[1] = u[end÷2][1] + π/2 # the solution at the middle of the time span should be -π/2
        residual[2] = u[end][1] - π/2 # the solution at the end of the time span should be π/2
    end
    bvp = BVProblem(simplependulum!, bc1!, u₀, tspan)
    solve(bvp, GeneralMIRK4(), dt=0.05)
end
pl2 = plot(sol2)
=#

##############################################
# 3 Modified to use ArrayPartition, dimensions
##############################################

const tspan2 = (0.0, π/2)s
const p2 = (9.81m/s², 1.0m) # g, L
let
    "Template for mutable local tuple at t = 0 - will be adapted to fit boundary conditions `bc1!`"
    u₀ = ArrayPartition([0.0], [π/2]s⁻¹)
    @inferred simplependulum´!(u₀ ./ 0.01s, u₀, p2, 0.01s)
    bvp2 = BVProblem(simplependulum´!, bc1!, u₀, tspan2, p2)
    solve(bvp2, Shooting(Tsit5()), dtmax = 0.05s    )
    #prob2 = ODEProblem(simplependulum´!, u₀, tspan2, p2) 
    #sol2 = solve(prob2, Tsit5(), dtmax = 0.05s);
    #plotsol(sol2)

    #@time sol3 = solve(bvp2, Shooting(Tsit5()), dtmax = 0.05s    ) # 0.012s
    #plotsol(sol3)
end















#=

# Boundary value
using MechGlueDiffEqBase, Test
using DifferentialEquations
#using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4
using MechanicalUnits: @import_expand, ∙
@import_expand(s, m, N, kg)
using Plots, MechGluePlots
#using OrdinaryDiffEq: ODEProblem
 # Dimensionless, ArrayPartition
sol1 = let 
    g = 9.81
    L = 1.0
    tspan = (0.0, π/2)
    packout(u) = u.x[1][1], u.x[2][1]
    function packin!(u´, u´1, u´2)
        u´.x[1][1] = u´1
        u´.x[2][1] = u´2
        return u´
    end
    function simplependulum!(u´, u, p, t)
        θ, θ´ = packout(u)
        packin!(u´, θ, -(g/L)*sin(θ))
    end
    prob = ODEProblem{true}(simplependulum!, ArrayPartition([0.5], [0.0]),(0.0, 1.0))
    sol1 = solve(prob, Tsit5());

    function bc1!(residual, u, p, t)
        r1 = u[end÷2][1] + π/2 # the solution at the middle of the time span should be -π/2
        r2 = u[end][1] - π/2 # the solution at the end of the time span should be π/2
        packin!(residual, r1, r2)
    end
    bvp = BVProblem(simplependulum!, bc1!, [π/2,π/2], tspan)
    sol1 = solve(bvp, GeneralMIRK4(), dt=0.05)
end
@test sol1(0.0)[1] ≈ -0.4426350362090615
@test sol1(0.0)[2] ≈ -4.659606328681781
plot(sol1)

# With units and ArrayPartition
sol1 = let 
    g = 9.81m/s²
    L = 1.0m
    tspan = (0.0, π/2)s
    function simplependulum!(u´,u,p,t)
        θ  = u[1]
        θ´ = u[2]
        u´[1] = θ´
        u´[2] = -(g/L)*sin(θ)
    end
    prob = ODEProblem{true}(simplependulum!,[0.5, 0.0/s],(0.0, 1.0)s)
    sol1 = solve(prob, Tsit5())


    function bc1!(residual, u, p, t)
        residual[1] = u[end÷2][1]s⁻¹ + π/2s⁻¹ # the solution at the middle of the time span should be -(π/2)s⁻¹
        residual[2] = u[end][1]s⁻² - π/2s⁻² # the solution at the end of the time span should be (π/2)s⁻²
    end
    bvp = BVProblem(simplependulum!, bc1!, [π/2,π/2], tspan)
    sol1 = solve(bvp, GeneralMIRK4(), dt=0.05s)
end
@test sol1(0.0)[1] ≈ -0.4426350362090615
@test sol1(0.0)[2] ≈ -4.659606328681781
plot(sol1)

nothing
=#