using Test
using MechGlueDiffEqBase, DiffEqBase, OrdinaryDiffEq
using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4, Shooting
using MechanicalUnits: @import_expand, ∙
@import_expand(m, kg, s)
#######################################################
# 1 Original BoundaryCondition problem from example
# https://diffeq.sciml.ai/stable/tutorials/bvp_example/
#######################################################
function simplependulum´!(u´, u , p, t)
    g, L = p
    @debug "simplependulum´! "  g L maxlog = 2
    θ, θ´ = u
    @debug "simplependulum´!" θ  θ´ maxlog = 2
    θ´´ = -(g/L) * sin(θ)
    @debug "simplependulum´!" θ´´  maxlog = 2
    u´ .= θ´, θ´´
    @debug "simplependulum´!" u´  maxlog = 2
    u´
end

"""
     bc!(u, sol::T, p, t) where {T<: DESolution}

Set `u` elements to zero when boundary conditions are met.

# Arguments
- `u`     When called from `solve(bvp::BVProblem, :BoundaryValueDiffEqAlgorithm, ...),
    the first value is `bvp.u0`. Avoid changing element types!
- sol     Solution, contains u for every step.

"""
function bc!(u, sol, p, t)
    @debug "bc!" u sol maxlog = 2
    umid = sol[end÷2]
    uend = sol[end]
    # The solution at the middle of the time span should be -π/2 (radians). 
    # We make this a little more unit-generic than necessary.
    u[1] = umid[1] + π/2 * oneunit(u[1])
    # The solution at the end of the time span should be π/2.
    # We need to express the evaluation in the units that u[2] has initially.
    u[2] = uend[1] * oneunit(u[2]) / oneunit(u[1]) - (π/2 * oneunit(u[2]))
    u
end


#=
@testset "BCP vector domain and range" begin  # 23.6s with compilation when run first
    u₀ = [0.0, π/2]  # θ, θ´
    g = 9.81
    L = 1.0
    p = g, L
    tspan = (0.0, π/2)
    bvp = BVProblem(simplependulum´!, bc!, u₀, tspan, p)
    sol = solve(bvp, Shooting(Tsit5()), dtmax = 0.05);
    @test sol[end÷2][1] ≈ -π/2
    @test sol[end][1] ≈ π/2
end


@testset "BCP, mixed mutable vector domain and range" begin # 32.8s with compilation when run first 
    u₀ = ArrayPartition([0.0], [π/2]) # θ, θ´
    g = 9.81
    L = 1.0
    p = convert_to_mixed(g, L) # convenience constructor; same structure as u₀.
    tspan = (0.0, π/2)
    bvp = BVProblem(simplependulum´!, bc!, u₀, tspan, p)
    sol = solve(bvp, Shooting(Tsit5()), dtmax = 0.05);
    @test sol[end÷2][1] ≈ -π/2
    @test sol[end][1] ≈ π/2
end


@testset "BCP vector domain and range, dimensional" begin# 1.8s with compilation when run first 
    g = 9.81m/s²
    u₀ = [0.0, 0.5π∙s⁻¹]  # θ, θ´
    L = 1.0m
    p = g, L
    tspan = (0.0, π/2)s
    bvp = BVProblem(simplependulum´!, bc!, u₀, tspan, p)
    @test_throws MethodError solve(bvp, Shooting(Tsit5()), dtmax = 0.05s);
end
=#

#@testset "BCP mixed mutable vector domain and range, dimensional" begin
    g = 9.81m/s²
    u₀ = ArrayPartition([0.0], [0.5π∙s⁻¹]) # θ, θ´
    L = 1.0m
    p = g, L
    tspan = (0.0, π/2)s
    bvp = BVProblem(simplependulum´!, bc!, u₀, tspan, p)
    #temp checks
    @isdefined(locfmt) || include("debug_logger.jl")
    #with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    #    @inferred simplependulum´!(u´₀, u₀ , p, tspan[1]);
    #end
#    with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
#        @inferred bc!(resi, u₀, p, tspan[1]);
#    end
  #  println("\n\n\nStarting solve with initial u₀  = $u₀")
  #  prob = ODEProblem{true}(simplependulum´!, bvp.u0, bvp.tspan, p)
  #  sol = solve(prob, Tsit5(), dtmax = 0.05s);
  #  with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
  #      @debug "Initial solution, solution is " sol
  #  end
    println("\n\n\nStarting BC solve")
    with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
        solve(bvp, Shooting(Tsit5();nlsolve=DIMENSIONAL_NLSOLVE), dtmax = 0.05s);
    end




    #
   # @test_throws MethodError solve(bvp, Shooting(Tsit5()), dtmax = 0.05s);
#end


#=

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
    #        with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    #            solve(prob, Tsit5())
    #        end
    #    else
    #        solve(prob, Tsit5())
    #    end
    #    sol
    #end
    #sol1 = solve_guarded(simplependulum´!; debug = false)
    #sol1 = solve(ODEProblem(simplependulum2´!, u₀, tspan), Tsit5())
#end


    "Template for mutable residual value"
    residual = 0.0 .* u₀
    bc!(residual, sol1, nothing, nothing)
    @info "Residual with u₀ = $u₀ is " residual
    bvp1 = BVProblem(simplependulum´!, bc!, u₀, tspan)
    sol1 = solve(bvp1, Shooting(Tsit5()))
    sol1 = solve(bvp1, GeneralMIRK4(), dt = 0.05)
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
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
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
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
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


#g = 9.81;  L = 1.0; tspan = (0.0, π/2)


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
#=
function bc2!(residual, u, p, t)
    # the solution at the middle of the time span should be -π/2
    mid = searchsortedfirst(t, t[end] / 2)
    # residual is a DE solution.
    residual[1] = u[mid][1] + π/2 
    # the solution at the end of the time span should be π/2
    residual[2] = u[end][1] - π/2 
    @debug "bc2!" string(residual) maxlog = 2
    residual
end

@testset "BoundaryCondition problem, ArrayPartition, dimensionless" begin
    "Template for mutable local tuple at t = 0 - will be adapted to fit boundary conditions `bc1!`"
    u₀ = ArrayPartition([0.0], [π/2]) # θ, θ´
    @inferred simplependulum´!(u₀ ./ 0.01, u₀, p1, 0.01)
    bvp2 = BVProblem(simplependulum´!, bc2!, u₀, tspan1, p1)
    sol2 = solve(bvp2, Shooting(Tsit5()), dtmax = 0.05); # 0.008982
    sol =plotsol(sol2)
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
    #        with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    #            solve(prob, Tsit5())
    #        end
    #    else
    #        solve(prob, Tsit5())
    #    end
    #    sol
    #end
    #sol1 = solve_guarded(simplependulum´!; debug = false)
    #sol1 = solve(ODEProblem(simplependulum2´!, u₀, tspan), Tsit5())
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
    sol1 = solve(bvp1, Shooting(Tsit5()))
    sol1 = solve(bvp1, GeneralMIRK4(), dt = 0.05)
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
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
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
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
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
@testset "BoundaryCondition problem, ArrayPartition, dimensions" begin
    const tspan3 = (0.0, π/2)s
    const p3 = (981cm/s², 100.0cm) # g, L
    "Template for mutable residual value"
    residual0 = ArrayPartition([0.0], [0.0])
    "Template for mutable local tuple at t = 0 - will be adapted to fit boundary conditions `bc1!`"
    u₃= ArrayPartition([0.0], [π/2]s⁻¹) # θ, θ´
    @inferred simplependulum´!(u₃ ./ 0.01s, u₃, p3, 0.01s)
    bvp3 = BVProblem(simplependulum´!, bc2!, (u₃, residual0), tspan3, p3)
    #with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    #    solve(bvp3, Shooting(Tsit5();nlsolve=DIMENSIONAL_NLSOLVE), dtmax = 0.05s    )
    #end

end






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
=#
nothing
