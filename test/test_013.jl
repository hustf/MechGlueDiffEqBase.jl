# Differential equations with mixed units.
# Re-formulation of test_005.jl; the solution is vector-like
# instead of matrix-like.

using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙, g
@import_expand(km, s, m, kg,  inch, °)
using OrdinaryDiffEq: ODEProblem
using DifferentialEquations: solve, Tsit5
using Logging: disable_logging, LogLevel, Debug, with_logger
import Logging
using Test
using Test: @inferred

@testset "Mixed mutable vector argument diffeq" begin
    # Constants we don't think we'll change ever
    x₀ = 0.0km
    y₀ = 0.0km
    ø = 15inch
    ρ = 1.225kg/m³

    # Calculated constants
    Aₚᵣ = π/4 * ø^2
    # Constants that we define as functions, because we may
    # want to modify them later in the same scripting session.
    α₀()  = 30°
    v₀()  = 1050m/s

    mₚ()   = 495kg
    C_s() = 0.4
    x´₀() = v₀() * cos(α₀())
    y´₀() = v₀() * sin(α₀())

    # Local tuple initial condition.
    u₀ = convert_to_mixed(x₀, y₀, x´₀(), y´₀())


    # Functions
    v(vx, vy) = √(vx^2 + vy^2)
    R(vx, vy) = 0.5∙ρ∙C_s()∙Aₚᵣ∙v(vx, vy)^2
    α(vx, vy) = atan(vy, vx)
    Rx(vx, vy) = R(vx, vy) * cos(α(vx, vy))
    Ry(vx, vy) = R(vx, vy) * sin(α(vx, vy))


    # Local tuple, i.e. the interesting degrees of freedom
    # and their derivatives
    function  Γ!(u´, u, p, t)
        x, y, x´, y´ = u
        @debug "Γ!"    x    y    x´   y´   maxlog = 2
        # Calculate the acceleration for this step
        x´´ =     -Rx(x´, y´) / mₚ()
        y´´ = -1g -Ry(x´, y´) / mₚ()
        @debug "! Acceleration"    x´´|> g    y´´|>g    maxlog = 2
        u´ .= x´, y´, x´´, y´´
        u´
    end

    function solve_guarded(u₀; alg = Tsit5(), debug=false)
        # Test the functions
        @inferred Γ!(u₀/s,u₀, nothing, nothing)
        !debug  && disable_logging(Debug)
        prob = ODEProblem( Γ!,u₀,(0.0,60)s)
        sol = if debug
            with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
                solve(prob, alg)
            end
        else
            solve(prob, alg)
        end
        # Re-enable
        disable_logging(LogLevel(Debug-1))
        sol
    end
    sol = solve_guarded(u₀, debug=true);
    @test sol(60s)[1] > 23km
end

nothing
