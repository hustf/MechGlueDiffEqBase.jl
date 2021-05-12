# Differential equations with mixed units using ArrayPartition and no specified algorithm
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙, g
@import_expand(km, N, s, m, km, kg, °, inch)
using OrdinaryDiffEq: ODEProblem
using DifferentialEquations: solve, Tsit5
using Logging: disable_logging, LogLevel, Debug, with_logger
import Logging

using Test: @inferred

# Constants we don't think we'll change ever
const x₀ = 0.0km
const y₀ = 0.0km
const ø = 15inch
const ρ = 1.225kg/m³

# Calculated constants
const Aₚᵣ = π/4 * ø^2
# Constants that we define as functions, because we may
# want to modify them later in the same scripting session.
α₀()  = 30°
v₀()  = 1050m/s

mₚ()   = 495kg
C_s() = 0.4
x´₀() = v₀() * cos(α₀())
y´₀() = v₀() * sin(α₀())

# Local tuple initial condition
u₀ = ArrayPartition([x₀, y₀], [x´₀(), y´₀()])

# Functions
v(vx, vy) = √(vx^2 + vy^2)
R(vx, vy) = 0.5∙ρ∙C_s()∙Aₚᵣ∙v(vx, vy)^2
α(vx, vy) = atan(vy, vx)
Rx(vx, vy) = R(vx, vy) * cos(α(vx, vy))
Ry(vx, vy) = R(vx, vy) * sin(α(vx, vy))


# Local tuple, i.e. the interesting degrees of freedom
# and their derivatives
function  Γ!(u´, u, p, t)
    (x, y), (x´, y´) = packout(u)
    @debug "Γ"    x    y    x´   y´   maxlog = 3
    # Calculate the acceleration for this step
    x´´ =     -Rx(x´, y´) / mₚ()
    y´´ = -1g -Ry(x´, y´) / mₚ()
    @debug "Γ Acceleration"    x´´|> g    y´´|>g    maxlog = 3
    packin!(u´, x´, y´, x´´, y´´)
end

packout(u) = u.x[1], u.x[2]
function packin!(u´, x´, y´, x´´, y´´)
    # We access the values through its field name, x (unrelated to the specific problem)
    u´.x[1][1] = x´;  u´.x[1][2] = y´
    u´.x[2][1] = x´´; u´.x[2][2] = y´´
    return u´
end

function solve_guarded(u₀; alg = Tsit5(), debug=false)
    # Test the functions
    @inferred packout(u₀)
    @inferred packout(0.01u₀/s)
    @inferred Γ!(u₀/s,u₀, nothing, nothing)
    !debug  && disable_logging(Debug)
    prob = ODEProblem( Γ!,u₀,(0.0,60)s)
    sol = if debug
        with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
            solve(prob, Tsit5())
        end
    else
        solve(prob, Tsit5())
    end
    # Re-enable
    disable_logging(LogLevel(Debug-1))
    sol
end
@time sol = solve_guarded(u₀, debug=true);
nothing