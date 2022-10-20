# Test BoundaryValueDiffEq with dimensional residuals and dimensional NLsolve
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙, ustrip, unit

using OrdinaryDiffEq
#using MechanicalUnits: preferunits, upreferred


# Original, from https://tutorials.sciml.ai/stable/advanced/02-advanced_ODE_solving/
function rober(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁+k₃*y₂*y₃
    du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    du[3] =  k₂*y₂^2
    du
end

prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
sol1 = @time solve(prob,Rosenbrock23())  # 3.434848 seconds (8.37 M allocations: 670.196 MiB, 3.96% gc time, 99.95% compilation time)
using BenchmarkTools
@btime solve(prob,Rosenbrock23()); #   57.600 μs (442 allocations: 39.19 KiB)
# Rosenbrock23 constructed Jacobians internally, and stored its values, nested, at each solution point:
np = length(sol1)
@test size(sol1.k) == (np,)
@test size(sol1.k[1]) == (1,)
@test size(sol1.k[1][1]) == (3,)
@test sol1.destats.nf == 243
function rober_jac(J,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    J[1,1] = k₁ * -1
    J[2,1] = k₁
    J[3,1] = 0
    J[1,2] = y₃ * k₃
    J[2,2] = y₂ * k₂ * -2 + y₃ * k₃ * -1
    J[3,2] = y₂ * 2 * k₂
    J[1,3] = k₃ * y₂
    J[2,3] = k₃ * y₂ * -1
    J[3,3] = 0
    nothing
  end
f = ODEFunction(rober, jac=rober_jac)
prob_jac = ODEProblem(f,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
sol2 = @btime solve(prob_jac, Rosenbrock23()); #   47.300 μs (355 allocations: 32.45 KiB)
# We provided the analytical Jacobian, but values are stored in the same manner as above:
np = length(sol2)
@test size(sol2.k) == (np,)
@test size(sol2.k[1]) == (1,)
@test size(sol2.k[1][1]) == (3,)
@test sol2.destats.nf == 183

# Use ArrayPartition as a preparation for using physical dimensions. This is a good deal slower without further adaptions.
prob = ODEProblem(rober,convert_to_mixed(1.0,0.0,0.0), (0.0,1e5), convert_to_mixed(0.04,3e7,1e4))
@btime solve(prob,Rosenbrock23()); #   259.900 μs (985 allocations: 76.17 KiB)
prob_jac = ODEProblem(f,convert_to_mixed(1.0,0.0,0.0),(0.0,1e5),convert_to_mixed(0.04,3e7,1e4))
sol_jac = @btime solve(prob_jac, Rosenbrock23()); #   47.300 μs (355 allocations: 32.45 KiB)

# Provide a placeholder for the Jacobian
Jp = jacobian_prototype_nan(x::MixedCandidate, vecfx::MixedCandidate)