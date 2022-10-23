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
x₀ = [1.0,0.0,0.0]
p = [0.04,3e7,1e4]
tspan = (0.0,1e5)
prob = ODEProblem(ODEFunction(rober),x₀,tspan,p)
sol1 = @time solve(prob,Rosenbrock23())  # 3.434848 seconds (8.37 M allocations: 670.196 MiB, 3.96% gc time, 99.95% compilation time)
using BenchmarkTools
@btime solve(prob,Rosenbrock23()); #   57.600 μs (442 allocations: 39.19 KiB)
# Rosenbrock23 constructed Jacobians internally, and stored its values, nested, at each solution point:
np = length(sol1)
@test size(sol1.k) == (np,)
@test size(sol1.k[1]) == (1,)
@test size(sol1.k[1][1]) == (3,)
@test sol1.k[1][1] isa Vector{Float64}

@test sol1.destats.nf == 243
# We supply the analytical Jacobian:
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
prob = ODEProblem(f,x₀,tspan,p)
sol2 = @btime solve(prob, Rosenbrock23()); #   47.300 μs (355 allocations: 32.45 KiB)
# We provided the analytical Jacobian. This was marginally faster, but values are stored in the 
# same manner as above:
np = length(sol2)
@test size(sol2.k) == (np,)
@test size(sol2.k[1]) == (1,)
@test size(sol2.k[1][1]) == (3,)
@test sol2.destats.nf == 183
@test sol2.k[1][1] isa ArrayPartition # x₀ is a normal vector, but this is the chosen 
                                        #structure when we supply an analytical Jacobian function
# Use ArrayPartition as a preparation for using physical dimensions. This is a good deal slower without further adaptions:
x₀ = convert_to_mixed(1.0,0.0,0.0)
p = convert_to_mixed(0.04,3e7,1e4)

prob = ODEProblem(rober,x₀, (0.0,1e5), p)
sol3 = @btime solve(prob,Rosenbrock23()); #   259.900 μs (985 allocations: 76.17 KiB)
prob = ODEProblem(f, x₀,(0.0,1e5), p)
sol4 = @btime solve(prob, Rosenbrock23()); #   47.300 μs (355 allocations: 32.45 KiB)
@test size(sol4.k) == (np,)
@test size(sol4.k[1]) == (1,)
@test size(sol4.k[1][1]) == (3,)
@test sol4.destats.nf == 183

# Write own Jacobian function using FiniteDiff!
function diy_jac(J,u,p,t)
    @debug string(J)
    throw("Ha! First call")
end
fnl = NonlinearFunction(f, Jp....??)
f = ODEFunction(rober, jac=diy_jac)
prob = ODEProblem(rober,x₀, (0.0,1e5), p)

# Provide a placeholder for the Jacobian (not working)
#=
Jp = jacobian_prototype_nan(x₀, rober(x₀, x₀, p, :t))
f = ODEFunction(rober)
prob = ODEProblem(f, x₀,(0.0,1e5), p)
solver = Rodas5(autodiff=Val(false), diff_type = Val(:central), concrete_jac = Val(true))
# That solver didn't quite fit. Let's try with a less complicated solver.
solver = Tsit5()
solve(prob, solver, wrap = Val(false))
=#





sol5 = @time solve(prob, solver) #  
np = length(sol5)
@test size(sol5.k) == (np,)
@test size(sol5.k[1]) == (1,)
@test size(sol5.k[1][1]) == (3,)
@test sol5.destats.nf == 183