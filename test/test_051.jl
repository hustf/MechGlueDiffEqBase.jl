# Stiff ODE exploration. 
# Currently, we found no way to implement stiff algorithms 
# with units. SciMLBase is in a bit of flux, too. 
# So avoid stiff equations so far. If equations are not stiff,
# ample warning about instability is given.
# Later, we would perhaps drop units in the left preconditioner,
# then apply it again in the "right preconditioner". There's a composite type 
# for this, so we could combine with other types of solvers as selected by DifferentialEquations.
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙, ustrip, unit
using OrdinaryDiffEq
include(joinpath(@__DIR__, "debug_logger.jl"))
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
sol1 = @time solve(prob,Rosenbrock23()) # 3.569596 seconds (8.63 M allocations: 665.816 MiB, 4.86% gc time, 99.95% compilation time)
using BenchmarkTools
@btime solve(prob,Rosenbrock23()); #  52.100 μs (428 allocations: 38.09 KiB)
np = length(sol1)
@test sol1.k[end][1] isa Vector{Float64} # k field undocumented
# See https://diffeq.sciml.ai/stable/basics/solution/#Special-Fields
# Rosenbrock23 constructed Jacobians internally:
@test sol1.destats.nf == 243
@test sol1.destats.njacs == 60 
@test sol1.destats.nsolve == 180

using DifferentialEquations
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    @enter solve(prob)  #  Debug: LinearSolve init
                 # alg = GenericLUFactorization{LinearAlgebra.RowMaximum}(LinearAlgebra.RowMaximum())
                 # typeof(prob) = LinearProblem{Vector{Float64}, true, Matrix{Float64}, Vector{Float64}, SciMLBase.NullParameters, Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{()"#undef", Tuple{}}}}
end;












# We define the analytical Jacobian:
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
    @debug "rober_jac " s_summary(J) string(J) maxlog = 2
    nothing
end
# We supply the Jacobian function (but not its form)
f = ODEFunction(rober, jac=rober_jac)
prob = ODEProblem(f, x₀, tspan, p)
sol2 = @btime solve(prob, Rosenbrock23()); #   63.800 μs (475 allocations: 38.91 KiB)
# We provided the analytical Jacobian. This was marginally faster (when @debug is commented out).
np = length(sol2)
@test sol2.k[end][1] isa Vector{Float64} # k field undocumented
@test sol2.destats.nf == 183
@test sol2.destats.njacs == 60 
@test sol2.destats.nsolve == 180
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, Rosenbrock23())                                                                                                  
end;

# Use ArrayPartition as a preparation for using physical dimensions. This is a good deal slower without further adaptions:
x₀ = convert_to_mixed(x₀)
p = convert_to_mixed(p)
f = ODEFunction(rober, jac=rober_jac)
prob = ODEProblem(f, x₀, tspan, p)
sol3 = @btime solve(prob, Rosenbrock23()); #  262.000 μs (1077 allocations: 80.69 KiB)
@test is_vector_mutable_stable(sol3.k[end][1])
@test sol3.destats.nf == 183
@test sol3.destats.njacs == 60 
@test sol3.destats.nsolve == 180
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, Rosenbrock23())                                                                                                  
end;

# Use ArrayPartition as a preparation for using physical dimensions. No Jacobian function.
f = with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
 ODEFunction(rober)
end

prob = with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
    ODEProblem(f, x₀, tspan, p)
end
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, Rodas4P())                                                                                                  
end;
solve(prob, Rodas4P())   








using ModelingToolkit

f_init = ODEFunction(rober)
prob = ODEProblem(f_init, x₀, tspan, p)
de_init = modelingtoolkitize(prob)
jacexpr = ModelingToolkit.generate_jacobian(de_init)[2] # Second is in-place
jacexpr.args[2].args[1] = :(@debug "auto_symb_jac " s_summary(ˍ₋out) string(ˍ₋out) maxlog = 2)
jac = eval(jacexpr)
Jp = jacobian_prototype_zero(x₀, x₀)
f = ODEFunction(rober; jac, jac_prototype = Jp, sparsity = nothing)
prob = ODEProblem(f, x₀, tspan, p)
de = modelingtoolkitize(prob)
# TODO: specify the linear solver to use!
# https://diffeq.sciml.ai/stable/features/linear_nonlinear/#Linear-Solvers:-linsolve-Specification
alg = Rosenbrock23(linsolve = nothing)
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, alg)                                                                                                  
end;








x₀ = convert_to_array(x₀)
p = convert_to_array(p)
prob = ODEProblem(f, x₀, tspan, p)
sol4 = @btime solve(prob, Rosenbrock23()); # 64.400 μs (475 allocations: 38.91 KiB)
# We provided the automatically found, analytical Jacobian. This was faster than 
# the manually typed 'rober_jac'.
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, Rosenbrock23())                                                                                                  
end;
np = length(sol4)
@test sol4.k[end][1] isa Vector{Float64} # k field undocumented
@test sol4.destats.nf == 183
@test sol4.destats.njacs == 60 
@test sol4.destats.nsolve == 180


x₀ = convert_to_mixed(x₀)
p = convert_to_mixed(p)
prob = ODEProblem(rober, x₀, tspan, p)
de = modelingtoolkitize(prob)

Jout = x₀

with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    jac(Jout, x₀, p, :t)
end
Jp = jacobian_prototype_zero(x₀, Jout)
DiffEqBase.WOperator(Jp)



jacexpr = ModelingToolkit.generate_jacobian(de)[2] # Second is in-place
jacexpr.args[2].args[1] = :(@debug "auto_symb_jac " s_summary(ˍ₋out) string(ˍ₋out) maxlog = 2)
jac = eval(jacexpr)


f = ODEFunction(f; jac = jac, jac_prototype = Jp)
prob = ODEProblem(f, x₀, tspan, p)

with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do                                         
    solve(prob, Rosenbrock23())                                                                                                  
end;

de = modelingtoolkitize(prob)
jac = calculate_jacobian(de)
jac_expr = generate_jacobian(de)[2]

# TODO: Make f.jac_prototype a DiffEqBase.AbstractDiffEqLinearOperator? E.g. WOperator

DiffEqBase.WOperator