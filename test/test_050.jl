# LinearSolvers use
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
include(joinpath(@__DIR__, "debug_logger.jl"))
using MechGlueDiffEqBase
using MechGlueDiffEqBase: determinant_dimension, determinant, mul! # Should probably export
using MechanicalUnits: @import_expand, ∙, ustrip, unit
@import_expand kN mm
using MechanicalUnits: 𝐋, 𝐋², 𝐌³, 𝐓, NoDims, dimension
using LinearSolve

n = 4
A = rand(n,n)
b = 1.0:n
prob = LinearProblem(A, b)
@test typeof(prob.A) == typeof(A)
@test typeof(prob.b) == typeof(b)
@test prob.A == A
@test prob.b == b
@test prob.p == SciMLBase.NullParameters()
@test length(prob.kwargs) == 0
linsolve = with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    init(prob)
end
alg = LinearSolve.defaultalg(A, b, linsolve.assumptions)
u0 = zero.(b)
Pl = LinearSolve.IterativeSolvers.Identity()
Pr = LinearSolve.IterativeSolvers.Identity()
maxiters = 4
abstol = 1.4901161193847656e-8
reltol = 1.4901161193847656e-8
verbose = false
assumptions = LinearSolve.OperatorAssumptions{Nothing}()
cacheval = LinearSolve.init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol, reltol, verbose,
    assumptions)
cache =  init(prob, alg)
sol1 = solve(cache)
@test all(A * sol1.u .≈ b)


# 
A = convert_to_mixed([   4.8kN∙mm⁻¹ -2400.0kN        0.0kN∙mm⁻¹;
    -2400.0kN          1.6e6mm∙kN   0.0kN;
    0.0kN∙mm⁻¹     0.0kN      200.0kN∙mm⁻¹])
@test determinant_dimension(A) == 𝐋²∙𝐌³∙𝐓^-6
b = convert_to_mixed([1kN, 2.0kN∙mm, 3kN])

u = A \ b
@test all(A * u .≈ b)

# Make preconditioners Pl and Pr to drop dimensions. We want 
# dimensionless expressions after preconditioning. But Pl and Pr
# do not seem to be taken into account in the intitialization phase
# of the default solver....
Pl = convert_to_mixed([1.0kN     0.0kN     0.0kN
                       0.0kN∙mm  1.0kN∙mm  0.0kN∙mm
                       0.0kN     0.0kN     1.0kN])
determinant_dimension(Pl)
Pli = inv(Pl) 
# The inplace 'mul' function has better error messages than 'Pli * b' for detailed dimensions.
bnd = zero.(ustrip.(b))
mul!(bnd, Pli,  b)
@test ustrip.(b) == bnd
@test all(isapprox.(Pli * (A * u - b),  0.0, atol = 1e-12))


prob = LinearProblem(convert_to_array(A), convert_to_array(b); u0 = u)
linsolve = with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    init(prob,  Pl=convert_to_array(Pl), Pr = convert_to_array(Pr))
end

with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    solve(linsolve,IterativeSolversJL_GMRES())
end



#=















# Right preconditioner which drops dimensions.
Pr = convert_to_mixed([ 1.0kN⁻¹  0.0kN⁻¹∙mm⁻¹  0.0kN⁻¹
                        0.0kN⁻¹  1.0kN⁻¹∙mm⁻¹  0.0kN⁻¹ 
                        0.0kN⁻¹  0.0kN⁻¹∙mm⁻¹  1.0kN⁻¹])
Pri = inv(Pr)
#Pri = convert_to_mixed([ 1.0kN  0.0kN  0.0kN
#                         0.0kN∙mm  1.0kN∙mm  0.0kN∙mm
#                         0.0kN  0.0kN  1.0kN])
mul!(bnd, Pl, b )


b
Pr * c

y = convert_to_mixed(u)
convert_to_mixed(A) * y



LinearSolve.init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol, reltol, verbose,
    assumptions)
prob = LinearProblem(A, b)

linsolve = with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    init(prob,  Pl=Pl, Pr = Pr)
end

Must perhaps dispatch on Residuals.....
solve(prob,IterativeSolversJL_GMRES(), Pl=Pl, Pr = Pr)
=#



