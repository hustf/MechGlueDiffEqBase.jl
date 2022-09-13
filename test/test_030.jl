# Test adaptions to NLSolversBase, NLSolve
using Test
using MechGlueDiffEqBase
using MechGlueDiffEqBase: nlsolve, converged
using MechanicalUnits: @import_expand, ∙
using Logging
import NLsolve

@import_expand(cm, kg, s)
######################
# 1 Utilities, NLsolve
######################
@testset "Utilities, NLsolve" begin
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1,NaN])
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1,Inf])
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
        convert_to_mixed([1,Inf]))
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
        convert_to_mixed([1s,Inf]))
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1 2;Inf 4])
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
        convert_to_mixed([1 2;Inf 4]))
    @test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
        convert_to_mixed([1s 2;Inf 4]))
end
#############################
# 2 Newton trust region solve
#   Vectors
#############################
function f_2by2!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end
@testset "Newton trust region solve, Vectors" begin
    F1 = [10.0, 20.0]
    # Evaluate implicitly at known zero
    f_2by2!(F1, [0,1]) 
    @test F1 == [0.0, 0.0]
    # OnceDifferentiable contains both f and df. We give prototype arguments.
    xprot1 = [NaN, NaN]
    df1 = OnceDifferentiable(f_2by2!, xprot1, F1; autodiff = :central)

    @test NLsolve.NewtonTrustRegionCache(df1) isa NLsolve.AbstractSolverCache
    @test MechGlueDiffEqBase.LenNTRCache(df1) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.
    r = nlsolve(df1, [ -0.5, 1.4], method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test r.zero ≈ [ 0, 1]
    @test r.iterations == 4
end
#############################
# 2 Newton trust region solve
#  ArrayPartition
#############################
@testset "Newton trust region solve, ArrayPartition" begin
    F2 = convert_to_mixed([10.0, 20.0])
    # Evaluate implicitly at known zero
    f_2by2!(F2, convert_to_mixed([0,1])) 
    @test F2 == convert_to_mixed([0.0, 0.0])
    # df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
    xprot2 = convert_to_mixed([NaN, NaN])
    df2 = OnceDifferentiable(f_2by2!, xprot2, F2; autodiff = :central)
    @test NLsolve.NewtonTrustRegionCache(df2) isa NLsolve.AbstractSolverCache
    @test MechGlueDiffEqBase.LenNTRCache(df2) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.

    with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
        nlsolve(df2, convert_to_mixed([ -0.5; 1.4]), method = :trust_region, autoscale = true)
    end

    r = nlsolve(df2, convert_to_mixed([ -0.5, 1.4]), method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test r.zero ≈ [ 0, 1]
    @test r.iterations == 4
end
#############################
# 3 Newton trust region cache
#  ArrayPartition dimensional
#############################
@testset "Newton trust region cache, ArrayPartition dimensional" begin
    function f_2by2a!(F, x)
        F[1] = (x[1]+3kg)*(x[2]^3-7s^3)+18kg∙s³      
        F[2] = sin(x[2]*exp(x[1]/kg)/s-1)s
    end
    F3 = convert_to_mixed([10.0kg∙s³, 20.0s])
    # Evaluate implicitly at known zero
    f_2by2a!(F3, convert_to_mixed([0kg,1s])) 
    @test F3 == convert_to_mixed([0.0kg∙s³, 0.0s])
    # df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
    xprot3 = convert_to_mixed([NaN∙kg, NaN∙s])
    df3 = OnceDifferentiable(f_2by2a!, xprot3, F3; autodiff = :central)
    @test_throws MethodError NLsolve.NewtonTrustRegionCache(df3)
    @test MechGlueDiffEqBase.LenNTRCache(df3) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.

    #with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do 
    #nlsolve(df3, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
    #end

    r = nlsolve(df3, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test r.zero ≈ [ 0, 1]
    @test r.iterations == 4
end
nothing