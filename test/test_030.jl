# Test adaptions to NLSolversBase, NLSolve
using Test
using MechGlueDiffEqBase
using MechGlueDiffEqBase: nlsolve, converged, MixedContent
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

function f_2by2!(F, x)
    F[1] = (x[1] + 3) * (x[2]^3 -7 ) + 18
    F[2] = sin(x[2] * exp(x[1]) - 1)
    F
end
function f_2by2!(F::ArrayPartition{<:Quantity}, x) where N
    F[1] = (x[1] + 3kg) * (x[2]^3 - 7s^3) + 18kg∙s³
    F[2] = sin(x[2] * exp(x[1]/kg) /s -1 )s
    F
end

#############################
# 2 Newton trust region solve
#   Vectors
#############################

@testset "Newton trust region solve, Vectors" begin
    F = [10.0, 20.0]
    # Evaluate implicitly at known zero
    f_2by2!(F, [0,1])
    @test F == [0.0, 0.0]
    # OnceDifferentiable contains both f and df. We give prototype arguments.
    xprot = [NaN, NaN]
    df = OnceDifferentiable(f_2by2!, xprot, F; autodiff = :central)

    @test NLsolve.NewtonTrustRegionCache(df) isa NLsolve.AbstractSolverCache
    @test MechGlueDiffEqBase.LenNTRCache(df) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.
    r = nlsolve(df, [ -0.5, 1.4], method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test all(isapprox.(r.zero, [ 0, 1], atol = 1e-12))
    @test r.iterations == 4
end
#############################
# 2 Newton trust region solve
#  ArrayPartition
#############################
@testset "Newton trust region solve, ArrayPartition" begin
    F = convert_to_mixed([10.0, 20.0])
    # Evaluate implicitly at known zero
    f_2by2!(F, convert_to_mixed([0,1]))
    @test F == convert_to_mixed([0.0, 0.0])
    # df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
    xprot = convert_to_mixed([NaN, NaN])
    df = OnceDifferentiable(f_2by2!, xprot, F; autodiff = :central)
    @test NLsolve.NewtonTrustRegionCache(df) isa NLsolve.AbstractSolverCache
    @test MechGlueDiffEqBase.LenNTRCache(df) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.
    nlsolve(df, convert_to_mixed([ -0.5; 1.4]), method = :trust_region, autoscale = true)
    r = nlsolve(df, convert_to_mixed([ -0.5, 1.4]), method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test all(isapprox.(r.zero, [ 0, 1], atol = 1e-12))
    @test r.iterations == 4
end
#############################
# 3 Newton trust region cache
#  ArrayPartition dimensional
#############################
@testset "Newton trust region cache, ArrayPartition dimensional" begin

    F = convert_to_mixed([10.0kg∙s³, 20.0s])
    # Evaluate implicitly at known zero
    f_2by2!(F, convert_to_mixed([0kg,1s]))
    @test F == convert_to_mixed([0.0kg∙s³, 0.0s])
    # df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
    xprot = convert_to_mixed([NaN∙kg, NaN∙s])
    df = OnceDifferentiable(f_2by2!, xprot, F; autodiff = :central)
    @test_throws MethodError NLsolve.NewtonTrustRegionCache(df)
    @test MechGlueDiffEqBase.LenNTRCache(df) isa NLsolve.AbstractSolverCache
    # Start at a point outside zero, iterate arguments until function value is zero.
    nlsolve(df, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
    r = nlsolve(df, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
    @test converged(r)
    # Did we find the correct arguments?
    @test isapprox(r.zero[1], 0kg, atol = 1e-12kg)
    @test isapprox(r.zero[2], 1s, rtol = 1e-12)
    @test r.iterations == 4
end
nothing