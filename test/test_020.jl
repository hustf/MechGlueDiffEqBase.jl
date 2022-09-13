# Basis functionality for finite differentiation with different types of vector.

using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
#using MechanicalUnits.Unitfu: DimensionError
#import OrdinaryDiffEq.FiniteDiff
#using OrdinaryDiffEq.FiniteDiff: finite_difference_derivative, default_relstep
#using OrdinaryDiffEq.FiniteDiff: finite_difference_jacobian, JacobianCache, finite_difference_jacobian!
#import OrdinaryDiffEq.ArrayInterface
#import Base: Broadcast
#using Base.Broadcast: Broadcasted, result_style, combine_styles, DefaultArrayStyle, BroadcastStyle
#import MechGlueDiffEqBase.RecursiveArrayTools
#using MechGlueDiffEqBase.RecursiveArrayTools: ArrayPartitionStyle, unpack
@import_expand(cm, kg, s)

################################################
# F 'ϵ' for finite differentiation of quantities
################################################
@testset "ϵ for finite differentiation of quantities" begin
    #     compute_epsilon(Val(:central}, x::T, relstep::Real, absstep::Quantity, dir=nothing)
    @test compute_epsilon(Val(:central), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg
    #     compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, 
    #           dir = nothing) where {T<:Number, T1<:Real, D, U}
    @test compute_epsilon(Val(:forward), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg

    #     compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,Quantity{T, D, U}}=nothing, 
    #        ::Union{Nothing,Quantity{T, D, U}}=nothing, dir=nothing)
    @test compute_epsilon(Val(:complex), 1.0kg, nothing, nothing, nothing) ≈ 2.220446049250313e-16
end

##############################################
# G Univariate finite derivative of quantities
##############################################
# Derivative, univariate real argument with units, out-of-place, dimensionless derivative
@testset "Univariate finite derivative of quantities" begin
    @test let 
        f = x -> 2x
        x = 2.0cm
        finite_difference_derivative(f, x)
    end ≈ 2.0
    # Derivative, univariate real, out-of-place, derivative with unit
    @test let 
        f = x -> 2kg * x
        x = 2.0cm
        return finite_difference_derivative(f, x)
    end ≈ 2.0kg
end
nothing