# Basis functionality for finite differentiation with different types of vector.

using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
@import_expand(cm, kg, s)

################################################
# F 'ϵ' for finite differentiation of quantities
################################################
@testset "ϵ for finite differentiation of quantities" begin
    #     compute_epsilon(Val(:central}, x::T, relstep::Real, absstep::Quantity, dir = nothing)
    @test compute_epsilon(Val(:central), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg
    #     compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U},
    #           dir = nothing) where {T<:Number, T1<:Real, D, U}
    @test compute_epsilon(Val(:forward), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg
    #     compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,Quantity{T, D, U}} = nothing,
    #        ::Union{Nothing,Quantity{T, D, U}} = nothing, dir = nothing)
    @test compute_epsilon(Val(:complex), 1.0kg, nothing, nothing, nothing) ≈ 2.220446049250313e-16kg
    #
    ## Dimensionless absstep (which is "wrong")
    @test compute_epsilon(Val(:central), 1.0kg, 0.001, 0.001, nothing) === 0.001kg
    @test compute_epsilon(Val(:forward), 1.0kg, 0.001, 0.001, nothing) === 0.001kg
    @test compute_epsilon(Val(:complex), 1.0kg, 0.001, 0.001, nothing) ≈ 2.220446049250313e-16kg

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
##############################################
# H Univariate finite derivative of quantities
#   Various fdtype
##############################################
@testset "Derivative single point f : R -> R tests" begin
    err_func(a, b) = maximum(abs.(a - b))
    @test err_func(finite_difference_derivative(x -> x^3, -1.0, Val{:forward}), 3) < 1e-7
    @test err_func(finite_difference_derivative(x -> x^3, -1.0, Val{:central}), 3) < 1e-10
    @test err_func(finite_difference_derivative(x -> x^3, -1.0, Val{:complex}), 3) < 1e-15

    @test err_func(finite_difference_derivative(x -> x^3, -1.0kg, Val{:forward}), 3kg²) < 1e-7kg²
    @test err_func(finite_difference_derivative(x -> x^3, -1.0kg, Val{:central}), 3kg²) < 1e-10kg²
    @test err_func(finite_difference_derivative(x -> x^3, -1.0kg, Val{:complex}), 3kg²) < 1e-15kg²
end
nothing