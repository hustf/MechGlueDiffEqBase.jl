# Finite difference extended function
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
using OrdinaryDiffEq.FiniteDiff: finite_difference_derivative, default_relstep
using OrdinaryDiffEq.FiniteDiff: finite_difference_jacobian, JacobianCache, finite_difference_jacobian!
import OrdinaryDiffEq.ArrayInterface
@import_expand(cm, kg, s)
####################
# Epsilon with units
####################
#     compute_epsilon(Val(:central}, x::T, relstep::Real, absstep::Quantity, dir=nothing)
@test compute_epsilon(Val(:central), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg
#     compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, 
#           dir = nothing) where {T<:Number, T1<:Real, D, U}
@test compute_epsilon(Val(:forward), 1.0kg, 0.001, 0.001kg, nothing) === 0.001kg

#     compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,Quantity{T, D, U}}=nothing, 
#        ::Union{Nothing,Quantity{T, D, U}}=nothing, dir=nothing)
@test compute_epsilon(Val(:complex), 1.0kg, nothing, nothing, nothing) ≈ 2.220446049250313e-16

###################################
# Matrix-like nested ArrayPartition
# [2, 2] = .x[2][2]
###################################
@test let
   x = ArrayPartition(1,2)
   f = x -> ArrayPartition(x[1] + 2x[2], 3x[1] + 4x[2])
   jacobian_prototype_zero(x, f(x))
end == matrixlike_arraypartition([0.0 0.0; 0.0 0.0])
@test let
    x = ArrayPartition(1,2)
    f = x -> ArrayPartition(x[1] + 2x[2], 3x[1] + 4x[2])
    similar_matrix(jacobian_prototype_zero(x, f(x)))
 end == [0.0 0.0
         0.0 0.0]
# With units
@test let
    x = ArrayPartition(1s, 2kg)
    f = x -> ArrayPartition(x[1]cm + 2s∙cm/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2])
    apa = jacobian_prototype_zero(x, f(x))
    matjac = [0.0cm    0.0cm∙s∙kg⁻¹
              0.0cm∙s  0.0cm∙s²∙kg⁻¹]
    similar_matrix(apa) == matjac || return false
    string(apa) == "MatrixLike ArrayPartition:[0.0cm 0.0cm∙s∙kg⁻¹; 0.0cm∙s 0.0cm∙s²∙kg⁻¹]"
end

 @test let
    x = ArrayPartition(1s, 2kg)
    f = x -> ArrayPartition(x[1]cm + 2s∙cm/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2])
    jacobian_prototype_zero(x, f(x))
 end == matrixlike_arraypartition([0.0cm   0.0cm∙s∙kg⁻¹
                                   0.0cm∙s  0.0cm∙s²∙kg⁻¹])


#######################################
# 3 Univariate finite difference, units
#######################################
# Derivative, univariate real argument with units, out-of-place, dimensionless derivative
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
####################################################
# 4 Dimensionless Jacobian, not using ArrayPartition
####################################################
# a) Jacobian cache, "f(x, y)-> z ", not using ArrayPartition, dimensionless
@test let 
    f = x -> x[1]^2 - 2x[1]∙x[2] + x[2]^3 # f = (x, y) -> x^2 - 2x∙y + y^3
    x = [1.0, 2.0]
    fx = f(x)   # 5.0
    fdtype     = Val(:forward)
    returntype = eltype(x)
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == [1.0, 2.0] || return false
    cache.fx == fx || return false
    cache.fx1 == 5.0 || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity == nothing
end 
# b) Jacobian cache, "f(x, y)-> [z] ", not using ArrayPartition, dimensionless
@test let 
    f = x -> x[1]^2 - 2x[1]∙x[2] + x[2]^3 # f = (x, y) -> x^2 - 2x∙y + y^3
    x = [1.0, 2.0]
    fx = [f(x)]   # [5.0]
    fdtype     = Val(:forward)
    returntype = eltype(f(x))
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == [1.0, 2.0] || return false
    cache.fx == fx || return false
    cache.fx1 == [5.0] || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# c) Jacobian cache, "f(x, y)-> [z] ", not using ArrayPartition, dimensionless, complex
@test let 
    f = x -> x[1]^2 - 2x[1]∙x[2] + x[2]^3 # f = (x, y) -> x^2 - 2x∙y + y^3
    x = [1.0, 2.0]
    fx = [f(x)]   # [5.0]
    fdtype     = Val(:complex)
    returntype = eltype(f(x))
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == [0.0 + 0.0im, 0.0 + 0.0im] || return false
    cache.fx == [0.0 + 0.0im] || return false
    cache.fx1 === nothing || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# d) finite difference Jacobian matrix, "f(x, y) -> z", not using ArrayPartition, dimensionless.
# Analytical result: f(x) = x[1]^2 - 2x[1]∙x[2] + x[2]^3
# =>                 J = [δf₁/ δx₁     δf₁/ δx₂    
#                         δf₂/ δx₁     δf₂/ δx₂]
# =                  J = [δf₁/ δx₁     δf₁/ δx₂    
#                         NaN          NaN]
# =>                 J = [δf/ δx₁      δf/ δx₂]
# =>     J([1.0, 2.0]) = [2∙1-2∙2      -2∙1+3∙2^2] = [-2    10]
@test isapprox(let 
    f = x -> x[1]^2 - 2x[1]∙x[2] + x[2]^3 # f = (x, y) -> x^2 - 2x∙y + y^3
    x = [1.0, 2.0]
    finite_difference_jacobian(f, x)
end, [-2.0 10.0]; atol = 1e-6)
# e) finite difference Jacobian matrix, "f(x, y) -> (u,v)", not using ArrayPartition, dimensionless.
# Analytical result: f(x) = [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
# =>                 J = [δf₁/ δx₁     δf₁/ δx₂    
#                         δf₂/ δx₁     δf₂/ δx₂]
# =>    J([1.0, 2.0]) =  [-2    10
#                          0    2]
isapprox(let 
    f = x -> [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
    x = [1.0, 2.0]
    finite_difference_jacobian(f, x)
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)
# f) fdtype Val(:complex)
@test isapprox(let 
    f = x -> [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
    x = [1.0, 2.0]
    fdtype     = Val(:complex)
    finite_difference_jacobian(f, x, fdtype)
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)
# g) fdtype Val(:central)
@test isapprox(let 
    f = x -> [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
    x = [1.0, 2.0]
    fdtype     = Val(:central)
    finite_difference_jacobian(f, x, fdtype)
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)

################################################
# 5 Dimensionless Jacobian, using ArrayPartition
################################################
# a) Jacobian cache, "f(x, y)-> z ", using ArrayPartition, dimensionless
@test let 
    f = x -> x[1]^2 - 2x[1]∙x[2] + x[2]^3 # f = (x, y) -> x^2 - 2x∙y + y^3
    x = ArrayPartition(1.0, 2.0)
    fx = f(x)   # 5.0
    fdtype     = Val(:forward)
    returntype = eltype(f(x))
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == ArrayPartition(1.0, 2.0) || return false
    cache.fx == fx || return false
    cache.fx1 == 5.0 || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# b) Jacobian cache, "f(x, y)-> ArrayPartition(z) ", using ArrayPartition, dimensionless
@test let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3) # f = (x, y) -> x^2 - 2x∙y + y^3
    x = ArrayPartition(1.0, 2.0)
    fx = f(x)   # ArrayPartition(5.0)
    fdtype     = Val(:forward)
    returntype = eltype(f(x))
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == ArrayPartition(1.0, 2.0) || return false
    cache.fx == fx || return false
    cache.fx1 == ArrayPartition(5.0) || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end
# c) Jacobian cache, "f(x, y)-> [z] ", using ArrayPartition, dimensionless, complex
@test let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3)
    x = ArrayPartition(1.0, 2.0)
    fx = f(x)   # ArrayPartition(5.0)
    fdtype     = Val(:complex)
    returntype = eltype(f(x))
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == ArrayPartition(0.0 + 0.0im, 0.0 + 0.0im) || return false
    cache.fx == ArrayPartition(0.0 + 0.0im) || return false
    cache.fx1 === nothing || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# d) finite difference Jacobian, "f(x, y) -> z", using ArrayPartition, dimensionless.
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3)
    x = ArrayPartition(1.0, 2.0)
    row_vector(finite_difference_jacobian(f, x))
end, [-2.0 10.0] ; atol = 1e-6)
# e) finite difference Jacobian matrix, "f(x, y) -> (u,v)", using ArrayPartition, dimensionless.
# Analytical result: f(x) = [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
# =>                 J = [δf₁/ δx₁     δf₁/ δx₂    
#                         δf₂/ δx₁     δf₂/ δx₂]
# =>    J([1.0, 2.0]) =  [-2    10
#                          0    2]
isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    similar_matrix(finite_difference_jacobian(f, x))
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)
# f) fdtype Val(:complex), using ArrayPartition, dimensionless.
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    fdtype     = Val(:complex)
    similar_matrix(finite_difference_jacobian(f, x, fdtype))
end, [-2.0 10.0 
       0.0 2.0]; atol = 1e-6)
# g) fdtype Val(:central), using ArrayPartition, dimensionless.
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    fdtype     = Val(:central)
    similar_matrix(finite_difference_jacobian(f, x, fdtype))
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)

#########################################
# 6 With dimensions, using ArrayPartition
#########################################
# a) Jacobian cache, "f(x, y)-> z ", using ArrayPartition, dimensions
@test let 
    f = x -> x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    fx = f(x)   # 5.0kg
    fdtype     = Val(:forward)
    returntype = eltype(fx)
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == ArrayPartition(1.0s⁻¹, 2.0s⁻²) || return false
    cache.fx == fx || return false
    cache.fx1 == 5.0kg || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# b) Jacobian cache, "f(x, y)-> ArrayPartition(z) ", using ArrayPartition, dimensions
@test let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6)
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    fx = f(x)   # ArrayPartition(5.0kg)
    fdtype     = Val(:forward)
    returntype = eltype(fx)
    cache = JacobianCache(x, fx, fdtype, returntype)
    cache.x1 == ArrayPartition(1.0s⁻¹, 2.0s⁻²) || return false
    cache.fx == fx || return false
    cache.fx1 == ArrayPartition(5.0kg) || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# c) Jacobian cache, "f(x, y)-> [z] ",  using ArrayPartition, dimensions, complex
@test let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6)
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    xcomp = complex.(zero(x))
    fx = f(xcomp)   # ArrayPartition((0.0 + 0.0im)kg)
    fdtype     = Val(:complex)
    returntype = MechGlueDiffEqBase.numtype(eltype(f(x)))
    cache = JacobianCache(xcomp, fx, fdtype, returntype)
    cache.x1 == ArrayPartition((0.0 + 0.0im)s⁻¹, (0.0 + 0.0im)s⁻²) || return false
    cache.fx == ArrayPartition((0.0 + 0.0im)kg) || return false
    cache.fx1 === nothing || return false
    cache.colorvec == 1:2 || return false
    cache.sparsity === nothing
end 
# d) finite difference Jacobian, "f(x, y) -> z", using ArrayPartition, dimensions
# Ref. test 5d)
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6)
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    row_vector(finite_difference_jacobian(f, x))
end ./ [-2.0kg∙s  10.0kg∙s²], 
                              [1.0 1.0]; atol = 1e-6)

# e) finite difference Jacobian matrix, "f(x, y) -> (u,v)", using ArrayPartition, dimensions.
# Ref. test 5e)
isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6, 2x[2]cm∙s²)
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    similar_matrix(finite_difference_jacobian(f, x))
end ./ [-2.0kg∙s  10.0kg∙s²
      0.0cm∙s   2.0cm∙s²], 
                           [1.0  1.0
                            NaN  1.0]; atol = 1e-6, nans = true)
# f) fdtype Val(:complex), using ArrayPartition, dimensions.
# Ref. test 5f)
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6, 2x[2]cm∙s²)
    x = ArrayPartition(1.0s⁻¹, 2.0s⁻²)
    fdtype     = Val(:complex)
    similar_matrix(finite_difference_jacobian(f, x, fdtype))
end ./ [-2.0kg∙s  10.0kg∙s²
        0.0cm∙s   2.0cm∙s²], 
                            [1.0  1.0
                             NaN  1.0]; atol = 1e-6, nans = true)
# g) fdtype Val(:central), using ArrayPartition, dimensions.
# Ref. test 5g)
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6, 2x[2]cm∙s²)
    x = ArrayPartition(1.0s⁻¹, 2.0s⁻²)
    fdtype     = Val(:central)
    similar_matrix(finite_difference_jacobian(f, x, fdtype))
end ./ [-2.0kg∙s  10.0kg∙s²
         0.0cm∙s   2.0cm∙s²], 
                            [1.0  1.0
                            NaN  1.0]; atol = 1e-6, nans = true)

nothing