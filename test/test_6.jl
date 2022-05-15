# Finite difference extended function
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
using MechanicalUnits.Unitfu: DimensionError
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


#######################################
# Univariate finite difference, units
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




#############################################
# Dispatch on square mutable matrix. 
# We want these to be inferrable,
# which the standard matrix type, e.g. 
#     Any[1 2; 3 4] 
# is mostly not. For inferrability, we implement
# as nested ArrayPartition. For mutability,
# the innermost type is a one-element vector.
#############################################
#=
let 
    M0 = ArrayPartition()
    M1 = ArrayPartition(ArrayPartition([1]))
    M2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]))
    M2u = ArrayPartition(ArrayPartition([1]kg, [2]s), ArrayPartition([3]s, [4]kg))
    M3 = ArrayPartition(ArrayPartition([1], [2], [3]), ArrayPartition([4], [5], [6]), ArrayPartition([7], [8], [9]))
    M3u = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    Mrect = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]), ArrayPartition([5], [6]))
    Vu = ArrayPartition([1.0]s⁻¹, [2.0]s⁻²)
    @test M0 isa MatrixMixedCandidate
    @test M1 isa MatrixMixedCandidate
    @test M2 isa MatrixMixedCandidate
    @test M2u isa MatrixMixedCandidate
    @test M3 isa MatrixMixedCandidate
    @test M3u isa MatrixMixedCandidate
    @test !(Mrect isa MatrixMixedCandidate)
    @test Vu isa MechGlueDiffEqBase.RW(N) where N
    @test !is_square_matrix_mutable(M0)
    @test !is_square_matrix_mutable(M1)
    @test is_square_matrix_mutable(M2)
    @test is_square_matrix_mutable(M2u)
    @test is_square_matrix_mutable(M3)
    @test is_square_matrix_mutable(M3u)
    @test !is_square_matrix_mutable(Mrect)
    @test repr(M0, context = :color=>true) == "ArrayPartition{Union{}, Tuple{}}(()\"#undef\")"
    @test repr(:"text/plain", M0, context = :color=>true) == "()\"#undef\""
    @test repr(M1, context = :color=>true) == "ArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}}}}}((ArrayPartition{Int64, Tuple{Vector{Int64}}}(([1],)),))"
    @test repr(:"text/plain", M1, context = :color=>true) == "(ArrayPartition{Int64, Tuple{Vector{Int64}}}(([1],)),)"
    @test repr(M2, context = :color=>true) == "\e[36mconvert_to_matrix_mixed(\e[39mAny[1 2; 3 4]\e[36m)\e[39m"
    @test repr(:"text/plain", M2, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}:\n 1  2\n 3  4"
    @test repr(M2u, context = :color=>true) == "\e[36mconvert_to_matrix_mixed(\e[39mAny[1\e[36mkg\e[39m 2\e[36ms\e[39m; 3\e[36ms\e[39m 4\e[36mkg\e[39m]\e[36m)\e[39m"
    @test repr(:"text/plain", M2u, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}}}}}:\n 1\e[36mkg\e[39m   2\e[36ms\e[39m\n  3\e[36ms\e[39m  4\e[36mkg\e[39m"
    @test repr(M3, context = :color=>true) == "\e[36mconvert_to_matrix_mixed(\e[39mAny[1 2 3; 4 5 6; 7 8 9]\e[36m)\e[39m"
    @test repr(:"text/plain", M3, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}}}:\n 1  2  3\n 4  5  6\n 7  8  9"
    @test repr(M3u, context = :color=>true) =="\e[36mconvert_to_matrix_mixed(\e[39mAny[1\e[36mkg\e[39m 2\e[36ms\e[39m 3\e[36mkg\e[39m; 4\e[36ms\e[39m 5\e[36mkg\e[39m 6\e[36ms\e[39m; 7\e[36mkg\e[39m 8\e[36ms\e[39m 9\e[36mkg\e[39m∙\e[36ms\e[39m]\e[36m)\e[39m"
    @test repr(:"text/plain", M3u, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ∙ ᵀ, Unitfu.FreeUnits{(\e[36mkg\e[39m, \e[36ms\e[39m),  ᴹ∙ ᵀ, nothing}}}}}}}:\n 1\e[36mkg\e[39m   2\e[36ms\e[39m              3\e[36mkg\e[39m\n  4\e[36ms\e[39m  5\e[36mkg\e[39m               6\e[36ms\e[39m\n 7\e[36mkg\e[39m   8\e[36ms\e[39m  9\e[36mkg\e[39m∙\e[36ms\e[39m"
    @test repr(Vu, context = :color=>true) == "\e[36m2-element mutable \e[39mArrayPartition(1.0\e[36ms⁻¹\e[39m, 2.0\e[36ms⁻²\e[39m)"
    @test repr(:"text/plain", Vu, context = :color=>true) == "2-element mutable ArrayPartition:\n 1.0\e[36ms⁻¹\e[39m\n 2.0\e[36ms⁻²\e[39m"
end
=#
# Indexing, mutating, type guarantee
let 
    M3 = ArrayPartition(ArrayPartition([1], [2], [3]), ArrayPartition([4], [5], [6]), ArrayPartition([7], [8], [9]))
    M3u = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    @test M3[2, 3] == 6
    @test M3u[2, 3] == 6s
    M3[2,3] = 7
    @test M3[2, 3] == 7
    @test_throws InexactError M3[2, 3] = 8.5
    M3u[2,3] = 7s
    @test M3u[2, 3] == 7s
    @test_throws DimensionError M3u[2, 3] = 8cm
end

############
# Conversion 
############
let
    A2 = [1 2; 3 4]
    M2 = convert_to_matrix_mixed(A2)
    @test convert_to_array(M2) == A2
    A2u = [1.0kg 2s; 3s 4kg]
    M2u = convert_to_matrix_mixed(A2u)
    @test convert_to_array(M2u) == A2u
    A3u = [1kg 2 3s; 4s 5kg 6; 7kg 8 9s] 
    M3u = convert_to_matrix_mixed(A3u)
    @test convert_to_array(M3u) == A3u
end

###########################################
# Jacobians with 'ordinary' x, f(x) types
# Default Finitediff.jl behaviour unchanged
###########################################
struct ImmutableVector <: DenseVector{Float64}
    x::Vector{Float64}
end
Base.size(x::ImmutableVector) = size(x.x)
Base.getindex(x::ImmutableVector, i::Integer) = x.x[i]
let
    x = ImmutableVector(ones(2))
    # Call to FiniteDiff\src\jacobians.jl:138, returns 2x2 Matrix{Float64}
    J = finite_difference_jacobian(identity, x, Val{:forward})
    @test J isa Matrix{Float64}
    @test J == [1.0 0.0; 0.0 1.0]
    J1 = finite_difference_jacobian(identity, [1.0, 1.0], Val{:forward})
    @test J1 isa Matrix{Float64}
    @test J1 == [1.0 0.0; 0.0 1.0]
    f = x -> [x[1] + 2x[2], 3x[1] + 4x[2]]
    #@test_throws AssertionError finite_difference_jacobian(f, x, Val{:forward})
    Jc = JacobianCache(x, f(x), Val{:forward})
    @test Jc.x1 == [1.0, 1.0]
    @test Jc.fx == [3.0, 7.0]
    @test Jc.fx1 == [3.0, 7.0]
    @test Jc.colorvec == 1:2
    @test Jc.sparsity === nothing
    Jc1 = JacobianCache([1.0, 2.0], f(x), Val{:forward})
    @test Jc1.x1 == Jc1.x1
    @test Jc1.fx == Jc.fx
    @test Jc1.fx1 == Jc.fx1
    @test Jc1.colorvec == Jc.colorvec
    @test Jc1.sparsity === nothing
end
####################################
# Create mutable Jacobian prototypes
####################################

# Dimensionless, mutable input
@test sum(isnan.(let
    x = ArrayPartition([0.0], [1.5])
    F = ArrayPartition([0.0], [1.5])
    jacobian_prototype_nan(x, F)
end)) == 4

# With units, mutable input
@test sum(isnan.(let
    x = ArrayPartition([0.0], [1.5]s⁻¹)
    F = ArrayPartition([0.0], [1.5]s⁻¹)
    jacobian_prototype_nan(x, F)
end)) == 4
# Immutable input
@test sum(isnan.(let
    x = ArrayPartition(0.0, 1.5s⁻¹)
    F = ArrayPartition(0.0kg, 1.5s⁻¹)
    jacobian_prototype_nan(x, F)
end)) == 4
# Mutable x, immutable f(x)
@test sum(isnan.(let
    x = ArrayPartition([0.0], [1.5s⁻¹])
    F = ArrayPartition(0.0kg, 1.5s⁻¹)
    jacobian_prototype_nan(x, F)
end)) == 4
# Immutable x, mutable f(x)
@test sum(isnan.(let
    x = ArrayPartition(0.0, 1.5s⁻¹)
    F = ArrayPartition([0.0]kg, [1.5s⁻¹])
    jacobian_prototype_nan(x, F)
end)) == 4
# Mutable x, inconsistent f(x)
@test_throws AssertionError let
    x = ArrayPartition([0.0], [1.5s⁻¹])
    F = ArrayPartition([0.0]kg, 1.5s⁻¹)
    jacobian_prototype_nan(x, F)
end
# Inconsistent x, mutable f(x)
@test_throws AssertionError let
    x = ArrayPartition(0.0, [1.5s⁻¹])
    F = ArrayPartition([0.0]kg, [1.5s⁻¹])
    jacobian_prototype_nan(x, F)
end 

# Immutable x, inconsistent f(x)
@test_throws AssertionError  let
    x = ArrayPartition(0.0, 1.5s⁻¹)
    F = ArrayPartition([0.0]kg, 1.5s⁻¹)
    jacobian_prototype_nan(x, F)
end
# Inconsistent x, immutable f(x)
@test_throws AssertionError  let
    x = ArrayPartition(0.0, [1.5s⁻¹])
    F = ArrayPartition(0.0kg, 1.5s⁻¹)
    jacobian_prototype_nan(x, F)
end
# Mutable x, too long f(x)
@test_throws MethodError let
    x = ArrayPartition([0.0], [1.5s⁻¹])
    F = ArrayPartition([0.0]kg, [1.5s⁻¹], [2.0cm])
    jacobian_prototype_nan(x, F)
end
# Mutable input, 2x2
@test sum(isnan.(let
    x = ArrayPartition([0.0]cm, [1.5]s⁻¹)
    F = ArrayPartition([0.0]kg, [1.5]s⁻¹)
    jacobian_prototype_nan(x, F)
end)) == 4
# Mutable input, 3x3, same type per row
@test  sum(isnan.(let
    x = ArrayPartition([0.0]cm, [1.5]s⁻¹, [1.0]cm)
    F = ArrayPartition([0.0], [1.5], [2.0])
    jacobian_prototype_nan(x, F)
end)) == 9
# Mutable input, 3x3, various types
@test  sum(isnan.(let
    x = ArrayPartition([0.0]cm, [1.5]s⁻¹, [1.0]cm)
    F = ArrayPartition([0.0]kg, [1.5]s⁻¹, [2.0]kg)
    jacobian_prototype_nan(x, F)
end)) == 9

#########################################
# Prototype mutable Jacobian, NaN values
# Note that (NaN == NaN) == false ! 
#########################################
@test sum(isnan.(let
    x = ArrayPartition(1, 2)
    f = x -> ArrayPartition(x[1] + 2x[2], 3x[1] + 4x[2])
    fx = f(x)
    jacobian_prototype_nan(x, fx)
end)) == 4
@test sum(isnan.(let
    x = ArrayPartition([1], [2])
    f = x -> ArrayPartition([x[1] + 2x[2]], [3x[1] + 4x[2]])
    fx = f(x)
    jacobian_prototype_nan(x, fx)
end)) == 4

@test sum(isnan.(let
    x = ArrayPartition(1s, 2kg)
    f = x -> ArrayPartition(x[1]cm + 2s∙cm/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2])
    fx = f(x)
    convert_to_array(jacobian_prototype_nan(x, fx))
end ./ [NaN∙cm NaN∙cm∙s∙kg⁻¹; NaN∙cm∙s NaN∙cm∙s²∙kg⁻¹])) == 4

# Zero values, x and f(x) ArrayPartition
@test let
    x = ArrayPartition(1, 2)
    f = x -> ArrayPartition(x[1] + 2x[2], 3x[1] + 4x[2])
    jacobian_prototype_zero(x, f(x))
 end == convert_to_matrix_mixed([0.0 0.0; 0.0 0.0])
 @test let
     x = ArrayPartition(1, 2)
     f = x -> ArrayPartition(x[1] + 2x[2], 3x[1] + 4x[2])
     convert_to_array(jacobian_prototype_zero(x, f(x)))
  end == [0.0 0.0
          0.0 0.0]
 @test let
     x = ArrayPartition(1s, 2kg)
     f = x -> ArrayPartition(x[1]cm + 2s∙cm/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2])
     jacobian_prototype_zero(x, f(x))
 end == convert_to_matrix_mixed([0.0cm   0.0cm∙s∙kg⁻¹
                             0.0cm∙s  0.0cm∙s²∙kg⁻¹])
# Internal conversion to ArrayPartition
 @test let
     x = [1s, 2kg]
     f = x -> [x[1]cm + 2cm∙s/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2]]
     jacobian_prototype_zero(x, f(x))
 end == convert_to_matrix_mixed([0.0cm   0.0cm∙s∙kg⁻¹
                             0.0cm∙s  0.0cm∙s²∙kg⁻¹])
# Internal conversion to ArrayPartition, Vector{Float64} and Vector{<:Quantity}
 @test let
     x = [1.0, 2.0]
     f = x -> [x[1]cm + 2cm * x[2], 3kg∙ x[1] + 4kg∙ x[2]]
     jacobian_prototype_zero(x, f(x))
 end == convert_to_matrix_mixed([0.0cm   0.0cm
                             0.0kg  0.0kg])

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
    cache.sparsity === nothing
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
# Not currently implemented. Square J only.
# e) finite difference Jacobian matrix, "f(x, y) -> (u,v)", using ArrayPartition, dimensionless.
# Analytical result: f(x) = [x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2]]
# =>                 J = [δf₁/ δx₁     δf₁/ δx₂    
#                         δf₂/ δx₁     δf₂/ δx₂]
# =>    J([1.0, 2.0]) =  [-2    10
#                          0    2]
isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    convert_to_array(finite_difference_jacobian(f, x))
end, [-2.0 10.0
       0.0 2.0]; atol = 1e-6)
# f) fdtype Val(:complex), using ArrayPartition, dimensionless.
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    fdtype     = Val(:complex)
    convert_to_array(finite_difference_jacobian(f, x, fdtype))
end, [-2.0 10.0 
       0.0 2.0]; atol = 1e-6)
# g) fdtype Val(:central), using ArrayPartition, dimensionless.
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2 - 2x[1]∙x[2] + x[2]^3, 2x[2])
    x = ArrayPartition(1.0, 2.0)
    fdtype     = Val(:central)
    convert_to_array(finite_difference_jacobian(f, x, fdtype))
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
# Not currently implemented. Square J only.

# e) finite difference Jacobian matrix, "f(x, y) -> (u,v)", using ArrayPartition, dimensions.
# Ref. test 5e)

isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6, 2x[2]cm∙s²)
    x = ArrayPartition(1.0s⁻¹    , 2.0s⁻²)
    jacobian_prototype_zero(x, f(x))
    convert_to_array(finite_difference_jacobian(f, x))
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
    convert_to_array(finite_difference_jacobian(f, x, fdtype))
end ./ [-2.0kg∙s  10.0kg∙s²
        0.0cm∙s   2.0cm∙s²], 
                            [1.0  1.0
                             NaN  1.0]; atol = 1e-6, nans = true)
#=
# g) fdtype Val(:central), using ArrayPartition, dimensions.
# Ref. test 5g)
@test isapprox(let 
    f = x -> ArrayPartition(x[1]^2∙kg∙s² - 2x[1]∙x[2]∙kg∙s³ + x[2]^3∙kg∙s^6, 2x[2]cm∙s²)
    x = ArrayPartition(1.0s⁻¹, 2.0s⁻²)
    fdtype     = Val(:central)
    convert_to_array(finite_difference_jacobian(f, x, fdtype))
end ./ [-2.0kg∙s  10.0kg∙s²
         0.0cm∙s   2.0cm∙s²], 
                            [1.0  1.0
                            NaN  1.0]; atol = 1e-6, nans = true)
=#

##############################
# Jacobian prototype as called
# from NLSolverbase. 
##############################
@test sum(isnan.(let
    x = ArrayPartition([0.0], [1.5])
    F = ArrayPartition([0.0], [1.5])
    # Jacobian prototype is a 2x2 matrix of NaN{Float64}
    alloc_DF(x, F)
end)) == 4
nothing