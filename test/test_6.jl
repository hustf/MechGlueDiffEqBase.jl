# Finite difference extended function
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
using MechanicalUnits.Unitfu: DimensionError
import OrdinaryDiffEq.FiniteDiff
using OrdinaryDiffEq.FiniteDiff: finite_difference_derivative, default_relstep
using OrdinaryDiffEq.FiniteDiff: finite_difference_jacobian, JacobianCache, finite_difference_jacobian!
import OrdinaryDiffEq.ArrayInterface
import Base: Broadcast
using Base.Broadcast: Broadcasted, result_style, combine_styles, DefaultArrayStyle, BroadcastStyle
import MechGlueDiffEqBase.RecursiveArrayTools
using MechGlueDiffEqBase.RecursiveArrayTools: ArrayPartitionStyle, unpack
@import_expand(cm, kg, s)
####################
# A Epsilon with units
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
# B Univariate finite difference, units
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
# C Mutable fixed-length mixed arrays. 
# We want these to be inferrable,
# which the standard matrix type, e.g. 
#     Any[1 2kg; 3 4] 
# is often not. For inferrability, we implement
# as nested ArrayPartitions. For mutability,
# the innermost type is a one-element vector.
#############################################

let 
    # n:dimensionless d: dimension, M: Matrix, V: Vector, E: empty, 0-3: size, i: immutable
    E = ArrayPartition()
    Mn1 = ArrayPartition(ArrayPartition([1]))
    Mu1 = ArrayPartition(ArrayPartition([1kg]))
    Mn2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]))
    Mu2 = ArrayPartition(ArrayPartition([1]kg, [2]s), ArrayPartition([3]s, [4]kg))
    Mn3 = ArrayPartition(ArrayPartition([1], [2], [3]), ArrayPartition([4], [5], [6]), ArrayPartition([7], [8], [9]))
    Mu3 = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    Mn3x2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]), ArrayPartition([5], [6]))
    Vn1 = ArrayPartition([1.0])
   ### Vn3 = ArrayPartition([1.0], [2.0], [3.0])
    Vu3 = ArrayPartition([1.0]s⁻¹, [2.0]s⁻², [3.0])
    @test E isa MixedCandidate
    @test Mn1 isa MixedCandidate
    @test Mn2 isa MixedCandidate
    @test Mu2 isa MixedCandidate
    @test Mn3 isa MixedCandidate
    @test Mu3 isa MixedCandidate
    @test !(Mn3x2 isa MixedCandidate)
    @test Vn1 isa MechGlueDiffEqBase.RW(N) where N
    @test Vu3 isa MechGlueDiffEqBase.RW(N) where N
    @test !is_square_matrix_mutable(E)
    @test is_square_matrix_mutable(Mn1)
    @test is_square_matrix_mutable(Mn2)
    @test is_square_matrix_mutable(Mu2)
    @test is_square_matrix_mutable(Mn3)
    @test is_square_matrix_mutable(Mu3)
    @test !is_square_matrix_mutable(Mn3x2)
    @test !is_square_matrix_mutable(Vn1)
    @test !is_square_matrix_mutable(Vu3)

    @test !is_vector_mutable_stable(E)
    @test !is_vector_mutable_stable(Mn1)
    @test !is_vector_mutable_stable(Mn2)
    @test !is_vector_mutable_stable(Mu2)
    @test !is_vector_mutable_stable(Mn3)
    @test !is_vector_mutable_stable(Mu3)
    @test !is_vector_mutable_stable(Mn3x2)
    @test !is_vector_mutable_stable(Vn1)
    @test is_vector_mutable_stable(Vu3)

    # trait from type
    @test !is_square_matrix_mutable(typeof(E))
    @test is_square_matrix_mutable(typeof(Mn1))
    @test is_square_matrix_mutable(typeof(Mn2))
    @test is_square_matrix_mutable(typeof(Mu2))
    @test is_square_matrix_mutable(typeof(Mn3))
    @test is_square_matrix_mutable(typeof(Mu3))
    @test !is_square_matrix_mutable(typeof(Mn3x2))
    @test !is_square_matrix_mutable(typeof(Vn1))
    @test !is_square_matrix_mutable(typeof(Vu3))

    @test !is_vector_mutable_stable(typeof(E))
    @test !is_vector_mutable_stable(typeof(Mn1))
    @test !is_vector_mutable_stable(typeof(Mn2))
    @test !is_vector_mutable_stable(typeof(Mu2))
    @test !is_vector_mutable_stable(typeof(Mn3))
    @test !is_vector_mutable_stable(typeof(Mu3))
    @test !is_vector_mutable_stable(typeof(Mn3x2))
    @test !is_vector_mutable_stable(typeof(Vn1))
    @test is_vector_mutable_stable(typeof(Vu3))

    @test mixed_array_trait(E) == Empty()
    @test mixed_array_trait([1, 2]) == NotMixed()
    @test mixed_array_trait(Mn1) == MatSqMut()
    @test mixed_array_trait(Vn1) == Single()
    @test mixed_array_trait(Vu3) == VecMut()

    @test mixed_array_trait(typeof(E)) == Empty()
    @test mixed_array_trait(typeof([1, 2])) == NotMixed()
    @test mixed_array_trait(typeof(Mn1)) == MatSqMut()
    @test mixed_array_trait(typeof(Vn1)) == Single()
    @test mixed_array_trait(typeof(Vu3)) == VecMut()

    @test repr(E, context = :color=>true) == "ArrayPartition{Union{}, Tuple{}}(()\"#undef\")"
    @test repr(:"text/plain", E, context = :color=>true) == "()\"#undef\""
    @test sprint(io -> print(IOContext(io, :color => true), E)) == "ArrayPartition{Union{}, Tuple{}}(()\"#undef\")"
    @test repr(Mn1, context = :color=>true) == "\e[36mconvert_to_mixed(\e[39m[1;;]\e[36m)\e[39m"
    @test repr(:"text/plain", Mn1, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}}}}}:\n 1"
    @test sprint(io -> print(IOContext(io, :color => true), Mn1)) == "\e[36mconvert_to_mixed(\e[39m[1;;]\e[36m)\e[39m"
    @test repr(Mn2, context = :color=>true) == "\e[36mconvert_to_mixed(\e[39m[1 2; 3 4]\e[36m)\e[39m"
    @test repr(:"text/plain", Mn2, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}:\n 1  2\n 3  4"
    @test sprint(io -> print(IOContext(io, :color => true), Mn2)) == "\e[36mconvert_to_mixed(\e[39m[1 2; 3 4]\e[36m)\e[39m"
    @test repr(Mu2, context = :color=>true) == "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m; 3\e[36ms\e[39m 4\e[36mkg\e[39m]\e[36m)\e[39m"
    @test repr(:"text/plain", Mu2, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}}}}}:\n 1\e[36mkg\e[39m   2\e[36ms\e[39m\n  3\e[36ms\e[39m  4\e[36mkg\e[39m"
    @test sprint(io -> print(IOContext(io, :color => true), Mu2)) == "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m; 3\e[36ms\e[39m 4\e[36mkg\e[39m]\e[36m)\e[39m"
    @test repr(Mn3, context = :color=>true) == "\e[36mconvert_to_mixed(\e[39m[1 2 3; 4 5 6; 7 8 9]\e[36m)\e[39m"
    @test repr(:"text/plain", Mn3, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}}}:\n 1  2  3\n 4  5  6\n 7  8  9"
    @test repr(Mu3, context = :color=>true) =="\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m 3\e[36mkg\e[39m; 4\e[36ms\e[39m 5\e[36mkg\e[39m 6\e[36ms\e[39m; 7\e[36mkg\e[39m 8\e[36ms\e[39m 9\e[36mkg\e[39m∙\e[36ms\e[39m]\e[36m)\e[39m"
    @test repr(:"text/plain", Mu3, context = :color=>true) == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}}}, ArrayPartition{Quantity{Int64}, Tuple{Vector{Quantity{Int64,  ᴹ, Unitfu.FreeUnits{(\e[36mkg\e[39m,),  ᴹ, nothing}}}, Vector{Quantity{Int64,  ᵀ, Unitfu.FreeUnits{(\e[36ms\e[39m,),  ᵀ, nothing}}}, Vector{Quantity{Int64,  ᴹ∙ ᵀ, Unitfu.FreeUnits{(\e[36mkg\e[39m, \e[36ms\e[39m),  ᴹ∙ ᵀ, nothing}}}}}}}:\n 1\e[36mkg\e[39m   2\e[36ms\e[39m              3\e[36mkg\e[39m\n  4\e[36ms\e[39m  5\e[36mkg\e[39m               6\e[36ms\e[39m\n 7\e[36mkg\e[39m   8\e[36ms\e[39m  9\e[36mkg\e[39m∙\e[36ms\e[39m"
    @test sprint(io -> print(IOContext(io, :color => true), Mu3)) == "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m 3\e[36mkg\e[39m; 4\e[36ms\e[39m 5\e[36mkg\e[39m 6\e[36ms\e[39m; 7\e[36mkg\e[39m 8\e[36ms\e[39m 9\e[36mkg\e[39m∙\e[36ms\e[39m]\e[36m)\e[39m"
    @test repr(Mn3x2, context = :color=>true) == "ArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}((\e[36m2-element mutable \e[39mArrayPartition(1, 2), \e[36m2-element mutable \e[39mArrayPartition(3, 4), \e[36m2-element mutable \e[39mArrayPartition(5, 6)))"
    @test repr(:"text/plain", Mn3x2, context = :color=>true) == "(\e[36m2-element mutable \e[39mArrayPartition(1, 2), \e[36m2-element mutable \e[39mArrayPartition(3, 4), \e[36m2-element mutable \e[39mArrayPartition(5, 6))"
    @test sprint(io -> print(IOContext(io, :color => true), Mn3x2)) == "ArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}((\e[36m2-element mutable \e[39mArrayPartition(1, 2), \e[36m2-element mutable \e[39mArrayPartition(3, 4), \e[36m2-element mutable \e[39mArrayPartition(5, 6)))"
    @test repr(Vn1, context = :color=>true) == "\e[36mSingle-element mutable matrix (discouraged) \e[39mArrayPartition(ArrayPartition([1.0]))"
    @test repr(:"text/plain", Vn1, context = :color=>true) == "Single-element (discouraged) ArrayPartition(ArrayPartition(Vector{<:Number})):\n 1.0"
    @test repr(Vu3, context = :color=>true) == "\e[36m3-element mutable \e[39mArrayPartition(1.0\e[36ms⁻¹\e[39m, 2.0\e[36ms⁻²\e[39m, 3.0)"
    @test repr(:"text/plain", Vu3, context = :color=>true) == "3-element mutable ArrayPartition:\n 1.0\e[36ms⁻¹\e[39m\n 2.0\e[36ms⁻²\e[39m\n                3.0"
    @test sprint(io -> print(IOContext(io, :color => true), Vu3)) == "\e[36mconvert_to_mixed(\e[39m1.0\e[36ms⁻¹\e[39m, 2.0\e[36ms⁻²\e[39m, 3.0\e[36m)\e[39m"
    @inferred transpose(E);
    @inferred transpose(Mn1)
    @inferred transpose(Mu1)
    @inferred transpose(Mn2)
    @inferred transpose(Mu2)
    @inferred transpose(Mn3)
    @inferred transpose(Mu3)
    @inferred transpose(Mn3x2)
    @inferred transpose(Vn1)
    @inferred transpose(Vu3)
end
#let     # Inferred broadcast of mixed matrices, also mapping.
    # n:dimensionless d: dimension, M: Matrix, V: Vector, E: empty, 0-3: size, i: immutable, 
    # a: array as normal, b: lazy broadcast prototype with function, B lazy broadcast prototype 
    # with two-argument function
    Vu3a = [1.0s⁻¹, 2.0s⁻², 3.0]
    Vn1 = ArrayPartition([1.0])
    Vn3 = ArrayPartition([1.0], [2.0], [3.0])
    Vu3 = convert_to_mixed(Vu3a)
    Mu1 = ArrayPartition(ArrayPartition([1kg]))
    Mn2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]))
    Mu2a = [1kg 2; 3 4cm]
    Mu2 = convert_to_mixed(Mu2a)
    M2a = convert_to_array(Mn2)
    Mu3 = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))

    @test axes(Vu3) == (Base.OneTo(3),)
    @test axes(Mn2) == (Base.OneTo(2), Base.OneTo(2))
    @test ndims(Vu3) == 1
    @test ndims(Mn2) == 2
    broadcast_style_Vu3 = BroadcastStyle(typeof(Vu3))
    @test broadcast_style_Vu3 == ArrayPartitionStyle{DefaultArrayStyle{1}}()
    broadcast_style_Mn2 = BroadcastStyle(typeof(Mn2))
    @test broadcast_style_Mn2 == ArrayPartitionStyle{DefaultArrayStyle{2}}()
    broadcast_style_Mu2 = BroadcastStyle(typeof(Mu2))
    @test broadcast_style_Mu2 == ArrayPartitionStyle{DefaultArrayStyle{2}}()
    @test typeof(Base.broadcastable(Vu3))  == typeof(Vu3)
    @test  typeof(Base.broadcastable(Mn2))  == typeof(Mn2)
    @test  typeof(Base.broadcastable(Mu2))  == typeof(Mu2)

    @test  combine_styles(2, Mu2a) != combine_styles(2, Mu2)
    @test  combine_styles(Mu2a, 2) != combine_styles(2, Mu2)
    @test combine_styles(Mu2a, 2) == combine_styles(2, Mu2a)
    @test combine_styles(Mu2, 2) == combine_styles(2, Mu2)
    Vn1b = Base.broadcasted(x -> 2x, Vn1)
    Vn3b = Base.broadcasted(x -> 2x, Vn3)
    Vu3ab = Base.broadcasted(x -> 2x, Vu3a)
    Vu3b = Base.broadcasted(x -> 2x, Vu3)
    Mu1b = Base.broadcasted(x -> 2x, Mu1)
    Mn2b = Base.broadcasted(x -> 2x, Mn2)
    Mu2b = Base.broadcasted(x -> 2x, Mu2)
    M2ab = Base.broadcasted(x -> 2x, M2a)


    Vn1x = Base.Broadcast.combine_axes(Vn1b.args...)
    Vn3x = Base.Broadcast.combine_axes(Vn3b.args...)
    Vu3x = Base.Broadcast.combine_axes(Vu3b.args...)
    Vu3ax = Base.Broadcast.combine_axes(Vu3ab.args...)
    Mu1x = Base.Broadcast.combine_axes(Mu1b.args...)
    Mn2x = Base.Broadcast.combine_axes(Mn2b.args...)
    Mu2x = Base.Broadcast.combine_axes(Mu2b.args...)
    M2ax = Base.Broadcast.combine_axes(M2ab.args...)
    @test Vu3x === Vu3ax
    @test M2ax === Mu2x
    Vu3l = Base.Broadcast.Broadcasted{typeof(broadcast_style_Vu3)}(Vu3b.f, Vu3b.args, Vu3x)
    Mn2l = Base.Broadcast.Broadcasted{typeof(broadcast_style_Mn2)}(Mn2b.f, Mn2b.args, Mn2x)
    Mu2l = Base.Broadcast.Broadcasted{typeof(broadcast_style_Mu2)}(Mn2b.f, Mu2b.args, Mu2x)

    @test copy(Vu3l) == convert_to_mixed(2Vu3a)
    @test copy(Mu2l) == convert_to_mixed(2Mu2a)

    @inferred Base.broadcasted(x -> x , Vu3)
    @inferred Base.broadcasted(x -> x, Mn2)
    @inferred broadcast(x -> x, Vn1)
    @inferred broadcast(x -> x, Vu3)
    @inferred broadcast(x -> x, Mu1)
    @inferred broadcast(x -> x, Mn2)
    @inferred broadcast(x -> x, Mu2)

    @test map(x -> 2x, Mu2) == convert_to_mixed([2kg 4; 6 8cm])
    @test map(x -> 2x, Mu3) == convert_to_mixed([ 2kg 4s 6kg
        8s  10kg  12s
      14kg   16s  18kg∙s])
    @inferred map(x -> 2x, Mu2)
    @inferred map(x -> 2x, Mu3)
    @test broadcast(x -> 2x, M2a) == [2 4; 6 8]
    @test broadcast(x -> 2x, Vu3) == convert_to_mixed(2Vu3a)
    @test broadcast(x -> 2x, Mu2) == convert_to_mixed(2Mu2a)
    @test Mu2 == Mu2

    Vn1B = Base.broadcasted((x, y)-> 2x * y, 2, Vn1)
    Vn3B = Base.broadcasted((x, y)-> 2x * y, 2, Vn3)
    Vu3aB = Base.broadcasted((x, y)-> 2x * y, 2, Vu3a)
    Vu3B = Base.broadcasted((x, y)-> 2x * y, 2, Vu3)
    Mu1B = Base.broadcasted((x, y)-> 2x * y, 2, Mu1)
    Mn2B = Base.broadcasted((x, y)-> 2x * y, 2, Mn2)
    Mu2B = Base.broadcasted((x, y)-> 2x * y, Mu2, Mu2)
    M2aB = Base.broadcasted((x, y)-> 2x * y, 2, M2a)
    Mu2Ba = Base.broadcasted((x, y)-> 2x * y, Mu2, M2a)


    @test unpack(Vn1B, 1).args[2][1] == Vn1[1]
    @test unpack(Vn3B, 1).args[2][1] == Vn3[1]
    @test unpack(Vu3aB, 1).args[2][1] == Vu3a[1]
    @test unpack(Vu3B, 1).args[2][1] ==  Vu3[1]
    @test unpack(Mu1B, 1).args[2][1, 1] ==  Mu1[1, 1]
    @test unpack(Mn2B, 1).args[2][1, 1] ==  Mn2[1, 1]
    @test unpack(Mu2B, 1).args[2][1, 1] ==  Mu2[1, 1]
    @test unpack(M2aB, 1).args[2][1, 1] ==  M2a[1, 1]
    
    @test broadcast((x, y)-> 2x * y, 2, Vu3) == convert_to_mixed([4.0s⁻¹, 8.0s⁻², 12.0])
    @test broadcast((x, y)-> 2x * y, 2, Mu2a) == Number[4kg 8; 12 16cm]
    @test broadcast((x, y)-> 2x * y, 2, Mu2) == convert_to_mixed([4kg 8; 12 16cm])
    @test broadcast((x, y)-> x * y, Mu2, Mu2) == convert_to_mixed([1kg² 4; 9 16cm²])
    @test broadcast((x, y)-> x * y, Mu2a, Mu2) == convert_to_mixed([1kg² 4; 9 16cm²])
    @test broadcast((x, y)-> x * y, Mu2, Mu2a) == convert_to_mixed([1kg² 4; 9 16cm²])

    type_mM = Broadcasted{ArrayPartitionStyle{Style}} where {Style <: DefaultArrayStyle{2}}
    @test !(typeof(Vu3) <: type_mM)
    @test !(typeof(M2aB) <: type_mM)
    @test typeof(Mn2B) <: type_mM
    @test typeof(Mu2B) <: type_mM
    type_mM2 = Broadcasted{ArrayPartitionStyle{Style}, Axes, F, Tuple{T1, T2}} where {Style <: DefaultArrayStyle{2}, Axes, F, T1, T2}
    @test !(typeof(Vu3) <: type_mM2)
    @test !(typeof(M2aB) <: type_mM2)
    @test typeof(Mn2B) <: type_mM2
    @test typeof(Mu2B) <: type_mM2
    @test !(typeof(Mn2b) <: type_mM2)

    @inferred copy(Vn1B)
    @inferred copy(Vn3B)
    @inferred Vector{Number} copy(Vu3aB)
    @inferred copy(Vu3B)
    @inferred copy(Mu1B)
    @inferred copy(Mn2B)
    @inferred copy(Mu2B)
    @inferred copy(M2aB)

    @test broadcast(*, 2, Mu2) == convert_to_mixed(2 .* Mu2a)
    @test broadcast(*, Mu2, 2) == convert_to_mixed(2 .* Mu2a)
    @test broadcast(*, Mu2, Mu2) == convert_to_mixed(Mu2a.* Mu2a)
    @inferred 2 .* Mu2
    @inferred Mu2 .* 2   
    @inferred Mu2 .* Mu2
    @inferred ArrayPartition Mu2 .^ 2

    # Zip with same order, when pairs of mixed matrix and normal matrix are zipped.
    # As a consequence of zipping to common ordering, we can compare matrices of
    # the mixed and normal types with an == sign.
    @test vec(collect(zip(Mu2))) == vec(collect(zip(Mu2)))
    @test vec(collect(zip(Mu2))) !== vec(collect(zip(Mu2a)))
    @test vec(collect(zip(Mu2))) == Tuple{Quantity{Int64}}[(1,)kg, (2,), (3,), (4,)cm]
    @test vec(collect(zip(Mu2, Mu2a))) == vec(collect(zip(Mu2a, Mu2)))
    @test vec(collect(zip(Mu2, Mu2))) == vec(collect(zip(Mu2a, Mu2)))
    # TODO find how to collect to a non-transposed form
    Mu2z = zip(Mu2, Mu2)
    M2az = zip(M2a, M2a)
    @test collect(zip(Mu2, Mu2))[1,2] == collect(zip(Mu2a, Mu2a))[2,1]
    # Map 1 arg
    @test is_vector_mutable_stable(map(x-> 1.2x, Vu3))
    @test map(x-> 1.2x, Vu3) == [1.2s⁻¹, 2.4s⁻², 3.5999999999999996]
    @inferred map(x-> 1.2x, Vu3)
    @inferred map(x-> 1.2x, Mu2)
    is_square_matrix_mutable(map(x-> 1.2x, Mu2))

    # Map 2 arg - not inferrable, recommend broadcasting

    @test map((x,y) -> x * y * kg, Vu3, convert_to_mixed([1, 2])) == [1.0, 4.0/s]kg∙s⁻¹
    @test map((x,y) -> x * y * kg, convert_to_mixed([1, 2]), Vu3) == [1.0, 4.0/s]kg∙s⁻¹
    @test map((x,y) -> x * y * kg, Vu3, Vu3) == [1.0kg∙s⁻², 4.0kg∙s⁻⁴, 9.0kg]
    @test map((x,y) -> x * y * kg, Mu2a, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
        # TODO shape of combination.
    @test map((x,y) -> x * y * kg, Mu2a, Mu2) == [1kg³ 4kg; 9kg 16kg∙cm²]
    f = (x,y) -> x * y * kg
    @enter map(f, Mu2a, Mu2)
#end
######################################
# D Indexing, mutating, type guarantee
######################################
let 
    Mn3 = ArrayPartition(ArrayPartition([1], [2], [3]), ArrayPartition([4], [5], [6]), ArrayPartition([7], [8], [9]))
    Mu3 = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    @test Mn3[2, 3] == 6
    @test Mu3[2, 3] == 6s
    Mn3[2,3] = 7
    @test Mn3[2, 3] == 7
    @test_throws InexactError Mn3[2, 3] = 8.5
    Mu3[2,3] = 7s
    @test Mu3[2, 3] == 7s
    @test_throws DimensionError Mu3[2, 3] = 8cm
end

##############
# E Conversion 
##############
let
    A2 = [1 2; 3 4]
    Mn2 = convert_to_mixed(A2)
    @test convert_to_array(Mn2) == A2
    A2u = [1.0kg 2s; 3s 4kg]
    Mu2 = convert_to_mixed(A2u)
    @test convert_to_array(Mu2) == A2u
    A3u = [1kg 2 3s; 4s 5kg 6; 7kg 8 9s] 
    Mu3 = convert_to_mixed(A3u)
    @test convert_to_array(Mu3) == A3u
    Vu3 = ArrayPartition([1.0]s⁻¹, [2.0]s⁻², [3.0])
    Vu3[3] = 4.0
    Vc = convert_to_mixed([1.0s⁻¹, 2.0s⁻², 3.0])
    Vc[3] = 4.0
    @test typeof(Vc) == typeof(Vu3)
    @test Vc[3] == Vu3[3]
end

###########################################
# 1 Jacobians with 'ordinary' x, f(x) types
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
    finite_difference_jacobian(f, x, Val{:forward})
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
# 2 Create mutable Jacobian prototypes
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
# 3 Prototype mutable Jacobian, NaN values
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
 end == convert_to_mixed([0.0 0.0; 0.0 0.0])
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
 end == convert_to_mixed([0.0cm   0.0cm∙s∙kg⁻¹
                             0.0cm∙s  0.0cm∙s²∙kg⁻¹])
# Internal conversion to ArrayPartition
 @test let
     x = [1s, 2kg]
     f = x -> [x[1]cm + 2cm∙s/kg * x[2], 3cm∙s ∙ x[1] + 4cm∙s²/kg∙ x[2]]
     jacobian_prototype_zero(x, f(x))
 end == convert_to_mixed([0.0cm   0.0cm∙s∙kg⁻¹
                             0.0cm∙s  0.0cm∙s²∙kg⁻¹])
# Internal conversion to ArrayPartition, Vector{Float64} and Vector{<:Quantity}
 @test let
     x = [1.0, 2.0]
     f = x -> [x[1]cm + 2cm * x[2], 3kg∙ x[1] + 4kg∙ x[2]]
     jacobian_prototype_zero(x, f(x))
 end == convert_to_mixed([0.0cm   0.0cm
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

#=
isapprox(let 
    f = x -> ArrayPartition([x[1]^2 - 2x[1]∙x[2] + x[2]^3], [2x[2]])
    x = ArrayPartition([1.0], [2.0])
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
=#

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
#=
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

################################
# 7 Jacobian prototype as called
# from NLSolverbase. 
################################

@test sum(isnan.(let
    x = ArrayPartition([0.0], [1.5])
    F = ArrayPartition([0.0], [1.5])
    # Jacobian prototype is a 2x2 matrix of NaN{Float64}
    alloc_DF(x, F)
end)) == 4

@test sum(isnan.(let
    x = ArrayPartition([0.0]cm, [1.5]s)
    F = ArrayPartition([0.0]kg, [1.5]cm)
    # Jacobian prototype is a 2x2 matrix of NaN{Float64} with units
    alloc_DF(x, F)
end)) == 4

@test sum(isnan.(convert_to_array(let
    x = ArrayPartition([0.0]cm, [1.5]s)
    F = ArrayPartition([0.0]kg, [1.5]cm)
    # Jacobian prototype is a 2x2 matrix of NaN{Float64} with units
    alloc_DF(x, F)
end) .+ [NaN∙kg∙cm⁻¹ NaN∙kg∙s⁻¹; NaN NaN∙cm∙s⁻¹])) == 4


nothing