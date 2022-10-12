# Concerning mutable fixed-length mixed arrays.
# This extends RecursiveArrayTools.jl and
# may fit more naturally in a separate glue package or as PRs.
#
using Test
using MechGlueDiffEqBase
import MechanicalUnits
using MechanicalUnits: @import_expand, ∙, DimensionError
using Base.Broadcast: BroadcastStyle, combine_styles, DefaultArrayStyle, Broadcasted
import MechGlueDiffEqBase.RecursiveArrayTools
using MechGlueDiffEqBase.RecursiveArrayTools: ArrayPartitionStyle, unpack
import LinearAlgebra
@import_expand(cm, kg, s)

#############################################
# A Mutable fixed-length mixed array.
# We want these to be inferrable,
# which the standard matrix type, e.g.
#     Any[1 2kg; 3 4]
# is often not. For inferrability, we implement
# as nested ArrayPartitions. For mutability,
# the innermost type is a one-element vector.
#############################################

@testset "A Mutable fixed-length mixed arrays - direct construction and printing" begin
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
    #
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
    #
    @test !is_vector_mutable_stable(typeof(E))
    @test !is_vector_mutable_stable(typeof(Mn1))
    @test !is_vector_mutable_stable(typeof(Mn2))
    @test !is_vector_mutable_stable(typeof(Mu2))
    @test !is_vector_mutable_stable(typeof(Mn3))
    @test !is_vector_mutable_stable(typeof(Mu3))
    @test !is_vector_mutable_stable(typeof(Mn3x2))
    @test !is_vector_mutable_stable(typeof(Vn1))
    @test is_vector_mutable_stable(typeof(Vu3))
    #
    @test mixed_array_trait(E) == Empty()
    @test mixed_array_trait([1, 2]) == NotMixed()
    @test mixed_array_trait(Mn1) == MatSqMut()
    @test mixed_array_trait(Vn1) == Single()
    @test mixed_array_trait(Vu3) == VecMut()
    #
    @test mixed_array_trait(typeof(E)) == Empty()
    @test mixed_array_trait(typeof([1, 2])) == NotMixed()
    @test mixed_array_trait(typeof(Mn1)) == MatSqMut()
    @test mixed_array_trait(typeof(Vn1)) == Single()
    @test mixed_array_trait(typeof(Vu3)) == VecMut()
    #
    shortp(x) = repr(x, context = :color=>true)
    longp(x) = repr(:"text/plain", x, context = :color=>true)
    # Note on 'replace' below: when this test is evaluated as in 'runtests.jl', RecursiveArrayTools.ArrayPartition is written including with originator's module name, hence 'replace'.
    sh(x) = replace(shortp(x), "RecursiveArrayTools." => "", "Unitfu." => "")
    lo(x) = replace(longp(x), "RecursiveArrayTools." => "", "Unitfu." => "")
    @test sh(E) == "ArrayPartition{Union{}, Tuple{}}(()\"#undef\")"
    @test lo(E) == "()\"#undef\""
    @test sh(E) == "ArrayPartition{Union{}, Tuple{}}(()\"#undef\")"
    @test sh(Mn1) == "\e[36mconvert_to_mixed(\e[39m[1;;]\e[36m)\e[39m"
    @test lo(Mn1) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}}}}}:\n 1"
    @test sh( Mn1) == "\e[36mconvert_to_mixed(\e[39m[1;;]\e[36m)\e[39m"
    @test sh(Mn2) == "\e[36mconvert_to_mixed(\e[39m[1 2; 3 4]\e[36m)\e[39m"
    @test lo(Mn2) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}:\n 1  2\n 3  4"
    @test sh( Mn2) == "\e[36mconvert_to_mixed(\e[39m[1 2; 3 4]\e[36m)\e[39m"
    @test sh(Mu2)== "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m; 3\e[36ms\e[39m 4\e[36mkg\e[39m]\e[36m)\e[39m"
    @test lo(Mu2)[1:80] == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Qu"
    @test sh( Mu2) == "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m; 3\e[36ms\e[39m 4\e[36mkg\e[39m]\e[36m)\e[39m"
    @test sh(Mn3) == "\e[36mconvert_to_mixed(\e[39m[1 2 3; 4 5 6; 7 8 9]\e[36m)\e[39m"
    @test lo(Mn3) == "\e[36mMatrixMixed as \e[39mArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}}}}:\n 1  2  3\n 4  5  6\n 7  8  9"
    @test sh(Mu3) =="\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m 3\e[36mkg\e[39m; 4\e[36ms\e[39m 5\e[36mkg\e[39m 6\e[36ms\e[39m; 7\e[36mkg\e[39m 8\e[36ms\e[39m 9\e[36mkg\e[39m∙\e[36ms\e[39m]\e[36m)\e[39m"
    @test lo(Mu3)[1:80] == "\e[36mMatrixMixed as \e[39mArrayPartition{Quantity{Int64}, Tuple{ArrayPartition{Qu"
    @test sh( Mu3) == "\e[36mconvert_to_mixed(\e[39m[1\e[36mkg\e[39m 2\e[36ms\e[39m 3\e[36mkg\e[39m; 4\e[36ms\e[39m 5\e[36mkg\e[39m 6\e[36ms\e[39m; 7\e[36mkg\e[39m 8\e[36ms\e[39m 9\e[36mkg\e[39m∙\e[36ms\e[39m]\e[36m)\e[39m"
    @test sh(Mn3x2)[1:80] == "ArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{In"
    @test lo(Mn3x2) == "(\e[36m2-element mutable \e[39mArrayPartition(1, 2), \e[36m2-element mutable \e[39mArrayPartition(3, 4), \e[36m2-element mutable \e[39mArrayPartition(5, 6))"
    @test sh( Mn3x2) == "ArrayPartition{Int64, Tuple{ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}, ArrayPartition{Int64, Tuple{Vector{Int64}, Vector{Int64}}}}}((\e[36m2-element mutable \e[39mArrayPartition(1, 2), \e[36m2-element mutable \e[39mArrayPartition(3, 4), \e[36m2-element mutable \e[39mArrayPartition(5, 6)))"
    @test sh(Vn1)== "\e[36mSingle-element mutable matrix (discouraged) \e[39mArrayPartition(ArrayPartition([1.0]))"
    @test sh( Vn1) == "\e[36mSingle-element mutable matrix (discouraged) \e[39mArrayPartition(ArrayPartition([1.0]))"
    @test lo(Vn1) == "Single-element (discouraged) ArrayPartition(ArrayPartition(Vector{<:Number})):\n 1.0"
    @test sh(Vu3) == "\e[36m3-element mutable \e[39mArrayPartition(1.0\e[36ms⁻¹\e[39m, 2.0\e[36ms⁻²\e[39m, 3.0)"
    @test lo(Vu3) == "3-element mutable ArrayPartition:\n 1.0\e[36ms⁻¹\e[39m\n 2.0\e[36ms⁻²\e[39m\n                3.0"
    @test sh(Vu3) == "\e[36m3-element mutable \e[39mArrayPartition(1.0\e[36ms⁻¹\e[39m, 2.0\e[36ms⁻²\e[39m, 3.0)"
    @test @inferred(transpose(Mn1)) isa LinearAlgebra.Transpose
    @test @inferred(transpose(Mu1)) isa LinearAlgebra.Transpose
    tMn2 = transpose(Mn2)
    @test tMn2 isa LinearAlgebra.Transpose
    @test @inferred(transpose(Mu2)) isa LinearAlgebra.Transpose
    tMu2 = transpose(Mu2)
    @test sh(tMu2) == "[1\e[36mkg\e[39m 3\e[36ms\e[39m; 2\e[36ms\e[39m 4\e[36mkg\e[39m]"
    @test @inferred(transpose(Mn3)) isa LinearAlgebra.Transpose
    @test @inferred(transpose(Mu3)) isa LinearAlgebra.Transpose
    @test @inferred(transpose(Mn3x2)) isa LinearAlgebra.Transpose
    @test @inferred(transpose(Vn1)) isa LinearAlgebra.Transpose
    @test @inferred(transpose(Vu3)) isa LinearAlgebra.Transpose
    # Coverage of print
    iob = IOBuffer()
    println(iob, E)
    println(iob, Mn1)
    println(iob, Vn1)
    println(iob, Mu2)
    println(iob, Vu3)
    String(take!(iob))
end

@testset "A Mutable fixed-length mixed arrays - Inferred broadcast and mapping" begin
     # Inferred broadcast of mixed matrices, also mapping.
    # n:dimensionless d: dimension, M: Matrix, V: Vector, E: empty, 0-3: size, i: immutable,
    # a: array as normal, b: lazy broadcast prototype with function, B lazy broadcast prototype
    # with two-argument function
    Vu3a = [1.0s⁻¹, 2.0s⁻², 3.0]
    Vn1 = ArrayPartition([1.0])
    Vn3 = ArrayPartition([1.0], [2.0], [3.0])
    Vu3 = convert_to_mixed(Vu3a)
    Mu1 = ArrayPartition(ArrayPartition([1kg]))
    Mn2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]))
    Mu2a = convert_to_array([1kg 2; 3 4cm])
    Mu2 = convert_to_mixed(Mu2a)
    M2a = convert_to_array(Mn2)
    Mu3 = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    #
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
    #
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
    #
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
    #
    @test is_vector_mutable_stable(copy(Vu3l))
    @test copy(Vu3l) == 2Vu3
    @test is_square_matrix_mutable(copy(Mu2l))
    @test copy(Mu2l) == 2Mu2
    #
    @test @inferred(Base.broadcasted(x -> x , Vu3)) isa Broadcasted
    @test @inferred(Base.broadcasted(x -> x, Mn2)) isa Broadcasted
    @test @inferred(broadcast(x -> x, Vn1)) == [1.0]
    @test @inferred(broadcast(x -> x, Vu3)) == Vu3
    @test @inferred(broadcast(x -> x, Mu1)) == Mu1
    @test @inferred(broadcast(x -> x, Mn2)) == Mn2
    @test @inferred(broadcast(x -> x, Mu2)) == Mu2
    #
    @test map(x -> 2x, Mu2) == convert_to_mixed([2kg 4; 6 8cm])
    @test map(x -> 2x, Mu3) == convert_to_mixed([ 2kg 4s 6kg
        8s  10kg  12s
      14kg   16s  18kg∙s])
    @test @inferred(map(x -> 2x, Mu2)) == 2Mu2
    @test @inferred(map(x -> 2x, Mu3)) == 2Mu3
    @test broadcast(x -> 2x, M2a) == [2 4; 6 8]
    @test broadcast(x -> 2x, Vu3) == convert_to_mixed(2Vu3a)
    @test broadcast(x -> 2x, Mu2) == convert_to_mixed(2Mu2a)
    @test Mu2 == Mu2
    #
    Vn1B = Base.broadcasted((x, y)-> 2x * y, 2, Vn1)
    Vn3B = Base.broadcasted((x, y)-> 2x * y, 2, Vn3)
    Vu3aB = Base.broadcasted((x, y)-> 2x * y, 2, Vu3a)
    Vu3B = Base.broadcasted((x, y)-> 2x * y, 2, Vu3)
    Mu1B = Base.broadcasted((x, y)-> 2x * y, 2, Mu1)
    Mn2B = Base.broadcasted((x, y)-> 2x * y, 2, Mn2)
    Mu2B = Base.broadcasted((x, y)-> 2x * y, Mu2, Mu2)
    M2aB = Base.broadcasted((x, y)-> 2x * y, 2, M2a)
    Mu2Ba = Base.broadcasted((x, y)-> 2x * y, Mu2, M2a)
    #
    @test unpack(Vn1B, 1).args[2][1] == Vn1[1]
    @test unpack(Vn3B, 1).args[2][1] == Vn3[1]
    @test unpack(Vu3aB, 1).args[2][1] == Vu3a[1]
    @test unpack(Vu3B, 1).args[2][1] ==  Vu3[1]
    @test unpack(Mu1B, 1).args[2][1, 1] ==  Mu1[1, 1]
    @test unpack(Mn2B, 1).args[2][1, 1] ==  Mn2[1, 1]
    @test unpack(Mu2B, 1).args[2][1, 1] ==  Mu2[1, 1]
    @test unpack(M2aB, 1).args[2][1, 1] ==  M2a[1, 1]
    #
    @test broadcast((x, y)-> 2x * y, 2, Vu3) == convert_to_mixed([4.0s⁻¹, 8.0s⁻², 12.0])
    @test broadcast((x, y)-> 2x * y, 2, Mu2a) == Number[4kg 8; 12 16cm]
    @test broadcast((x, y)-> 2x * y, 2, Mu2) == convert_to_mixed([4kg 8; 12 16cm])
    @test broadcast((x, y)-> x * y, Mu2, Mu2) == convert_to_mixed([1kg² 4; 9 16cm²])
    @test broadcast((x, y)-> x * y, Mu2a, Mu2) == convert_to_mixed([1kg² 4; 9 16cm²])
    @test broadcast((x, y)-> x * y, Mu2, Mu2a) == convert_to_mixed([1kg² 4; 9 16cm²])
    #
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
    #
    @test @inferred(copy(Vn3B)) == 4Vn3
    @test copy(Vu3aB) == 4Vu3
    @test @inferred(copy(Vu3B)) == 4Vu3
    @test @inferred(copy(Mu1B)) == 4Mu1
    @test @inferred(copy(Mn2B))  == 4Mn2
    @test @inferred(copy(Mu2B)) == 2Mu2.^2
    @test @inferred(copy(M2aB)) == 4M2a
    #
    @test broadcast(*, 2, Mu2) == convert_to_mixed(2 .* Mu2a)
    @test broadcast(*, Mu2, 2) == convert_to_mixed(2 .* Mu2a)
    @test broadcast(*, Mu2, Mu2) == convert_to_mixed(Mu2a.* Mu2a)
    @test is_square_matrix_mutable(@inferred( 2 .* Mu2))
    @test is_square_matrix_mutable(@inferred( Mu2 .* 2 ))
    @test is_square_matrix_mutable(@inferred( Mu2 .* Mu2))
    @test_throws ErrorException @inferred( Mu2 .^ 2)
    # Zip with same order, when pairs of mixed matrix and normal matrix are zipped.
    # As a consequence of zipping to common ordering, we can compare matrices of
    # the mixed and normal types with an == sign.
    @test vec(collect(zip(Mu2))) == vec(collect(zip(Mu2)))
    @test vec(collect(zip(Mu2))) !== vec(collect(zip(Mu2a)))
    @test vec(collect(zip(Mu2))) == Tuple{Quantity{Int64}}[(1,)kg, (2,), (3,), (4,)cm]
    @test vec(collect(zip(Mu2, Mu2a))) == vec(collect(zip(Mu2a, Mu2)))
    @test vec(collect(zip(Mu2, Mu2))) == vec(collect(zip(Mu2a, Mu2)))
    #
    Mu2z = zip(Mu2, Mu2)
    M2az = zip(M2a, M2a)
    @test collect(zip(Mu2, Mu2))[2,1] == collect(zip(Mu2a, Mu2a))[1,2]
    # Map 1 arg
    @test is_vector_mutable_stable(map(x-> 1.2x, Vu3))
    @test map(x-> 1.2x, Vu3) == [1.2s⁻¹, 2.4s⁻², 3.5999999999999996]
    @test is_vector_mutable_stable(@inferred( map(x-> 1.2x, Vu3)))
    @test is_square_matrix_mutable(@inferred( map(x-> 1.2x, Mu2)))
    #
    # Map 2 arg - not inferrable, prefer broadcasting!
    #
    @test map((x,y) -> x * y * kg, Vu3, convert_to_mixed([1, 2])) == [1.0, 4.0/s]kg∙s⁻¹
    @test map((x,y) -> x * y * kg, convert_to_mixed([1, 2]), Vu3) == [1.0, 4.0/s]kg∙s⁻¹
    @test map((x,y) -> x * y * kg, Vu3, Vu3) == [1.0kg∙s⁻², 4.0kg∙s⁻⁴, 9.0kg]
    @test map((x,y) -> x * y * kg, Mu2a, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
    @test map((x,y) -> x * y * kg, Mu2, Mu2a) == [1kg³ 4kg; 9kg 16kg∙cm²]
end
######################################
# B Indexing, mutating, type guarantee
######################################
@testset "B Indexing, mutating, type guarantee - Direct construction, getindex, setindex" begin
    E = ArrayPartition()
    Vn1 = ArrayPartition([1.0])
    Mn3 = ArrayPartition(ArrayPartition([1], [2], [3]), ArrayPartition([4], [5], [6]), ArrayPartition([7], [8], [9]))
    Mu3 = ArrayPartition(ArrayPartition([1]kg, [2]s, [3]kg), ArrayPartition([4]s, [5]kg, [6]s), ArrayPartition([7]kg, [8]s, [9]kg*s))
    Vu3 = ArrayPartition([1.0]s⁻¹, [2.0]s⁻², [3.0])
    Mn3x2 = ArrayPartition(ArrayPartition([1], [2]), ArrayPartition([3], [4]), ArrayPartition([5], [6]))
    @test Mn3[2, 3] == 6
    @test Mu3[2, 3] == 6s
    Mn3[2,3] = 7
    @test Mn3[2, 3] == 7
    @test_throws InexactError Mn3[2, 3] = 8.5
    Mu3[2,3] = 7s
    @test Mu3[2, 3] == 7s
    @test_throws DimensionError Mu3[2, 3] = 8cm
    @test Mn3[2, 3] == 7
    @test Vn1[1] == 1.0
    @test transpose(Mn3)[3, 2] == 7
    @test transpose(Vn1)[1] == 1.0
    @test transpose(Vu3)[1,2] == 2.0s⁻²
    @test_throws BoundsError transpose(Vu3)[2,1] == 2.0s⁻²
    @test size(transpose(Mn3)) == (3, 3)
    @test size(Mn3x2) == (6, )           # We don't care about rectangle matrices, not implemented
    @test size(transpose(Mn3x2)) == (1, 6)
    @test_throws BoundsError transpose(Vn1)[1,1 ] == 1.0
    # transposed setindex
    transpose(Mn3)[1, 2] = 40
    @test Mn3[2,1] == 40
    transpose(Vu3)[1, 2] = 20.0s⁻²
    @test Vu3[2] == 20.0s⁻²
end
##################################
# C More conversion / construction
##################################
@testset "C More conversion / construction  - Back and forward conversion, alternative construction" begin
    # Back and forward conversion
    A2 = [1 2; 3 4]
    Mn2 = convert_to_mixed(A2)
    @test convert_to_array(Mn2) == A2
    A2u = [1.0kg 2s; 3s 4kg]
    Mu2 = convert_to_mixed(A2u)
    @test convert_to_array(Mu2) == A2u
    A3u = [1kg 2 3s; 4s 5kg 6; 7kg 8 9s]
    Mu3 = convert_to_mixed(A3u)
    @test convert_to_array(Mu3) == A3u
    # setindex
    Vu3 = ArrayPartition([1.0]s⁻¹, [2.0]s⁻², [3.0])
    Vu3[3] = 4.0
    Vc = convert_to_mixed([1.0s⁻¹, 2.0s⁻², 3])
    Vc[3] = 4
    @test typeof(Vc) == typeof(Vu3)
    @test Vc[3] == Vu3[3]
    # Alternative methods for mixed vectors construction / conversion
    @test @inferred(convert_to_mixed(Vc)) == Vc
    @test @inferred(convert_to_mixed(1.0s⁻¹, 2.0s⁻², 4)) == Vc
    @test_throws MethodError convert_to_mixed(tuple([1.0s⁻¹], [2.0s⁻²], [4.0]))
    @test_throws MethodError convert_to_mixed([1.0s⁻¹], [2.0s⁻²], [4.0])
    @test_throws MethodError convert_to_mixed(tuple(1.0s⁻¹, 2.0s⁻², 4.0))
    @test_throws MethodError MechGlueDiffEqBase.ArrayPartition_from_single_element_vectors([1.0s⁻¹], [2.0s⁻²], [4.0])
    @test is_vector_mutable_stable(convert_to_mixed((1.0s, 2.0s, 4.0s)))
    @test is_vector_mutable_stable(convert_to_mixed([1.0s, 2.0, 4]))
    @test_throws AssertionError convert_to_mixed(transpose([1.0s, 2.0, 4]))
    @test @inferred(convert_to_mixed((1.0s, 2.0s, 4.0s))) == convert_to_mixed(1.0s, 2.0s, 4.0s)
    @test is_vector_mutable_stable( convert_to_mixed(tuple([1.0s], [2.0s], [4.0s])))
    @test convert_to_mixed(1.0s⁻¹, 2.0s⁻², 4.0) == Vu3
    @test convert_to_mixed(1.0s⁻¹)[1] == 1.0s⁻¹ # Discouraged type, warned in printing statement
    # More mixed matrices
    @test @inferred(convert_to_mixed(Mu2)) == Mu2
    @test Base.return_types(convert_to_mixed, (typeof(A2),))[1] == ArrayPartition{<:Number, <:Tuple{Vararg{Union{ArrayPartition{<:Number, <:Tuple{Vararg{Vector{<:Number}, N}}}, Vector{<:Number}}, N}}} where N
    ArrayPartition(ArrayPartition([1]kg, [3]), ArrayPartition([5.0], [4.0]cm))
end

nothing