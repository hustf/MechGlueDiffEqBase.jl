# Test "multiplication" and "division" and similar operations.
# For recursive ArrayPartition unitful mutable representations
# of vectors, matrices and transposed versions of these.
# Informally, these are 'mixed' matrices and vectors.
using Test
using MechGlueDiffEqBase # exports ArrayPartition
using MechGlueDiffEqBase: determinant_dimension, determinant, mul!
import MechanicalUnits
using MechanicalUnits: @import_expand, ∙, ustrip, unit, 𝐋², 𝐋⁴, 𝐌³, 𝐓, NoDims, NoUnits
using MechanicalUnits: 𝐋, 𝐌, 𝐓
using MechanicalUnits: DimensionError, Dimension, Dimensions
import LinearAlgebra
@import_expand(dam, cm, kg, s, GPa, mm, N, m, kN)

##################
# Matrix inversion
##################
@testset "> Matrix inversion" begin
    # Reference
    @test all(isapprox.(inv([1 1; -1 1]),[0.5 -0.5; 0.5 0.5], rtol = 1e-6))
    @inferred inv([1 1; -1 1])
    # Same unit
    # WAS @test inv([1 1; -1 1]kg)≈[0.5 -0.5; 0.5 0.5]kg^-1
    @test all(isapprox.(inv([1 1; -1 1]kg),[0.5 -0.5; 0.5 0.5]kg^-1, rtol = 1e-6))
    @inferred inv([1 1; -1 1]kg)
    # Same dimension
    @test all(isapprox.(inv([1cm 0.001dam; -0.001dam 1cm]),[0.5 -0.5; 0.5 0.5]cm^-1, rtol = 1e-6))
    @inferred inv([1cm 0.001dam; -0.001dam 1cm])
    # Not implemented: Inverse symmetric unit (this is dimensionally sound)
    @test_throws ArgumentError inv([1 1cm; -1cm^-1 1])
end

@testset "> MixedMatrix inversion" begin
    # Reference
    Am = convert_to_mixed([1 1; -1 1])
    Ai =  [0.5 -0.5; 0.5 0.5]
    @test all(isapprox.(inv(Am),Ai, rtol = 1e-6))
    @test @inferred(ArrayPartition{<:Number}, inv(Am)) == Ai
    # Consistent unit over elements
    Amu = convert_to_mixed([1 1; -1 1]kg)
    Aiu =  [0.5 -0.5; 0.5 0.5]/kg
    @test @inferred(ArrayPartition{<:Number}, inv(Amu)) == Aiu
    # Same dimension (not much of a test - the preferred common unit is used anyway)
    Amc = convert_to_mixed([1cm 0.001dam; -0.001dam 1cm])
    Aiuc =  [0.5 -0.5; 0.5 0.5]cm^-1
    @test @inferred(ArrayPartition{<:Number}, inv(Amc)) == Aiuc
    # Inverse symmetric units (this is dimensionally sound)
    Amm = convert_to_mixed([1 1cm; -1cm^-1 1])
    Aium =  [0.5 -0.5cm; 0.5/cm 0.5]
    @test @inferred(ArrayPartition{<:Number}, inv(Amm)) == Aium
end


#################################################
# MixedMatrix inversion with dimensional diagonal
#################################################
@testset "> MixedMatrix inversion with dimensional diagonal" begin
    A = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
    Am = convert_to_mixed(A)
    # We can't do this in Base.
    @test_throws ArgumentError inv(A)
    # But nested ArrayPartitions implements the method:
    Ami = inv(Am)
    @test Ami == [-1.7777777777777777kg⁻¹ 0.8888888888888888s⁻¹ -0.1111111111111111cm;
                 1.5555555555555556kg⁻¹∙cm⁻¹ -0.7777777777777778cm⁻¹∙s⁻¹ 0.2222222222222222;
                -0.11111111111111119s⁻¹ 0.22222222222222227kg∙s⁻² -0.11111111111111112kg∙cm∙s⁻¹]
    # Verify correctness.
    Ident = convert_to_array(Ami) * A
    @test Ident[1, 1] ≈ 1
    @test Ident[2, 2] ≈ 1
    @test Ident[3, 3] ≈  1
    @test abs(Ident[1, 2]) < 1e-12cm
    @test abs(Ident[1, 3]) < 1e-12s∙kg⁻¹
end
###############################################
# Pre-multiply with the inverse of mixed matrix
###############################################
@testset "> Pre-multiply with the inverse of mixed matrix" begin
    w = [1 1; 1 10]cm
    x = [1cm 1s; 1s 10cm]
    y = [1cm 1cm∙s; 1cm/s 10cm]
    z = [1 1s; 1/s 10]
    ka = [   4.8kN∙mm⁻¹ -2400.0kN        0.0kN∙mm⁻¹
         -2400.0kN          1.6e6mm∙kN   0.0kN
             0.0kN∙mm⁻¹     0.0kN      200.0kN∙mm⁻¹]
    @test_throws DimensionError w^2
    @test_throws DimensionError x^2
    @test y^2 == [2cm² 11cm²∙s; 11cm²∙s⁻¹ 101cm²]
    @test z^2 == [2 11s;  11s⁻¹ 101 ]
    @test_throws DimensionError ka^2
    wm = convert_to_mixed(w)
    xm = convert_to_mixed(x)
    ym = convert_to_mixed(y)
    zm = convert_to_mixed(z)
    k = convert_to_mixed(ka)

    @test determinant_dimension(wm) == 𝐋²
    @test determinant_dimension(xm) == Dimensions{(Dimension{Missing}(1//1),)}
    @test determinant_dimension(ym) == 𝐋²
    @test determinant_dimension(zm) == NoDims
    @test determinant_dimension(k) ==  𝐋²∙ 𝐌³∙ 𝐓^-6

    @test determinant(wm) == 9cm²
    @test_throws DimensionMismatch determinant(xm)
    @test determinant(ym) == 9cm²
    @test determinant(zm) == 9
    @test determinant(k) == 3.84e8kN³∙mm⁻¹
    Fm = convert_to_mixed(4.56kN, -2240.0mm∙kN, 200.0kN)
    c = inv(k)
    @test all(isapprox.(c * Fm, [1.0mm, 1e-4, 1.0mm]))
    @test all(isapprox.(k \ Fm, [1.0mm, 1e-4, 1.0mm]))
end

#############################################
# Multiplication mixed matrix by mixed vector
#############################################
@testset "Multiplication" begin
    A3 = [1 2 3; 4 5 6; 7 8 0]
    b3 = [1, 20, 300]
    c3 = [NaN, NaN, NaN]
    A3m = convert_to_mixed(A3)
    b3m = convert_to_mixed(b3)
    c3m = convert_to_mixed(c3)
    A3u = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
    b3u = [1cm, 20, 300kg∙cm∙s⁻¹]
    c3u = [1kg∙cm, 2s∙cm,3]
    A3mu = convert_to_mixed(A3u)
    b3mu = convert_to_mixed(b3u)
    c3mu = convert_to_mixed(c3u)
    @testset "> Multiplication mixed matrix by mixed vector" begin
        @test A3 * b3 == [1*1  + 2*20 + 3*300, 1904, 167]
        @test mul!(c3, A3, b3) == [941, 1904, 167] # Float64 on left, though, as NaN{Float64}
        c3 = [NaN, NaN, NaN]
        @test mul!(c3m, A3m, b3m) == [941, 1904, 167]
        c3m = convert_to_mixed(c3)
        @test A3m * b3m == [941, 1904, 167]
        @test mul!(c3mu, A3mu, b3mu) == [941kg∙cm, 1904cm∙s, 167]
        c3mu = convert_to_mixed(c3u)
        @test A3u * b3u == [941kg∙cm, 1904cm∙s, 167]
        @test A3mu * b3mu == [941kg∙cm, 1904cm∙s, 167]
    end
    #########################################################
    # Multiplication, transposed mixed matrix by mixed vector
    #########################################################
    @testset "> Multiplication, transposed mixed matrix by mixed vector" begin
        @test_throws DimensionError transpose(A3u) * b3u
        @test_throws DimensionMismatch transpose(A3mu) * b3mu
        tA3mu = transpose(A3mu)
        d3 = convert_to_mixed([1cm, 20kg∙cm∙s⁻¹, 300kg∙cm²])
        e3 = convert_to_mixed([NaN∙kg∙cm, NaN∙kg∙cm², NaN∙s∙cm])
        @test mul!(e3, tA3mu, d3) == [2181.0kg∙cm, 2502.0kg∙cm², 123.0cm∙s]
        @test convert_to_array(tA3mu) * convert_to_array(d3) == [2181.0kg∙cm, 2502.0kg∙cm², 123.0cm∙s]
    end
    @testset "> Informative error message" begin
        fpth, io = mktemp(tempdir(); cleanup=true)
        redirect_stdio(stderr = io) do
            try
                A3mu * c3mu
            catch
            end
        end
        close(io)
        msg = readlines(fpth)
        @test msg[1] == "┌ Warning: mul! DimensionError hint"
        @test msg[2] == "│   (i, j) = (1, 2)"
        @test msg[3] == "│   A[i, j] = 2kg∙cm"
        @test msg[4] == "│   B[j] = 2cm∙s"
        @test msg[5] == "│   A[i, j] * B[j] = 4kg∙cm²∙s"
        @test msg[6] == "│   C[i] = 1kg²∙cm"
        @test msg[7] == "│   oneunit(C[i]) / oneunit(A[i, j]) = 1.0kg"
        @test msg[8] == "│   oneunit(C[i]) / oneunit(B[j]) = 1.0kg²∙s⁻¹"
    end
end

@testset "> Beam stiffness and flexibility matrices" begin
    # Beam with six degrees of freedom (2d, shear modes excluded)
    E = 200GPa
    I = 2e6mm⁴
    A = 1000mm²
    l = 100cm
    # NoUnits simplifies "fraction units", cm/mm.
    kx = E∙A / l |> NoUnits |> kN
    ky = 12∙E∙I / l^3 |> NoUnits |> kN
    kθ = 2∙E∙I / l |> NoUnits |> kN
    kθy = 6∙E∙I / l^2  |> NoUnits |> kN
    # The stiffness matrix with six degrees of freedom.
    K = convert_to_mixed([kx        0kN/mm     0kN        -kx       0kN/mm     0kN;
                0kN/mm     ky        -kθy      0kN/mm     -ky       -kθy;
                0kN        -kθy      2∙kθ      0kN        kθy       kθ  ;
                -kx       0kN/mm     0kN        kx        0kN/mm     0kN  ;
                0kN/mm     -ky       kθy       0kN/mm     ky        kθy ;
                0kN        -kθy      kθ        0kN        kθy       2∙kθ])
    # The system is dimensionally sound:
    @test determinant_dimension(K) == 𝐋^4 * 𝐌^6 * 𝐓^-12
    @test determinant_dimension(K) == 𝐋⁴∙𝐌^6 ∙𝐓^-12
    # With six degrees of freedom, the system is statically determinate. There's nothing interesting
    # to be found with these equations.
    @test determinant(K) == 0.0kN^6∙mm⁻²
    # Restraining translation at one end, rotation at the other end:
    Kᵣ = convert_to_mixed(K[3:5, 3:5])
    @test determinant(Kᵣ) == 3.84e8kN³∙mm⁻¹
    # Flexibility matrix
    Cᵣ = inv(Kᵣ)
    # Matrix multiplication has not been defined for mixed matrices, so convert to Any[] for this check:
    ustrip.(convert_to_array(Cᵣ) * convert_to_array(Kᵣ)) ≈ LinearAlgebra.I(3)
    # Forced rotation on one end, dislocation at the other
    υ = convert_to_mixed([0.01, 1mm, 1mm])
    # Corresponding moment and force vector:
    S = Kᵣ* υ .|> kN
    @test S[1] isa  MechanicalUnits.Unitfu.Energy # A moment or torque has the dimensions of energy. MechanicalUnits don't import the ambiguity.
    @test S[2] isa  MechanicalUnits.Force
    @test S[3] isa  MechanicalUnits.Force
    @test S[2] == 200kN
    # Let's go the other way and re-calculate deformations from forces:
    @test all(NoUnits.(Cᵣ * S) .≈ υ)
    @test all(NoUnits.(Kᵣ \ S) .≈ υ)
end


nothing