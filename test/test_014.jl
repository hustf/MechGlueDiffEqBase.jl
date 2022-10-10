# Type-stable getindex
#= Test in conjuction with
test_004.jl
test_010.jl
test_011.jl
test_012.jl
=#

using Test
using MechanicalUnits: @import_expand
@import_expand kg m s
using MechGlueDiffEqBase


module t1
    using MechGlueDiffEqBase
    using MechanicalUnits:@import_expand
    @import_expand kg m s
    const p = convert_to_mixed(1.0kg/s, 0.1)
    const ptyp = typeof(p)
    const M = convert_to_mixed([1kg 2s; 3/s 4kg])
    const mtyp = typeof(M)
    fa(p::ptyp) = first(p.x[1])
    fb() = fa(p)
    fc(p) = first(p.x[1])
    fd() = fc(p)
    ga(p::ptyp) = p[1]
    gb() = ga(p)
    gc(p) = p[1]
    gd() = gc(p)
    ha(M::mtyp) = M[2, 2]
    hb() = ha(M)
    hc(M) = M[2, 2]
    hd() = hc(M)
end


@testset "getindex inference" begin
    @testset "Contextless" begin
        @testset "(Type-in-stable) return is correct" begin
            # n:dimensionless d: dimension, M: Matrix, V: Vector, i: immutable,
            # a: array as normal
            Mn2a = [1 2; 3 4]
            Mn2 = convert_to_mixed(Mn2a)
            Md2a = [1.0kg 2s; 3s 4kg]
            Md2 = convert_to_mixed(Md2a)
            Vd3a = [1.0s⁻¹, 2.0s⁻², 3.0]
            Vd3 = convert_to_mixed(1.0s⁻¹, 2s⁻², 3.0)
            Vd3i = ArrayPartition(1.0s⁻¹, 2s⁻², 3.0)
            Md3 = ArrayPartition(Vd3, Vd3, Vd3)
            Md3i = ArrayPartition(Vd3i, Vd3i, Vd3i)
            @test @inferred(Mn2a[2,2]) === 4
            @test @inferred(Mn2[2,2]) === 4
            @test_throws ErrorException @inferred(Md2a[2,2])
            @test_throws ErrorException @inferred(Md2a[2,2])
            @test_throws ErrorException @inferred(Md2[2,2])
            @test_throws ErrorException @inferred(Vd3a[2])
            @test_throws ErrorException @inferred(Vd3[2])
            @test_throws ErrorException @inferred(Vd3i[2])
            @test_throws ErrorException @inferred(Md3[2,2])
            @test_throws ErrorException @inferred(Md3i[2,2])
        end
    end
    @testset "Compiled module inferrable" begin
        p1 = convert_to_mixed(2.0kg/s, 0.3)
        M1 = convert_to_mixed([1kg 2s; 3/s 4kg])
        @testset "Long form vector type-stable" begin
            @test @inferred(t1.fa(p1)) === 2.0kg∙s⁻¹
            @test @inferred(t1.fb()) === 1.0kg∙s⁻¹
            @test @inferred(t1.fc(p1)) === 2.0kg∙s⁻¹
            @test @inferred(t1.fd()) === 1.0kg∙s⁻¹
        end
        @testset "Normal form vector type-stable" begin
            @test @inferred(t1.ga(p1)) === 2.0kg∙s⁻¹
            @test @inferred(t1.gb()) === 1.0kg∙s⁻¹
            @test @inferred(t1.gc(p1)) === 2.0kg∙s⁻¹
            @test @inferred(t1.gd()) === 1.0kg∙s⁻¹
        end
        @testset "Normal form matrix type-stable" begin
            @test @inferred(t1.ha(M1)) === 4kg
            @test @inferred(t1.hb()) === 4kg
            @test @inferred(t1.hc(M1)) === 4kg
            @test @inferred(t1.hd()) === 4kg
        end
    end
    p = convert_to_mixed(1.0kg/s, 0.1)
    ptyp = typeof(p)
    M = convert_to_mixed([1kg 2s; 3/s 4kg])
    mtyp = typeof(M)
    @testset "Functions in Main inferrable (don't run as compiled test file)" begin
        fa(p::ptyp) = first(p.x[1])
        fb() = fa(p)
        fc(p) = first(p.x[1])
        fd() = fc(p)
        ga(p::ptyp) = p[1]
        gb() = ga(p)
        gc(p) = p[1]
        gd() = gc(p)
        ha(M::mtyp) = M[2, 2]
        hb() = ha(M)
        hc(M) = M[2, 2]
        hd() = hc(M)
        @testset "Long form type-stable" begin
            @test @inferred(fa(p)) === 1.0kg∙s⁻¹
            @test @inferred(fb()) === 1.0kg∙s⁻¹
            @test @inferred(fc(p)) === 1.0kg∙s⁻¹
            @test @inferred(fd()) === 1.0kg∙s⁻¹
        end
        @testset "Normal form type-stable" begin
            @test @inferred(ga(p)) === 1.0kg∙s⁻¹
            @test @inferred(gb()) === 1.0kg∙s⁻¹
            @test @inferred(gc(p)) === 1.0kg∙s⁻¹
            @test @inferred(gd()) === 1.0kg∙s⁻¹
        end
        @testset "Normal form matrix type-stable" begin
            @test @inferred(ha(M)) === 4kg
            @test @inferred(hb()) === 4kg
            @test @inferred(hc(M)) === 4kg
            @test @inferred(hd()) === 4kg
        end
    end
end

