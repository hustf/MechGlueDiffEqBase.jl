# Check type stability of ODE_DEFAULT_NORM with mixed units and with ArrayPartition
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, dimension, NoDims, ∙
import MechanicalUnits: g, g⁻¹
@import_expand(km, N, s, m, km, kg, °, inch)
using DiffEqBase, OrdinaryDiffEq

@testset "Inferrable norm Unitless ArrayPartition" begin
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = @inferred ArrayPartition(r0ul,v0ul)
    @test @inferred(ODE_DEFAULT_NORM(rv0ul, 0.0)) === 2915.770473301504
end
@testset "Inferrable norm  ArrayPartition units" begin
    r0 = [1131.340, -2282.343, 6672.423]∙km
    v0 = [-5.64305, 4.30333, 2.42879]∙km/s
    rv0 = @inferred ArrayPartition(r0, v0)
    @test @inferred(ODE_DEFAULT_NORM(rv0, 0.0s)) === 41.02536038842316
end
@testset "Inferrable norm  ArrayPartition mixed units" begin
    r0 = [1.0km, 2.0km, 3.0m/s, 4.0m/s]
    v0 = [1.0km/s, 2.0km/s, 3.0m/s², 4m/s²]
    rv0 = @inferred ArrayPartition(r0, v0)
    @test @inferred(ODE_DEFAULT_NORM(rv0, 0.0s)) === 1.5811388300841898
end


@testset "Inferrable zero Unitless ArrayPartition" begin
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = ArrayPartition(r0ul, v0ul)
    @test @inferred(zero(rv0ul)) == ArrayPartition([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    @test ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}}} == typeof(zero(rv0ul))
end
@testset "Inferrable zero compatible units" begin
    r0 = [1131.340, -2282.343, 6672.423]∙km
    r1 = [1km, 2.0m]
    @test @inferred(zero(r0)) == [0.0, 0.0, 0.0]km
    @test @inferred(zero(r1)) == [0.0, 0.0]km
    rv0 = ArrayPartition(r0)
    @test @inferred(zero(rv0)) == [0.0, 0.0, 0.0]km
    @test typeof(zero(rv0)) == typeof(rv0)
    rv1 = ArrayPartition(r1)
    @test @inferred(zero(rv1)) == [0.0, 0.0]km
    @test typeof(zero(rv0)) == typeof(rv0)
end
@testset "Inferrable zero vector and ArrayPartition mixed units" begin
    r0 = [1.0km, -2km, 3m/s, 4m/s]
    @test @inferred(zero(r0)) == [0.0km, 0.0km, 0.0m/s, 0.0m/s]
    rv0 = @inferred ArrayPartition(r0)
    zer = @inferred zero(rv0)
    @test zer == [0.0km, 0.0km, 0.0m/s, 0.0m/s]
    @test typeof(zer) === typeof(rv0)
end

@testset "Inferrable similar vector and ArrayPartition compatible units" begin
    r0 = [1131.340, -2282.343, 6672.423]∙km
    simi = @inferred(similar(r0))
    @test all(typeof.(r0) == typeof.(simi))
    @test simi !== r0
    rv0 = @inferred ArrayPartition(r0)
    sima = @inferred similar(rv0)
    @test all(typeof.(rv0) == typeof.(sima))
    @test sima !== rv0
end


@testset "Inferrable similar vector and ArrayPartition mixed units" begin
    r0 = [1.0km, -2km, 3m/s, 4m/s]
    simi = @inferred(similar(r0))
    @test all(typeof.(r0) == typeof.(simi))
    @test simi !== r0
    rv0 = @inferred ArrayPartition(r0)
    sima = @inferred similar(rv0)
    @test all(typeof.(rv0) == typeof.(sima))
    @test sima !== rv0
end

@testset "Inferrable UNITLESS_ABS2 Unitless ArrayPartition" begin
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = ArrayPartition(r0ul,v0ul)
    @test @inferred(UNITLESS_ABS2(rv0ul)) == 5.101030471786125e7
end

@testset "Inferrable UNITLESS_ABS2  ArrayPartition mixed units" begin
    r0 = [1.0km, 2.0km, 3.0m/s, 4.0m/s]
    v0 = [1.0km/s, 2.0km/s, 3.0m/s², 4m/s²]
    rv0 = ArrayPartition(r0, v0)
    @test @inferred(UNITLESS_ABS2(1.0km)) == 1.0
    @test @inferred(UNITLESS_ABS2(r0)) == [1.0, 4.0, 9.0, 16.0]
    @test @inferred(UNITLESS_ABS2(rv0)) == 2 .* [1.0, 4.0, 9.0, 16.0]
end