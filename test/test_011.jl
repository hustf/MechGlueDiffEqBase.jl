# Norm, ABS2 zero for ArrayPartions and 'mixed matrices'.

using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, ∙
using BenchmarkTools # temp
@import_expand(cm, mm, kg, s)
######################################################
# D ABS2, zero
# Similar to test_004, but with new argument structure
######################################################
@testset "Extended UNITLESS_ABS2, selected argument structures " begin
    Vn6a = [1131.34, -2282.343, 6672.423, -5.64305, 4.30333, 2.42879]
    Vn6 = convert_to_mixed(Vn6a)
    Vu6ac = Vn6a∙kg
    Vu6c = convert_to_mixed(Vu6ac)
    Vu6 = convert_to_mixed(Vn6.*[kg, cm, s, mm, kg/kg, 1])
    Vu6r = convert_to_mixed(Vn6.* reverse([kg, cm, s, mm, kg/kg, 1]))
    Vn4a = [1, 3, 5, 7]
    Vn4 = convert_to_mixed(Vn4a)
    Vu4 = Vn4a∙kg
    Mn2a = [1 3; 5 7]
    Mn2 = convert_to_mixed(Mn2a)
    Mu2a = [1kg 3; 5 7cm]
    Mu2 = convert_to_mixed(Mu2a)
    normval = 5.101030471786125e7
    @testset "Correctness " begin
        for v in (Vn6a, Vn6, Vu6ac, Vu6c, Vu6, Vu6r)
            @test UNITLESS_ABS2(v) === normval
        end
        for v in (Vn4a, Vn4, Vu4, Mn2a, Mn2, Mu2a, Mu2)
            #println(v)
            @test UNITLESS_ABS2(v) === 84
        end
    end
    @testset "Inferrability" begin
        for v in (Vn6a, Vn6, Vu6ac, Vu6c, Vu6, Vu6r, Vn4a, Vn4, Vu4, Mn2a, Mn2, Mu2a, Mu2)
            if v isa Vector{<:Quantity} ||
                v isa ArrayPartition{<:Quantity} ||
                v isa ArrayPartition{<:Real} ||
                v isa Matrix{Int64} ||
                v isa Matrix{Quantity{Int64}}
                    #println("inferred ", typeof(v) )
                    #println(v)
                    @test @inferred(UNITLESS_ABS2(v)) isa Real
                    #println()
            else
                #println("v = ", v)
                #println(typeof(v), "\t\t", Base.return_types(UNITLESS_ABS2, (typeof(v),))[1])
                #println()
                @test true
            end
        end
    end
end
####################
# E ODE_DEFAULT_NORM
####################
@testset "Extended ODE_DEFAULT_NORM, selected argument structures " begin
    Vn6a = [1131.34, -2282.343, 6672.423, -5.64305, 4.30333, 2.42879]
    Vn6 = convert_to_mixed(Vn6a)
    Vu6ac = Vn6a∙kg
    Vu6c = convert_to_mixed(Vu6ac)
    Vu6 = convert_to_mixed(Vn6.*[kg, cm, s, mm, kg/kg, 1])
    Vu6r = convert_to_mixed(Vn6.* reverse([kg, cm, s, mm, kg/kg, 1]))
    Vn4a = [5, 5, 5, 5]
    Vn4 = convert_to_mixed(Vn4a)
    Vu4 = Vn4a∙kg
    Mn2a = [5 5; 5 5]
    Mn2 = convert_to_mixed(Mn2a)
    Mu2a = [5kg 5; 5 5cm]
    Mu2 = convert_to_mixed(Mu2a)
    normval = 2915.770473301504
    @testset "Correctness " begin
        for v in (Vn6a, Vn6, Vu6ac, Vu6c, Vu6, Vu6r)
            @test ODE_DEFAULT_NORM(v, 0) === normval
        end
        for v in (Vn4a, Vn4, Vu4, Mn2a, Mn2, Mu2a, Mu2)
            @test ODE_DEFAULT_NORM(v, 0) === 5.0
        end
    end
    @testset "Inferrability" begin
        for v in (Vn6a, Vn6, Vu6ac, Vu6c, Vu6, Vu6r, Vn4a, Vn4, Vu4, Mn2a, Mn2, Mu2a, Mu2)
            if v isa Vector{<:Quantity} ||
                v isa ArrayPartition{<:Quantity} ||
                v isa ArrayPartition{<:Real} ||
                v isa Matrix{Int64} ||
                v isa Matrix{Quantity{Int64}} ||
                v isa Vector{Float64}
                    @test @inferred(ODE_DEFAULT_NORM(v, 0)) isa Real
            else
                @test true
            end
        end
    end
end

nothing