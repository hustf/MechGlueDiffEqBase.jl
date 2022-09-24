using Test
using FiniteDiff, StaticArrays
using MechGlueDiffEqBase
using MechGlueDiffEqBase: finite_difference_jacobian, finite_difference_jacobian!
using MechanicalUnits: @import_expand, ustrip
#using FiniteDiff: JacobianCache
#using MechGlueDiffEqBase.RecursiveArrayTools: ArrayPartition

@import_expand kg s m
#####################################
# I Jacobian, not as in NLSolve calls
#####################################
# Based on
# https://github.com/JuliaDiff/jl/README.md

function jacob_static(vconstructor, N; x = vconstructor(rand(N)))
    gcalls = 0
    handleleft(x, i) = i==1 ? zero(eltype(x)) : x[i-1]
    handleright(x, i) = i==length(x) ? zero(eltype(x)) : x[i+1]
    function g(x) # out-of-place
        gcalls += 1
        vconstructor([handleleft(x, i) - 2x[i] + handleright(x,i) for i in 1:N])
    end
    J = finite_difference_jacobian(g, x)
    J, gcalls
end
function jacob_mutating(vconstructor, mconstructor, N; x = vconstructor(rand(N)), J = mconstructor(rand(N, N)))
    gcalls = 0
    handleleft(x, i) = i==1 ? zero(eltype(x)) : x[i-1]
    handleright(x, i) = i==length(x) ? zero(eltype(x)) : x[i+1]
    function g!(dx, x) # in-place, mutating
        gcalls += 1
        for i in 2:length(x) - 1
            dx[i] = x[i-1] - 2x[i] + x[i+1]
        end
        dx[1] = -2x[1] + x[2]
        dx[end] = x[end-1] - 2x[end]
        dx
    end
    finite_difference_jacobian!(J, g!, x)
    J, gcalls
end
Jcompare = [-2.0 1.0 0.0; 1.0 -2.0 1.0; 0.0 1.0 -2.0]
@testset "Dimensionless, allocating and mutating Jacobians" begin
    @testset "Allocating" begin
        @testset "Jacobian, allocating, Static Vector" begin
            vconstructor = SVector{3, Float64}
            J, gcalls = jacob_static(vconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
        @testset "Jacobian, allocating, vector" begin
            vconstructor = Vector{Float64}
            J, gcalls = jacob_static(vconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
        @testset "Jacobian, allocating, mutable ArrayPartition" begin
            vconstructor = convert_to_mixed
            J, gcalls = jacob_static(vconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
    end
    @testset "Mutating" begin
        @testset "Jacobian, mutating MMatrix" begin
            vconstructor = MVector{3, Float64}
            mconstructor = MMatrix{3, 3, Float64}
            J, gcalls = jacob_mutating(vconstructor, mconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
        @testset "Jacobian, mutating, Matrix" begin
            vconstructor = Vector{Float64}
            mconstructor = Matrix{Float64}
            J, gcalls = jacob_mutating(vconstructor, mconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
        @testset "Jacobian, mutating, mutable ArrayPartition" begin
            vconstructor = convert_to_mixed
            mconstructor = convert_to_mixed
            J, gcalls = jacob_mutating(vconstructor, mconstructor, 3)
            @test J == Jcompare
            @test gcalls == 4
        end
    end
end
@testset "f(x) and x same dimensions, allocating and mutating Jacobians" begin
    @testset "Mutating" begin
        @testset "Jacobian, mutating MMatrix" begin
            vconstructor = MVector{3, typeof(1.0m)}
            mconstructor = MMatrix{3, 3, typeof(1.0m)}
            x = vconstructor([1.0m, 2.0m, 3.0m])
            J = mconstructor(rand(3, 3)m)
            @test_throws MethodError jacob_mutating(vconstructor, mconstructor, 3; x, J)
        end
        @testset "Jacobian, mutating, Matrix" begin
            vconstructor = Vector{typeof(1.0m)}
            mconstructor = Matrix{typeof(1.0m)}
            x = vconstructor([1.0m, 2.0m, 3.0m])
            J = mconstructor(rand(3, 3)m)
            @test_throws MethodError jacob_mutating(vconstructor, mconstructor, 3; x, J)
        end
        @testset "Jacobian, mutating, mutable ArrayPartition" begin
            vconstructor = convert_to_mixed
            mconstructor = convert_to_mixed
            x = vconstructor([1.0, 2.0, 3.0]m)
            J = mconstructor(rand(3, 3))
            J, gcalls = jacob_mutating(vconstructor, mconstructor, 3; x, J)
            @test J == Jcompare
            @test gcalls == 4
        end
    end
    @testset "Allocating" begin
        @testset "Jacobian, allocating, Static Vector" begin
            vconstructor = SVector{3, typeof(1.0m)}
            x = vconstructor(1.0m, 2.0m, 3.0m)
            @test_throws MethodError jacob_static(vconstructor, 3; x)
        end
        @testset "Jacobian, allocating, vector" begin
            vconstructor = Vector{typeof(1.0m)}
            x = vconstructor([1.0m, 2.0m, 3.0m])
            @test_throws MethodError jacob_static(vconstructor, 3; x)
        end
        @testset "Jacobian, allocating, mutable ArrayPartition" begin
            vconstructor = convert_to_mixed
            x = vconstructor(1.0m, 2.0m, 3.0m)
            J, gcalls = jacob_static(vconstructor, 3; x)
            @test J == Jcompare
            @test gcalls == 4
        end
    end
end

@testset "JacobianCache fdtype :complex" begin
    function paramsdic(cache::JacobianCache{CacheType1, CacheType2, CacheType3, CacheType4, ColorType, SparsityType, fdtype, returntype}) where {CacheType1, CacheType2, CacheType3, CacheType4, ColorType, SparsityType, fdtype, returntype}
        Dict(:CacheType1 => CacheType1, :CacheType2 => CacheType2, :CacheType3 => CacheType3, :CacheType4 => CacheType4, 
             :ColorType => ColorType, :SparsityType => SparsityType, :fdtype => fdtype, :returntype => returntype)
    end
    @testset "Real, default" begin
        x = rand(3)
        ca = JacobianCache(x)
        c = JacobianCache(convert_to_mixed(x))
        @test c.x1 == ca.x1
        @test c.x2 == ca.x2
        @test c.fx == ca.fx
        @test c.fx1 == ca.fx1
        @test paramsdic(ca)[:fdtype] == paramsdic(c)[:fdtype]
        @test paramsdic(ca)[:returntype] == paramsdic(c)[:returntype]
        @test paramsdic(ca)[:CacheType1] == Vector{Float64}
        @test paramsdic(c)[:CacheType1] == ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    end
    @testset "Complex, default" begin
        x = rand(ComplexF64, 3)
        ca = JacobianCache(x)
        c = JacobianCache(convert_to_mixed(x))
        @test c.x1 == ca.x1
        @test c.x2 == ca.x2
        @test c.fx == ca.fx
        @test c.fx1 == c.fx1
        @test paramsdic(ca)[:fdtype] == paramsdic(c)[:fdtype]
        @test paramsdic(ca)[:returntype] == paramsdic(c)[:returntype]
    end

    @testset "Real, :complex" begin
        fdtype = Val{:complex}
        x = rand(3)
        ca = JacobianCache(x, fdtype)
        c = JacobianCache(convert_to_mixed(x), fdtype)
        @test c.x1 == ca.x1
        @test c.x2 == ca.x2
        @test c.fx == ca.fx
        @test ca.fx1 isa Nothing
        @test c.fx1 isa Nothing
        @test paramsdic(ca)[:fdtype] == paramsdic(c)[:fdtype]
        @test paramsdic(ca)[:returntype] == paramsdic(c)[:returntype]
        @test paramsdic(ca)[:CacheType1] == paramsdic(ca)[:CacheType1]
    end
    @testset "Complex, :complex" begin
        fdtype = Val{:complex}
        x = rand(ComplexF64, 3)
        @test_throws ErrorException JacobianCache(x, fdtype)
        @test_throws ErrorException JacobianCache(convert_to_mixed(x), fdtype)
    end
    @testset "Real quantity, default" begin
        x = rand(3) .* [kg, m, s]
        ca = JacobianCache(x)
        c = JacobianCache(convert_to_mixed(x))
        @test c.x1 == ca.x1
        @test c.x2 == ca.x2
        @test c.fx == ca.fx
        @test c.fx1 == ca.fx1
        @test paramsdic(ca)[:fdtype] == paramsdic(c)[:fdtype]
        @test paramsdic(ca)[:returntype] == paramsdic(c)[:returntype]
    end
    @testset "Real quantity, :complex" begin
        fdtype = Val{:complex}
        xa = rand(3) .* [kg, m, s]
        x = convert_to_mixed(xa)
        fx = copy(x)
        @test_throws ErrorException JacobianCache(xa, fdtype)
        @test_throws ErrorException JacobianCache(x, fdtype)
        # Same temp structure as in 'finite_difference_jacobian'. We avoid making
        # another outside constructor, (which may be necessary to do anyway).
        returntype = eltype(x)
        _x = zero.(complex.(x))
        _fx = zero.(complex.(fx))
        c = JacobianCache(_x, _fx, fdtype, numtype(returntype))
        @test c.x1 == zero.(complex.(x))
        @test c.x2 == zero.(complex.(x))
        @test c.fx == zero.(complex.(fx))
        @test c.fx1 isa Nothing
        @test paramsdic(c)[:fdtype] == Val(:complex)
        @test paramsdic(c)[:returntype] == Float64 # Is changed to Quantity{Float64} by callee.
    end
end


@testset "Vary dims, allocating and mutating Jacobians, fdtype" begin
    foocalls = 0
    # Non-dimensional
    function f_nd(x)
        foocalls += 1
        convert_to_mixed((x[1] + 3) * (x[2]^3 - 343) + 18,
                         sin(x[2] * exp(x[1]) -1 ))
    end
    function f(x)
        foocalls += 1
        convert_to_mixed((x[1] + 3kg) * (x[2]^3 - 343s^3) + 18kg∙s³,
                         sin(x[2] * exp(x[1]/kg)/s -1 )s)
    end
    function f!(F, x)
        foocalls += 1
        F[1] = (x[1] + 3kg) * (x[2]^3 - 343∙s^3) + 18kg∙s³
        F[2] = sin(x[2] * exp(x[1]/kg)/s - 1 )s
        F
    end
    function f_nd!(F, x)
        foocalls += 1
        F[1] = (x[1] + 3) * (x[2]^3 - 343) + 18
        F[2] = sin(x[2] * exp(x[1]) - 1 )
        F
    end

    Jman = [-342s³ 9kg∙s²; 1s∙kg⁻¹ 1] # [df1/dx1 df1/dx2; df2/dx1 df2/dx2]
    x = convert_to_mixed(0.0kg, 7.0s)
    F = convert_to_mixed(0.0kg∙s³, 0.0s)
    f!(F, x)
    @test F[1] == 18.0kg∙s³
    @test f(x)[1] == 18.0kg∙s³
    x = convert_to_mixed(0.0kg, 1.0s)
    x_nd = ustrip(x)
    Jman_nd = ustrip.(Jman)
    for fdsymb in [:forward, :central, :complex]
        @testset "Allocating, $fdsymb" begin
            @testset "Jacobian, allocating, mutable ArrayPartition, $fdsymb" begin
                foocalls = 0
                J = finite_difference_jacobian(f, x)
                relerr = (J - Jman) ./ Jman
                @test all(relerr .< 1e-6)
                @test foocalls == 3
            end
        end
        @testset "Mutating" begin
            @testset "Jacobian, mutating, mutable ArrayPartition, $fdsymb" begin
                foocalls = 0
                J = convert_to_mixed(Jman * 1.0NaN) # Fill with NaN values
                finite_difference_jacobian!(J, f!, x, Val(fdsymb))
                relerr = (J - Jman) ./ Jman
                @test all(relerr .< 1e-6)
                @test foocalls == (fdsymb == :central ? 5 : 3) 
            end
        end
        @testset "Allocating dimensionless, $fdsymb" begin
            @testset "Jacobian, allocating, mutable ArrayPartition, $fdsymb" begin
                foocalls = 0
                J = finite_difference_jacobian(f_nd, x_nd)
                relerr = (J - Jman_nd) ./ Jman_nd
                @test all(relerr .< 1e-6)
                @test foocalls == 3
            end
        end
        @testset "Mutating dimensionless, $fdsymb" begin
            @testset "Jacobian, mutating, mutable ArrayPartition, $fdsymb" begin
                foocalls = 0
                J = convert_to_mixed(Jman_nd * 1.0NaN) # Fill with NaN values
                finite_difference_jacobian!(J, f_nd!, x_nd, Val(fdsymb))
                relerr = (J - Jman_nd) ./ Jman_nd
                @test all(relerr .< 1e-6)
                @test foocalls == (fdsymb == :central ? 5 : 3) 
            end
        end
    end
end
