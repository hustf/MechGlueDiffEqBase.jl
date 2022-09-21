using Test
using FiniteDiff, StaticArrays
import MechGlueDiffEqBase
using MechGlueDiffEqBase: convert_to_mixed
using MechanicalUnits: @import_expand, ∙
@import_expand kg s m
#####################################
# I Jacobian, not as in NLSolve calls
#####################################
# Based on
# https://github.com/JuliaDiff/FiniteDiff.jl/README.md

function jacob_static(vconstructor, N)
    fcalls = 0
    handleleft(x, i) = i==1 ? zero(eltype(x)) : x[i-1]
    handleright(x, i) = i==length(x) ? zero(eltype(x)) : x[i+1]
    function g(x) # out-of-place
        fcalls += 1
        vconstructor([handleleft(x, i) - 2x[i] + handleright(x,i) for i in 1:N])
    end
    x = vconstructor(rand(N))
    J = FiniteDiff.finite_difference_jacobian(g, x)
    J, fcalls
end
Jcompare = [-2.0 1.0 0.0; 1.0 -2.0 1.0; 0.0 1.0 -2.0]
@time @testset "Jacobian, allocating, Static Vector" begin
    J, fcalls = jacob_static(SVector{3, Float64}, 3)
    @test J == Jcompare
    @test fcalls == 4
end
@time @testset "Jacobian, allocating, vector" begin
    J, fcalls = jacob_static(Vector{Float64}, 3)
    @test J == Jcompare
    @test fcalls == 4
end
@time @testset "Jacobian, allocating, mutable ArrayPartition" begin
    J, fcalls = jacob_static(convert_to_mixed, 3)
    @test J == Jcompare
    @test fcalls == 4
end
#=
# TODO fix fcalls thing from above.
# Now test in-place and out-of-place
# Also fdtypes, and complex functions.*

# Here, the types of f(x) are different from the types of f.
function f!(F, x)
    F[1] = (x[1] + 3kg) + (x[2] + 1s)∙kg/s + x[3]∙kg/m
    F[2] = x[2]
    F[3] = (x[3] + 1m) ∙ s
    F
end
function f(x)
    Fv = convert_to_mixed(1.0kg, 2.0s, 3.0m∙s)
    f!(Fv, x)
end

vx = [1.0kg, 2s, 3m]
x = convert_to_mixed(1.0kg, 2s, 3m)

with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
    FiniteDiff.finite_difference_jacobian(f, x)
end
FiniteDiff.finite_difference_jacobian(f, x)




f_3by3!(Fv, xv)

=#