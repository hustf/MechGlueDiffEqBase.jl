# Test BoundaryValueDiffEq with dimensional residuals and dimensional NLsolve
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase
using OrdinaryDiffEq
using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4, Shooting
using MechanicalUnits: @import_expand, ∙, ustrip, unit
import NLSolversBase
using NLSolversBase: value_jacobian!!
@import_expand(cm, kg, s)


function simplependulum´!(u´, u , p, t)
    g, L = p
    θ  = u[1]
    θ´ = u[2]
    u´[1] = θ´
    u´[2] = -(g/L) * sin(θ) # θ´´
    @debug "simplependulum´!" typeof(u´) maxlog = 1
    u´
end

function bca!(u, sol, p, t)
    @debug "bca!" string(u) supertype(typeof(sol)) maxlog = 2
    umid = sol[end÷2]
    uend = sol[end]
    # The solution at the middle of the time span should be -π/2 (radians).
    # We make this a little more unit-generic than necessary.
    u[1] = umid[1] + π/2 * oneunit(u[1])
    # The solution at the end of the time span should be π/2.
    # We need to express the evaluation in the units that u[2] has initially.
    u[2] = uend[1] * oneunit(u[2]) / oneunit(u[1]) - (π/2 * oneunit(u[2]))
    @debug "bca! -> " string(u) maxlog = 2
    u
end



# The internal loss function defined by BoundaryValueDiffEq, used to minimize
# the residual would look somewhat like this.
f!(bvp) = function (resid, minimizer)
    tmp_prob = remake(bvp, u0 = minimizer)
    sol = solve(tmp_prob, alg.ode_alg;dtmax = 0.05)
    bca!(resid, sol, nothing, sol.t)
    resid
end


alg =  Shooting(Tsit5())
jamate = [-0.4157163819928267 0.28583325159711503; -0.5503793078791553 -0.3188487145980475]

#############################################################
# Vector types - simplest but hard to compile with quantities
#############################################################
xv = [-π/4, π/8]  # θ, θ´
@testset "OnceDifferentiable values, vector types" begin
    x = xv
    bvp = BVProblem(simplependulum´!, bca!, x, (0.0, 1.5), (9.81, 1.0))
    resid = [0.0, 0.0] # Residual dummy
    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df = OnceDifferentiable(f!(bvp), x, resid)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df.x_f)) == 2
    @test sum(isnan.(df.x_df)) == 2
    @test df.F isa Vector{Float64}
    @test df.DF isa Matrix{Float64}
    @test sum(iszero.(df.F)) == 2
    @test sum(isnan.(df.DF)) == 4
    @test zero.(df.F) == [0.0, 0.0]
    @test zero.(df.DF) == [0.0 0.0; 0.0 0.0]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial f values.
    prob = ODEProblem(simplependulum´!, bvp.u0, bvp.tspan, bvp.p)
    solguess = solve(prob, alg.ode_alg, dtmax = 0.05);
    @inferred bca!(resid, solguess, NaN, solguess.t)
    @test all(isapprox.(resid, [2π/3, -π/2], rtol = 0.02))
    #    Check if OnceDifferentiable finds the same residual internally.
    @test all(isapprox.(resid, NLSolversBase.value!!(df, x)))
    #    Jacobian of the residual
    @test all(NLSolversBase.jacobian!!(df, x) .≈ jamate)
    # c) Minimize the residual. Is the solution satisfying boundary conditions?
    sol = solve(bvp, alg, dtmax = 0.05);
    @test all(isapprox.(bca!(resid, sol, nothing, sol.t), [0.0, 0.0]; atol = 1e-6))
    # d) Test the OnceDifferentiable interface more
    @test all(NLSolversBase.value_jacobian!!(df, x)[2] .≈ jamate)
end
    ########################
    # Mutable ArrayPartition
    ########################
@testset "OnceDifferentiable values, mutable ArrayParitition, nondimensional" begin
    x = ArrayPartition([xv[1]], [xv[2]])  # θ, θ´
    bvp = BVProblem(simplependulum´!, bca!, x, (0.0, 1.5), (9.81, 1.0))
    resid = ArrayPartition([0.0], [0.0])
    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df = OnceDifferentiable(f!(bvp), x, resid)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df.x_f)) == 2
    @test sum(isnan.(df.x_df)) == 2
    @test is_vector_mutable_stable(df.F)
    @test is_square_matrix_mutable(df.DF)
    @test sum(iszero.(df.F)) == 2
    @test sum(isnan.(df.DF)) == 4
    @test zero.(df.F) == [0.0, 0.0]
    @test zero.(df.DF) == [0.0 0.0; 0.0 0.0]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial f values.
    prob = ODEProblem(simplependulum´!, bvp.u0, bvp.tspan, bvp.p)
    solguess = solve(prob, alg.ode_alg, dtmax = 0.05);
    @inferred bca!(resid, solguess.u, NaN, solguess.t)
    @test all(isapprox.(resid, [2π/3, -π/2], rtol = 0.02))
    #    Check if OnceDifferentiable finds the same residual internally.
    @test all(isapprox.(resid, NLSolversBase.value!!(df, x)))
    #    Jacobian of the residual
    @test all(NLSolversBase.jacobian!!(df, x) .≈ jamate)

    # c) Minimize the residual. Is the solution satisfying boundary conditions?
    sol = solve(bvp, alg, dtmax = 0.05);
    @test all(isapprox.(bca!(resid, sol, nothing, sol.t), [0.0, 0.0]; atol = 1e-6))
    # d) Test the OnceDifferentiable interface more
    @test all(NLSolversBase.value_jacobian!!(df, x)[2] .≈ jamate)
end

    #######################################
    # Mutable ArrayPartition with dimension
    #######################################
@testset "OnceDifferentiable values, mutable ArrayParitition, dimensional" begin
    x = ArrayPartition([xv[1]], [xv[2]]s⁻¹)  # θ, θ´
    bvp = BVProblem(simplependulum´!, bca!, x, (0.0, 1.5)s, (981cm/s², 100.0cm))
    resid = ArrayPartition([0.0], [0.0]s⁻¹) # Dimensionless loss function, not the hardest case

    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df = OnceDifferentiable(f!(bvp), x, resid)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df.x_f)) == 2
    @test unit.(df.x_f)[2] == s⁻¹
    @test sum(isnan.(df.x_df)) == 2
    @test unit.(df.x_df)[2] == s⁻¹
    @test is_vector_mutable_stable(df.F)
    @test is_square_matrix_mutable(df.DF)
    @test sum(iszero.(df.F)) == 2
    @test sum(isnan.(df.DF)) == 4
    @test zero.(df.F) == [0.0, 0.0s⁻¹]
    @test zero.(df.DF) == [0.0 0.0s; 0.0s⁻¹ 0.0]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial f values.
    prob = ODEProblem(simplependulum´!, bvp.u0, bvp.tspan, bvp.p)
    solguess = solve(prob, alg.ode_alg, dtmax = 0.05);
    @inferred bca!(resid, solguess.u, NaN, solguess.t)
    @test all(isapprox.(resid, [2π/3, -(π/2)s⁻¹], rtol = 0.02))
    #    Check if OnceDifferentiable finds the same residual internally.
    @test all(isapprox.(resid, NLSolversBase.value!!(df, x)))
    #    Jacobian of the residual
    @test all(NLSolversBase.jacobian!!(df, x) .≈ convert_to_mixed(jamate .* [1 s;s⁻¹ 1]))
    # c) Minimize the residual. Is the solution satisfying boundary conditions?
     sol = solve(bvp, alg, dtmax = 0.05s);
     # Comparing quantities close to zero requires absolute tolerance, not relative. 
     @test isapprox(bca!(resid, sol, nothing, sol.t)[1], 0.0; atol = 1e-6)
     @test isapprox(bca!(resid, sol, nothing, sol.t)[2], 0.0s⁻¹; atol = 1e-6s⁻¹)
     # d) Test the OnceDifferentiable interface more
     J = NLSolversBase.value_jacobian!!(df, x)[2]
     @test all(J .≈ convert_to_mixed(jamate .* [1 s;s⁻¹ 1]))
     # e) Printing
     iob = IOBuffer()
     println(iob, sol)
     @test length(String(take!(iob))) > 100
end

