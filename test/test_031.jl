# Test OnceDifferentiable construction and display with mutable ArrayParititions
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase # exports ArrayPartition
using OrdinaryDiffEq
using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4, Shooting
using MechanicalUnits: @import_expand, ∙, ustrip, unit
import NLSolversBase
using NLSolversBase: value_jacobian!!
@import_expand(cm, kg, s)

"""
Debug formatter, highlight NLSolversBase. To use:
```
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
    @test ...
end
```
"""
function locfmt(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    if repr(_module) == "FiniteDiff"
        color = :green
    elseif repr(_module) == "Main"
        color = :176
    elseif repr(_module) ==  "MechGlueDiffEqBase"
        color = :magenta
    else
        color = :blue
    end
    prefix = string(level == Logging.Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end

function simplependulum´!(u´, u , p, t)
    g, L = p
    θ  = u[1]
    θ´ = u[2]
    u´[1] = θ´
    u´[2] = -(g/L) * sin(θ) # θ´´
    @debug "simplependulum´!" typeof(u´) maxlog=1
    u´
end

function bca!(residual, u, p, t)
    # the solution at the middle of the time span should be 0
    mid = searchsortedfirst(t, t[end] / 2)
    # residual is an ODESolution.
    residual[1] = u[mid][1]
    # the solution at the end of the time span should be π/2
    residual[2] = u[end][1] - π / 2
    @debug "bca!" string(u[1]) string(residual) maxlog = 1
    residual
end



# The internal loss function defined by BoundaryValueDiffEq, used to minimize 
# the residual would look somewhat like this.
f!(bvp) = function (resid, minimizer)
    tmp_prob = remake(bvp, u0=minimizer)
    sol = solve(tmp_prob, alg.ode_alg;dtmax = 0.05)
    bca!(resid, sol, nothing, sol.t)
    resid
end


alg =  Shooting(Tsit5())

#############################################################
# Vector types - simplest but hard to compile with quantities
#############################################################
x1 = [-π/4, π/8]  # θ, θ´
@testset "OnceDifferentiable values, vector types" begin
    bvp1 = BVProblem(simplependulum´!, bca!, x1, (0.0, 1.5), (9.81, 1.0))

    resid1 = [0.0, 0.0] # Residual dummy
    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df1 = OnceDifferentiable(f!(bvp1), x1, resid1)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df1.x_f)) == 2
    @test sum(isnan.(df1.x_df)) == 2
    @test df1.F isa Vector{Float64}
    @test df1.DF isa Matrix{Float64}
    @test sum(iszero.(df1.F)) == 2
    @test sum(isnan.(df1.DF)) == 4
    @test zero.(convert_to_array(df1.F)) == [0.0, 0.0]
    @test zero.(convert_to_array(df1.DF)) == [0.0 0.0; 0.0 0.0]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial df values.
    prob1 = ODEProblem(simplependulum´!, bvp1.u0, bvp1.tspan, bvp1.p) 
    solguess1 = solve(prob1, Tsit5(), dtmax = 0.05);
    bca!(resid1, solguess1, NaN, solguess1.t) 
    @test isapprox(resid1, [0.6, -π/2], atol = 0.1)
    #    Check if OnceDifferentiable finds the same residual internally.
    @test isapprox(resid1, NLSolversBase.value!!(df1, x1))
    #    Jacobian of the residual
    jamate = [-0.5458791359906279 0.2554698462543077; -0.5503793078424869 -0.3188487145980475]
    @test isapprox(convert_to_array(NLSolversBase.jacobian!!(df1, x1)) ./
        jamate, [1.0 1.0; 1.0 1.0])
    # c) Minimize the residual. Is the solution satisfying boundary conditions?
    sol1 = solve(bvp1, alg, dtmax = 0.05);
    @test isapprox(bca!(resid1, sol1, nothing, sol1.t), [0.0, 0.0]; atol = 1e-6)
    # d) Test the OnceDifferentiable interface more
    @test isapprox(convert_to_array(NLSolversBase.value_jacobian!!(df1, x1)[2]) ./
        jamate, [1.0 1.0; 1.0 1.0])
end
    ########################
    # Mutable ArrayPartition 
    ########################
@testset "OnceDifferentiable values, mutable ArrayParitition, nondimensional" begin
    x2 = ArrayPartition([x1[1]], [x1[2]])  # θ, θ´
    bvp2 = BVProblem(simplependulum´!, bca!, x2, (0.0, 1.5), (9.81, 1.0))
    resid2 = ArrayPartition([0.0], [0.0])
    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df2 = OnceDifferentiable(f!(bvp2), x2, resid2)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df2.x_f)) == 2
    @test sum(isnan.(df2.x_df)) == 2
    @test is_vector_mutable_stable(df2.F)
    @test is_square_matrix_mutable(df2.DF)
    @test sum(iszero.(df2.F)) == 2
    @test sum(isnan.(df2.DF)) == 4
    @test zero.(convert_to_array(df2.F)) == [0.0, 0.0]
    @test zero.(convert_to_array(df2.DF)) == [0.0 0.0; 0.0 0.0]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial df values.
    prob2 = ODEProblem(simplependulum´!, bvp2.u0, bvp2.tspan, bvp2.p) 
    solguess2 = solve(prob2, Tsit5(), dtmax = 0.05);
    bca!(resid2, solguess2.u, NaN, solguess2.t) 
    @test isapprox(resid2, [0.6, -π/2], atol = 0.05)
    #    Check if OnceDifferentiable finds the same residual internally.
    @test isapprox(resid2, NLSolversBase.value!!(df2, x2))
    #    Jacobian of the residual
    @test isapprox(convert_to_array(NLSolversBase.jacobian!!(df2, x2)) ./
            jamate, [1.0 1.0; 1.0 1.0])
    # c) Minimize the residual. Is the solution satisfying boundary conditions?
    sol2 = solve(bvp2, alg, dtmax = 0.05);
    @test isapprox(bca!(resid2, sol2, nothing, sol2.t), [0.0, 0.0]; atol = 1e-6)
    # d) Test the OnceDifferentiable interface more
    @test isapprox(convert_to_array(NLSolversBase.value_jacobian!!(df2, x2)[2]) ./
        jamate, [1.0 1.0; 1.0 1.0])
end

    #######################################
    # Mutable ArrayPartition with dimension 
    #######################################
@testset "OnceDifferentiable values, mutable ArrayParitition, dimensional" begin
    x3 = ArrayPartition([x1[1]], [x1[2]]s⁻¹)  # θ, θ´
    bvp3 = BVProblem(simplependulum´!, bca!, x3, (0.0, 1.5)s, (981cm/s², 100.0cm))
    resid3 = ArrayPartition([0.0], [0.0]) # Dimensionless loss function, not the hardest case

    # NLSolversBase\src\objective_types\oncedifferentiable.jl:23 (-> :98)
    df3 = OnceDifferentiable(f!(bvp3), x3, resid3)
    # a) Is the intitalization of OnceDifferentiable as expected?
    @test sum(isnan.(df3.x_f)) == 2
    @test unit.(df3.x_f)[2] == s⁻¹
    @test sum(isnan.(df3.x_df)) == 2
    @test unit.(df3.x_df)[2] == s⁻¹
    @test is_vector_mutable_stable(df3.F)
    @test is_square_matrix_mutable(df3.DF)
    @test sum(iszero.(df3.F)) == 2
    @test sum(isnan.(df3.DF)) == 4
    @test zero.(convert_to_array(df3.F)) == [0.0, 0.0]
    @test zero.(convert_to_array(df3.DF)) == [0.0 0.0s; 0.0 0.0s]
    # b) Is the residual and Jacobian of residual as expected?
    #    Update resid 'manually', using the initial df values.
    prob3 = ODEProblem(simplependulum´!, bvp3.u0, bvp3.tspan, bvp3.p) 
    solguess3 = solve(prob3, Tsit5(), dtmax = 0.05);
    bca!(resid3, solguess3.u, NaN, solguess3.t) 
    @test isapprox(resid3, [0.6, -π/2], atol = 0.05)
    #    Check if OnceDifferentiable finds the same residual internally.
    @test isapprox(resid3, NLSolversBase.value!!(df3, x3))
    #    Jacobian of the residual
    @test isapprox(convert_to_array(NLSolversBase.jacobian!!(df3, x3)) ./
        hcat(jamate[:,1], jamate[:,2]s), [1.0 1.0; 1.0 1.0]) 
    # c) Minimize the residual. Is the solution satisfying boundary conditions?
    alg =  Shooting(Tsit5(), nlsolve=DIMENSIONAL_NLSOLVE)

    sol3 = solve(bvp3, alg, dtmax = 0.05s);
    @test isapprox(bca!(resid3, sol3, nothing, sol3.t), [0.0, 0.0]; atol = 1e-6)
    # d) Test the OnceDifferentiable interface more
    @test isapprox(convert_to_array(NLSolversBase.value_jacobian!!(df3, x3)[2]) ./
        hcat(jamate[:,1], jamate[:,2]s), [1.0 1.0; 1.0 1.0])
end


#=

# Temp
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
    NLSolversBase.jacobian!!(df3, x3)
end
df3.x_df = x3
@enter df3.df(df3.DF, df3.x_df)

=#