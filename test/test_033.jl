# Test BoundaryValueDiffEq with dimensional residuals and dimensional NLsolve
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase # exports ArrayPartition
using OrdinaryDiffEq
using BoundaryValueDiffEq: BVProblem,  GeneralMIRK4, Shooting
using MechanicalUnits: @import_expand, ∙, ustrip, unit
using MechanicalUnits: preferunits, upreferred
import NLSolversBase
using NLSolversBase: value_jacobian!!
@import_expand(W, m, cm, K, g, kg, J, s)
using MechanicalUnits: °C

include(joinpath(@__DIR__, "debug_logger.jl"))
function samedim´!(u´, u , p, t)
    h, A, T_e, Cp, m1, m2  = p
    T1, T2  = u
    Q´1 = h * A *(T1 − T_e) / (Cp * m1)
    Q´2 = h * A *(T2 − T1) / (Cp * m2)
    u´ .= (Q´1, Q´2)
    u´
end

function bco!(u, sol, p, t)
    u_bc = sol[end]
    u[1] = u_bc[1] + u_bc[2] - 2 * 320K
    u[2] = u_bc[1] - 330K
    u
end




#######################################
# Mutable ArrayPartition with equal dimension
#######################################
#@testset "OnceDifferentiable values, mutable ArrayParitition, dimensional" begin
    alg = Shooting(Tsit5())
    p = convert_to_mixed(-40W/(m²∙K), 100cm², 298K, 	0.466J/(g∙K), 1.0kg, 1.0kg)
    u0 = convert_to_mixed(398.0K, 498.0K)
    @inferred samedim´!(u0/s, u0 , p, :t)
    bvp = BVProblem(samedim´!, bco!, u0, (0.0, 3600.0)s, p)

    # We could now skip to SKIPTO below.
    #=
    pro = ODEProblem(samedim´!, bvp.u0, bvp.tspan, bvp.p)
    solguess = solve(pro, alg.ode_alg, dtmax = 10.0);

    bco!(resid, solguess, p, :t)
    @test resid[1] ≈ -16.292291651286405K
    @test resid[2] ≈ -27.45039038389706K

    # The internal loss function defined by BoundaryValueDiffEq, used to minimize
    # the residual would look somewhat like this.
    g!(bvp) = function (resid, minimizer)
        tmp_prob = remake(bvp, u0 = minimizer)
        sol = solve(tmp_prob, alg.ode_alg;dtmax = 0.05)
        bco!(resid, sol, nothing, sol.t)
        resid
    end
    dg = OnceDifferentiable(g!(bvp), u0, solguess[end])
    @test sum(isnan.(dg.x_f)) == 2
    @test sum(isnan.(dg.x_df)) == 2
    @test is_vector_mutable_stable(dg.F)
    @test is_square_matrix_mutable(dg.DF)
    @test zero.(dg.F) == [0.0K, 0.0K]
    @test zero.(dg.DF) == [0.0 0.0; 0.0 0.0]
    dg = "forget it, test ok"
    =#
    # SKIPTO 
    sol = solve(bvp, alg, dtmax = 10.0);
    initialvalues = sol[1]
    endvalues = sol[end]
    # Residual (i.e. the misfit to boundary conditions) dummy
    resid = convert_to_mixed(NaN, NaN)K
    bco!(resid, sol, nothing, sol.t)
    @test all(isapprox.(resid, [0.0, 0.0]K; atol = 1e-6K))

#end

