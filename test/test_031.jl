# Test adaptions to NLSolversBase, NLSolve
# Another NewtonTrustRegion problem, to increase
# test coverage.
# When e.g. x = convert_to_mixed(1.0kg, 2.0kg),
# nlsolve will find: xtol = 0.0kg, which
# fails the immediate type check: xtol<:Real.
# In these cases, user must provide tolerances
# explicitly. Tolerances should be unitless. 
using Test
using MechanicalUnits: @import_expand, ∙
using MechGlueDiffEqBase
using MechGlueDiffEqBase: nlsolve, converged, MixedContent
using Logging
using NLsolve
@import_expand(cm, kg, s, m, N)

@testset "Non-mixed mixed vectors" begin
    function rosenbrock(; x0= [-1.2, 1.0], fx0 = [-1.2, 1.0])
        function f!(fvec, x)
            fvec[1] = (oneunit(eltype(x)) - x[1])m/s
            fvec[2] = 10m/kg * (oneunit(eltype(x)) * x[2] - x[1]^2)
            fvec
        end
        function j!(fjac, x)
            fjac[1,1] = -1.0m/s
            fjac[1,2] = 0.0m/s
            fjac[2,1] = -20x[1]m/kg
            fjac[2,2] = 10.0m/s
            fjac
        end
        (OnceDifferentiable(f!, j!, x0, fx0), x0, "Rosenbrock")
    end
    # With ArrayPartition and dimensions
    x0 = convert_to_mixed(-1.2kg/s, 1.0kg/s)
    fx0 = convert_to_mixed(2.2N, -4.4N)
    probd = rosenbrock(;x0, fx0)
    resd = @time nlsolve(probd[1], probd[2]; xtol = 0.0, ftol = 1e-8, droptol = 1.0e10)
end

nothing