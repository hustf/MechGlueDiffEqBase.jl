# Differential equations with mixed units using ArrayPartition
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, dimension, NoDims, ∙
@import_expand(N, s, m, km, kg)
using Unitfu, DiffEqBase, OrdinaryDiffEq

@testset "Initial checks ArrayPartition" begin
    r0 = [1131.340, -2282.343, 6672.423]∙km
    v0 = [-5.64305, 4.30333, 2.42879]∙km/s
    Δt = 86400.0*365∙s
    μ = 398600.4418∙km³/s²
    rv0 = ArrayPartition(r0,v0)

    function f(dy, y, μ, t)
        r = norm(y.x[1])
        dy.x[1] .= y.x[2]
        dy.x[2] .= -μ .* y.x[1] / r^3
    end
    prob = ODEProblem(f,rv0,(0.0, 1.0)s,μ)

    integrator = DiffEqBase.__init(prob, Tsit5(), dt = 0.1s)
    # Internalnorm's range should be dimensionless
    internalnorm = integrator.opts.internalnorm
    @test dimension(internalnorm(rv0, 0.0s)) == NoDims

    sol = solve(prob, Tsit5(), dt=0.1s)
    @test sol(0.0s) == rv0
    sol1 = sol(Δt)
    expected = ArrayPartition([5.699838107777531e19, -1.1522704782345573e20, 3.372551297486921e20]km, [-2.987239096323713e17, 2.2791800330017168e17, 1.2826177222463973e17]km∙s⁻¹)
    boolpart = sol1 .≈ expected
    @test all(boolpart)
end

@testset "With ArrayPartition" begin
    r0 = [1131.340, -2282.343, 6672.423]∙km
    v0 = [-5.64305, 4.30333, 2.42879]∙km/s
    Δt = 86400.0*365∙s
    μ = 398600.4418∙km³/s²
    rv0 = ArrayPartition(r0,v0)

    function f(dy, y, μ, t)
        r = norm(y.x[1])
        dy.x[1] .= y.x[2]
        dy.x[2] .= -μ .* y.x[1] / r^3
    end

    prob = ODEProblem(f,rv0,(0.0, 1.0)s,μ)
    for alg in [Tsit5(), AutoVern6(Rodas5(autodiff=false)),
                AutoVern7(Rodas5(autodiff=false)),
                AutoVern8(Rodas5(autodiff=false)),
                AutoVern9(Rodas5(autodiff=false))]
        println(alg, "   ")
        @test solve(prob,alg).retcode === :Success
    end
    print("\n")
end