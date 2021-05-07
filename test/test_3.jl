# Differential equations with mixed units using ArrayPartition
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, dimension, NoDims, ∙
import MechanicalUnits: g, g⁻¹
@import_expand(km, N, s, m, km, kg, °, inch)
using DiffEqBase, OrdinaryDiffEq # Unitfu,

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

@testset "zero of ArrayPartition with mixed units" begin
    @test zero.([0.0m∙s⁻¹, 0.0m∙s⁻¹, 909.3266739736605m∙s⁻², 525.0m∙s⁻²],) == [0.0m∙s⁻¹, 0.0m∙s⁻¹, 0.0m∙s⁻², 0.0m∙s⁻²]
    @test zero(ArrayPartition([0.0m∙s⁻¹, 0.0m∙s⁻¹, 909.3266739736605m∙s⁻², 525.0m∙s⁻²],)) ==  [0.0m∙s⁻¹, 0.0m∙s⁻¹, 0.0m∙s⁻², 0.0m∙s⁻²]
    α₀() = 30°
    x₀() = 0.0m
    y₀() = 0.0m
    v₀() = 1050m/s
    v₀x() = v₀() * cos(α₀())
    v₀y() = v₀() * sin(α₀())
    d() = 15inch
    A() = π/4 * d()^2
    mₚ() = 495kg
    ρ() = 1.225kg/m³
    C_s() = 0.4
    v(vx, vy) = √(vx^2 + vy^2)
    R(vx, vy) = 0.5∙ρ()∙C_s()∙A()∙v(vx, vy)^2
    α(vx, vy) = atan(vy, vx)
    Rx(vx, vy) = R(vx, vy) * cos(α(vx, vy))
    Ry(vx, vy) = R(vx, vy) * sin(α(vx, vy))

    function f(du,u,p,t)
        x, y, vx,vy = u
        du[1] = dx = vx
        du[2] = dy = vy
        du[3] = dvx = -Rx(vx, vy) / mₚ()
        du[4] = dvy = -1g -Ry(vx, vy) / mₚ()
    end

    tspan = (0.0, 60)s
    u₀ = ArrayPartition([x₀(), y₀(), v₀x(), v₀y()])
    prob = ODEProblem(f,u₀,tspan)

    algs = [Euler(),Midpoint(),Heun(),Ralston(),RK4(),SSPRK104(),SSPRK22(),SSPRK33(),
        SSPRK43(),SSPRK432(),BS3(),BS5(),DP5(),DP8(),Feagin10(),Feagin12(),
        Feagin14(),TanYam7(),Tsit5(),TsitPap8(),Vern6(),Vern7(),Vern8(),Vern9()]
    algs = vcat(algs, [Tsit5(), AutoVern6(Rodas5(autodiff=false)),
        AutoVern7(Rodas5(autodiff=false)),
        AutoVern8(Rodas5(autodiff=false)),
        AutoVern9(Rodas5(autodiff=false))])


    using OrdinaryDiffEq: OrdinaryDiffEqAdaptiveAlgorithm, OrdinaryDiffEqCompositeAlgorithm, DAEAlgorithm, FunctionMap,LinearExponential

    function requires_stepsize(alg)
        adaptive = OrdinaryDiffEq.isadaptive(alg)
        (((!(typeof(alg) <: OrdinaryDiffEqAdaptiveAlgorithm) && !(typeof(alg) <: OrdinaryDiffEqCompositeAlgorithm) && !(typeof(alg) <: DAEAlgorithm)) || !adaptive) ) && !(typeof(alg) <: Union{FunctionMap,LinearExponential})
    end

    for alg in algs
        println(alg, "   ")
        if requires_stepsize(alg)
            @test solve(prob,alg, dt = 1.0s).retcode === :Success
        else
            @test solve(prob,alg).retcode === :Success
        end
    end
end
