# Adapted from 'Ordinary Differential Equations' test/units_tests.jl
using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, dimension, NoDims, ∙
using OrdinaryDiffEq
@import_expand(N, s, m, km, kg)

#using OrdinaryDiffEq, Unitfu
algs = [Euler(),Midpoint(),Heun(),Ralston(),RK4(),SSPRK104(),SSPRK22(),SSPRK33(),
        SSPRK43(),SSPRK432(),BS3(),BS5(),DP5(),DP8(),Feagin10(),Feagin12(),
        Feagin14(),TanYam7(),Tsit5(),TsitPap8(),Vern6(),Vern7(),Vern8(),Vern9()]


@testset "Unitful time with unitless state" begin
    u0 = 30.0
    tspan = (0.0, 10.0)s
    prob1 = ODEProblem((du,u,t,p) -> (du[1] = -0.2∙s⁻¹ * u[1]), [u0], tspan)
    prob2 = ODEProblem((u,t,p)    -> (-0.2∙s⁻¹ * u[1]), u0, tspan)
    prob3 = ODEProblem((u,t,p)    -> [-0.2∙s⁻¹* u[1]], [u0],tspan)
    for prob in [prob1, prob2, prob3]
        @test solve(prob, Tsit5()).retcode === :Success
    end
end

@testset "Scalar units" begin
    f(y,p,t) = 0.5*y / 3.0s
    u0 = 1.0∙N
    prob = ODEProblem(f,u0,(0.0,1.0)s)

    for alg in algs
        @show alg
        @test solve(prob,alg,dt=1∙s/10).retcode === :Success
    end
end

@testset "2D units" begin
    f(dy,y,p,t) = (dy .= 0.5.*y ./ 3.0s)
    u0 = [1.0 2.0
          3.0 1.0]N
    prob = ODEProblem(f,u0,(0.0, 1.0)s)

    for alg in algs
        @show alg
        sol = solve(prob,alg,dt=1∙s/10)
    end
    sol = solve(prob,ExplicitRK())
end

@testset "Without ArrayPartition" begin
    # coordinate: u = [position, momentum]
    # parameters: p = [mass, force constanst]
    function f_harmonic!(du,u,p,t)
        du[1] = u[2]/p[1]
        du[2] = -p[2]*u[1]
    end

    mass = 1.0∙kg
    k = 1.0∙N/m
    p = [mass, k]

    u0 = [1.0m, 0.0∙kg∙m/s] # initial values (position, momentum)
    tspan = (0.0∙s, 10.0∙s)
    prob = ODEProblem(f_harmonic!, u0, tspan, p)
    @test solve(prob, Tsit5()).retcode == :Success
end