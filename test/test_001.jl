using Test
using MechGlueDiffEqBase
using MechanicalUnits: @import_expand, dimension, NoDims
@import_expand(N, s, m)
using DifferentialEquations
@testset "ODE" begin
    f = (y, p, t) -> 0.5y / 3.0  # The derivative, yeah
    u0 = 1.5
    @test zero(u0) == 0 * u0
    @test value(u0) == 1.5
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, u0, tspan)
    integrator = DiffEqBase.__init(prob, Tsit5(), dt = 0.1)
    # Internalnorm's range should be dimensionless
    internalnorm = integrator.opts.internalnorm
    @test dimension(internalnorm(u0, tspan[1])) == NoDims
    sol = solve(prob, Tsit5(), dt=0.1)
    @test sol(0.0) == u0
    @test sol(1.0) ≈ 1.7720406
    @time solve(prob, Tsit5()) # 23 alloc, 4797 KiB
end
@testset "ODE quantity" begin
    f = (y, p, t) -> 0.5y / 3.0s  # The derivative, yeah
    u0 = 1.5N
    @test zero(u0) == 0 * u0
    @test value(u0) == 1.5
    tspan = (0.0s, 1.0s)
    prob = ODEProblem(f, u0, tspan)
    integrator = DiffEqBase.__init(prob, Tsit5(), dt = 0.1s)
    # Internalnorm's range should be dimensionless
    internalnorm = integrator.opts.internalnorm
    @test dimension(internalnorm(u0, tspan[1])) == NoDims
    sol = solve(prob, Tsit5(), dt=0.1s)
    @test sol(0.0s) == u0
    @test sol(1.0s) ≈ 1.7720406N
    @time solve(prob, Tsit5()) # 29 alloc, 5641KiB
end

@testset "ODE quantity algorithms" begin
    f = (y, p, t) -> 0.5y / 3.0s  # The derivative, yeah
    u0 = 1.5N
    @test zero(u0) == 0 * u0
    @test value(u0) == 1.5
    tspan = (0.0s, 1.0s)
    prob = ODEProblem(f, u0, tspan)
    integrator = DiffEqBase.__init(prob, Tsit5(), dt = 0.1s)
    # Internalnorm's range should be dimensionless
    internalnorm = integrator.opts.internalnorm
    @test dimension(internalnorm(u0, tspan[1])) == NoDims
    sol = solve(prob, Tsit5(), dt=0.1s)
    @test sol(0.0s) == u0
    @test sol(1.0s) ≈ 1.7720406N
    @time solve(prob, Tsit5()) # 29 alloc, 5641KiB
end
