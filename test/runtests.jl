using MechGlueDiffEqBase
using MechGluePlots
import MechanicalUnits: @import_expand, Quantity, Time, Force, ∙
import Unitfu.AbstractQuantity
import DifferentialEquations
import DifferentialEquations: SciMLBase, solve, ODEProblem, Tsit5
import SciMLBase
import SciMLBase: AbstractTimeseriesSolution, RecipesBase
import SciMLBase: AbstractDiscreteProblem, AbstractRODESolution, SensitivityInterpolation
using Test
import Plots.plot

@import_expand(N, s, m)
@testset "ODE" begin
    f = (y, p, t) -> 0.5y / 3.0s  # The derivative, yeah
    u0 = 1.5N
    tspan = (0.0s, 1.0s)
    prob = ODEProblem(f, u0, tspan)
    solve(prob, Tsit5())
    sol = solve(prob)
    @test sol.t isa Vector{<:Time}
    @test sol.u isa Vector{<:Force}
end

#@testset "Plot ODE quantity" begin
    SciMLBase.RecipesBase.debug(true)
    f1 = (y, p, t) -> -1.5y / 0.3s  # The derivative, yeah
    u0 = 1.5N
    tspan = (0.0s, 1.0s)
    prob = ODEProblem(f1, u0, tspan)
    solve(prob, Tsit5())
    sol = solve(prob)
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T, N, A}
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T<:Quantity, N, A}
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T<:Quantity, N, A<:Array{<:Quantity}}
    # Here, we only test this part of the dispatch mechanism.
    # This is intended to work with MechGluePlots also loaded.
    # See other tests in MechGluecode.
    plot(sol)
#end

