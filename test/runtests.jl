using MechGlueDiffEqBase
import MechanicalUnits: @import_expand, Quantity, Time, Force, ∙, Quantity
import Unitfu.AbstractQuantity
import DifferentialEquations
import DifferentialEquations: SciMLBase, solve, ODEProblem, Tsit5
import SciMLBase
import SciMLBase: AbstractTimeseriesSolution, RecipesBase
import SciMLBase: AbstractDiscreteProblem, AbstractRODESolution, SensitivityInterpolation
using Test
import Plots.plot
@import_expand(N, s)
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

@testset "Plot ODE quantity" begin
    SciMLBase.RecipesBase.debug(true)
    f1 = (y, p, t) -> -1.5y / 0.3s  # The derivative, yeah
    u0 = 1.5N
    tspan = (0.0s, 1.0s)
    prob = ODEProblem(f1, u0, tspan)
    solve(prob, Tsit5())
    sol = solve(prob)
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T, N, A}
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T<:AbstractQuantity, N, A}
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T<:AbstractQuantity, N, A<:AbstractArray{<:AbstractQuantity}}
    @test sol isa AbstractTimeseriesSolution{T, N, A} where {T<:Quantity, N, A<:Array{<:Quantity}}
    plot(sol)
    # TODO implement in MechGlueDiffEqBase
    # the recipe found in SciMLBase/solutions/solution_interface.jl:151
    # Type to dispatch on:
    # AbstractTimeseriesSolution{T, N, A}) where {T, N, A}
    # T = Quantity{Float64,  ᴸ∙ ᴹ∙ ᵀ⁻², Unitfu.FreeUnits{(N,),  ᴸ∙ ᴹ∙ ᵀ⁻², nothing}}
    # N = 1
    # A = Vector{Quantity{Float64,  ᴸ∙ ᴹ∙ ᵀ⁻², Unitfu.FreeUnits{(N,),  ᴸ∙ ᴹ∙ ᵀ⁻², nothing}}}

end

