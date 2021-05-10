# MechGlueDiffEqBase
Glue code for making [DiffEqBase](https://github.com/SciML/DiffEqBase.jl) work with units.

This defines how to calculate the vector norm when the vector is given in units compatible with [Unitfu.jl](https://github.com/hustf/Unitfu.jl), from registy [M8](https://github.com/hustf/M8). The differential equation algorithms expects the norm to be unitless, as can be seen in e.g. step size estimators:

It also used to include glue code for [RecursiveArrayTools](https://github.com/SciML/RecursiveArrayTools.jl), which enables type-stable solution of equations with mixed units. This may not be needed after a change to Unitfu.jl v1.7.7, but the depencency is kept until further upstream testing.

err_scaled = **error** / (**abstol** + norm(u) * **reltol**)

where **bold** indicates unitful objects.

The functions are adaptions of corresponding code from [DiffEqBase](https://github.com/SciML/DiffEqBase.jl/blob/6bb8830711e729ef513f2b1beb95853e4a691375/src/init.jl).
