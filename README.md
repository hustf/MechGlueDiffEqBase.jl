# MechGlueDiffEqBase
Glue code for making [DiffEqBase](https://github.com/SciML/DiffEqBase.jl) work with units.

It also includes glue code for [RecursiveArrayTools](https://github.com/SciML/RecursiveArrayTools.jl), which enables type-stable solution of equations with mixed units. We define zero for mixed-dimension vectors.

This defines how to calculate the vector norm when the vector is given in units compatible with [Unitfu.jl](https://github.com/hustf/Unitfu.jl), from registy [M8](https://github.com/hustf/M8). The differential equation algorithms expects the norm to be unitless, as can be seen in e.g. step size estimators:

err_scaled = **error** / (**abstol** + norm(u) * **reltol**)

where **bold** indicates unitful objects.

The functions are adaptions of corresponding code from [DiffEqBase](https://github.com/SciML/DiffEqBase.jl/blob/6bb8830711e729ef513f2b1beb95853e4a691375/src/init.jl).



