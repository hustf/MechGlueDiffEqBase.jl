#####################
# Jacobian prototypes
#####################

# TODO: Don't dispatch on quantity. Dispatch on ArrayPartition if necessary.

"""
    jacobian_prototype_zero(x, vecfx)
-> type-stable, mutable Jacobian container prototype.
Assumes x is a vector of arguments to function f, and vecfx = f(x).
The Jacobian associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
jacobian_prototype_zero(x, vecfx) = zero(jacobian_prototype_nan(x, vecfx))


#=
"""
    jacobian_prototype_nan(x, vecfx)
-> type-stable, mutable Jacobian container prototype.
Assumes x is a vector of arguments to function f, and vecfx = f(x).
The Jacobian associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
function jacobian_prototype_nan(x::ArrayPartition, vecfx::ArrayPartition)
    # Immutable input
    typednan = (xel, fxel) -> [NaN * (fxel / xel)]
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    N = length(x)
    @assert N == length(vecfx)
    @assert x.x isa NTuple{N, Q}
    @assert vecfx.x isa NTuple{N, Q}
    map(genrow, vecfx)
end
=#
jacobian_prototype_nan(x::MixedCandidate, vecfx::MixedCandidate) = 
    jacobian_prototype_nan(mixed_array_trait(x), mixed_array_trait(vecfx), x, vecfx)

function jacobian_prototype_nan(::VecMut, ::VecMut, x::RW(N), vecfx::RW(N)) where N
    @debug "jacobian_prototypes/jacobian_prototype_nan:45" maxlog=2 
    # Mutable input
    typednan = (xel, fxel) -> NaN * (fxel / xel)
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    Jint = map(genrow, vecfx)
    ArrayPartition(Jint...)
end