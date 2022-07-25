# Extend Unitifu function for non-quantities
numtype(x::T) where {T<:Union{DataType, Number}} = isbitstype(T) ? T : x
# Extends NLsolve.jl/src/ultils.jl:21
check_isfinite(x::MixedCandidate) = check_isfinite(mixed_array_trait(x), x)

function check_isfinite(::VecMut, x)
    if any(any.(!isfinite, x))
        i = findall(!isfinite, x)
        throw(NLsolve.IsFiniteException(i))
    end
end
function check_isfinite(::MatSqMut, x)
    if any(any.(!isfinite, x))
        i = findall(!isfinite, convert_to_array(x))
        throw(NLsolve.IsFiniteException(i))
    end
end
function check_isfinite(::NotMixed, x)
    if any(!isfinite, x)
        i = findall(!isfinite, x)
        throw(IsFiniteException(i))
    end
end


"""
    is_reciprocal_symmetric(A)

    A[i, j] == 1 / A[j, i]

If the matrix of units is reciprocal symmetric, all exponents Aᴺ of it carry the same dimensions.
if A is not singular and is square, and units are reciprocal symmetric, A is invertible to A⁻¹.
Other cases may (?) also fit with N = -1, so this check is restrictive.
"""
function is_reciprocal_symmetric(A)
    m, n = size(A)
    if m != n
        @debug "is_reciprocal_symmetric" A false
        return false
    end
    for i = 1:n
        for j = i:n
            if A[i, j] != A[j, i]^(-1)
                @debug "is_reciprocal_symmetric" i j A[i,j] A[j, i]^(-1)
                return false
            end
        end
    end
    true
end



# Extends NLsolve.jl/src/utils.jl:21
function ____assess_convergence(x::Tx,
                            x_previous::Tx,
                            f::F,
                            xtol::Real,
                            ftol::Real) where {N, Tx<: RW(N), F<: RW(N)}
    x_converged, f_converged = false, false
    @debug "assess_convergence " string(x) string(f) string(xtol) string(ftol)
    Δx = x-x_previous
    if norm(Δx ./ oneunit.(Δx)) <= xtol
        x_converged = true
    end
    if all(f.<=ftol)
        f_converged = true
    end
    return x_converged, f_converged
end