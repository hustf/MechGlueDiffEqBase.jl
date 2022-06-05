
# Extends LinearAlgebra.mul! and RecursiveArrayTools
#################################
# Matrix by vector multiplication
#################################
# Arguments have the same number N of rows. A is 'mixed matrix', B and C are 'mixed' vectors,
# but B and C do not necessarily have equal types element by element.
# We do not use traits to distinguish the arguments, so zero-length tuples may
# possibly be dispatched here.
function mul!(C::RW(N), A::ArrayPartition{<:Q, <:NTuple{N, RW(N)}}, B::RW(N)) where {N}
    for i = 1:N
        C[i] = zero(C[i])
        for j = 1:N
            # We use try-catch here in order to pinpoint better where units are inconsistent.
            # The DimensionError message won't report where in an array this occurs.
            try
                C[i] += A[i, j] * B[j]
            catch e
                @warn("mul! DimensionError hint",
                    (i, j),
                    A[i, j], 
                    B[j],
                    A[i, j] * B[j], 
                    C[i],
                    oneunit(C[i]) / oneunit(A[i,j]),
                    oneunit(C[i]) / oneunit(B[j]))
                rethrow(e)
            end
        end
    end
    C
end
function mul!(C::RW(N), A::Transpose, B::RW(N)) where {N}
    # Remove lazy transpose
    Afull = convert_to_mixed(A)
    mul!(C, Afull, B)
end
function (*)( A::ArrayPartition{<:Q, <:NTuple{N, RW(N)}}, B::RW(N)) where {N}
    Avu = oneunit.(A[:, 1])
    Bu = oneunit(first(B))
    Cu = convert_to_mixed(Avu * Bu)
    @debug "*" string(Avu) string(Bu) string(Cu) maxlog=1
    mul!(Cu, A, B)
end

########################################
# "Division", pre-multiply the inverse A
########################################
# Extends Base: \
(\)(A::MixedCandidate, B::AbstractArray) = premul_inv(mixed_array_trait(A), A, B)
premul_inv(::VecMut, A, B) = A \ B
function premul_inv(::MatSqMut, Q, B::AbstractVecOrMat)
    require_one_based_indexing(Q, B)
    # Split numeric unit matrices, treat separately
    Qar = convert_to_array(Q)
    Qn = ustrip.(Qar)
    Qu = oneunit.(Qar)
    @assert is_reciprocal_symmetric(Qu) "Can't invert matrix because element dimensions are not reciprocal symmetric around the diagonal"
    Bar = convert_to_array(B)
    Bn = ustrip.(Bar)
    Bu = oneunit.(Bar)
    @debug "premul_inv " string(Qn) string(Bn) maxlog=1
    # Numeric result
    Mn = Qn \ Bn
    # The reciprocal of Q has the same units as Q. Hence:
    @debug "" string(Q) string(Qar) string(Qu) string(Bu) maxlog=1
    Mu = oneunit.(Qu * Bu)
    # Elementwise facor number and unit
    @debug "" string(Mn) string(Mu) maxlog=1
    convert_to_mixed(Mn .* Mu)
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
