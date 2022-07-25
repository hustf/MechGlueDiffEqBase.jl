
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
    # We take the units from the first column of A multiplied by B. 
    # The following terms must be dimensionally compatible. If not,
    # the error message should be helpful.
    # Note that text books often implicitly allow operations like 0.0 * 0.0 + 1kg * 2.0m,
    # implying that a zero element in a matrix 'doesn't matter'.
    Avu = oneunit.(A[:, 1])
    Bu = oneunit(first(B))
    Cu = convert_to_mixed(Avu * Bu)
    @debug "*" string(Avu) string(Bu) string(Cu) maxlog=1
    mul!(Cu, A, B)
end

##########################
# Vector space operations
##########################
for op in (:+, :-)
    @eval begin
        function Base.$op(A::MixedCandidate, B::AbstractMatrix)
            row_major = convert_to_mixed(B)
            ArrayPartition(map((x, y)->Base.broadcast($op, x, y), A.x, row_major.x))
        end
    end
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
    @debug "premul_inv" oneunit.(Qar)
    Qu = oneunit.(Qar) # TODO Revisit, alloc?
    Qu = 1 ./ Qu
    @debug "premul_inv" oneunit.(Qar)
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
    # Elementwise factor number and unit
    convert_to_mixed(Mn .* Mu)
end

#####################################
# "Division", allocating inverse of A
#####################################
inv(A::MixedCandidate) = isbitstype(eltype(A)) ? _inv_consistent(mixed_array_trait(A), A) : _inv_mixed(mixed_array_trait(A), A)
_inv(::Mixed, Q) = throw(DimensionMismatch("inversion candidate is not square: dimensions are $(size(Q))"))
function _inv_mixed(::MatSqMut, Q)
    require_one_based_indexing(Q)
    # Split numeric unit matrices, treat separately
    Qar = convert_to_array(Q)
    Qn = ustrip.(Qar)
    @debug "_inv_mixed" oneunit.(Qar)
    Qu = oneunit.(Qar)
    Qu = 1 ./ Qu
    @debug "_inv_mixed" oneunit.(Qar)
    @assert is_reciprocal_symmetric(Qu) "Can't invert matrix because element dimensions are not reciprocal symmetric around the diagonal"
    @debug "_inv_mixed" string(Qn) maxlog=1
    # The reciprocal of Q has the same units as Q. Hence:
    @debug "_inv_mixed" string(Q) string(Qar) string(Qu) maxlog=1
     # Elementwise facor number and unitog=1
    convert_to_mixed(inv(Qn) .* Qu)
end
function _inv_consistent(::MatSqMut, Q)
    require_one_based_indexing(Q)
    # Split off units, invert separately
    Qn = convert_to_array(ustrip(Q))
    Qu = 1 / oneunit(eltype(Q))
    convert_to_mixed(inv(Qn) * Qu)
end
#=
    checksquare(A)
    S = typeof((one(T)*zero(T) + one(T)*zero(T))/one(T))
    AA = convert(AbstractArray{S}, A)
    if istriu(AA)
        Ai = triu!(parent(inv(UpperTriangular(AA))))
    elseif istril(AA)
        Ai = tril!(parent(inv(LowerTriangular(AA))))
    else
        Ai = inv!(lu(AA))
        Ai = convert(typeof(parent(Ai)), Ai)
    end
    return Ai
end
=#