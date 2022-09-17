
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
    @debug "*" string(Avu) string(Bu) string(Cu) maxlog = 1
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
(\)(A::MixedCandidate, B::AbstractVector) = premul_inv(mixed_array_trait(A), A, B)
premul_inv(::VecMut, A, B) = A \ B
function premul_inv(::MatSqMut, Q, B::AbstractVector)
    # TODO compare with inv(Q) and possibly use that.
    require_one_based_indexing(Q, B)
    # Split numeric unit matrices, treat separately
    Qn = ustrip.(Q)
    # => Units of the inverse(Q) are the inverse units of transpose(Q).
    Qu = 1 ./ transpose(Q)
    @debug "premul_inv" string(Qn) string(Qu)
    # If determinant(Q) is dimensionless, this assertion would return Unitfu.NoDims.
    # If the Q entries have mismatched dimensions, the determinant is undefined
    # and its dimension is missing. The error is thrown here. Call determinant(Q)
    # to locate the mismatched dimensions.
    @assert determinant_dimension(Q) !== Dimensions{(Dimension{Missing}(1//1),)} "Can't invert matrix because of mismatched dimensions (units)."
    Bn = ustrip.(B)
    Bu = oneunit.(B)
    @debug "premul_inv " string(Bn) string(Bu) maxlog = 1
    # Numeric result
    Mn = convert_to_array(Qn) \ convert_to_array(Bn)
    Mu = oneunit.(Qu * Bu)
    @debug "premul_inv " string(Mn) string(Mu) maxlog = 1
    # Elementwise factor number and unit
    Mn .* Mu
end

#####################################
# "Division", allocating inverse of A
#####################################
inv(A::MixedCandidate) = isbitstype(eltype(A)) ? _inv_consistent(mixed_array_trait(A), A) : _inv_mixed(mixed_array_trait(A), A)
_inv(::Mixed, Q) = throw(DimensionMismatch("inversion candidate is not square: dimensions are $(size(Q))"))
function _inv_mixed(::MatSqMut, Q)
    require_one_based_indexing(Q)
    # If determinant(Q) is dimensionless, this assertion would return Unitfu.NoDims.
    # If the Q entries have mismatched dimensions, the determinant is undefined
    # and its dimension is missing. The error is thrown here. Call determinant(Q)
    # to locate the mismatched dimensions.
    @assert determinant_dimension(Q) !== Dimensions{(Dimension{Missing}(1//1),)} "Can't invert because of mismatched dimensions (units)."
    # Split numeric unit matrices, treat separately
    # Qar = convert_to_array(Q)
    Qn = ustrip.(Q)
    Qu = oneunit.(Q)
    @debug "_inv_mixed" string(Qn) string(Qu) maxlog = ""
    # Dimensions of the inverse(Q) are the inverse dimensions of transpose(Q).
    # => Units of the inverse(Q) are the inverse units of transpose(Q).
    Qui = convert_to_mixed(1 ./ Qu)
     # Elementwise facor number and unitog = 1
    inv(Qn) .* transpose(Qui)
end
function _inv_consistent(::MatSqMut, Q)
    require_one_based_indexing(Q)
    # Split off units, invert separately
    Qn = convert_to_array(ustrip(Q))
    Qu = 1 / oneunit(eltype(Q))
    convert_to_mixed(inv(Qn) * Qu)
end
