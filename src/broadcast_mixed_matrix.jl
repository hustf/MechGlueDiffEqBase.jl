#####################################################
# Inferrability of mixed matrices during broadcasting
#####################################################
# Types and traits are defined in 'io_traits_conversion'
# These definitions tells the compiler more type
# details in advance of mapping or broadcasting.

# Overload MixedMatrices.
function Broadcast.BroadcastStyle(::Type{ArrayPartition{T,S}}) where {T, S<:NTuple{N, RW(N)} where N}
    ArrayPartitionStyle{Base.Broadcast.DefaultArrayStyle{2}}()
end

@inline function Base.copy(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Style}}) where {Style <: Base.Broadcast.DefaultArrayStyle{2}}
    # TODO: Make this type-inferrable by extracting N and
    N = npartitions(bc)
    source = bc.args[1]
    Am = ArrayPartition(map(eachrow(source)) do rw
        
        this = copy(rw)
        @debug "copy " this typeof(this)
        @inline function f(i)
            copy(rw)
        end
        #mutable_row = ntuple(rw-> vpack, rw)
        ArrayPartition(fmap(vpack, rw)...)
    end...)
    @assert Am isa MixedCandidate "Cannot copy $(typeof(bc))"
    Am



    ArrayPartition(f, N)

 #  Am = ArrayPartition(map(eachrow(source)) do rw
#        mutable_row = ArrayPartition(vpack, rw)
#        @debug "copy" mutable_row typeof(mutable_row)
#        ArrayPartition(mutable_row...)
#    end...)
end
