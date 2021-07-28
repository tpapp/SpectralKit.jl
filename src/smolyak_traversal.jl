#####
##### Smolyak implementation details
#####

####
#### Block sizes and shuffling
####

"""
$(SIGNATURES)

Length of each block `b`.

!!! note
    Smolyak grids use “blocks” of polynomials, each indexed by ``b == 0, …, B`, with an
    increasing number of points in each.
"""
@inline __block_length(b::Int) = b ≤ 1 ? b + 1 : 1 << (b - 1)

"""
$(SIGNATURES)

Cumulative block length at block `b`.
"""
@inline __cumulative_block_length(b::Int) = b == 0 ? 1 : ((1 << b) + 1)

"""
$(TYPEDEF)

An iterator of indices for picking elements from a grid of length `len`, which should be a
valid cumulative block length.
"""
struct SmolyakGridShuffle
    len::Int
end

Base.length(ι::SmolyakGridShuffle) = ι.len

Base.eltype(::Type{SmolyakGridShuffle}) = Int

function Base.iterate(ι::SmolyakGridShuffle)
    @unpack len = ι
    i = (len + 1) ÷ 2
    i, (0, 0)                   # step = 0 is special-cased
end

function Base.iterate(ι::SmolyakGridShuffle, (i, step))
    @unpack len = ι
    i == 0 && return len > 1 ? (1, (1, len - 1)) : nothing
    i′ = i + step
    if i′ ≤ len
        i′, (i′, step)
    else
        step′ = step ÷ 2
        if step′ ≥ 2
            i′ = step′ ÷ 2 + 1
            i′, (i′, step′)
        else
            nothing
        end
    end
end

####
#### index traversal
####

"""
$(SIGNATURES)

Internal implementation of the Smolyak indexing iterator.

# Arguments

- `M`: upper bound on each block index, remains constant during the iteration.

- `slack`: `B - sum(blocks)`, cached

- `indices`: current indices

- `blocks`: block indexes

- `limits`: `__cumulative_block_length.(blocks)`, cached

# Return values

- `valid::Bool`: `false` iff there is no next element, in which case the following values
  should be ignored

- `Δ::Int`: change in `slack`

- `indices′`, `blocks′, `limits′`: next values for corresponding arguments above, each an
  `::NTuple{N,Int}`
"""
@inline function __inc(cumulative_block_lengths::NTuple{M,Int}, slack::Int,
                       indices::NTuple{N,Int}, blocks::NTuple{N,Int},
                       limits::NTuple{N,Int}) where {M,N}
    i1, iτ... = indices
    b1, bτ... = blocks
    l1, lτ... = limits
    if i1 < l1                  # increment i1, same block
        true, 0, (i1 + 1, iτ...), blocks, limits
    elseif b1 < M && slack > 0  # increment i1, next block
        b1′ = b1 + 1
        true, -1, (i1 + 1, iτ...), (b1′, bτ...), (cumulative_block_lengths[b1′], lτ...)
    else
        if N == 1               # end of iteration, arbitrary value since !valid
            false, 0, indices, blocks, limits
        else                    # i1 = 1, increment tail if applicable
            Δ1 = b1
            valid, Δτ, iτ′, bτ′, lτ′ = __inc(cumulative_block_lengths, slack + Δ1, iτ, bτ, lτ)
            valid, Δ1 + Δτ, (1, iτ′...), (0, bτ′...), (__cumulative_block_length(0), lτ′...)
        end
    end
end

function __inc_init(::Val{N}, ::Val{B}) where {N,B}
    indices = ntuple(_ -> 1, Val(N))
    blocks = ntuple(_ -> 0, Val(N))
    l = __cumulative_block_length(0)
    limits = ntuple(_ -> l, Val(N))
    slack = B
    slack, indices, blocks, limits
end

"""
$(SIGNATURES)

Calculate the length of a [`SmolyakIndices`](@ref) iterator. Argument as in the latter.
"""
function __smolyak_length(::Val{N}, ::Val{B}, M::Int) where {N,B}
    # implicit assumption: M ≤ B
    c = zeros(MVector{B+1,Int}) # indexed as 0, …, B
    for b in 0:M
        c[b + 1] = __block_length(b)
    end
    for n in 2:N
        for b in B:(-1):0            # blocks with indices that sum to b
            s = 0
            for a in 0:min(b, M)
                s += __block_length(a) * c[b - a + 1]
            end
            # can safely overwrite since they will not be used again for n + 1
            c[b + 1] = s
        end
    end
    sum(c)
end
