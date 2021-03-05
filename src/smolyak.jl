#####
##### Smolyak bases NOTE WIP
#####

####
#### Utility functions for traversal
####

"""
$(SIGNATURES)

Internal implementation of the Smolyak indexing iterator.

# Arguments

- `M`, `ℓ`: see [`SmolyakIndices`](@ref). These remain constant during the iteration.

- `slack`: `B - sum(blocks)`

- `indices`: current indices

- `blocks`: block indexes

- `limits`: `ℓ.(blocks)`, cached

# Return values

- `valid::Bool`: `false` iff there is no next element, in which case the following values
  should be ignored

- `Δ::Int`: change in `slack`

- `indices′`, `blocks′, `limits′`: next values for corresponding arguments above, each an
  `::NTuple{N,Int}`
"""
function __inc(M::Int, ℓ::L, slack::Int, indices::NTuple{N,Int}, blocks::NTuple{N,Int},
               limits::NTuple{N,Int}) where {L,N}
    i1, iτ... = indices
    b1, bτ... = blocks
    l1, lτ... = limits
    if i1 < l1                  # increment i1, same block
        true, 0, (i1 + 1, iτ...), blocks, limits
    elseif b1 < M && slack > 0  # increment i1, next block
        b1′ = b1 + 1
        true, -1, (i1 + 1, iτ...), (b1′, bτ...), (ℓ(b1′), lτ...)
    else
        if N == 1               # end of the line, arbitrary value since !valid
            false, 0, indices, blocks, limits
        else                    # i1 = 1, increment tail if applicable
            Δ1 = b1
            valid, Δτ, iτ′, bτ′, lτ′ = __inc(M, ℓ, slack + Δ1, iτ, bτ, lτ)
            valid, Δ1 + Δτ, (1, iτ′...), (0, bτ′...), (ℓ(0), lτ′...)
        end
    end
end

struct SmolyakIndices{N,L}
    ℓ::L
    B::Int
    M::Int
    @doc """
    Iteration over indices in a Smolyak basis/interpolation.

    # Arguments

    - `ℓ` calculates cumulative block lengths for block `b` (counting from 0), mapping to
      positive integers

    - `N`: the dimension of indices

    - `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

    - `M`: upper bound on each block index

    # Details

    Consider positive integer indices `(i1, …, iN)`, each starting at one.

    Let `b1` denote the smallest integer such that `i1 ≤ ℓ(b1)`, and similarly for `i2, …,
    iN`.

    An index `(i1, …, iN)` is visited iff all of the following hold:

    1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
    2. `0 ≤ b1 ≤ M`, …, `1 ≤ bN ≤ M`,
    3. `b1 + … + bN ≤ B`

    Visited indexes are in *column-major* order.
    """
    function SmolyakIndices{N}(ℓ::L, B::Int, M::Int = B) where {N,L}
        @argcheck N ≥ 1
        @argcheck B ≥ M ≥ 0
        new{N,L}(ℓ, B, M)
    end
end

Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

Base.IteratorSize(::Type{<:SmolyakIndices{N}}) where N = Base.SizeUnknown() # FIXME

@inline function Base.iterate(ι::SmolyakIndices{N}) where N
    indices = ntuple(_ -> 1, Val(N))
    blocks = ntuple(_ -> 0, Val(N))
    l = ι.ℓ(0)
    limits = ntuple(_ -> l, Val(N))
    slack = ι.B
    indices, (slack, indices, blocks, limits)
end

@inline function Base.iterate(ι::SmolyakIndices{N}, (slack, indices, blocks, limits)) where N
    valid, Δ, indices′, blocks′, limits′ = __inc(ι.M, ι.ℓ, slack, indices, blocks, limits)
    valid || return nothing
    slack′ = slack + Δ
    indices′, (slack′, indices′, blocks′, limits′)
end

"""
$(SIGNATURES)

Cumulative block length for Chebyshev nodes on open intervals.
"""
_chebyshev_open_ℓ(b::Int) = b == 0 ? 1 : (2^b + 1)
