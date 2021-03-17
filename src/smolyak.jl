#####
##### Smolyak bases NOTE WIP
#####

struct ChebyshevOpen end

block_length(::ChebyshevOpen, b::Int) = b ≤ 1 ? b + 1 : 2^(b - 1)

"""
$(SIGNATURES)

Cumulative block length for Chebyshev nodes on open intervals.
"""
cumulative_block_length(::ChebyshevOpen, b::Int) =  b == 0 ? 1 : (2^b + 1)

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
function __inc(M::Int, kind, slack::Int, indices::NTuple{N,Int}, blocks::NTuple{N,Int},
               limits::NTuple{N,Int}) where {N}
    i1, iτ... = indices
    b1, bτ... = blocks
    l1, lτ... = limits
    if i1 < l1                  # increment i1, same block
        true, 0, (i1 + 1, iτ...), blocks, limits
    elseif b1 < M && slack > 0  # increment i1, next block
        b1′ = b1 + 1
        true, -1, (i1 + 1, iτ...), (b1′, bτ...), (cumulative_block_length(kind, b1′), lτ...)
    else
        if N == 1               # end of the line, arbitrary value since !valid
            false, 0, indices, blocks, limits
        else                    # i1 = 1, increment tail if applicable
            Δ1 = b1
            valid, Δτ, iτ′, bτ′, lτ′ = __inc(M, kind, slack + Δ1, iτ, bτ, lτ)
            valid, Δ1 + Δτ, (1, iτ′...), (0, bτ′...), (cumulative_block_length(kind, 0), lτ′...)
        end
    end
end

struct SmolyakIndices{N,B,K}
    kind::K
    M::Int
    @doc """
    Indexing specification in a Smolyak basis/interpolation.

    # Arguments

    - `N`: the dimension of indices

    - `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

    - `kind` (eg [`ChebyshevOpen`]) for calculating block sizes

    - `M`: upper bound on each block index

    # Details

    Consider positive integer indices `(i1, …, iN)`, each starting at one.

    Let `ℓ(b) = cumulative_block_length(kind, b)`, and `b1` denote the smallest integer such
    that `i1 ≤ ℓ(b1)`, and similarly for `i2, …, iN`. Extend this with `ℓ(-1) = 0` for the purposes of notation.

    An index `(i1, …, iN)` is visited iff all of the following hold:

    1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
    2. `0 ≤ b1 ≤ M`, …, `1 ≤ bN ≤ M`,
    3. `b1 + … + bN ≤ B`

    Visited indexes are in *column-major* order.
    """
    function SmolyakIndices{N,B}(kind::K, M::Int = B) where {N,B,K}
        @argcheck N ≥ 1
        @argcheck B ≥ M ≥ 0
        new{N,B,K}(kind, M)
    end
end

Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

"""
$(SIGNATURES)

Calculate the length of a [`SmolyakIndices`](@ref) iterator. Argument as in the latter.
"""
function smolyak_length(::Val{N}, ::Val{B}, kind, M::Int) where {N,B}
    # implicit assumption: M ≤ B
    c = zeros(MVector{B+1,Int}) # indexed as 0, …, B
    for b in 0:M
        c[b + 1] = block_length(kind, b)
    end
    for n in 2:N
        for b in B:(-1):0            # blocks with indices that sum to b
            s = 0
            for a in 0:min(b, M)
                s += block_length(kind, a) * c[b - a + 1]
            end
            # can safely overwrite since they will not be used again for n + 1
            c[b + 1] = s
        end
    end
    sum(c)
end

function Base.length(ι::SmolyakIndices{N,B}) where {N,B}
    smolyak_length(Val(N), Val(B), ι.kind, ι.M)
end

@inline function Base.iterate(ι::SmolyakIndices{N,B}) where {N,B}
    indices = ntuple(_ -> 1, Val(N))
    blocks = ntuple(_ -> 0, Val(N))
    l = cumulative_block_length(ι.kind, 0)
    limits = ntuple(_ -> l, Val(N))
    slack = B
    indices, (slack, indices, blocks, limits)
end

@inline function Base.iterate(ι::SmolyakIndices, (slack, indices, blocks, limits))
    valid, Δ, indices′, blocks′, limits′ = __inc(ι.M, ι.kind, slack, indices, blocks, limits)
    valid || return nothing
    slack′ = slack + Δ
    indices′, (slack′, indices′, blocks′, limits′)
end


# struct SmolyakBasis{N,B,I<:SmolyakIndices{N,B},T<:NTuple{N}}
#     ι::I
#     transformations::T
# end

# struct SmolyakBasisAt{S,U}
#     smolyak_basis::S
#     univariate_bases_at::U
# end

# function basis_at(smolyak_basis::SmolyakBasis{N,B}, x::SVector{N,R},
#                   order::Order{0}()) where {N,B,R<:Real}
#     @unpack ι, univariate_bases = smolyak_basis
#     L = cumulative_block_length(ι.kind, ι.M)
#     _f(b, x) = sacollect(SVector{L}, basis_function(b, x, order))
#     univariate_bases_at = map(_f, x, t)
#     SmolyakBasis(basis, univariate_bases_at)
# end

# function Base.iterate(ι::SmolyakBasisAt, state = ())
#     @unpack univariate_basis, smolyak_basis = ι
#     indices, state′ = iterate(smolyak_basis.ι, state...)
#     v = mapfoldl(getindex, *, univariate_basis, SVector(indices))
#     v, state′
# end

####
#### collocation points
####

struct NestedExtremaIndices
    len::Int
end

Base.length(ι::NestedExtremaIndices) = ι.len

Base.eltype(::Type{NestedExtremaIndices}) = Int

"""
$(SIGNATURES)

Return an iterator that traverses indexes of extrema.
"""
function nested_extrema_indices(kind::ChebyshevOpen, b::Int)
    @argcheck b ≥ 0
    NestedExtremaIndices(cumulative_block_length(kind, b))
end

function Base.iterate(ι::NestedExtremaIndices)
    @unpack len = ι
    i = (len + 1) ÷ 2
    i, (0, 0)                   # step = 0 is special-cased
end

function Base.iterate(ι::NestedExtremaIndices, (i, step))
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
