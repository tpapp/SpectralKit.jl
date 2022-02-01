#####
##### Smolyak implementation details
#####

####
#### Nesting sizes and shuffling
####

###
### nesting grids
###
### NOTE: This is not exported as we have no API for nesting univariate bases, only
### Smolyak. When refactoring, consider exporting with a unified API.

"""
$(SIGNATURES)

Cumulative block length at block `b`.
"""
@inline function nesting_total_length(::Type{Chebyshev}, ::EndpointGrid, b::Int)
    b == 0 ? 1 : ((1 << b) + 1)
end

"""
$(SIGNATURES)

Length of each block `b`.

!!! note
    Smolyak grids use “blocks” of polynomials, each indexed by ``b == 0, …, B`, with an
    increasing number of points in each.
"""
@inline function nesting_block_length(::Type{Chebyshev}, ::EndpointGrid, b::Int)
    b ≤ 1 ? b + 1 : 1 << (b - 1)
end

@inline nesting_total_length(::Type{Chebyshev}, ::InteriorGrid, b::Int) = 3^b

@inline function nesting_block_length(::Type{Chebyshev}, ::InteriorGrid, b::Int)
    b == 0 ? 1 : 2 * 3^(b - 1)
end

"""
$(TYPEDEF)

An iterator of indices for picking elements from a grid of length `len`, which should be a
valid cumulative block length.
"""
struct SmolyakGridShuffle{K}
    grid_kind::K
    len::Int
end

Base.length(ι::SmolyakGridShuffle) = ι.len

Base.eltype(::Type{<:SmolyakGridShuffle}) = Int

function Base.iterate(ι::SmolyakGridShuffle{InteriorGrid})
    @unpack len = ι
    i0 = (len + 1) ÷ 2          # first index at this level
    Δ = len                     # basis for step size
    a = 2                       # alternating as 2Δa and Δa
    i0, (i0, i0, Δ, a)
end

function Base.iterate(ι::SmolyakGridShuffle{InteriorGrid}, (i, i0, Δ, a))
    @unpack len = ι
    i′ = i + a * Δ
    if i′ ≤ len
        i′, (i′, i0, Δ, 3 - a)
    else
        if Δ == 1
            nothing
        else
            Δ = Δ ÷ 3
            i0 -= Δ
            i0, (i0, i0, Δ, 2)
        end
    end
end

function Base.iterate(ι::SmolyakGridShuffle{EndpointGrid})
    @unpack len = ι
    i = (len + 1) ÷ 2
    i, (0, 0)                   # step = 0 is special-cased
end

function Base.iterate(ι::SmolyakGridShuffle{EndpointGrid}, (i, step))
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

function __inc_init(nesting_total_lengths, ::Val{N}, ::Val{B}) where {N,B}
    indices = ntuple(_ -> 1, Val(N))
    blocks = ntuple(_ -> 0, Val(N))
    l = first(nesting_total_lengths)
    limits = ntuple(_ -> l, Val(N))
    slack = B
    slack, indices, blocks, limits
end

"""
$(SIGNATURES)

Internal implementation of the Smolyak indexing iterator.

# Arguments

- `nesting_total_lengths`: precalculated nesting total lengths, constant during iteration,
  indexes with an offset of `1`

- `slack`: `B - sum(blocks)`, cached

- `indices`: current indices

- `blocks`: block indexes

- `limits`: limit for each index (for column-major reset)

# Return values

- `valid::Bool`: `false` iff there is no next element, in which case the following values
  should be ignored

- `Δ::Int`: change in `slack`

- `indices′`, `blocks′, `limits′`: next values for corresponding arguments above, each an
  `::NTuple{N,Int}`
"""
@inline function __inc(nesting_total_lengths::NTuple{Mp1,Int}, slack::Int,
                       indices::NTuple{N,Int}, blocks::NTuple{N,Int},
                       limits::NTuple{N,Int}) where {Mp1,N}
    i1, iτ... = indices
    b1, bτ... = blocks
    l1, lτ... = limits
    if i1 < l1                  # increment i1, same block
        true, 0, (i1 + 1, iτ...), blocks, limits
    elseif b1 < (Mp1 - 1) && slack > 0  # increment i1, next block
        b1′ = b1 + 1
        true, -1, (i1 + 1, iτ...), (b1′, bτ...), (nesting_total_lengths[b1′ + 1], lτ...)
    else
        if N == 1               # end of iteration, arbitrary value since !valid
            false, 0, indices, blocks, limits
        else                    # i1 = 1, increment tail if applicable
            Δ1 = b1
            valid, Δτ, iτ′, bτ′, lτ′ = __inc(nesting_total_lengths, slack + Δ1, iτ, bτ, lτ)
            valid, Δ1 + Δτ, (1, iτ′...), (0, bτ′...), (nesting_total_lengths[1], lτ′...)
        end
    end
end

"""
$(SIGNATURES)

Calculate the length of a [`SmolyakIndices`](@ref) iterator. Argument as in the latter.
"""
function __smolyak_length(grid_kind::AbstractGrid, ::Val{N}, ::Val{B}, M::Int) where {N,B}
    # implicit assumption: M ≤ B, enforced by the SmolyakParameters constructor
    _bl(b) = nesting_block_length(Chebyshev, grid_kind, b)
    c = zeros(MVector{B+1,Int}) # indexed as 0, …, B
    for b in 0:M
        c[b + 1] = _bl(b)
    end
    for n in 2:N
        for b in B:(-1):0            # blocks with indices that sum to b
            s = 0
            for a in 0:min(b, M)
                s += _bl(a) * c[b - a + 1]
            end
            # can safely overwrite since they will not be used again for n + 1
            c[b + 1] = s
        end
    end
    sum(c)
end
