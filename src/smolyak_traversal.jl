#####
##### Smolyak implementation details
#####

export SmolyakParameters

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

struct SmolyakParameters{B,M}
    function SmolyakParameters{B,M}() where {B,M}
        @argcheck B isa Int && B ≥ 0
        @argcheck M isa Int && M ≥ 0
        new{B,min(B,M)}()       # maintain M ≤ B
    end
end

"""
$(SIGNATURES)

Parameters for Smolyak grids that are independent of the dimension of the domain.

Polynomials are organized into blocks of `1, 2, 2, 4, 8, 16, …` polynomials (and
corresponding gridpoints), indexed with a *block index* `b` that starts at `0`. `B ≥ ∑ bᵢ`
and `0 ≤ bᵢ ≤ M` constrain the number of blocks along each dimension `i`.
"""
@inline function SmolyakParameters(B::Integer, M::Integer = B)
    SmolyakParameters{Int(B),Int(M)}()
end

"""
$(TYPEDEF)

Indexing specification in a Smolyak basis/interpolation.

# Type parameters

- `N`: the dimension of indices

- `H`: highest index visited for all dimensions

- `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

- `M`: upper bound on each block index

# Constructor

Takes the dimension `N` as a parameter and a `SmolyakParameters` object, calculating
everything else.

# Details

Consider positive integer indices `(i1, …, iN)`, each starting at one.

Let `ℓ(b) = __cumulative_block_length(b)`, and `b1` denote the smallest integer such
that `i1 ≤ ℓ(b1)`, and similarly for `i2, …, iN`. Extend this with `ℓ(-1) = 0` for the
purposes of notation.

An index `(i1, …, iN)` is visited iff all of the following hold:

1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
2. `0 ≤ b1 ≤ M`, …, `1 ≤ bN ≤ M`,
3. `b1 + … + bN ≤ B`

Visited indexes are in *column-major* order.
"""

struct SmolyakIndices{N,H,B,M}
    len::Int
    cumulative_block_lengths::NTuple{M,Int}
    function SmolyakIndices{N}(smolyak_parameters::SmolyakParameters{B,M}) where {N,B,M}
        @argcheck N ≥ 1
        H = __cumulative_block_length(M)
        len = __smolyak_length(Val(N), Val(B), M)
        new{N,H,B,M}(len, ntuple(__cumulative_block_length, Val(M)))
    end
end

function Base.show(io::IO, smolyak_indices::SmolyakIndices{N,H,B,M}) where {N,H,B,M}
    @unpack len = smolyak_indices
    print(io, "Smolyak indexing, B=$(B) total blocks, capped at M=$(M), dimension $(len)")
end

@inline highest_visited_index(::SmolyakIndices{N,H}) where {N,H} = H

Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

@inline Base.length(ι::SmolyakIndices) = ι.len

@inline function Base.iterate(ι::SmolyakIndices{N,H,B}) where {N,H,B}
    slack, indices, blocks, limits = __inc_init(Val(N), Val(B))
    indices, (slack, indices, blocks, limits)
end

@inline function Base.iterate(ι::SmolyakIndices, (slack, indices, blocks, limits))
    valid, Δ, indices′, blocks′, limits′ = __inc(ι.cumulative_block_lengths, slack, indices, blocks, limits)
    valid || return nothing
    slack′ = slack + Δ
    indices′, (slack′, indices′, blocks′, limits′)
end

####
#### product traversal
####

struct SmolyakProduct{I<:SmolyakIndices,S}
    smolyak_indices::I
    sources::S
    @doc """
    $(SIGNATURES)

    An iterator equivalent to

    ```
    [prod(getindex.(sources, indices)) for indices in smolyak_indices]
    ```

    implemented to perform the minimal number of multiplications. Detailed docs of the
    arguments are in [`SmolyakIndices`](@ref).

    Caller should arrange the elements of `sources` in the correct order, see
    [`nested_extrema_indices`](@ref). Each element in `sources` should have at least
    `H` elements (cf type parameters of [`SmolyakIndices`](@ref)), this is not checked.
    """
    function SmolyakProduct(smolyak_indices::SmolyakIndices{N},
                            sources::S) where {N,S<:SVector{N,<:SVector}}
        @argcheck length(sources) == N
        new{typeof(smolyak_indices),S}(smolyak_indices, sources)
    end
end

Base.length(smolyak_product::SmolyakProduct) = length(smolyak_product.smolyak_indices)

Base.eltype(::SmolyakProduct{I,S}) where {I,S} = eltype(eltype(S))

@inline function Base.iterate(smolyak_product::SmolyakProduct, state...)
    @unpack smolyak_indices, sources = smolyak_product
    itr = iterate(smolyak_indices, state...)
    itr ≡ nothing && return nothing
    indices, state′ = itr
    prod(getindex.(sources, indices)), state′
end
