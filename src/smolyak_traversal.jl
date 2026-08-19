#####
##### Smolyak implementation details
#####

####
#### index traversal
####

function __inc_init(family, kind, ::Val{N}, total) where N
    indices = ntuple(_ -> 1, Val(N))
    levels = ntuple(_ -> 0, Val(N))
    b0 = block_length(family, kind, 0)
    limits = ntuple(_ -> b0, Val(N))
    slack = total
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

"""
$(TYPEDEF)

Indexing specification in a Smolyak basis/interpolation.

# Type parameters

- `N`: the dimension of indices

- `H`: highest index visited for all dimensions

- `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

- `M`: upper bound on each block index

# Constructor

Takes the dimension `N` as a parameter, `grid_kind`, and a `SmolyakParameters` object,
calculating everything else.

# Details

Consider positive integer indices `(i1, …, iN)`, each starting at one.

Let `ℓ(b) = nesting_total_length(Chebyshev, grid_knid, kind, b)`, and `b1` denote the
smallest integer such that `i1 ≤ ℓ(b1)`, and similarly for `i2, …, iN`. Extend this with
`ℓ(-1) = 0` for the purposes of notation.

An index `(i1, …, iN)` is visited iff all of the following hold:

1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
2. `0 ≤ b1 ≤ M`, …, `1 ≤ bN ≤ M`,
3. `b1 + … + bN ≤ B`

Visited indexes are in *column-major* order.
"""
struct SmolyakIndices{N,H,B,M,Mp1}
    "number of coefficients (cached)"
    len::Int
    "nesting total lengths (cached)"
    nesting_total_lengths::NTuple{Mp1,Int}
    function SmolyakIndices{N}(grid_kind::AbstractGrid,
                               smolyak_parameters::SmolyakParameters{B,M}) where {N,B,M}
        @argcheck N ≥ 1
        Mp1 = M + 1
        len = __smolyak_length(grid_kind, Val(N), Val(B), M)
        first_block_length = nesting_total_length(Chebyshev, grid_kind, 0)
        nesting_total_lengths = ntuple(bp1 -> nesting_total_length(Chebyshev, grid_kind, bp1 - 1),
                                       Val(Mp1))
        H = last(nesting_total_lengths)
        new{N,H,B,M,Mp1}(len, nesting_total_lengths)
    end
end

function Base.show(io::IO, smolyak_indices::SmolyakIndices{N,H,B,M}) where {N,H,B,M}
    (; len) = smolyak_indices
    print(io, "Smolyak indexing, ∑bᵢ ≤ $(B), all bᵢ ≤ $(M), dimension $(len)")
end

@inline highest_visited_index(::SmolyakIndices{N,H}) where {N,H} = H

Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

@inline Base.length(ι::SmolyakIndices) = ι.len

@inline function Base.iterate(ι::SmolyakIndices{N,H,B}) where {N,H,B}
    slack, indices, blocks, limits = __inc_init(ι.nesting_total_lengths, Val(N), Val(B))
    indices, (slack, indices, blocks, limits)
end

@inline function Base.iterate(ι::SmolyakIndices, (slack, indices, blocks, limits))
    valid, Δ, indices′, blocks′, limits′ = __inc(ι.nesting_total_lengths, slack, indices,
                                                 blocks, limits)
    valid || return nothing
    slack′ = slack + Δ
    indices′, (slack′, indices′, blocks′, limits′)
end

####
#### product traversal
####

struct SmolyakProduct{I<:SmolyakIndices,S<:Tuple,P}
    smolyak_indices::I
    sources::S
    product_kind::P
    @doc """
    $(SIGNATURES)

    An iterator conceptually equivalent to

    ```
    [prod(getindex.(sources, indices)) for indices in smolyak_indices]
    ```

    using [`_product`](@ref) instead to account for derivatives. Detailed docs of the
    arguments are in [`SmolyakIndices`](@ref).

    Caller should arrange the elements of `sources` in the correct order, see
    [`nested_extrema_indices`](@ref). Each element in `sources` should have at least
    `H` elements (cf type parameters of [`SmolyakIndices`](@ref)), this is not checked.
    """
    function SmolyakProduct(smolyak_indices::I, sources::S,
                            product_kind::P) where {N,I<:SmolyakIndices{N},S,P}
        @argcheck length(sources) == N
        new{I,S,P}(smolyak_indices, sources, product_kind)
    end
end

Base.length(smolyak_product::SmolyakProduct) = length(smolyak_product.smolyak_indices)

function Base.eltype(::Type{SmolyakProduct{I,S,P}}) where {I,S,P}
    _product_type(P, fieldtypes(S))
end

@inline function Base.iterate(smolyak_product::SmolyakProduct, state...)
    (; smolyak_indices, sources, product_kind) = smolyak_product
    itr = iterate(smolyak_indices, state...)
    itr ≡ nothing && return nothing
    indices, state′ = itr
    _product(product_kind, map(getindex, sources, indices)), state′
end
