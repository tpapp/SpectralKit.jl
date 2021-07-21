#####
##### traversing Smolyak indices
#####

####
#### index traversal
####

"""
$(SIGNATURES)

Internal implementation of the Smolyak indexing iterator.

# Arguments

- `M`: upper bound on each block index, remains constant during the iteration.

- `grid_kind`: used in [`cumulative_block_length`](@ref), remains constant during the iteration.

- `slack`: `B - sum(blocks)`, cached

- `indices`: current indices

- `blocks`: block indexes

- `limits`: `ℓ.(blocks)`, cached

# Return values

- `valid::Bool`: `false` iff there is no next element, in which case the following values
  should be ignored

- `C::Int`: index of the last element in `indices′` that changed compared to `indices`

- `Δ::Int`: change in `slack`

- `indices′`, `blocks′, `limits′`: next values for corresponding arguments above, each an
  `::NTuple{N,Int}`
"""
function __inc(M::Int, grid_kind, slack::Int, indices::NTuple{N,Int},
               blocks::NTuple{N,Int}, limits::NTuple{N,Int}) where N
    i1, iτ... = indices
    b1, bτ... = blocks
    l1, lτ... = limits
    if i1 < l1                  # increment i1, same block
        true, 1, 0, (i1 + 1, iτ...), blocks, limits
    elseif b1 < M && slack > 0  # increment i1, next block
        b1′ = b1 + 1
        true, 1, -1, (i1 + 1, iτ...), (b1′, bτ...), (cumulative_block_length(grid_kind, b1′), lτ...)
    else
        if N == 1               # end of the line, arbitrary value since !valid
            false, 0, 0, indices, blocks, limits
        else                    # i1 = 1, increment tail if applicable
            Δ1 = b1
            valid, C, Δτ, iτ′, bτ′, lτ′ = __inc(M, grid_kind, slack + Δ1, iτ, bτ, lτ)
            (valid, C + 1, Δ1 + Δτ, (1, iτ′...), (0, bτ′...),
             (cumulative_block_length(grid_kind, 0), lτ′...))
        end
    end
end

function __inc_init(::Val{N}, ::Val{B}, grid_kind) where {N,B}
    indices = ntuple(_ -> 1, Val(N))
    blocks = ntuple(_ -> 0, Val(N))
    l = cumulative_block_length(grid_kind, 0)
    limits = ntuple(_ -> l, Val(N))
    slack = B
    slack, indices, blocks, limits
end

"""
$(SIGNATURES)

Calculate the length of a [`SmolyakIndices`](@ref) iterator. Argument as in the latter.
"""
function smolyak_length(::Val{N}, ::Val{B}, grid_kind, M::Int) where {N,B}
    # implicit assumption: M ≤ B
    c = zeros(MVector{B+1,Int}) # indexed as 0, …, B
    for b in 0:M
        c[b + 1] = block_length(grid_kind, b)
    end
    for n in 2:N
        for b in B:(-1):0            # blocks with indices that sum to b
            s = 0
            for a in 0:min(b, M)
                s += block_length(grid_kind, a) * c[b - a + 1]
            end
            # can safely overwrite since they will not be used again for n + 1
            c[b + 1] = s
        end
    end
    sum(c)
end

struct SmolyakIndices{N,B,H,K}
    grid_kind::K
    M::Int
    len::Int
    @doc """
    Indexing specification in a Smolyak basis/interpolation.

    # Parameters

    - `N`: the dimension of indices

    - `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

    - `H`: highest index visited for all dimensions

    # Arguments

    - `grid_kind` (eg [`InteriorGrid`]) for calculating block sizes

    - `M`: upper bound on each block index

    # Details

    Consider positive integer indices `(i1, …, iN)`, each starting at one.

    Let `ℓ(b) = cumulative_block_length(grid_kind, b)`, and `b1` denote the smallest integer such
    that `i1 ≤ ℓ(b1)`, and similarly for `i2, …, iN`. Extend this with `ℓ(-1) = 0` for the
    purposes of notation.

    An index `(i1, …, iN)` is visited iff all of the following hold:

    1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
    2. `0 ≤ b1 ≤ M`, …, `1 ≤ bN ≤ M`,
    3. `b1 + … + bN ≤ B`

    Visited indexes are in *column-major* order.
    """
    function SmolyakIndices{N,B}(grid_kind::K, M::Int) where {N,B,K}
        @argcheck N ≥ 1
        @argcheck B ≥ M ≥ 0
        H = cumulative_block_length(grid_kind, min(M,B))
        len = smolyak_length(Val(N), Val(B), grid_kind, M)
        new{N,B,H,K}(grid_kind, M, len)
    end
end

@inline highest_visited_index(::SmolyakIndices{N,B,H}) where {N,B,H} = H

Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

@inline Base.length(ι::SmolyakIndices) = ι.len

@inline function Base.iterate(ι::SmolyakIndices{N,B}) where {N,B}
    slack, indices, blocks, limits = __inc_init(Val(N), Val(B), ι.grid_kind)
    indices, (slack, indices, blocks, limits)
end

@inline function Base.iterate(ι::SmolyakIndices, (slack, indices, blocks, limits))
    valid, _, Δ, indices′, blocks′, limits′ = __inc(ι.M, ι.grid_kind, slack, indices, blocks, limits)
    valid || return nothing
    slack′ = slack + Δ
    indices′, (slack′, indices′, blocks′, limits′)
end

####
#### product traversal
####

"""
$(SIGNATURES)

Return the reverse cumulative products of the first elements of `sources`.
"""
function __initialize_products(::Val{N}, sources::SVector{N}) where N
    reverse(cumprod(reverse(map(first, sources))))
end

"""
$(SIGNATURES)

Return the reverse cumulative products of `getindex.(sources, indices)`, assuming that
`products[(C+1):end]` already has the correct cumulative product.
"""
function __update_products(C, indices, sources, products::SVector{N,T}) where {N,T}
    result = MVector(products)
    Π = C >= N ? one(T) : products[C + 1]
    for j in C:-1:firstindex(result)
        Π *= sources[j][indices[j]]
        result[j] = Π
    end
    SVector(result)
end

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

@inline function Base.iterate(smolyak_product::SmolyakProduct{<:SmolyakIndices{N,B}}) where {N,B}
    @unpack smolyak_indices, sources = smolyak_product
    @unpack grid_kind = smolyak_indices
    slack, indices, blocks, limits = __inc_init(Val(N), Val(B), grid_kind)
    products = __initialize_products(Val(N), sources)
    first(products), (slack, indices, blocks, limits, products)
end

@inline function Base.iterate(smolyak_product::SmolyakProduct,
                              (slack, indices, blocks, limits, products))
    @unpack smolyak_indices, sources = smolyak_product
    @unpack M, grid_kind = smolyak_indices
    valid, C, Δ, indices′, blocks′, limits′ = __inc(M, grid_kind, slack, indices, blocks, limits)
    valid || return nothing
    products′ = __update_products(C, indices′, sources, products)
    slack′ = slack + Δ
    first(products′), (slack′, indices′, blocks′, limits′, products′)
end
