#####
##### Smolyak bases
#####

export SmolyakParameters, smolyak_basis

struct SmolyakParameters{B,M}
    function SmolyakParameters{B,M}() where {B,M}
        @argcheck B isa Int && B ≥ 0
        @argcheck M isa Int && M ≥ 0
        new{B,min(B,M)}()       # maintain M ≤ B
    end
end

function Base.show(io::IO, ::SmolyakParameters{B,M}) where {B,M}
    print(io, "Smolyak parameters, ∑bᵢ ≤ $(B), all bᵢ ≤ $(M)")
end

"""
$(SIGNATURES)

Parameters for Smolyak grids that are independent of the dimension of the domain.

Polynomials are organized into blocks of `1, 2, 2, 4, 8, 16, …` polynomials (and
corresponding gridpoints), indexed with a *block index* `b` that starts at `0`. `B ≥ ∑ bᵢ`
and `0 ≤ bᵢ ≤ M` constrain the number of blocks along each dimension `i`.

`M > B` is not an error, but will be normalized to `M = B`.
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
    "number of coefficients (cached)"
    len::Int
    "cumulative block lengths (cached)"
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
    print(io, "Smolyak indexing, ∑bᵢ ≤ $(B), all bᵢ ≤ $(M), dimension $(len)")
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

struct SmolyakBasis{I<:SmolyakIndices,U,T<:Tuple} <: FunctionBasis
    smolyak_indices::I
    univariate_parent::U
    transformations::T
end

function Base.show(io::IO, smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}}) where N
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    print(io, "Sparse multivariate basis on ℝ^$N\n  ", smolyak_indices,
          "\n  using ", univariate_parent,
          "\n  domain transformations")
    for transformation in transformations
        print(io, "\n    ", transformation)
    end
end

"""
$(SIGNATURES)

Create a sparse Smolyak basis using `univariate_family` (eg `Chebyshev`), which takes
`grid_kind` and a dimension parameter. `B > 0` caps the *sum* of blocks, while `M > 0` caps
blocks along each dimension separately, and `transformations` is a tuple of transformations
applied coordinate-wise.

## Example

```jldoctest
julia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3),
                             (BoundedLinear(2, 3), SemiInfRational(3.0, 4.0)))
Sparse multivariate basis on ℝ^2
  Smolyak indexing, ∑bᵢ ≤ 3, all bᵢ ≤ 3, dimension 29
  using Chebyshev polynomials (1st kind), interior grid, dimension: 9
  domain transformations
    (2.0,3.0) [linear transformation]
    (3.0,∞) [rational transformation with scale 4.0]

julia> dimension(basis)
29

julia> domain(basis)
((2.0, 3.0), (3.0, Inf))
```
"""
function smolyak_basis(univariate_family, grid_kind::AbstractGrid,
                       smolyak_parameters::SmolyakParameters,
                       transformations::NTuple{N,Any}) where {N}
    smolyak_indices = SmolyakIndices{N}(smolyak_parameters)
    univariate_parent = univariate_family(grid_kind, highest_visited_index(smolyak_indices))
    SmolyakBasis(smolyak_indices, univariate_parent, transformations)
end

function domain(smolyak_basis::SmolyakBasis)
    @unpack univariate_parent, transformations = smolyak_basis
    D = domain(univariate_parent)
    map(transformation -> from_domain.(Ref(transformation), Ref(univariate_parent), D),
        transformations)
end

dimension(smolyak_basis::SmolyakBasis) = length(smolyak_basis.smolyak_indices)

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}},
                  x::SVector{N,<:Real}) where {N,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    function _f(transformation, x)
        sacollect(SVector{H}, basis_at(UnivariateBasis(univariate_parent, transformation), x))
    end
    univariate_bases_at = SVector{N}(map(_f, transformations, Tuple(x)))
    SmolyakProduct(smolyak_indices, univariate_bases_at)
end

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}}, x) where N
    basis_at(smolyak_basis, SVector{N}(x))
end

function grid(::Type{T},
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}}) where {T<:Real,N,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    x = sacollect(SVector{H}, gridpoint(T, univariate_parent, i)
                  for i in SmolyakGridShuffle(H))
    ys = map(transformation -> from_domain.(Ref(transformation), Ref(univariate_parent), x),
             transformations)
    [SVector{N}(getindex.(ys, ι)) for ι in smolyak_indices]
end
