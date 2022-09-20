#####
##### Smolyak bases
#####

export SmolyakParameters, smolyak_basis

struct SmolyakParameters{B,M}
    function SmolyakParameters{B,M}() where {B,M}
        @argcheck B isa Int && B ≥ 0
        @argcheck M isa Int && M ≥ 0
        M > B && @warn "M > B replaced with M = B" M B
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

`M > B` is not an error, but will be normalized to `M = B` with a warning.
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
    @unpack len = smolyak_indices
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

struct SmolyakProduct{I<:SmolyakIndices,S<:Tuple}
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
                            sources::S) where {N,S}
        @argcheck length(sources) == N
        new{typeof(smolyak_indices),S}(smolyak_indices, sources)
    end
end

Base.length(smolyak_product::SmolyakProduct) = length(smolyak_product.smolyak_indices)

@generated function Base.eltype(::Type{SmolyakProduct{I,S}}) where {I,S}
    T = mapfoldl(eltype, _mul_type, fieldtypes(S))
    :($T)
end

@inline function Base.iterate(smolyak_product::SmolyakProduct, state...)
    @unpack smolyak_indices, sources = smolyak_product
    itr = iterate(smolyak_indices, state...)
    itr ≡ nothing && return nothing
    indices, state′ = itr
    reduce(_mul, getindex.(sources, indices)), state′
end

struct SmolyakBasis{I<:SmolyakIndices,U} <: FunctionBasis
    smolyak_indices::I
    univariate_parent::U
end

function Base.show(io::IO, smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}}) where N
    @unpack smolyak_indices, univariate_parent = smolyak_basis
    print(io, "Sparse multivariate basis on ℝ")
    print_unicode_superscript(io, N)
    print(io, "\n  ", smolyak_indices, "\n  using ", univariate_parent)
end

"""
$(SIGNATURES)

Create a sparse Smolyak basis.

# Arguments

- `univariate_family`: should be a callable that takes a `grid_kind` and a `dimension`
  parameter, eg `Chebyshev`.

- `grid_kind`: the grid kind, eg `InteriorGrid()` etc.

- `smolyak_parameters`: the Smolyak grid specification parameters, see
  [`SmolyakParameters`](@ref).

- `N`: the dimension. wrapped in a `Val` for type stability, a convenience constructor also
  takes integers.

## Example

```jldoctest
julia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)
Sparse multivariate basis on ℝ²
  Smolyak indexing, ∑bᵢ ≤ 3, all bᵢ ≤ 3, dimension 81
  using Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 27

julia> dimension(basis)
81

julia> domain(basis)
((-1, 1), (-1, 1))
```

## Properties

*Grids nest*: increasing arguments of `SmolyakParameters` result in a refined grid that
contains points of the cruder grid.
"""
function smolyak_basis(univariate_family, grid_kind::AbstractGrid,
                       smolyak_parameters::SmolyakParameters, ::Val{N}) where {N}
    @argcheck N ≥ 1
    smolyak_indices = SmolyakIndices{N}(grid_kind, smolyak_parameters)
    univariate_parent = univariate_family(grid_kind, highest_visited_index(smolyak_indices))
    SmolyakBasis(smolyak_indices, univariate_parent)
end

# convenience constructor
@inline function smolyak_basis(univariate_family, grid_kind::AbstractGrid,
                       smolyak_parameters::SmolyakParameters, N::Integer)
    smolyak_basis(univariate_family, grid_kind, smolyak_parameters, Val(N))
end

function domain(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}}) where {N}
    @unpack univariate_parent= smolyak_basis
    D = domain(univariate_parent)
    ntuple(_ -> D, Val(N))
end

dimension(smolyak_basis::SmolyakBasis) = length(smolyak_basis.smolyak_indices)

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}}, x) where {N,H}
    @argcheck length(x) == N
    @unpack smolyak_indices, univariate_parent = smolyak_basis
    function _f(x)
        sacollect(SVector{H}, basis_at(univariate_parent, x))
    end
    univariate_bases_at = map(_f, replace_zero_tags(Tuple(x)))
    SmolyakProduct(smolyak_indices, univariate_bases_at)
end

struct SmolyakGridIterator{T,I,S}
    smolyak_indices::I
    sources::S
end

Base.eltype(::Type{<:SmolyakGridIterator{T}}) where {T} = T

Base.length(itr::SmolyakGridIterator) = length(itr.smolyak_indices)

function grid(::Type{T},
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}}) where {T<:Real,N,H}
    @unpack smolyak_indices, univariate_parent = smolyak_basis
    sources = sacollect(SVector{H}, gridpoint(T, univariate_parent, i)
                        for i in SmolyakGridShuffle(univariate_parent.grid_kind, H))
    SmolyakGridIterator{NTuple{N,T},typeof(smolyak_indices),typeof(sources)}(smolyak_indices, sources)
end

function Base.iterate(itr::SmolyakGridIterator, state...)
    @unpack smolyak_indices, sources = itr
    result = iterate(smolyak_indices, state...)
    result ≡ nothing && return nothing
    ι, state′ = result
    map(i -> sources[i], ι), state′
end

"""
$(SIGNATURES)

Utility function to check is `basis1` is a subset of `basis2` with shared indices.
"""
function _is_shared_index_subset(basis1::Chebyshev{K1}, basis2::Chebyshev{K2}) where {K1,K2}
    K1 == K2 && basis1.N ≤ basis2.N
end

function is_subset_basis(basis1::SmolyakBasis{<:SmolyakIndices{N1,H1,B1,M1}},
                         basis2::SmolyakBasis{<:SmolyakIndices{N2,H2,B2,M2}}) where {N1,H1,B1,M1,N2,H2,B2,M2}
    (N1 == N2 && B2 ≥ B1 && M2 ≥ M1 &&
        # NOTE: traversal relies on the same (column major) ordering of indices in both
        # bases. Testing for this is currently innocuous, as Chebyshev has this property.
        # If some basis is added to the code which doesn't this should be tested for in
        # `augment_coefficients` which should then use a different code path.
        _is_shared_index_subset(basis1.univariate_parent, basis2.univariate_parent))
end

"""
$(TYPEDEF)

Given two iterations `ι1 ∈ itr1` and `ι2 ∈ itr2`, and a vector `θ1` such that `length(θ1) ==
length(itr1)`, return an iterator that returns elements of `θ1` when `ι1 == ι2` and zero
otherwise.

# Internals

state is a tuple of:

- index for the next upcoming element of `θ1`,
- the next item in `itr1`, set to `(0, 0, …)` after all of them are used
- the corresponding iterator state (ignore for sentinel value `(0, 0, …)`
- state of `itr2` (only after the first call to `iterate`)
"""
struct PaddingIterator{V1,I1,I2}
    θ1::V1
    itr1::I1
    itr2::I2
end

Base.length(itr::PaddingIterator) = length(itr.itr2)

Base.eltype(itr::PaddingIterator) = eltype(itr.θ1)

function Base.iterate(itr::PaddingIterator, state = (firstindex(itr.θ1),
                                                     iterate(itr.itr1)...))
    @unpack θ1, itr1, itr2 = itr
    i, ι1, state1, state2... = state
    res2 = iterate(itr2, state2...)
    res2 ≡ nothing && return nothing
    ι2, state2 = res2
    if ι1 == ι2
        x = itr.θ1[i]
        res1 = iterate(itr.itr1, state1)
        if res1 ≡ nothing
            ι1 = map(_ -> 0, ι1) # sentinel ensures never visiting ι1 == ι2 branch again
        else
            ι1, state1 = res1
            i += 1
        end
    else
        x = zero(eltype(itr.θ1))
    end
    x, (i, ι1, state1, state2)
end

function augment_coefficients(basis1::SmolyakBasis, basis2::SmolyakBasis, θ1::AbstractVector)
    @argcheck is_subset_basis(basis1, basis2)
    @argcheck dimension(basis1) == length(θ1)
    collect(PaddingIterator(θ1, basis1.smolyak_indices, basis2.smolyak_indices))
end
