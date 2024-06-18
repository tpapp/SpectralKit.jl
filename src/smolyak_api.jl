#####
##### Smolyak bases
#####

export smolyak_basis

struct SmolyakBasis{I<:SmolyakIndices,U<:UnivariateBasis} <: MultivariateBasis
    smolyak_indices::I
    univariate_parent::U
end

function Base.show(io::IO, smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}}) where N
    (; smolyak_indices, univariate_parent) = smolyak_basis
    print(io, "Sparse multivariate basis on ℝ", SuperScript(N), "\n  ", smolyak_indices,
          "\n  using ", univariate_parent)
end

Base.length(basis::SmolyakBasis{<:SmolyakIndices{N}}) where N = N

function Base.getindex(basis::SmolyakBasis, i::Int)
    @argcheck 1 ≤ i ≤ length(basis) BoundsError(basis, i)
    basis.univariate_parent
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
[-1,1]²
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
    D = domain(smolyak_basis.univariate_parent)
    coordinate_domains(Val(N), D)
end

dimension(smolyak_basis::SmolyakBasis) = length(smolyak_basis.smolyak_indices)

"""
$(SIGNATURES)

Helper function to make univariate bases for a Smolyak basis.
"""
function _univariate_bases_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}},
                              x) where {N,H}
    (; univariate_parent) = smolyak_basis
    map(x -> sacollect(SVector{H}, basis_at(univariate_parent, x)), x)
end

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}},
                  x::Union{Tuple,AbstractVector}) where {N}
    @argcheck length(x) == N
    SmolyakProduct(smolyak_basis.smolyak_indices,
                   _univariate_bases_at(smolyak_basis, NTuple{N}(x)),
                   nothing)
end

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N}},
                  Dx::∂CoordinateExpansion) where {N}
    (; ∂D, x) = Dx
    @argcheck length(x) == N
    SmolyakProduct(smolyak_basis.smolyak_indices,
                   _univariate_bases_at(smolyak_basis, x),
                   ∂D)
end

struct SmolyakGridIterator{T,I,S}
    smolyak_indices::I
    sources::S
end

Base.eltype(::Type{<:SmolyakGridIterator{T}}) where {T} = T

Base.length(itr::SmolyakGridIterator) = length(itr.smolyak_indices)

function grid(::Type{T},
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,H}}) where {T<:Real,N,H}
    (; smolyak_indices, univariate_parent) = smolyak_basis
    sources = sacollect(SVector{H}, gridpoint(T, univariate_parent, i)
                        for i in SmolyakGridShuffle(univariate_parent.grid_kind, H))
    SmolyakGridIterator{NTuple{N,T},typeof(smolyak_indices),typeof(sources)}(smolyak_indices, sources)
end

function Base.iterate(itr::SmolyakGridIterator, state...)
    (; smolyak_indices, sources) = itr
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
    (; θ1, itr1, itr2) = itr
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
