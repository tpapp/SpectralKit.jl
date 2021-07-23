#####
##### Smolyak bases
#####

export smolyak_basis

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
julia> basis = smolyak_basis(Chebyshev, InteriorGrid(), Val(3),
                             (BoundedLinear(2, 3), SemiInfRational(3.0, 4.0)))
Sparse multivariate basis on ℝ^2
  Smolyak indexing, 3 total blocks, capped at 3, dimension 29
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
function smolyak_basis(univariate_family, grid_kind::AbstractGrid, ::Val{B},
                       transformations::NTuple{N,Any}, M = B) where {B,N}
    smolyak_indices = SmolyakIndices{N,B}(M)
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

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,B,H}},
                  x::SVector{N,<:Real}) where {N,B,H}
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
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,B,H}}) where {T<:Real,N,B,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    x = sacollect(SVector{H}, gridpoint(T, univariate_parent, i)
                  for i in SmolyakGridShuffle(H))
    ys = map(transformation -> from_domain.(Ref(transformation), Ref(univariate_parent), x),
             transformations)
    [SVector{N}(getindex.(ys, ι)) for ι in smolyak_indices]
end
