#####
##### Smolyak bases
#####

export smolyak_basis

struct SmolyakBasis{I<:SmolyakIndices,U,T<:Tuple} <: FunctionBasis
    smolyak_indices::I
    univariate_parent::U
    transformations::T
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
                             (BoundedLinear(2, 3), SemiInfRational(3.0, 4.0)));

julia> dimension(basis)
29

julia> domain(basis)
((2.0, 3.0), (3.0, Inf))
```
"""
function smolyak_basis(univariate_family, grid_kind, ::Val{B}, transformations::NTuple{N,Any},
                       M = B) where {B,N}
    smolyak_indices = SmolyakIndices{N,B}(grid_kind, M)
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

struct SmolyakBasisAt{S,A}
    smolyak_basis::S
    univariate_bases_at::A
end

function basis_at(smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,B,H}},
                  x::SVector{N,<:Real}) where {N,B,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    function _f(transformation, x)
        sacollect(SVector{H}, basis_at(UnivariateBasis(univariate_parent, transformation), x))
    end
    univariate_bases_at = SVector{N}(map(_f, transformations, Tuple(x)))
    SmolyakProduct(smolyak_indices, univariate_bases_at)
end

function grid(::Type{T},
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,B,H}}) where {T<:Real,N,B,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    x = sacollect(SVector{H}, gridpoint(T, univariate_parent, i)
                  for i in block_shuffle(smolyak_indices.grid_kind, H))
    ys = map(transformation -> from_domain.(Ref(transformation), Ref(univariate_parent), x),
             transformations)
    [SVector{N}(getindex.(ys, ι)) for ι in smolyak_indices]
end
