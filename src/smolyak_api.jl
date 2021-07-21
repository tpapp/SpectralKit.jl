#####
##### Smolyak bases
#####

export smolyak_basis

struct SmolyakBasis{I<:SmolyakIndices,U,T<:Tuple} <: FunctionBasis
    smolyak_indices::I
    univariate_parent::U
    transformations::T
end

function smolyak_basis(univariate_family, kind, ::Val{B}, transformations::NTuple{N},
                       M = B) where {B,N}
    smolyak_indices = SmolyakIndices{N,B}(kind, M)
    univariate_parent = univariate_family(highest_visited_index(smolyak_indices))
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
        sacollect(SVector{H}, basis_at(univariate_basis(univariate_parent, transformation), x))
    end
    univariate_bases_at = SVector{N}(map(_f, transformations, Tuple(x)))
    SmolyakProduct(smolyak_indices, univariate_bases_at)
end

function grid(::Type{T},
              smolyak_basis::SmolyakBasis{<:SmolyakIndices{N,B,H}}, kind) where {T<:Real,N,B,H}
    @unpack smolyak_indices, univariate_parent, transformations = smolyak_basis
    @argcheck kind == smolyak_indices.kind # FIXME remove when kind is moved to family
    x = sacollect(SVector{H}, gridpoint(T, univariate_parent, kind, i)
                  for i in block_shuffle(kind, H))
    ys = map(transformation -> from_domain.(Ref(transformation), Ref(univariate_parent), x),
             transformations)
    [SVector{N}(getindex.(ys, ι)) for ι in smolyak_indices]
end
