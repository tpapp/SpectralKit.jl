#####
##### Smolyak bases
#####

export SmolyakLevel, SmolyakBasis

struct SmolyakLevel
    total::Int
    each::Int
    @doc """
    $(SIGNATURES)

    Level specification for a Smolyak basis.

    `total` constrains the *sum* of levels in all axes.

    `each` constrains the level along *each* axis.

    Formally, ``0 ≤ ℓᵢ ≤ each ∀ i; ∑ᵢ ℓᵢ ≤ total`` where each `i` is an axis.

    If `each > total`, the normalization `each = total` is set with a warning.
    """
    function SmolyakLevel(; total::Int, each::Int = total)
        @argcheck total ≥ 0
        @argcheck each ≥ 0
        if each > total
            @warn "‘each’ normalized to ‘total’" each total
            each = total
        end
        new(total, each)
    end
end

"""
$(SIGNATURES)

A wrapper for iterating through Smolyak indices. See [`__smolyak__init`](@ref) and
[`__smolyak_step`](@ref). Can be plugged straight into `iterate`.

`g` transforms the value.
"""
function __smolyak_iterate(family, kind, level::SmolyakLevel, f, itrs, g,
                           state = nothing)
    if state ≡ nothing
        accum, state... = __smolyak_init(family, kind, level.total, f, itrs)
        g(accum), (accum, state...)
    else
        result = __smolyak_step(family, kind, level.each, f, itrs,
                                state...)
        if result ≡ nothing
            nothing
        else
            slack = state[2]
            accum, Δ, rest... = result
            g(accum), (accum, slack + Δ, rest...)
        end
    end
end

function Base.show(io::IO, level::SmolyakLevel)
    (; total, each) = level
    print(io, "Smolyak parameters, ∑ℓᵢ ≤ $(total), all ℓᵢ ≤ $(each)")
end

struct SmolyakBasis{F,K,D} <: MultivariateBasis
    family::F
    kind::K
    domain_transformations::D
    level::SmolyakLevel
    @doc """
    $(SIGNATURES)
    """
    function SmolyakBasis(family::F, kind::K, domain_transformations::D,
                          level::SmolyakLevel) where {F,K,D<:Tuple}
        new{F,K,D}(family, kind, domain_transformations, level)
    end
end

function Base.show(io::IO, basis::SmolyakBasis)
    (; family, kind, domain_transformations, level) = basis
    lead = "SmolyakBasis("
    next = ",\n" * ' '^length(lead)
    print(io, "SmolyakBasis(", family, next, kind, next, domain_transformations, next,
          level, ") # dimension: ", dimension(basis))
end


# """
# $(SIGNATURES)

# Create a sparse Smolyak basis.

# # Arguments

# - `family`: univariate function family, eg `Chebyshev`.

# - `kind`: the grid kind, eg `Interior()` or `Endpoints()`.

# - `domain_transformations`

# - `smolyak_level`: the Smolyak level specificaion, see [`SmolyakLevel`](@ref).

# - `N`: the dimension. wrapped in a `Val` for type stability, a convenience constructor also
#   takes integers.

# ## Example

# FIXME these examples need to be updated
# ```jldoctest
# julia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)
# Sparse multivariate basis on ℝ²
#   Smolyak indexing, ∑bᵢ ≤ 3, all bᵢ ≤ 3, dimension 81
#   using Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 27

# julia> dimension(basis)
# 81

# julia> domain(basis)
# [-1,1]²
# ```

# ## Properties

# *Grids nest*: increasing arguments of `SmolyakParameters` result in a refined grid that
# contains points of the cruder grid.
# """

function domain(smolyak_basis::SmolyakBasis)
    map(domain, smolyak_basis.domain_transformations)
end

function dimension(smolyak_basis::SmolyakBasis)
    (; family, kind, domain_transformations, level) = smolyak_basis
    N = length(domain_transformations)
    __smolyak_length(family, kind, Val(N), level.total, level.each)
end

struct SmolyakBasisAt{I,P,F,K,L<:SmolyakLevel}
    family::F
    kind::K
    level::L
    itrs::I
    product_kind::P
end

function Base.eltype(::Type{<:SmolyakBasisAt{I,P}}) where {I,P}
    _product_type(P, map(eltype, fieldtypes(I)))
end

function Base.length(itr::SmolyakBasisAt)
    (; family, kind, level, itrs, product_kind) = itr
    N = length(itrs)
    __smolyak_length(family, kind, Val(N), level.total, level.each)
end

function Base.iterate(itr::SmolyakBasisAt, state = nothing)
    (; family, kind, level, itrs, product_kind) = itr
    # FIXME this does not yet work for derivatives
    __smolyak_iterate(family, kind, level,
                      (a, b) -> isempty(b) ? (a,) : (a * first(b), b...),
                      itrs, first, state)
end

function basis_at(smolyak_basis::SmolyakBasis, x::Tuple)
    (; family, kind, domain_transformations, level) = smolyak_basis
    @argcheck length(x) == length(domain_transformations)
    itrs = map((x, d) -> basis_at(UnivariateBasis(family, kind, d, level.each), x),
               x, domain_transformations)
    SmolyakBasisAt(family, kind, level, itrs, nothing)
end

function basis_at(smolyak_basis::SmolyakBasis, x::SVector)
    basis_at(smolyak_basis, Tuple(x))
end

# function basis_at(smolyak_basis::SmolyakBasis, Dx::∂CoordinateExpansion)
#     (; family, kind, domain_transformations, level) = smolyak_basis
#     (; ∂D, x) = Dx
#     itrs = map((x, d) -> basis_at(UnivariateBasis(family, kind, d, each), x),
#                x, domain_transformations)
#     BasisAt(family, kind, total, each, itrs, ∂D)
# end

struct SmolyakGrid{I,F,K,S<:SmolyakLevel}
    family::F
    kind::K
    level::S
    itrs::I
end

function Base.eltype(::Type{<:SmolyakGrid{I}}) where {I}
    Tuple{map(eltype, fieldtypes(I))...}
end

function Base.length(itr::SmolyakGrid)
    __smolyak_length(itr.family, itr.kind, Val(length(itr.itrs)), itr.level.total, itr.level.each)
end

function Base.iterate(itr::SmolyakGrid, state = nothing)
    (; family, kind, level, itrs) = itr
    __smolyak_iterate(family, kind, level, (a, b) -> (a, b...), itrs, identity, state)
end

function grid(::Type{T}, smolyak_basis::SmolyakBasis) where {T<:AbstractFloat}
    (; family, kind, domain_transformations, level) = smolyak_basis
    (; each) = level
    itrs = map(d -> grid(T, UnivariateBasis(family, kind, d, each)), domain_transformations)
    SmolyakGrid(family, kind, level, itrs)
end

####
#### index traversal
####

@concrete struct SmolyakIndices{I}
    family
    kind
    level
    itrs::I
end

Base.eltype(::Type{<:SmolyakIndices{I}}) where I = NTuple{length(fieldtypes(I)),Int}

Base.IteratorSize(::SmolyakIndices) = Base.SizeUnknown()

"""
$(SIGNATURES)

Iterate through indices of a Smolyak basis. For each value, the indices in the tuples
correspond to the index of that univariate basis function. Not part of the API.

Note: only used as a building block for [`adjust_coefficients`](@ref).
"""
function SmolyakIndices(basis::SmolyakBasis)
    (; family, kind, domain_transformations, level) = basis
    itr1 = Iterators.countfrom(1, 1)
    itrs = ntuple(_ -> itr1, Val(length(domain_transformations)))
    SmolyakIndices(family, kind, level, itrs)
end

function Base.iterate(itr::SmolyakIndices, state = nothing)
    (; family, kind, level, itrs) = itr
    __smolyak_iterate(family, kind, level, (a, b) -> (a, b...), itrs, identity, state)
end

function adjust_coefficients(θ1::AbstractVector{T}, basis1::SmolyakBasis,
                             basis2::SmolyakBasis) where T
    (; family, kind, level, domain_transformations) = basis1
    @argcheck family ≡ basis2.family
    @argcheck domain_transformations == basis2.domain_transformations
    if kind ≡ basis2.kind && level == basis2.level
        # these are the same bases
        return copy(θ1)
    end
    θ = Dict{NTuple{length(domain_transformations),Int},T}()
    for (x, ι) in zip(θ1, smolyak_indices(basis1))
        θ[ι] = x
    end
    z = zero(T)
    [get(θ, ι, z) for ι in smolyak_indices(basis2)]
end
