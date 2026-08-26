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
        @argcheck total ≥ 1
        @argcheck each ≥ 1
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

@concrete struct SmolyakIndices
    family
    kind
    level
    itrs
end

"""
$(SIGNATURES)

Iterate through indices of
"""
function smolyak_indices(basis::SmolyakBasis)
    (; family, kind, domain_transformations, level) = basis
    itr1 = 1:level.each
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

# """
# $(SIGNATURES)

# Utility function to check is `basis1` is a subset of `basis2` with shared indices.
# """
# function _is_shared_index_subset(basis1::Chebyshev{K1}, basis2::Chebyshev{K2}) where {K1,K2}
#     K1 == K2 && basis1.N ≤ basis2.N
# end

# function is_subset_basis(basis1::SmolyakBasis{<:SmolyakIndices{N1,H1,B1,M1}},
#                          basis2::SmolyakBasis{<:SmolyakIndices{N2,H2,B2,M2}}) where {N1,H1,B1,M1,N2,H2,B2,M2}
#     (N1 == N2 && B2 ≥ B1 && M2 ≥ M1 &&
#         # NOTE: traversal relies on the same (column major) ordering of indices in both
#         # bases. Testing for this is currently innocuous, as Chebyshev has this property.
#         # If some basis is added to the code which doesn't this should be tested for in
#         # `augment_coefficients` which should then use a different code path.
#         _is_shared_index_subset(basis1.univariate_parent, basis2.univariate_parent))
# end

# """
# $(TYPEDEF)

# Given two iterations `ι1 ∈ itr1` and `ι2 ∈ itr2`, and a vector `θ1` such that `length(θ1) ==
# length(itr1)`, return an iterator that returns elements of `θ1` when `ι1 == ι2` and zero
# otherwise.

# # Internals

# state is a tuple of:

# - index for the next upcoming element of `θ1`,
# - the next item in `itr1`, set to `(0, 0, …)` after all of them are used
# - the corresponding iterator state (ignore for sentinel value `(0, 0, …)`
# - state of `itr2` (only after the first call to `iterate`)
# """
# struct PaddingIterator{V1,I1,I2}
#     θ1::V1
#     itr1::I1
#     itr2::I2
# end

# Base.length(itr::PaddingIterator) = length(itr.itr2)

# Base.eltype(itr::PaddingIterator) = eltype(itr.θ1)

# function Base.iterate(itr::PaddingIterator, state = (firstindex(itr.θ1),
#                                                      iterate(itr.itr1)...))
#     (; θ1, itr1, itr2) = itr
#     i, ι1, state1, state2... = state
#     res2 = iterate(itr2, state2...)
#     res2 ≡ nothing && return nothing
#     ι2, state2 = res2
#     if ι1 == ι2
#         x = itr.θ1[i]
#         res1 = iterate(itr.itr1, state1)
#         if res1 ≡ nothing
#             ι1 = map(_ -> 0, ι1) # sentinel ensures never visiting ι1 == ι2 branch again
#         else
#             ι1, state1 = res1
#             i += 1
#         end
#     else
#         x = zero(eltype(itr.θ1))
#     end
#     x, (i, ι1, state1, state2)
# end

# function augment_coefficients(basis1::SmolyakBasis, basis2::SmolyakBasis, θ1::AbstractVector)
#     @argcheck is_subset_basis(basis1, basis2)
#     @argcheck dimension(basis1) == length(θ1)
#     collect(PaddingIterator(θ1, basis1.smolyak_indices, basis2.smolyak_indices))
# end
