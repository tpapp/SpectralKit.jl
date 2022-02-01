####
#### Chebyshev on [-1,1]
####

export Chebyshev

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev{K} <: FunctionBasis
    "Grid specification."
    grid_kind::K
    "The number of basis functions."
    N::Int
    @doc """
    $(SIGNATURES)
    `N` Chebyshev polynomials (of the first kind) on ``[-1, 1]``, with the associated
    grid of `grid_kind`.

    !!! note
        This is not meant to be used directly as a basis, but as a building block, eg in
        [`univariate_basis`](@ref) and [`smolyak_basis`](@ref).
    """
    function Chebyshev(grid_kind::K, N::Int) where K
        @argcheck N ≥ 1
        new{K}(grid_kind, N)
    end
end

@inline dimension(basis::Chebyshev) = basis.N

@inline domain(::Chebyshev) = (-1, 1)

function Base.show(io::IO, chebyshev::Chebyshev)
    @unpack grid_kind, N = chebyshev
    print(io, "Chebyshev polynomials (1st kind), ", grid_kind, ", dimension: ", N)
end

####
#### basis function iterator
####

struct ChebyshevIterator{T}
    x::T
    N::Int
end

Base.eltype(::Type{<:ChebyshevIterator{T}}) where {T} = T

Base.length(itr::ChebyshevIterator) = itr.N

basis_at(basis::Chebyshev, x::Real) = ChebyshevIterator(x, basis.N)

function Base.iterate(itr::ChebyshevIterator)
    @unpack x = itr
    one(x), (2, one(x), x)
end

function Base.iterate(itr::ChebyshevIterator{T}, (i, fp, fpp)) where T
    @unpack x, N = itr
    i > N && return nothing
    f = 2 * x * fp - fpp
    f::T, (i + 1, f, fp)
end

####
#### grids
####

function gridpoint(::Type{T}, basis::Chebyshev{InteriorGrid}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    sinpi((N - 2 * i + 1) / T(2 * N)) # use formula Xu (2016)
end

function gridpoint(::Type{T}, basis::Chebyshev{EndpointGrid}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    if N == 1
        cospi(1/T(2))           # 0.0 as a fallback, even though it does not have endpoints
    else
        cospi((N - i) ./ T(N - 1))
    end
end

function gridpoint(::Type{T}, basis::Chebyshev{InteriorGrid2}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    cospi(((N - i + 1) ./ T(N + 1)))
end


####
#### augmenting
####

function augment_coefficients(basis1::Chebyshev{K1}, basis2::Chebyshev{K2},
                              θ1::AbstractVector) where {K1,K2}
    @argcheck is_subset_basis(basis1, basis2)
    @argcheck length(θ1) == dimension(basis1)
    vcat(θ1, zeros(eltype(θ1), basis2.N - basis1.N))
end

function is_subset_basis(basis1::Chebyshev{K1}, basis2::Chebyshev{K2}) where {K1,K2}
    K1 == K2 && basis2.N ≥ basis1.N
end
