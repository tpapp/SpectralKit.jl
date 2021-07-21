####
#### Chebyshev on [-1,1]
####

export Chebyshev

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev <: FunctionBasis
    "The number of basis functions."
    N::Int
    function Chebyshev(N::Int)
        @argcheck N ≥ 1
        new(N)
    end
end

@inline dimension(basis::Chebyshev) = basis.N

@inline domain(::Chebyshev) = (-1, 1)

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

function gridpoint(::Type{T}, basis::Chebyshev, ::InteriorGrid, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    cospi((2*(N - i) + 1) / T(2 * N))
end

function gridpoint(::Type{T}, basis::Chebyshev, ::EndpointGrid, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    @argcheck N ≥ 2             # FIXME move validation to constructor
    cospi((N - i) ./ T(N - 1))
end
