####
#### Chebyshev on [-1,1]
####

export Chebyshev

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev <: FunctionFamily
    "The number of basis functions."
    N::Int
    function Chebyshev(N::Int)
        @argcheck N ≥ 1
        new(N)
    end
end

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

basis_at(family::Chebyshev, x::Real) = ChebyshevIterator(x, family.N)

function Base.iterate(itr::ChebyshevIterator)
    @unpack x = itr
    one(x), (2, one(x), x)
end

function Base.iterate(itr::ChebyshevIterator, (i, fp, fpp))
    @unpack x, N = itr
    i > N && return nothing
    f = 2 * x * fp - fpp
    f, (i + 1, f, fp)
end

####
#### grids
####

function grid(::Type{T}, family::Chebyshev, ::InteriorGrid) where {T <: Real}
    @unpack N = family
    cospi.(((2 * N - 1):-2:1) ./ T(2 * N))
end

function grid(::Type{T}, family::Chebyshev, ::EndpointGrid) where {T <: Real}
    @unpack N = family
    @argcheck N ≥ 2
    cospi.(((N-1):-1:0) ./ T(N - 1))
end
