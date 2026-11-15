#####
##### domains
#####

export domain_kind

####
#### generic
####

"""
$(SIGNATURES)

Return the *kind* of a domain type (preferred) or value. Errors for objects/types which
are not domains. Also works for domains of transformations.

The following return values are possible:

1. `:univariate`, the bounds of which can be accessed using `minimum`, `maximum`, and
`extrema`,

2. `:multivariate`, which supports `length`, `getindex` (`[]`), and conversion with `Tuple`.
"""
@inline domain_kind(x) = domain_kind(typeof(x))

domain_kind(::Type{T}) where T = throw(MethodError(domain_kind, (T,)))

####
#### univariate domains
####

"""
Univariate domain representation. Supports `extrema`, `minimum`, `maximum`.

!!! note
    Implementations only need to define `extrema`.
"""
abstract type AbstractUnivariateDomain end

@inline domain_kind(::Type{<:AbstractUnivariateDomain}) = :univariate

Broadcast.broadcastable(domain::AbstractUnivariateDomain) = Ref(domain)

Base.minimum(domain::AbstractUnivariateDomain) = extrema(domain)[1]

Base.maximum(domain::AbstractUnivariateDomain) = extrema(domain)[2]

"""
Represents a univariate domain. Use `extrema`, `minimum`, or `maximum` to access the
bounds. Internal, not exported.
"""
struct UnivariateDomain{T} <: AbstractUnivariateDomain
    min::T
    max::T
end

Base.extrema(domain::UnivariateDomain) = (domain.min, domain.max)

function Base.show(io::IO, domain::UnivariateDomain)
    mi, ma = extrema(domain)
    print(io, "[")
    mi == -Inf ? print(io, "-∞") : print(io, mi)
    print(io, ",")
    ma == Inf ? print(io, "∞") : print(io, ma)
    print(io, "]")
end

"""
Represents the interval ``[-1, 1]``.

This is the domain of the Chebyshev polynomials. For internal use, not exported.
"""
struct PM1 <: AbstractUnivariateDomain end

Base.extrema(::PM1) = (-1, 1)

Base.show(io::IO, ::PM1) = print(io, "[-1,1]")
