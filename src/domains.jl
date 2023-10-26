#####
##### domains
#####

export domain_kind, coordinate_domains

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

"""
Represents the interval ``[-1, 1]``.

This is the domain of the Chebyshev polynomials. For internal use, not exported.
"""
struct PM1 <: AbstractUnivariateDomain end

Base.extrema(::PM1) = (-1, 1)

Base.show(io::IO, ::PM1) = print(io, "[-1,1]")

###
### multivariate domains
###

"""
Representation of a multivariate domain as the product of coordinate domains.
"""
struct CoordinateDomains{T<:Tuple}
    domains::T
    function CoordinateDomains(domains::Tuple)
        @argcheck all(d -> domain_kind(d) ≡ :univariate, domains)
        new{typeof(domains)}(domains)
    end
end

@inline domain_kind(::Type{<:CoordinateDomains}) = :multivariate

function Base.show(io::IO, domain::CoordinateDomains)
    (; domains) = domain
    if allequal(domains)
        print(io, domains[1], SuperScript(length(domains)))
    else
        join(io, domains, "×")
    end
end

@inline Base.length(domain::CoordinateDomains) = length(domain.domains)

@inline Base.getindex(domain::CoordinateDomains, i) = getindex(domain.domains, i)

Base.Tuple(domain::CoordinateDomains) = domain.domains

"""
$(SIGNATURES)

Create domains which are the product of univariate domains. The result support `length`,
indexing with integers, and `Tuple` for conversion.
"""
function coordinate_domains(domains::Tuple)
    CoordinateDomains(domains)
end

"""
$(SIGNATURES)
"""
function coordinate_domains(domains::Vararg)
    CoordinateDomains(domains)
end

"""
$(SIGNATURES)

Create a coordinate domain which is the product of `domain` repeated `N` times.
"""
function coordinate_domains(::Val{N}, domain) where N
    @argcheck N isa Integer && N ≥ 1
    CoordinateDomains(ntuple(_ -> domain, Val(N)))
end

"""
$(SIGNATURES)
"""
@inline function coordinate_domains(N::Integer, domain)
    coordinate_domains(Val(N), domain)
end
