#####
##### transformations
#####

export transform_to, transform_from, PM1, coordinate_domains,
    coordinate_transformations, BoundedLinear, InfRational, SemiInfRational

####
#### domains
####

"""
A supertype for domains. Not part of the API, only for organizing code.
"""
abstract type AbstractDomain end

###
### univariate domains
###

"""
Univariate domain representation. Supports `extrema`, `minimum`, `maximum`.

!!! note
    Implementations only need to define `extrema`.
"""
abstract type AbstractUnivariateDomain <: AbstractDomain end

Broadcast.broadcastable(domain::AbstractUnivariateDomain) = Ref(domain)

Base.minimum(domain::AbstractUnivariateDomain) = extrema(domain)[1]

Base.maximum(domain::AbstractUnivariateDomain) = extrema(domain)[2]

struct UnivariateDomain{T}
    min::T
    max::T
end

Base.extrema(domain::UnivariateDomain) = (domain.min, domain.max)

"""
Represents the interval ``[-1, 1]``.

This is the domain of the Chebyshev polynomials.
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
struct CoordinateDomains{T<:Tuple{Vararg{AbstractUnivariateDomain}}} <: AbstractDomain
    domains::T
end

function Base.show(io::IO, domain::CoordinateDomains)
    (; domains) = domain
    if allequal(domains)
        print(io, domains[1])
        print_unicode_superscript(io, length(domains))
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
function coordinate_domains(domains::Tuple{Vararg{AbstractUnivariateDomain}})
    CoordinateDomains(domains)
end

"""
$(SIGNATURES)
"""
function coordinate_domains(domains::Vararg{AbstractUnivariateDomain})
    CoordinateDomains(domains)
end

"""
$(SIGNATURES)

Create a coordinate domain which is the product of `domain` repeated `N` times.
"""
function coordinate_domains(::Val{N}, domain::AbstractUnivariateDomain) where N
    @argcheck N isa Integer && N ≥ 1
    CoordinateDomains(ntuple(_ -> domain, Val(N)))
end

"""
$(SIGNATURES)
"""
@inline function coordinate_domains(N::Integer, domain::AbstractUnivariateDomain)
    coordinate_domains(Val(N), domain)
end

####
#### generic api
####

"""
A transformation maps values between a *domain*, usually specified by the basis, and the
(co)domain that is specified by a transformation.  Transformations are not required to be
subtypes, this just documents the interface they need to support:

- [`transform_to`](@ref)

- [`transform_from`](@ref)

- [`domain`](@ref)

!!! note
    Abstract type used for code organization, not exported.
"""
abstract type AbstractTransformation end

"""
$(TYPEDEF)

An abstract type for univariate transformations.
"""
abstract type UnivariateTransformation end

Broadcast.broadcastable(transformation::UnivariateTransformation) = Ref(transformation)

"""
`$(FUNCTIONNAME)(domain, transformation, x)`

Transform `x` to `domain` using `transformation`.

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function transform_to end

"""
`$(FUNCTIONNAME)(domain, transformation, x)`

Transform `x` from `domain` using `transformation`.

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function transform_from end

####
#### coordinate transformations
####

abstract type MultivariateTransformation end

struct CoordinateTransformations{T<:Tuple} <: MultivariateTransformation
    transformations::T
end

function Base.Tuple(coordinate_transformations::CoordinateTransformations)
    coordinate_transformations.transformations
end

function Base.show(io::IO, ct::CoordinateTransformations)
    print(io, "coordinate transformations")
    for t in ct.transformations
        print(io, "\n  ", t)
    end
end

Broadcast.broadcastable(ct::CoordinateTransformations) = Ref(ct)

"""
$(SIGNATURES)

Wrapper for coordinate-wise transformations.

```jldoctest
julia> using StaticArrays

julia> ct = coordinate_transformations(BoundedLinear(0, 2), SemiInfRational(2, 3))
coordinate transformations
  (0.0,2.0) ↔ domain [linear transformation]
  (2,∞) ↔ domain [rational transformation with scale 3]

julia> dom = coordinate_domains(PM1(), PM1())
[-1,1]²

julia> x = transform_from(dom, ct, (0.4, 0.5))
(1.4, 11.0)

julia> y = transform_to(dom, ct, x)
(0.3999999999999999, 0.5)
```
"""
function coordinate_transformations(transformations::Tuple)
    CoordinateTransformations(transformations)
end

coordinate_transformations(transformations...) = coordinate_transformations(transformations)

function transform_to(domain::CoordinateDomains, ct::CoordinateTransformations, x::Tuple)
    @unpack domains = domain
    @unpack transformations = ct
    @argcheck length(domains) == length(transformations) == length(x)
    map((d, t, x) -> transform_to(d, t, x), domains, transformations, x)
end

function transform_to(domain::CoordinateDomains, ct::CoordinateTransformations,
                   x::AbstractVector)
    SVector(transform_to(domain, ct, Tuple(x)))
end

function transform_from(domain::CoordinateDomains, ct::CoordinateTransformations, x::Tuple)
    @unpack domains = domain
    @unpack transformations = ct
    @argcheck length(domains) == length(transformations) == length(x)
    map((d, t, x) -> transform_from(d, t, x), domains, transformations, x)
end

function transform_from(domain::CoordinateDomains, ct::CoordinateTransformations, x::AbstractVector)
    SVector(transform_from(domain, ct, Tuple(x)))
end

####
#### specific transformations
####

###
### bounded linear
###

struct BoundedLinear{T <: Real} <: UnivariateTransformation
    "Midpoint `m`."
    m::T
    "Scale `s`."
    s::T
    function BoundedLinear(a::T, b::T) where {T <: Real}
        @argcheck isfinite(a) && isfinite(b) DomainError
        s = (b - a) / 2
        m = (a + b) / 2
        @argcheck s > 0 DomainError((; a, b), "Need `a < b`.")
        m, s = promote(m, s)
        new{typeof(m)}(m, s)
    end
end

function Base.show(io::IO, transformation::BoundedLinear)
    @unpack m, s = transformation
    print(io, "(", m - s, ",", m + s, ") ↔ domain [linear transformation]")
end

"""
$(TYPEDEF)

Transform the domain to `y ∈ (a, b)`, using ``y = x ⋅ s + m``.

`m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
"""
BoundedLinear(a::Real, b::Real) = BoundedLinear(promote(a, b)...)

function transform_from(::PM1, t::BoundedLinear, x::Scalar)
    @unpack m, s = t
    _add(_mul(x, s), m)
end

function transform_to(::PM1, t::BoundedLinear, y::Scalar)
    @unpack m, s = t
    _div(_sub(y, m), s)
end

function domain(t::BoundedLinear)
    @unpack m, s = t
    UnivariateDomain(m - s, m + s)
end

###
### semi-infinite interval
###

struct SemiInfRational{T<:Real} <: UnivariateTransformation
    "The finite endpoint `A`."
    A::T
    "Scale factor `L ≠ 0`."
    L::T
    function SemiInfRational(A::T, L::T) where {T <: Real}
        @argcheck isfinite(A) DomainError
        @argcheck isfinite(L) && L ≠ 0 DomainError
        new{T}(A, L)
    end
end

function Base.show(io::IO, transformation::SemiInfRational)
    @unpack A, L = transformation
    if L > 0
        D = "($A,∞)"
    else
        D = "(-∞,A)"
    end
    print(io, D, " ↔ domain [rational transformation with scale ", L, "]")
end

"""
$(SIGNATURES)

The domian transformed to  `[A, Inf)` (when `L > 0`) or `(-Inf,A]`
(when `L < 0`) using ``y = A + L ⋅ (1 + x) / (1 - x)``.

When used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.

# Example mappings for the domain ``(-1,1)``

- ``-1/2 ↦ A + L / 3``
- ``0 ↦ A + L``
- ``1/2 ↦ A + 3 ⋅ L``
"""
SemiInfRational(A::Real, L::Real) = SemiInfRational(promote(A, L)...)

transform_from(::PM1, t::SemiInfRational, x) = t.A + t.L * (1 + x) / (1 - x)

function transform_to(::PM1, t::SemiInfRational, y)
    @unpack A, L = t
    z = _sub(y, A)
    x = _div(_sub(z, L), _add(z, L))
    if (y == Inf && L > 0) || (y == -Inf && L < 0)
        one(x)                  # works for derivatives, because they are 0 at ±∞
    else
        x
    end
end

function domain(t::SemiInfRational)
    @unpack L, A = t
    ∞ = oftype(A, Inf)
    L > 0 ? UnivariateDomain(A, ∞) : UnivariateDomain(-∞, A)
end

###
### infinite interval
###

struct InfRational{T <: Real} <: UnivariateTransformation
    "The center `A`."
    A::T
    "Scale factor `L > 0`."
    L::T
    function InfRational(A::T, L::T) where {T <: Real}
        @argcheck isfinite(A) DomainError
        @argcheck isfinite(L) && L > 0 DomainError
        new{T}(A, L)
    end
end

function Base.show(io::IO, transformation::InfRational)
    @unpack A, L = transformation
    print(io, "(-∞,∞) ↔ domain [rational transformation with center ", A, ", scale ", L, "]")
end

"""
$(SIGNATURES)

The domain transformed to `(-Inf, Inf)` using ``y = A + L ⋅ x / √(1 - x^2)``, with `L > 0`.

# Example mappings (for domain ``(-1,1)``)

- ``0 ↦ A``
- ``±0.5 ↦ A ± L / √3``
"""
InfRational(A::Real, L::Real) = InfRational(promote(A, L)...)

transform_from(::PM1, T::InfRational, x::Real) = T.A + T.L * x / √(1 - abs2(x))

function transform_to(::PM1, t::InfRational, y::Real)
    @unpack A, L = t
    z = _sub(y, A)
    # FIXME implement for derivatives
    x = z / hypot(z, L)
    if isinf(y)
        y > 0 ? one(x) : -one(x)
    else
        x
    end
end

domain(::InfRational) = UnivariateDomain(-Inf, Inf)
