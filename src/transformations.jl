#####
##### transformations
#####

export domain, domain_kind, transform_to, transform_from, coordinate_transformations,
    BoundedLinear, InfRational, SemiInfRational

####
#### generic api
####

"""
$(TYPEDEF)

An abstract type for univariate transformations.
"""
abstract type AbstractUnivariateTransformation end

Broadcast.broadcastable(transformation::AbstractUnivariateTransformation) = Ref(transformation)

domain_kind(::Type{<:AbstractUnivariateTransformation}) = :univariate

"""
`$(FUNCTIONNAME)(domain, transformation, x)`

Transform `x` to `domain` using `transformation`.

`domain` can be replaced by `basis` for a shortcut which uses `domain(basis)`.

Transformations to infinity make sure that ``\\pm\\infty`` is mapped to the limit for
values and derivatives.
"""
function transform_to end

"""
`$(FUNCTIONNAME)(domain, transformation, x)`

Transform `x` from `domain` using `transformation`.

`domain` can be replaced by `basis` for a shortcut which uses `domain(basis)`.

Transformations to infinity make sure that ``\\pm\\infty`` is mapped to the limit for
values and derivatives.
"""
function transform_from end

####
#### coordinate transformations
####

struct CoordinateTransformations{T<:Tuple}
    transformations::T
end

domain_kind(::Type{<:CoordinateTransformations}) = :multivariate

function domain(coordinate_transformations::CoordinateTransformations)
    coordinate_domains(map(domain, coordinate_transformations.transformations))
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

Wrapper for coordinate-wise transformations. To extract components, convert to Tuple.

```jldoctest
julia> using StaticArrays

julia> ct = coordinate_transformations(BoundedLinear(0, 2), SemiInfRational(2, 3))
coordinate transformations
  (0.0,2.0) ↔ domain [linear transformation]
  (2,∞) ↔ domain [rational transformation with scale 3]

julia> d1 = domain(Chebyshev(InteriorGrid(), 5))
[-1,1]

julia> dom = coordinate_domains(d1, d1)
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
    (; domains) = domain
    (; transformations) = ct
    @argcheck length(domains) == length(transformations) == length(x)
    map((d, t, x) -> transform_to(d, t, x), domains, transformations, x)
end

function transform_to(domain::CoordinateDomains{T}, ct::CoordinateTransformations,
                      x::AbstractVector) where T
    SVector(transform_to(domain, ct, _ntuple_like(T, x)))
end

function transform_to(domain::CoordinateDomains, ct::CoordinateTransformations,
                      Dx::∂CoordinateExpansion)
    ∂CoordinateExpansion(Dx.∂D, transform_to(domain, ct, Dx.x))
end

function transform_from(domain::CoordinateDomains, ct::CoordinateTransformations, x::Tuple)
    (; domains) = domain
    (; transformations) = ct
    @argcheck length(domains) == length(transformations) == length(x)
    map((d, t, x) -> transform_from(d, t, x), domains, transformations, x)
end

function transform_from(domain::CoordinateDomains{T}, ct::CoordinateTransformations,
                        x::AbstractVector) where {T}
    SVector(transform_from(domain, ct, _ntuple_like(T, x)))
end

####
#### specific transformations
####

###
### bounded linear
###

struct BoundedLinear{T <: Real} <: AbstractUnivariateTransformation
    "Midpoint `m`."
    m::T
    "Scale `s`."
    s::T
    @doc """
    $(SIGNATURES)

    Transform the domain to `y ∈ (lower, upper)`, using a linear mapping.

    `lower < upper` is enforced.
    """
    function BoundedLinear(; lower::Real, upper::Real)
        @argcheck isfinite(lower) && isfinite(upper) DomainError
        lower, upper = promote(lower, upper)
        s = (upper - lower) / 2
        m = (lower + upper) / 2
        @argcheck s > 0 DomainError((; lower, upper), "Need `lower < upper`.")
        m, s = promote(m, s)
        new{typeof(m)}(m, s)
    end
end

function Base.show(io::IO, transformation::BoundedLinear)
    (; m, s) = transformation
    print(io, "(", m - s, ",", m + s, ") ↔ domain [linear transformation]")
end

function transform_from(::PM1, t::BoundedLinear, x::Scalar)
    (; m, s) = t
    x * s + m
end

function transform_to(::PM1, t::BoundedLinear, y::Real)
    (; m, s) = t
    (y - m) / s
end

function transform_to(domain::PM1, t::BoundedLinear, y::𝑑Expansion{Dp1}) where Dp1
    (; m, s) = t
    (; coefficients) = y
    y0, yD... = coefficients
    x0 = transform_to(domain, t, y0)
    xD = map(y -> y / s, yD)
    𝑑Expansion(SVector(x0, xD...))
end

function domain(t::BoundedLinear)
    (; m, s) = t
    UnivariateDomain(m - s, m + s)
end

###
### semi-infinite interval
###

struct SemiInfRational{T<:Real} <: AbstractUnivariateTransformation
    "The finite endpoint."
    endpoint::T
    "Scale factor."
    scale::T
    @doc """
    $(SIGNATURES)

    The domian transformed to  `[endpoint, Inf)` (when `scale > 0`) or `(-Inf,endpoint]`
    (when `scale < 0`) using ``y = endpoint + scale ⋅ (1 + x) / (1 - x)``.

    When used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.

    # Example mappings for the domain ``(-1,1)``

    - ``-1/2 ↦ endpoint + scale / 3``
    - ``0 ↦ endpoint + scale``
    - ``1/2 ↦ endpoint + 3 ⋅ scale``
    """
    function SemiInfRational(; endpoint::Real = 0, scale::Real = 1)
        @argcheck isfinite(endpoint) DomainError
        @argcheck isfinite(scale) && scale ≠ 0 DomainError
        endpoint, scale = promote(endpoint, scale)
        new{typeof(endpoint)}(endpoint, scale)
    end
end

function Base.show(io::IO, transformation::SemiInfRational)
    (; endpoint, scale) = transformation
    if scale > 0
        D = "($(endpoint),∞)"
    else
        D = "(-∞,$(endpoint))"
    end
    print(io, D, " ↔ domain [rational transformation with scale ", scale, "]")
end

transform_from(::PM1, t::SemiInfRational, x) = t.endpoint + t.scale * (1 + x) / (1 - x)

function transform_to(::PM1, t::SemiInfRational, y::Real)
    (; endpoint, scale) = t
    z = y - endpoint
    x = (z - scale) / (z + scale)
    if y == Inf || y == -Inf
        one(x)
    else
        x
    end
end

function transform_to(domain::PM1, t::SemiInfRational, y::𝑑Expansion{Dp1}) where Dp1
    (; scale) = t
    (; coefficients) = y
    x0 = transform_to(domain, t, coefficients[1])
    Dp1 == 1 && return 𝑑Expansion(SVector(x0))
    # based on Boyd (2001), Table E.7
    Q = abs2(x0 - 1)
    x1 = (coefficients[2] * Q) / (2*scale)
    Dp1 == 2 && return 𝑑Expansion(SVector(x0, x1))
    error("$(Dp1-1)th derivative not implemented yet, open an issue.")
end

function domain(t::SemiInfRational)
    (; endpoint, scale) = t
    endpoint = float(endpoint)
    ∞ = oftype(endpoint, Inf)
    scale > 0 ? UnivariateDomain(endpoint, ∞) : UnivariateDomain(-∞, endpoint)
end

###
### infinite interval
###

struct InfRational{T <: Real} <: AbstractUnivariateTransformation
    "The center"
    center::T
    "Scale factor"
    scale::T
    @doc """
    $(SIGNATURES)

    The domain transformed to `(-Inf, Inf)` using
    ``y = center + scale ⋅ x / √(1 - x^2)``, with `scale > 0`.

    # Example mappings (for domain ``(-1,1)``)

    - ``0 ↦ center``
    - ``±0.5 ↦ center ± scale / √3``
    """
    function InfRational(center::T, scale::T) where {T <: Real}
        @argcheck isfinite(center) DomainError
        @argcheck isfinite(scale) && scale > 0 DomainError
        new{T}(center, scale)
    end
end

function Base.show(io::IO, transformation::InfRational)
    (; center, scale) = transformation
    print(io, "(-∞,∞) ↔ domain [rational transformation with center ", center, ", scale ", scale, "]")
end

InfRational(; center::Real = 0.0, scale::Real = 1.0) = InfRational(promote(center, scale)...)

transform_from(::PM1, T::InfRational, x::Real) = T.center + T.scale * x / √(1 - abs2(x))

function transform_to(::PM1, t::InfRational, y::Real)
    (; center, scale) = t
    z = y - center
    x = z / hypot(z, scale)
    if isinf(y)
        y > 0 ? one(x) : -one(x)
    else
        x
    end
end

function transform_to(domain::PM1, t::InfRational, y::𝑑Expansion{Dp1}) where Dp1
    (; scale) = t
    (; coefficients) = y
    x0 = transform_to(domain, t, coefficients[1])
    Dp1 == 1 && return SVector(x0)
    # based on Boyd (2001), Table E.5
    Q = 1 - abs2(x0)
    sQ = √Q
    x1 = (coefficients[2] * Q * sQ) / scale
    Dp1 == 2 && return 𝑑Expansion(SVector(x0, x1))
    error("$(Dp1-1)th derivative not implemented yet, open an issue.")
end

domain(::InfRational) = UnivariateDomain(-Inf, Inf)
