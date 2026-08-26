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
#### specific transformations
####

###
### bounded linear
###

struct BoundedLinear{T <: Real} <: AbstractUnivariateTransformation
    "Lower limit."
    lower::T
    "Upper limit."
    upper::T
    @doc """
    $(SIGNATURES)

    Transform the domain to `y ∈ (lower, upper)`, using a linear mapping.

    `lower < upper` is enforced.
    """
    function BoundedLinear(; lower::Real, upper::Real)
        @argcheck isfinite(lower) && isfinite(upper) DomainError
        lower, upper = promote(lower, upper)
        @argcheck upper > lower DomainError((; lower, upper), "Need `lower < upper`.")
        new{typeof(lower)}(lower, upper)
    end
end

function Base.show(io::IO, transformation::BoundedLinear)
    (; lower, upper) = transformation
    print(io, "BoundedLinear(lower = ", lower, ", upper = ", upper, ")")
end

function transform_from(::PM1, t::BoundedLinear, x::Scalar)
    (; lower, upper) = t
    (x+1) / 2 * (upper-lower)  + lower
end

function transform_to(::PM1, t::BoundedLinear, y::Real)
    (; lower, upper) = t
    (y-lower) / (upper-lower) * 2 - 1
end

function transform_to(domain::PM1, t::BoundedLinear, y::𝑑Expansion{Dp1}) where Dp1
    (; lower, upper) = t
    (; coefficients) = y
    y0, yD... = coefficients
    x0 = transform_to(domain, t, y0)
    s = (upper - lower) / 2
    xD = map(y -> y / s, yD)
    𝑑Expansion(SVector(x0, xD...))
end

function domain(t::BoundedLinear)
    (; lower, upper) = t
    UnivariateDomain(lower, upper)
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
    print(io, "SemiInfRational(endpoint = ", endpoint, ", scale = ", scale, ")")
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
    print(io, "InfRational(; center = ", center, ", scale = ", scale, ")")
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
