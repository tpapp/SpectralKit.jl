#####
##### transformations
#####

export domain, domain_kind, transform_to, transform_from, coordinate_transformations,
    BoundedLinear, InfRational, SemiInfRational

####
#### generic api
####

"""

!!! note
    Abstract type used for code organization, not exported.
"""

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
    ∂CoordinateExpansion(x.∂D, transform_to(domain, ct, x.x))
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
    (; m, s) = transformation
    print(io, "(", m - s, ",", m + s, ") ↔ domain [linear transformation]")
end

"""
$(TYPEDEF)

Transform the domain to `y ∈ (a, b)`, using ``y = x ⋅ s + m``.

`m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
"""
BoundedLinear(a::Real, b::Real) = BoundedLinear(promote(a, b)...)

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
    (; A, L) = transformation
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

function transform_to(::PM1, t::SemiInfRational, y::Real)
    (; A, L) = t
    z = y - A
    x = (z - L) / (z + L)
    if y == Inf || y == -Inf
        one(x)
    else
        x
    end
end

function transform_to(domain::PM1, t::SemiInfRational, y::𝑑Expansion{Dp1}) where Dp1
    (; A, L) = t
    (; coefficients) = y
    x0 = transform_to(domain, t, coefficients[1])
    Dp1 == 1 && return 𝑑Expansion(SVector(x0))
    # based on Boyd (2001), Table E.7
    Q = abs2(x0 - 1)
    x1 = (coefficients[2] * Q) / (2*L)
    Dp1 == 2 && return 𝑑Expansion(SVector(x0, x1))
    error("$(Dp1-1)th derivative not implemented yet, open an issue.")
end

function domain(t::SemiInfRational)
    (; L, A) = t
    A = float(A)
    ∞ = oftype(A, Inf)
    L > 0 ? UnivariateDomain(A, ∞) : UnivariateDomain(-∞, A)
end

###
### infinite interval
###

struct InfRational{T <: Real} <: AbstractUnivariateTransformation
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
    (; A, L) = transformation
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
    (; A, L) = t
    z = y - A
    x = z / hypot(z, L)
    if isinf(y)
        y > 0 ? one(x) : -one(x)
    else
        x
    end
end

function transform_to(domain::PM1, t::InfRational, y::𝑑Expansion{Dp1}) where Dp1
    (; A, L) = t
    (; coefficients) = y
    x0 = transform_to(domain, t, coefficients[1])
    Dp1 == 1 && return Coefficients(SVector(x0))
    # based on Boyd (2001), Table E.5
    Q = 1 - abs2(x0)
    sQ = √Q
    x1 = (coefficients[2] * Q * sQ) / L
    Dp1 == 2 && return 𝑑Expansion(SVector(x0, x1))
    error("$(Dp1-1)th derivative not implemented yet, open an issue.")
end

domain(::InfRational) = UnivariateDomain(-Inf, Inf)
