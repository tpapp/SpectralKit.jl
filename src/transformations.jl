#####
##### transformations
#####

export to_pm1, from_pm1, coordinate_transformations,
    BoundedLinear, InfRational, SemiInfRational

####
#### generic api
####

"""
$(TYPEDEF)

An abstract type for univariate transformations. Transformations are not required to be
subtypes, this just documents the interface they need to support:

- [`to_pm1`](@ref)

- [`from_pm1`](@ref)

- [`domain`](@ref)

!!! NOTE
    Abstract type used for code organization, not exported.
"""
abstract type UnivariateTransformation end

Broadcast.broadcastable(transformation::UnivariateTransformation) = Ref(transformation)

function domain(transformation::UnivariateTransformation)
    from_pm1.(transformation, (-1, 1))
end

"""
`$(FUNCTIONNAME)(transformation, x)`

Transform `x` to ``[-1, 1]`` using `transformation`.

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function to_pm1 end

"""
`$(FUNCTIONNAME)(transformation, x)`

Transform `x` from ``[-1, 1]`` using `transformation`.

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function from_pm1 end

####
#### coordinate transformations
####

struct CoordinateTransformations{T<:Tuple}
    transformations::T
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
julia> ct = coordinate_transformations(BoundedLinear(0, 2), SemiInfRational(2, 3))
coordinate transformations
  (0.0,2.0) ↔ (-1, 1) [linear transformation]
  (2,∞) ↔ (-1, 1) [rational transformation with scale 3]

julia> x = from_pm1(ct, SVector(0.4, 0.5))
2-element SVector{2, Float64} with indices SOneTo(2):
  1.4
 11.0

julia> y = to_pm1(ct, x)
2-element SVector{2, Float64} with indices SOneTo(2):
 0.3999999999999999
 0.5
```
"""
function coordinate_transformations(transformations::Tuple)
    CoordinateTransformations(transformations)
end

coordinate_transformations(transformations...) = coordinate_transformations(transformations)

function to_pm1(ct::CoordinateTransformations, x::SVector{N}) where N
    SVector{N}(map((t, x) -> to_pm1(t, x), ct.transformations, Tuple(x)))
end

function to_pm1(ct::CoordinateTransformations, x::AbstractVector)
    to_pm1(ct, SVector{length(ct.transformations)}(x))
end

function from_pm1(ct::CoordinateTransformations, x::SVector{N}) where N
    SVector{N}(map((t, x) -> from_pm1(t, x), ct.transformations, Tuple(x)))
end

function from_pm1(ct::CoordinateTransformations, x::AbstractVector)
    from_pm1(ct, SVector{length(ct.transformations)}(x))
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
    print(io, "(", m - s, ",", m + s, ") ↔ (-1, 1) [linear transformation]")
end

"""
$(TYPEDEF)

Transform `x ∈ (-1,1)` to `y ∈ (a, b)`, using ``y = x ⋅ s + m``.

`m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
"""
BoundedLinear(a::Real, b::Real) = BoundedLinear(promote(a, b)...)

function from_pm1(T::BoundedLinear, x::Real)
    @unpack m, s = T
    x * s + m
end

function to_pm1(T::BoundedLinear, y::Real)
    @unpack m, s = T
    (y - m) / s
end

###
### semi-infinite interval
###

struct SemiInfRational{T <: Real} <: UnivariateTransformation
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
    print(io, D, " ↔ (-1, 1) [rational transformation with scale ", L, "]")
end

"""
$(SIGNATURES)

`[-1,1]` transformed to the domain `[A, Inf)` (when `L > 0`) or `(-Inf,A]`
(when `L < 0`) using ``y = A + L ⋅ (1 + x) / (1 - x)``.

When used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.

# Example mappings

- ``-1/2 ↦ A + L / 3``
- ``0 ↦ A + L``
- ``1/2 ↦ A + 3 ⋅ L``
"""
SemiInfRational(A::Real, L::Real) = SemiInfRational(promote(A, L)...)

from_pm1(T::SemiInfRational, x) = T.A + T.L * (1 + x) / (1 - x)

function to_pm1(T::SemiInfRational, y)
    @unpack A, L = T
    z = y - A
    x = (z - L) / (z + L)
    if (y == Inf && L > 0) || (y == -Inf && L < 0)
        one(x)
    else
        x
    end
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
    print(io, "(-∞,∞) ↔ (-1, 1) [rational transformation with center ", A, ", scale ", L, "]")
end

"""
$(SIGNATURES)

Chebyshev polynomials transformed to the domain `(-Inf, Inf)`
using ``y = A + L ⋅ x / √(1 - x^2)``, with `L > 0`.

# Example mappings

- ``0 ↦ A``
- ``±0.5 ↦ A ± L / √3``
"""
InfRational(A::Real, L::Real) = InfRational(promote(A, L)...)

from_pm1(T::InfRational, x::Real) = T.A + T.L * x / √(1 - abs2(x))

function to_pm1(T::InfRational, y::Real)
    @unpack A, L = T
    z = y - A
    x = z / hypot(L, z)
    if isinf(y)
        y > 0 ? one(x) : -one(x)
    else
        x
    end
end
