#####
##### Transformation of the Chebyshev polynomials — generic code
#####

export univariate_basis, BoundedLinear, InfRational, SemiInfRational

abstract type UnivariateTransformation end

####
#### transformations
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
        @argcheck isfinite(a) && isfinite(b)
        s = (b - a) / 2
        m = (a + b) / 2
        @argcheck s > 0 "Need `a < b`."
        m, s = promote(m, s)
        new{typeof(m)}(m, s)
    end
end

"""
$(TYPEDEF)

Transform `x ∈ (-1,1)` to `y ∈ (a, b)`, using ``y = x ⋅ s + m``.

`m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
"""
BoundedLinear(a::Real, b::Real) = BoundedLinear(promote(a, b)...)

function from_domain(T::BoundedLinear, ::Chebyshev, x::Real)
    @unpack m, s = T
    x * s + m
end

function to_domain(T::BoundedLinear, ::Chebyshev, y::Real)
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
        @argcheck L ≠ 0
        new{T}(A, L)
    end
end

"""
$(SIGNATURES)

`[-1,1]` transformed to the domain `[A, Inf)` (when `L > 0`) or `(-Inf,A]`
(when `L < 0`) using ``y = A + L ⋅ (1 + x) / (1 - x)``.

When used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.
"""
SemiInfRational(A::Real, L::Real) = SemiInfRational(promote(A, L)...)

from_domain(T::SemiInfRational, ::Chebyshev, x) = T.A + T.L * (1 + x) / (1 - x)

function to_domain(T::SemiInfRational, ::Chebyshev, y)
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
        @argcheck L > 0
        new{T}(A, L)
    end
end

"""
$(TYPEDEF)

Chebyshev polynomials transformed to the domain `(-Inf, Inf)`
using ``y = A + L ⋅ x / √(1 - x^2)``, with `L > 0`.

`0` is mapped to `A`.
"""
InfRational(A::Real, L::Real) = InfRational(promote(A, L)...)

from_domain(T::InfRational, ::Chebyshev, x::Real) = T.A + T.L * x / √(1 - abs2(x))

function to_domain(T::InfRational, ::Chebyshev, y::Real)
    @unpack A, L = T
    z = y - A
    x = z / hypot(L, z)
    if isinf(y)
        y > 0 ? one(x) : -one(x)
    else
        x
    end
end

####
#### univariate basis
####

struct UnivariateBasis{P,T}
    parent::P
    transformation::T
end

"""
$(SIGNATURES)

Create a univariate basis from `parent`, transforming the domain with `transformation`.

# Example

The following is a basis with 10 transformed Chebyshev polynomials of the first kind on
``(3,∞)``, with equal amounts of nodes on both sides of `7 = 3 + 4`:
```julia
univariate_basis(Chebyshev(10), SemiInfRational(3.0, 4.0))
```
"""
function univariate_basis(parent, transformation)
    UnivariateBasis(parent, transformation)
end

@inline dimension(basis::UnivariateBasis) = dimension(basis.parent)

function domain(basis::UnivariateBasis)
    @unpack parent, transformation = basis
    from_domain.(Ref(transformation), Ref(parent), domain(parent))
end

function basis_at(basis::UnivariateBasis, x::Real)
    @unpack parent, transformation = basis
    basis_at(parent, to_domain(transformation, parent, x))
end

function grid(::Type{T}, basis::UnivariateBasis, kind) where T
    @unpack parent, transformation = basis
    map(x -> from_domain(transformation, parent, x), grid(T, parent, kind))
end
