#####
##### Transformation of the Chebyshev polynomials — generic code
#####

export univariate_basis, BoundedLinear, InfRational, SemiInfRational

####
#### simplified transformation API
####

"""
$(TYPEDEF)

An abstract type for univariate transformations. Transformations are not required to be
subtypes, this just documents the interface they need to support:

- [`to_domain`](@ref)

- [`from_domain`](@ref)
"""
abstract type UnivariateTransformation end

"""
`$(FUNCTIONNAME)(transformation, parent, x)`

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function to_domain end

"""
`$(FUNCTIONNAME)(transformation, parent, x)`

!!! FIXME
    document, especially differentiability requirements at infinite endpoints
"""
function from_domain end

####
#### univariate basis
####

struct UnivariateBasis{P,T} <: FunctionBasis
    parent::P
    transformation::T
end

function Base.show(io::IO, univariate_basis::UnivariateBasis)
    @unpack parent, transformation = univariate_basis
    print(io, parent, "\n  domain ", transformation)
end


"""
$(SIGNATURES)

Create a univariate basis from `univariate_family`, with the specified `grid_kind` and
dimension `N`, transforming the domain with `transformation`.

`parent` is a univariate basis, `transformation` is a univariate transformation (supporting
the interface described by [`UnivariateTransformation`](@ref), but not necessarily a
subtype). Univariate bases support [`SpectralKit.gridpoint`](@ref).

# Example

The following is a basis with 10 transformed Chebyshev polynomials of the first kind on
``(3,∞)``, with equal amounts of nodes on both sides of `7 = 3 + 4` and an interior grid:

```jldoctest
julia> basis = univariate_basis(Chebyshev, InteriorGrid(), 10, SemiInfRational(3.0, 4.0))
Chebyshev polynomials (1st kind), interior grid, dimension: 10
  domain (3.0,∞) [rational transformation with scale 4.0]

julia> dimension(basis)
10

julia> domain(basis)
(3.0, Inf)
```
"""
function univariate_basis(univariate_family, grid_kind::AbstractGrid, N::Integer,
                          transformation)
    UnivariateBasis(univariate_family(grid_kind, Int(N)), transformation)
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

function gridpoint(::Type{T}, basis::UnivariateBasis, i) where T
    @unpack parent, transformation = basis
    from_domain(transformation, parent, gridpoint(T, parent, i))
end

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

function Base.show(io::IO, transformation::BoundedLinear)
    @unpack m, s = transformation
    print(io, "(", m - s, ",", m + s, ") [linear transformation]")
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

function Base.show(io::IO, transformation::SemiInfRational)
    @unpack A, L = transformation
    if L > 0
        D = "($A,∞)"
    else
        D = "(-∞,A)"
    end
    print(io, D, " [rational transformation with scale ", L, "]")
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

function Base.show(io::IO, transformation::InfRational)
    @unpack A, L = transformation
    print(io, "(-∞,∞) [rational transformation with center ", A, ", scale ", L, "]")
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
