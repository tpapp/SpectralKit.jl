#####
##### Transformation of the Chebyshev polynomials — generic code
#####

export SemiInfinite, univariate_basis

abstract type UnivariateTransformation end

"""
$(TYPEDEF)

`[-1,1]` transformed to the domain `[A, Inf)` (when `L > 0`) or `(-Inf,A]`
(when `L < 0`) using ``y = A + L ⋅ (1 + x) / (1 - x)``.
"""
struct SemiInfinite{T <: Real} <: UnivariateTransformation
    "The finite endpoint `A`."
    A::T
    "Scale factor `L ≠ 0`."
    L::T
    function SemiInfinite(A::T, L::T) where {T <: Real}
        @argcheck L ≠ 0
        new{T}(A, L)
    end
end

SemiInfinite(A::Real, L::Real) = SemiInfinite(promote(A, L)...)

from_domain(T::SemiInfinite, ::Chebyshev, x) = T.A + T.L * (1 + x) / (1 - x)

function to_domain(T::SemiInfinite, ::Chebyshev, y)
    @unpack A, L = T
    z = y - A
    x = (z - L) / (z + L)
    if (y == Inf && L > 0) || (y == -Inf && L < 0)
        one(x)
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
univariate_basis(Chebyshev(10), SemiInfinite(3.0, 4.0))
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

# """
# $(TYPEDEF)

# Define a function family by transforming from the Chebyshev polynomials on `[-1,1]`.

# For `family::TransformedChebyshev`, subtypes implement:

# - `from_chebyshev(family, x)`, for transforming from `x ∈ [-1,1]` to `y` of the domain,

# - `to_chebyshev(family, y, order)` for transforming `y` the domain to `x ∈ [-1,1]`, where
#   `order` follows the semantics of [`basis_function`](@ref) and returns `x`, `∂x/∂y`, … as
#   requested.

# - `domain_extrema(family)` is optional.
# """
# abstract type TransformedChebyshev <: FunctionFamily end

# function roots(::Type{T}, family::TransformedChebyshev, N) where {T}
#     from_chebyshev.(family, roots(T, Chebyshev(), N))
# end

# function extrema(::Type{T}, family::TransformedChebyshev, N) where {T}
#     from_chebyshev.(family, extrema(T, Chebyshev(), N))
# end

# """
# $(SIGNATURES)

# Return the first argument transformed to the domain `[-1,1]`, and a callable that transform
# Chebyshev basis functions and derivatives back to the original domain.
# """
# function chebyshev_transform_helpers(family, y, order::Order{0})
#     to_chebyshev(family, y, order), identity
# end

# struct TransformOrder1{O,T}
#     order::O
#     x′::T
# end

# (τ::TransformOrder1{Order{1}})(t′::Real) = τ.x′ * t′
# (τ::TransformOrder1{OrdersTo{1}})(t::SVector{2}) = SVector(t[1], τ.x′ * t[2])

# function chebyshev_transform_helpers(family, y, order::Union{Order{1},OrdersTo{1}})
#     x, x′ = to_chebyshev(family, y, OrdersTo(1))
#     x, TransformOrder1(order, x′)
# end

# function basis_function(family::TransformedChebyshev, y::Real, order, k)
#     x, τ = chebyshev_transform_helpers(family, y, order)
#     b = basis_function(Chebyshev(), x, order, k)
#     if k isa Int
#         τ(b)
#     elseif k isa Val
#         map(τ, b)
#     end
# end

# function basis_function(family::TransformedChebyshev, y::Real, order)
#     x, τ = chebyshev_transform_helpers(family, y, order)
#     (τ(x) for x in basis_function(Chebyshev(), x, order))
# end

# function domain_extrema(family::TransformedChebyshev)
#     from_chebyshev.(family, domain_extrema(Chebyshev()))
# end

####
#### Rational Chebyshev on [A,Inf) or (-Inf,A]
####


# function to_chebyshev(TL::ChebyshevSemiInf, y, ::Order{1})
#     @unpack A, L = TL
#     2 * L / abs2(y - A + L)
# end

# function to_chebyshev(TL::ChebyshevSemiInf, y, ::OrdersTo{1})
#     @unpack A, L = TL
#     z = y - A
#     num = z - L
#     den = z + L
#     SVector(chebyshev_semiinf_endpoints(y, L, num/den), 2 * L / abs2(den))
# end

####
#### Rational Chebyshev on (-Inf,Inf)
####

# """
# $(TYPEDEF)

# Chebyshev polynomials transformed to the domain `(-Inf, Inf)`
# using ``y = A + L ⋅ x / √(L^2 + x^2)``, with `L > 0`.

# `0` is mapped to `A`.
# """
# struct ChebyshevInf{T <: Real} <: TransformedChebyshev
#     "The center `A`."
#     A::T
#     "Scale factor `L > 0`."
#     L::T
#     function ChebyshevInf(A::T, L::T) where {T <: Real}
#         @argcheck L > 0
#         new{T}(A, L)
#     end
# end

# ChebyshevInf(A::Real, L::Real) = ChebyshevInf(promote(A, L)...)

# function Base.show(io::IO, TB::ChebyshevInf)
#     print(io, "ChebyshevInf($(TB.A), $(TB.L))")
# end

# domain_extrema(TB::ChebyshevInf) = (-Inf, Inf)

# function from_chebyshev(TB::ChebyshevInf, x)
#     TB.A + TB.L * x / √(1 - abs2(x))
# end

# function chebyshev_inf_endpoints(y, x::T) where {T}
#     if y == Inf
#         one_like(T)
#     elseif y == -Inf
#         -one_like(T)
#     else
#         x
#     end
# end

# function to_chebyshev(TB::ChebyshevInf, y)
#     @unpack A, L = TB
#     z = y - A
#     x = z / hypot(L, z)
#     chebyshev_inf_endpoints(y, x)
# end

# function to_chebyshev(TB::ChebyshevInf, y)
#     @unpack A, L = TB
#     abs2(L) / hypot(L, z)^3
# end

# function to_chebyshev(TB::ChebyshevInf, y)
#     @unpack A, L = TB
#     z = y - A
#     den = hypot(L, z)
#     x = z / den
#     x′ = abs2(L) / den^3
#     SVector(chebyshev_inf_endpoints(y, x), x′)
# end

####
#### Chebyshev on a finite interval `[a,b]`
####

# """
# $(TYPEDEF)

# Chebyshev polynomials transformed to the domain `(a, b)`. using ``y = x ⋅ s + m``.

# `m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
# """
# struct ChebyshevInterval{T <: Real} <: TransformedChebyshev
#     "Start of interval `a`."
#     a::T
#     "End of interval `b`."
#     b::T
#     "Midpoint `m`."
#     m::T
#     "Scale `s`."
#     s::T
#     function ChebyshevInterval(a::T, b::T) where {T <: Real}
#         @argcheck isfinite(a) && isfinite(b)
#         s = (b - a) / 2
#         m = (a + b) / 2
#         @argcheck s > 0 "Need `a < b`."
#         new{promote_type(T,typeof(m),typeof(s))}(a, b, m, s)
#     end
# end

# function Base.show(io::IO, TI::ChebyshevInterval)
#     print(io, "ChebyshevInterval($(TI.a), $(TI.b))")
# end

# ChebyshevInterval(A::Real, L::Real) = ChebyshevInterval(promote(A, L)...)

# domain_extrema(TI::ChebyshevInterval) = (TI.a, TI.b)

# function from_chebyshev(TI::ChebyshevInterval, x)
#     @unpack m, s = TI
#     x * s + m
# end

# function to_chebyshev(TI::ChebyshevInterval, y)
#     @unpack m, s = TI
#     (y - m) / s
# end
