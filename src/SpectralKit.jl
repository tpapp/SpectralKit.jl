module SpectralKit

export domain_extrema, roots, augmented_extrema, evaluate, Chebyshev, SemiInfChebyshev

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using Parameters: @unpack

####
#### utilities
####

flip_if(condition, x) = condition ? -x : x

one_like(::Type{T}) where {T} = cos(zero(T))

"""
$(TYPEDEF)

Abstract type for function families.

# Supported interface

For `F::FunctionFamily`, the supported API functions are

- [`domain_extrema`](@ref) for querying the domain,

- [`evaluate`](@ref) for function evaluation,

- [`roots`](@ref) and [`augmented_extrema`](@ref) to obtain collocation points.
"""
abstract type FunctionFamily end

Broadcast.broadcastable(family::FunctionFamily) = Ref(family)

"""
`$(FUNCTIONNAME)(family)`

Return the extrema of the domain of the given `family` as an ordered tuple.

Type is not specified, but guaranteed to be the same for both endpoints.
"""
function domain_extrema end

"""
`$(FUNCTIONNAME)(family, n, x, order)`

Evaluate the `n`th (starting from 0) function in `family` at `x`.

`order` determines the derivatives:

- `Val(d)` returns the `d`th derivatives, starting from `0` (for the function value)

- `Val(0:d)` returns the derivatives up to `d`, starting from the function value, as
   multiple values (ie a tuple).

The implementation is intended to be type stable.

Evaluating outside the domain should not error, but may not return correct values for all
inputs, especially when far from the domain endpoints.
"""
function evaluate end

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the roots of the `N`th function in `family`, as a vector of `N` numbers with element
type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Chebyshev” grid.

Order is monotone, but not guaranteed to be increasing.
"""
roots(family, N::Integer) = roots(Float64, family, N)

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the augmented extrema (extrema + boundary values) of the `N-1`th function in
`family`, as a vector of `N` numbers with element type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Lobatto” grid.

Order is monotone, but not guaranteed to be increasing.
"""
function augmented_extrema(family::FunctionFamily, N::Integer)
    augmented_extrema(Float64, family, N)
end

####
#### Chebyshev on [-1,1]
####

"""
$(TYPEDEF)

Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev <: FunctionFamily end

@inline domain_extrema(::Chebyshev) = (-1, 1)

"""
$(SIGNATURES)

Evaluate the `n`th Chebyshev polynomial at `-1` for the given orders.
"""
chebyshev_min(::Type{T}, n::Integer, ::Val{0}) where {T} = flip_if(isodd(n), one_like(T))

"""
$(SIGNATURES)

Evaluate the `n`th Chebyshev polynomial at `1` for the given orders.
"""
chebyshev_max(::Type{T}, n::Integer, ::Val{0}) where {T} = one_like(T)

function chebyshev_min(::Type{T}, n::Integer, ::Val{1}) where {T}
    flip_if(iseven(n), one_like(T) * abs2(n))
end

chebyshev_max(::Type{T}, n::Integer, ::Val{1}) where {T} = one_like(T) * abs2(n)

function chebyshev_min(::Type{T}, n::Integer, ::Val{0:1}) where {T}
    chebyshev_min(T, n, Val(0)), chebyshev_min(T, n, Val(1))
end

function chebyshev_max(::Type{T}, n::Integer, ::Val{0:1}) where {T}
    chebyshev_max(T, n, Val(0)), chebyshev_max(T, n, Val(1))
end

"""
$(SIGNATURES)

Evaluate the `n`th Chebyshev polynomial at `-1 < x < 1` for the given orders.
"""
chebyshev_interior(n::Integer, x::Real, ::Val{0}) = cos(n * acos(x))

function chebyshev_interior(n::Integer, x::Real, ::Val{1})
    t = acos(x)
    n * sin(n * t) / sin(t)
end

function chebyshev_interior(n::Integer, x::Real, ::Val{0:1})
    t = acos(x)
    t′ = 1 / sin(t)
    s, c = sincos(n * t)
    c, s * n * t′
end

function evaluate(family::Chebyshev, n::Integer, x::T, order) where {T <: Real}
    if x == -1
        chebyshev_min(T, n, order)
    elseif x == 1
        chebyshev_max(T, n, order)
    else
        chebyshev_interior(n, x, order)
    end
end

function roots(::Type{T}, ::Chebyshev, N) where {T <: Real}
    cospi.(((2 * N - 1):-2:1) ./ T(2 * N))
end

function augmented_extrema(::Type{T}, family::Chebyshev, N) where {T}
    cospi.(((N-1):-1:0) ./ T(N - 1))
end

####
#### Transformation of the Chebyshev polynomials — generic code
####

"""
$(TYPEDEF)

Define a function family by transforming from the Chebyshev polynomials on `[-1,1]`.

For `family::TransformedChebyshev`, subtypes implement:

- `from_chebyshev(family, x)`, for transforming from `x ∈ [-1,1]` to `y` of the domain,

- `to_chebyshev(family, y, order)` for transforming `y` the domain to `x ∈ [-1,1]`, where
  `order` follows the semantics of [`evaluate`](@ref) and returns `x`, `∂x/∂y`, … as
  requested.

- `domain_extrema(family)` is optional.
"""
abstract type TransformedChebyshev <: FunctionFamily end

function roots(::Type{T}, family::TransformedChebyshev, N) where {T}
    from_chebyshev.(family, roots(T, Chebyshev(), N))
end

function augmented_extrema(::Type{T}, family::TransformedChebyshev, N) where {T}
    from_chebyshev.(family, augmented_extrema(T, Chebyshev(), N))
end

function evaluate(family::TransformedChebyshev, n::Integer, x::Real, order::Val{0})
    evaluate(Chebyshev(), n, to_chebyshev(family, x, order), order)
end

function evaluate(family::TransformedChebyshev, n::Integer, x::Real, ::Val{1})
    x, x′ = to_chebyshev(family, x, Val(0:1))
    t′ = evaluate(Chebyshev(), n, x, Val(1))
    t′ * x′
end

function evaluate(family::TransformedChebyshev, n::Integer, x::Real, ::Val{0:1})
    x, x′ = to_chebyshev(family, x, Val(0:1))
    t, t′ = evaluate(Chebyshev(), n, x, Val(0:1))
    t, t′ * x′
end

function domain_extrema(family::TransformedChebyshev)
    from_chebyshev.(family, domain_extrema(Chebyshev()))
end

####
#### Rational Chebyshev on [A,Inf) or (-Inf,A]
####

"""
$(TYPEDEF)

Chebyshev polynomials transformed to the domain `[A, Inf)` (when `L > 0`) or `(-Inf,A]`
(when `L < 0`) using ``y = A + L * (1 + x) / (1 - x)``.
"""
struct SemiInfChebyshev{T <: Real} <: TransformedChebyshev
    "The finite endpoint."
    A::T
    "Scale factor."
    L::T
    function SemiInfChebyshev(A::T, L::T) where {T <: Real}
        @argcheck L ≠ 0
        new{T}(A, L)
    end
end

SemiInfChebyshev(A::Real, L::Real) = SemiInfChebyshev(promote(A, L)...)

function domain_extrema(TL::SemiInfChebyshev)
    @unpack A, L = TL
    if L > 0
        promote(A, Inf)
    else
        promote(-Inf, A)
    end
end

function from_chebyshev(TL::SemiInfChebyshev, x)
    TL.A + TL.L * (1 + x) / (1 - x)
end

function semiinf_chebyshev_endpoints(y, L, x::T) where {T}
    if (y == Inf && L > 0) || (y == -Inf && L < 0)
        one_like(T)
    else
        x
    end
end

function to_chebyshev(TL::SemiInfChebyshev, y, ::Val{0})
    @unpack A, L = TL
    z = y - A
    x = (z - L) / (z + L)
    semiinf_chebyshev_endpoints(y, L, x)
end

function to_chebyshev(TL::SemiInfChebyshev, y, ::Val{1})
    @unpack A, L = TL
    2 * L / abs2(y - A + L)
end

function to_chebyshev(TL::SemiInfChebyshev, y, ::Val{0:1})
    @unpack A, L = TL
    z = y - A
    num = z - L
    den = z + L
    semiinf_chebyshev_endpoints(y, L, num/den), 2 * L / abs2(den)
end

end # module
