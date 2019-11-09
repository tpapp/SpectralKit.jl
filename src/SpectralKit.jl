module SpectralKit

export Chebyshev, SemiInfChebyshev, roots, augmented_extrema, evaluate, domain_extrema

using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF

####
#### general
####

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

Broadcast.broadcastable(f::FunctionFamily) = Ref(f)

"""
`$(FUNCTIONNAME)(F::FunctionFamily, x, order)`

Internal function for implementing conversion of `x` to radians according to the family `F`.

`order` follows the same semantics as [`evaluate`](@ref).
"""
function to_radian end

"""
`$(FUNCTIONNAME)(F::FunctionFamily, θ)

Internal function for implementing conversion from half-turn units (multiples of `π`) to the
domain of the family `F`.
"""
function from_halfturn end

####
#### Chebyshev on [-1,1]
####

"""
$(TYPEDEF)

Chebyhev polynomials of the first kind.
"""
struct Chebyshev <: FunctionFamily end

@inline domain_extrema(::Chebyshev) = (-1.0, 1.0)

_flip_if(condition, x) = condition ? -x : x

_evaluate_min(::Chebyshev, x::T, n, ::Val{1}) where {T} = _flip_if(iseven(n), T(abs2(n)))

_evaluate_max(::Chebyshev, x::T, n, ::Val{1}) where {T} = T(abs2(n))

to_radian(::Chebyshev, x, ::Val{0}) = acos(x)

to_radian(family, x, ::Val{1}) = last(to_radian(family, x, Val{0:1}))

function to_radian(::Chebyshev, x, ::Val{0:1})
    t = acos(x)
    t′ = 1 / sin(t)
    t, t′
end

from_halfturn(::Chebyshev, θ) = cospi(θ)

####
#### Rational Chebyshev on [0,Inf)
####

struct SemiInfChebyshev{T} <: FunctionFamily
    L::T
end

to_radian(TL::SemiInfChebyshev, y, ::Val{0}) = 2 * acot(√(y / TL.L))

function to_radian(TL::SemiInfChebyshev, y, ::Val{0:1})
    ryL = √(y / L)
    t = 2 * acot(ryL)
    t′ = -1 / (ryL * (y + L))
    t, t′
end

from_halfturn(TL::SemiInfChebyshev, θ) = TL.L * abs2(cot(θ * π / 2))

####
#### generic API and its implementation
####

"""
$(SIGNATURES)

Helper function for [`roots`](@ref), returning the location of `N` roots as angles, in
halfturn units.

Implementation ensures that mathematical zeros are numerically `0`. `T` determines the
resulting float type.

Internal, not exported.
"""
function halfturn_roots(::Type{T}, N::Integer) where {T <: Real}
    ((2 * N - 1):-2:1) ./ T(2 * N)
end

"""
$(SIGNATURES)

Helper function for [`augmented_extrema`](@ref), returning the location of `N` augmented
extrema as angles, in halfturn units.

Implementation ensures that mathematical zeros are numerically `0`. `T` determines the
resulting float type.

Internal, not exported.
"""
function halfturn_augmented_extrema(::Type{T}, N::Integer) where {T <: Real}
    ((N-1):-1:0) ./ T(N - 1)
end

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the roots of the `N`th function in `family`, as a vector of `N` numbers with element
type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Chebyshev” grid.
"""
roots(T, family, N) = from_halfturn.(Ref(family), halfturn_roots(T, N))

roots(family, N::Integer) = roots(Float64, family, N)

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the augmented extrema (extrema + boundary values, in increasing order) of the `N-1`th
function in `family`, as a vector of `N` numbers with element type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Lobatto” grid.
"""
function augmented_extrema(::Type{T}, family::FunctionFamily, N) where {T}
    from_halfturn.(Ref(family), halfturn_augmented_extrema(T, N))
end

function augmented_extrema(family::FunctionFamily, N::Integer)
    augmented_extrema(Float64, family, N)
end

"""
$(SIGNATURES)

Evaluate the `n` (starting from 0) function in `family` at `x`.

`order` determines the derivatives:

- `Val(d)` returns the `d`th derivatives, starting from `0` (for the function value)

- `Val(0:d)` returns the derivatives up to `d`, starting from the function value, as
  multiple values (ie a tuple).
"""
evaluate(family::FunctionFamily, x, n, order::Val{0}) = cos(n * to_radian(family, x, order))

function evaluate(family::FunctionFamily, x, n, order::Val{1})
    mi, ma = domain_extrema(family)
    if x == mi
        _evaluate_min(family, x, n, Val(1))
    elseif x == ma
        _evaluate_max(family, x, n, Val(1))
    else
        t, t′ = to_radian(family, x, Val(0:1))
        n * sin(n * t) * t′
    end
end

function evaluate(family::FunctionFamily, x, n, order::Val{0:1})
    mi, ma = domain_extrema(family)
    if x == mi
        evaluate(family, x, n, Val(0)), _evaluate_min(family, x, n, Val(1))
    elseif x == ma
        evaluate(family, x, n, Val(0)), _evaluate_max(family, x, n, Val(1))
    else
        t, t′ = to_radian(family, x, Val(0:1))
        s, c = sincos(n * t)
        c, s * n * t′
    end
end

end # module
