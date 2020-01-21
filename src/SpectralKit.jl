module SpectralKit

export is_function_family, domain_extrema, roots, augmented_extrema, evaluate, Chebyshev,
    ChebyshevSemiInf, ChebyshevInf, ChebyshevInterval

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


Not part of the API, just used internally for dispatch. See [`is_function_family`](@ref).
"""
abstract type FunctionFamily end

Broadcast.broadcastable(family::FunctionFamily) = Ref(family)

"""
`$(FUNCTIONNAME)(F)`

`$(FUNCTIONNAME)(f::F)`

Test if the argument is a *function family*, supporting the following interface:

- [`domain_extrema`](@ref) for querying the domain,

- [`evaluate`](@ref) for function evaluation,

- [`roots`](@ref) and [`augmented_extrema`](@ref) to obtain collocation points.

Can be used on both types (preferred) and values (for convenience).
"""
is_function_family(::Type{Any}) = false

is_function_family(::Type{<:FunctionFamily}) = true

is_function_family(x) = is_function_family(typeof(x))

"""
`$(FUNCTIONNAME)(family)`

Return the extrema of the domain of the given `family` as a tuple.

Type can be arbitrary, but guaranteed to be the same for both endpoints, and type stable.
"""
function domain_extrema end

"""
`$(FUNCTIONNAME)(family, k, x, order)`

Evaluate the `k`th (starting from 1) function in `family` at `x`.

`order` determines the derivatives:

- `Val(d)` returns the `d`th derivative, starting from `0` (for the function value)

- `Val(0:d)` returns the derivatives up to `d`, starting from the function value, as
   multiple values (ie a tuple).

The implementation is intended to be type stable.

!!! note
    Consequences are undefined for evaluating outside the domain.

## Note about indexing

Most texts index polynomial families with `n = 0, 1, …`. Following the Julia array indexing
convention, this package uses `k = 1, 2, …`. Some code may use `n = k - 1` internally for
easier comparison with well-known formulas.
"""
function evaluate end

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the roots of the `K = N + 1`th function in `family`, as a vector of `N` numbers with
element type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Chebyshev” grid.

Order is monotone, but not guaranteed to be increasing.
"""
roots(family, N::Integer) = roots(Float64, family, N)

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the augmented extrema (extrema + boundary values) of the `N`th function in `family`,
as a vector of `N` numbers with element type `T` (default `Float64`).

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

Evaluate the `k`th Chebyshev polynomial at `-1` for the given orders.
"""
chebyshev_min(::Type{T}, k::Integer, ::Val{0}) where {T} = flip_if(iseven(k), one_like(T))

"""
$(SIGNATURES)

Evaluate the `k`th Chebyshev polynomial at `1` for the given orders.
"""
chebyshev_max(::Type{T}, k::Integer, ::Val{0}) where {T} = one_like(T)

function chebyshev_min(::Type{T}, k::Integer, ::Val{1}) where {T}
    flip_if(isodd(k), one_like(T) * abs2(k - 1))
end

chebyshev_max(::Type{T}, k::Integer, ::Val{1}) where {T} = one_like(T) * abs2(k - 1)

function chebyshev_min(::Type{T}, k::Integer, ::Val{0:1}) where {T}
    chebyshev_min(T, k, Val(0)), chebyshev_min(T, k, Val(1))
end

function chebyshev_max(::Type{T}, k::Integer, ::Val{0:1}) where {T}
    chebyshev_max(T, k, Val(0)), chebyshev_max(T, k, Val(1))
end

"""
$(SIGNATURES)

Evaluate the `k`th Chebyshev polynomial at `-1 < x < 1` for the given orders.
"""
chebyshev_interior(k::Integer, x::Real, ::Val{0}) = cos((k - 1) * acos(x))

function chebyshev_interior(k::Integer, x::Real, ::Val{1})
    n = k - 1
    t = acos(x)
    n * sin(n * t) / sin(t)
end

function chebyshev_interior(k::Integer, x::Real, ::Val{0:1})
    n = k - 1
    t = acos(x)
    t′ = 1 / sin(t)
    s, c = sincos(n * t)
    c, s * n * t′
end

function evaluate(family::Chebyshev, k::Integer, x::T, order) where {T <: Real}
    @argcheck k > 0
    if x == -1
        chebyshev_min(T, k, order)
    elseif x == 1
        chebyshev_max(T, k, order)
    else
        chebyshev_interior(k, x, order)
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

function evaluate(family::TransformedChebyshev, k::Integer, x::Real, order::Val{0})
    evaluate(Chebyshev(), k, to_chebyshev(family, x, order), order)
end

function evaluate(family::TransformedChebyshev, k::Integer, x::Real, ::Val{1})
    x, x′ = to_chebyshev(family, x, Val(0:1))
    t′ = evaluate(Chebyshev(), k, x, Val(1))
    t′ * x′
end

function evaluate(family::TransformedChebyshev, k::Integer, x::Real, ::Val{0:1})
    x, x′ = to_chebyshev(family, x, Val(0:1))
    t, t′ = evaluate(Chebyshev(), k, x, Val(0:1))
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
(when `L < 0`) using ``y = A + L ⋅ (1 + x) / (1 - x)``.
"""
struct ChebyshevSemiInf{T <: Real} <: TransformedChebyshev
    "The finite endpoint `A`."
    A::T
    "Scale factor `L ≠ 0`."
    L::T
    function ChebyshevSemiInf(A::T, L::T) where {T <: Real}
        @argcheck L ≠ 0
        new{T}(A, L)
    end
end

ChebyshevSemiInf(A::Real, L::Real) = ChebyshevSemiInf(promote(A, L)...)

function Base.show(io::IO, TL::ChebyshevSemiInf)
    print(io, "ChebyshevSemiInf($(TL.A), $(TL.L))")
end

function domain_extrema(TL::ChebyshevSemiInf)
    @unpack A, L = TL
    if L > 0
        promote(A, Inf)
    else
        promote(-Inf, A)
    end
end

function from_chebyshev(TL::ChebyshevSemiInf, x)
    TL.A + TL.L * (1 + x) / (1 - x)
end

function chebyshev_semiinf_endpoints(y, L, x::T) where {T}
    if (y == Inf && L > 0) || (y == -Inf && L < 0)
        one_like(T)
    else
        x
    end
end

function to_chebyshev(TL::ChebyshevSemiInf, y, ::Val{0})
    @unpack A, L = TL
    z = y - A
    x = (z - L) / (z + L)
    chebyshev_semiinf_endpoints(y, L, x)
end

function to_chebyshev(TL::ChebyshevSemiInf, y, ::Val{1})
    @unpack A, L = TL
    2 * L / abs2(y - A + L)
end

function to_chebyshev(TL::ChebyshevSemiInf, y, ::Val{0:1})
    @unpack A, L = TL
    z = y - A
    num = z - L
    den = z + L
    chebyshev_semiinf_endpoints(y, L, num/den), 2 * L / abs2(den)
end

####
#### Rational Chebyshev on (-Inf,Inf)
####

"""
$(TYPEDEF)

Chebyshev polynomials transformed to the domain `(-Inf, Inf)`.
(when `L < 0`) using ``y = A + L ⋅ x / √(L^2 + x^2)``.
"""
struct ChebyshevInf{T <: Real} <: TransformedChebyshev
    "The center `A`."
    A::T
    "Scale factor `L > 0`."
    L::T
    function ChebyshevInf(A::T, L::T) where {T <: Real}
        @argcheck L > 0
        new{T}(A, L)
    end
end

ChebyshevInf(A::Real, L::Real) = ChebyshevInf(promote(A, L)...)

function Base.show(io::IO, TB::ChebyshevInf)
    print(io, "ChebyshevInf($(TB.A), $(TB.L))")
end

domain_extrema(TB::ChebyshevInf) = (-Inf, Inf)

function from_chebyshev(TB::ChebyshevInf, x)
    TB.A + TB.L * x / √(1 - abs2(x))
end

function chebyshev_inf_endpoints(y, x::T) where {T}
    if y == Inf
        one_like(T)
    elseif y == -Inf
        -one_like(T)
    else
        x
    end
end

function to_chebyshev(TB::ChebyshevInf, y, ::Val{0})
    @unpack A, L = TB
    z = y - A
    x = z / hypot(L, z)
    chebyshev_inf_endpoints(y, x)
end

function to_chebyshev(TB::ChebyshevInf, y, ::Val{1})
    @unpack A, L = TB
    abs2(L) / hypot(L, z)^3
end

function to_chebyshev(TB::ChebyshevInf, y, ::Val{0:1})
    @unpack A, L = TB
    z = y - A
    den = hypot(L, z)
    x = z / den
    x′ = abs2(L) / den^3
    chebyshev_inf_endpoints(y, x), x′
end

####
#### Chebyshev on a finite interval `[a,b]`
####

"""
$(TYPEDEF)

Chebyshev polynomials transformed to the domain `(a, b)`. using ``y = x ⋅ s + m``.

`m` and `s` are calculated and checked by the constructor; `a < b` is enforced.
"""
struct ChebyshevInterval{T <: Real} <: TransformedChebyshev
    "Start of interval `a`."
    a::T
    "End of interval `b`."
    b::T
    "Midpoint `m`."
    m::T
    "Scale `s`."
    s::T
    function ChebyshevInterval(a::T, b::T) where {T <: Real}
        @argcheck isfinite(a) && isfinite(b)
        s = (b - a) / 2
        m = (a + b) / 2
        @argcheck s > 0 "Need `a < b`."
        new{promote_type(T,typeof(m),typeof(s))}(a, b, m, s)
    end
end

function Base.show(io::IO, TI::ChebyshevInterval)
    print(io, "ChebyshevInterval($(TI.a), $(TI.b))")
end

ChebyshevInterval(A::Real, L::Real) = ChebyshevInterval(promote(A, L)...)

domain_extrema(TI::ChebyshevInterval) = (TI.a, TI.b)

function from_chebyshev(TI::ChebyshevInterval, x)
    @unpack m, s = TI
    x * s + m
end

function to_chebyshev(TI::ChebyshevInterval, y, ::Val{0})
    @unpack m, s = TI
    (y - m) / s
end

to_chebyshev(TI::ChebyshevInterval, y, ::Val{1}) = 1 / TI.s

function to_chebyshev(TI::ChebyshevInterval, y, ::Val{0:1})
    @unpack m, s = TI
    (y - m) / s, 1 / s
end

end # module
