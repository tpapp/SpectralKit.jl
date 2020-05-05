#####
##### Transformation of the Chebyshev polynomials — generic code
#####

"""
$(TYPEDEF)

Define a function family by transforming from the Chebyshev polynomials on `[-1,1]`.

For `family::TransformedChebyshev`, subtypes implement:

- `from_chebyshev(family, x)`, for transforming from `x ∈ [-1,1]` to `y` of the domain,

- `to_chebyshev(family, y, order)` for transforming `y` the domain to `x ∈ [-1,1]`, where
  `order` follows the semantics of [`basis_function`](@ref) and returns `x`, `∂x/∂y`, … as
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

"""
$(SIGNATURES)

Return the first argument transformed to the domain `[-1,1]`, and a callable that transform
Chebyshev basis functions and derivatives back to the original domain.
"""
function chebyshev_transform_helpers(family, y, order::Order{0})
    to_chebyshev(family, y, order), identity
end

struct TransformOrder1{O,T}
    order::O
    x′::T
end

(τ::TransformOrder1{Order{1}})(t′::Real) = τ.x′ * t′
(τ::TransformOrder1{OrdersTo{1}})(t::SVector{2}) = SVector(t[1], τ.x′ * t[2])

function chebyshev_transform_helpers(family, y, order::Union{Order{1},OrdersTo{1}})
    x, x′ = to_chebyshev(family, y, OrdersTo(1))
    x, TransformOrder1(order, x′)
end

function basis_function(family::TransformedChebyshev, k, y::Real, order)
    x, τ = chebyshev_transform_helpers(family, y, order)
    b = basis_function(Chebyshev(), k, x, order)
    if k isa Int
        τ(b)
    elseif k isa Val
        map(τ, b)
    end
end

function basis_iterator(family::TransformedChebyshev, y::Real, order)
    x, τ = chebyshev_transform_helpers(family, y, order)
    (τ(x) for x in basis_iterator(Chebyshev(), x, order))
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

function to_chebyshev(TL::ChebyshevSemiInf, y, ::Order{0})
    @unpack A, L = TL
    z = y - A
    x = (z - L) / (z + L)
    chebyshev_semiinf_endpoints(y, L, x)
end

function to_chebyshev(TL::ChebyshevSemiInf, y, ::Order{1})
    @unpack A, L = TL
    2 * L / abs2(y - A + L)
end

function to_chebyshev(TL::ChebyshevSemiInf, y, ::OrdersTo{1})
    @unpack A, L = TL
    z = y - A
    num = z - L
    den = z + L
    SVector(chebyshev_semiinf_endpoints(y, L, num/den), 2 * L / abs2(den))
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

function to_chebyshev(TB::ChebyshevInf, y, ::Order{0})
    @unpack A, L = TB
    z = y - A
    x = z / hypot(L, z)
    chebyshev_inf_endpoints(y, x)
end

function to_chebyshev(TB::ChebyshevInf, y, ::Order{1})
    @unpack A, L = TB
    abs2(L) / hypot(L, z)^3
end

function to_chebyshev(TB::ChebyshevInf, y, ::OrdersTo{1})
    @unpack A, L = TB
    z = y - A
    den = hypot(L, z)
    x = z / den
    x′ = abs2(L) / den^3
    SVector(chebyshev_inf_endpoints(y, x), x′)
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

function to_chebyshev(TI::ChebyshevInterval, y, ::Order{0})
    @unpack m, s = TI
    (y - m) / s
end

to_chebyshev(TI::ChebyshevInterval, y, ::Order{1}) = 1 / TI.s

function to_chebyshev(TI::ChebyshevInterval, y, ::OrdersTo{1})
    @unpack m, s = TI
    SVector((y - m) / s, 1 / s)
end
