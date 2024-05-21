#####
##### utility functions for tests
#####

struct KroneckerVector{T} <: AbstractVector{T}
    i::Int
    len::Int
end

Base.size(v::KroneckerVector) = (v.len, )

@inline function Base.getindex(v::KroneckerVector{T}, i::Int) where {T}
    @boundscheck 1 ≤ i ≤ v.len
    i == v.i ? one(T) : zero(T)
end

chebyshev_cos(x, n) = cos((n - 1) * acos(x))

function chebyshev_cos_deriv(x, n)
    z = cos(zero(x)) * abs2(n - 1)
    if x == -1
        isodd(n) ? -z : z
    elseif x == 1
        z
    else
        t = acos(x)
        (n - 1) * sin((n - 1) * t) / sin(t)
    end
end

"""
$(SIGNATURES)

Return a random scalar in `(-1.0, 1.0)` (the default), except when `i == 1`, return `-1.0`,
when `i == 2`, return `1.0`.

The intention is to provide comprehensive testing for endpoints.
"""
function rand_pm1(i = 3)
    i == 1 && return -1.0
    i == 2 && return 1.0
    rand() * 2 - 1.0
end

"Flags (`true`) for elements in `a` that are within `atol` of some element in `b`."
function is_approximately_in(a, b; atol = √eps())
    _same(a::Real, b::Real) = a == b || abs(a - b) ≤ atol # Inf = Inf, etc
    _same(a::AbstractVector, b::AbstractVector) = all(_same.(a, b))
    _same(a::Tuple, b::Tuple) = mapreduce((x, y) -> abs(x - y), max, a, b) ≤ atol
    map(a -> any(b -> _same(a, b), b), a)
end

"Are elements in `a` in `b`, approximately."
function is_approximate_subset(a, b; atol = √eps())
    sum(is_approximately_in(a, b; atol = atol)) == length(a)
end

"""
$(SIGNATURES)

A vector of coefficients compatible with `basis`, with zeros, except for `i` where it is 1.
"""
function e_i(basis, i)
    θ = zeros(dimension(basis))
    θ[i] = 1.0
    θ
end

"""
Some sanity checks for iterators.
"""
function iterator_sanity_checks(itr)
    T = eltype(typeof(itr))
    @test eltype(itr) ≡ T
    @test all(x -> typeof(x) ≡ T, itr)
    @test count(_ -> true, itr) == length(itr)
end

"nth derivative of f at x."
function DD(f, x, n = 1; p = 10)
    if n == 0
        f(x)
    else
        central_fdm(p, n)(f, x)
    end
end
