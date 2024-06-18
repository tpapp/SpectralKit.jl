#####
##### utility functions for tests
#####

using SpectralKit: TransformedBasis, SmolyakBasis, SmolyakIndices # dispatch for rand_in_domain

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

Return a random value in [-1,1], putting an atomic mass on endpoints.

The intention is to provide comprehensive testing for endpoints.
"""
rand_pm1() = clamp((rand() - 0.5) * 2.5, -1, 1)

"""
$(SIGNATURES)

Return a random value in the domain of the given basis, putting an atomic mass on endpoints.

The intention is to provide comprehensive testing for endpoints.
"""
rand_in_domain(::Chebyshev) = rand_pm1()

function rand_in_domain(basis::SmolyakBasis{<:SmolyakIndices{N}}) where N
    (; univariate_parent) = basis
    SVector(ntuple(_ -> rand_in_domain(univariate_parent), Val(N)))
end

function rand_in_domain(basis::TransformedBasis)
    (; parent, transformation) = basis
    transform_from(parent, transformation, rand_in_domain(parent))
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
