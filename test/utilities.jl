#####
##### utility functions for tests
#####

"kinds we test on"
const KINDS = [Endpoints(), Interior()]

# FIXME reenable what is needed
# using SpectralKit: TransformedBasis, SmolyakBasis, SmolyakIndices # dispatch for rand_in_domain

chebyshev_cos(x, n) = cos((n - 1) * acos(x))

"""
$(SIGNATURES)

Derivative of the `n`th Chebyshev polynomial at `x`, using the cosine formula (special
cased at ±1).
"""
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

Test if `x` is an extrema of the `n`th Chebyshev polynomial, by checking derivatives to
be within tolerance (±1 special cased).
"""
function is_chebyshev_extrema(x, n; tol = 1e-8)
    if abs(abs(x) - 1) ≤ tol
        true
    else
        abs(chebyshev_cos_deriv(x, n)) ≤ tol
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
function rand_in_domain(basis::SpectralKit.UnivariateBasis{Chebyshev})
    transform_from(PM1(), basis.domain_transformation, rand_pm1())
end

# function rand_in_domain(basis::SmolyakBasis{<:SmolyakIndices{N}}) where N
#     (; univariate_parent) = basis
#     SVector(ntuple(_ -> rand_in_domain(univariate_parent), Val(N)))
# end

# FIXME remove
# function rand_in_domain(basis::TransformedBasis)
#     (; parent, transformation) = basis
#     transform_from(parent, transformation, rand_in_domain(parent))
# end

"""
$(SIGNATURES)

Flags (`true`) for elements in `a` that are within `atol` of some element in `b`.
"""
function is_approximately_in(a, b; atol = √eps())
    _same(a::Real, b::Real) = a == b || abs(a - b) ≤ atol # Inf = Inf, etc
    _same(a::AbstractVector, b::AbstractVector) = all(_same.(a, b))
    _same(a::Tuple, b::Tuple) = mapreduce((x, y) -> abs(x - y), max, a, b; init = 0.0) ≤ atol
    map(a -> any(b -> _same(a, b), b), a)
end

"""
$(SIGNATURES)

Are elements in `a` in `b`, approximately.
"""
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

"""
$(SIGNATURES)

A vector of Smolyak indices traversed in column-major order. For testing.
"""
function naive_smolyak_indices(family, kind, ::Val{N}, total, each) where N
    # map indices to levels
    I = SpectralKit.grid_length(family, kind, total)
    index_to_level = zeros(I)
    i = 0
    for level in 0:total
        b = SpectralKit.block_length(family, kind, level)
        index_to_level[(i+1):(i+b)] .= level
        i += b
    end
    # collect indices
    indices = Vector{NTuple{N,Int}}()
    for ι in Iterators.product(ntuple(_ -> 1:I, Val(N))...)
        levels = map(i -> index_to_level[i], ι)
        if all(levels .≤ each) && sum(levels) ≤ total
            push!(indices, ι)
        end
    end
    indices
end

"""
$(SIGNATURES)

Get the `n`th element from an infinite iterator. For unit testing.
"""
function _nth(infinite_itr, i)
    x, state = SpectralKit._start(infinite_itr)
    for _ in 2:i
        x, state = SpectralKit._next(infinite_itr, state)
    end
    x
end

"""
$(SIGNATURES)

Test the Smolyak iterator implementation building blocks directly using a parallel,
naive calculation and checking invariants.
"""
function test_smolyak_step(family, kind, total, each, f, itrs::NTuple{N,Any}) where N
    i = 1
    reference = naive_smolyak_indices(family, kind, Val(N), total, each)
    (accum, slack, remainders, states, levels) = SpectralKit.__smolyak_init(family, kind,
                                                                            total, f, itrs)
    while true
        @test slack + sum(levels) == total # simple sanity check
        cs = map(_nth, itrs, reference[i])
        @test accum == foldr(f, cs; init = ())
        next = SpectralKit.__smolyak_step(family, kind, each, f, itrs,
                                          accum, slack, remainders, states, levels)
        (accum, Δ, remainders, states, levels) = next
        Δ == SpectralKit.Δ_DONE && break
        slack += Δ
        i += 1
    end
    @test i == length(reference) # we used up all elements
    nothing
end

"""
$(SIGNATURES)

Implementation of non-increasing Smolyak levels, for unit testing.
"""
function nonincreasing_smolyak_levels(::Val{N}, total, each)  where N
    I = 0:each
    c = Vector{NTuple{N,Int}}()
    for i in Iterators.product(ntuple(_ -> I, Val(N))...)
        if sum(i) ≤ total && issorted(i, rev = true)
            push!(c, i)
        end
    end
    c
end
