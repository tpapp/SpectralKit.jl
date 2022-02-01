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

Return a random scalar in `(left, right)`. When `i == 1`, return `left`, when `i == 2`,
return `right`, each converted to `Float64` for consistency.
"""
function rand_in_domain(i, left, right)
    i == 1 && return Float64(left)
    i == 2 && return Float64(right)
    if isfinite(left) && isfinite(right) && left ≤ right
        Δ = right - left
        rand() * Δ + left
    elseif isfinite(left) && right == Inf
        abs2(randn()) + left
    elseif isfinite(right) && left == -Inf
        right - abs2(randn())
    elseif left == -Inf && right == Inf
        randn()
    else
        error("Don't know how to generate random values in [$(left),$(ma)]")
    end
end

"Replace elements `[begin:C]` with zero, return the new vector."
function zero_upto(x::SVector{N,T}, C) where {N,T}
    y = MVector(x)
    y[begin:C] .= zero(T)
    SVector(y)
end

"Flags (`true`) for elements in `a` that are within `atol` of some element in `b`."
function is_approximately_in(a, b; atol = √eps())
    _same(a::Real, b::Real) = a == b || abs(a - b) ≤ atol # Inf = Inf, etc
    _same(a::AbstractVector, b::AbstractVector) = all(_same.(a, b))
    map(a -> any(b -> _same(a, b), b), a)
end

"Are elements in `a` in `b`, approximately."
function is_approximate_subset(a, b; atol = √eps())
    sum(is_approximately_in(a, b; atol = atol)) == length(a)
end

# """
# $(SIGNATURES)

# Tests for function families.
# """
# function test_is_function_family(family::F) where F
#     @test is_function_family(family)
#     @test is_function_family(F)
# end

# """
# $(SIGNATURES)

# Test the `N` roots of `family`.
# """
# function test_roots(family, N; atol = 1e-13)
#     r = roots(family, N)
#     @test r isa Vector{Float64}
#     @test length(r) == N
#     @test all(abs.(basis_function.(family, r, Order(0), N + 1)) .≤ atol)
# end

# """
# $(SIGNATURES)

# Test the `N` extrema of `family`.
# """
# function test_extrema(family, N; atol = 1e-13)
#     a = extrema(family, N)
#     @test a isa Vector{Float64}
#     @test length(a) == N
#     @test all(abs.(basis_function.(family, a[2:(end-1)], Order(1), N)) .≤ atol)
#     mi, ma = domain_extrema(family)
#     â = sort(a)                 # non-increasing transformations switch signs
#     @test mi ≈ â[1] atol = 1e-16
#     @test ma ≈ â[end] atol = 1e-16
# end

# """
# $(SIGNATURES)

# Test continuity of `family` at endpoints using a heuristic.

# `expected_extrema` is the expected domain extrema. The function `k ∈ ks` is basis_functiond at the
# extrema and tested for continuity:

# 1. finite extrema are adjusted by `eps_step`,
# 2. infinite extrema are proxied for by a signed `inf_proxy`.
# """
# function test_endpoint_continuity(family, expected_extrema, ks;
#                                   eps_step = eps(), inf_proxy = 1e6, atol = eps()^0.25)
#     mi, ma = domain_extrema(family)
#     @test mi == expected_extrema[1]
#     @test ma == expected_extrema[2]
#     for k in ks
#         ev(x, order) = basis_function(family, x, order, k)

#         # at domain minimum
#         f_mi, f′_mi = @inferred ev(mi, OrdersTo(1))
#         mi_approx = isfinite(mi) ? mi + eps_step : -inf_proxy
#         @test f_mi isa Float64
#         @test f′_mi isa Float64
#         @test f_mi ≈ ev(mi, Order(0))
#         @test f_mi ≈ ev(mi_approx, Order(0)) atol = atol
#         @test f′_mi ≈ ev(mi, Order(1))
#         @test f′_mi ≈ ev(mi_approx, Order(1)) atol = atol

#         # at domain maximum
#         f_ma, f′_ma = @inferred ev(ma, OrdersTo(1))
#         ma_approx = isfinite(ma) ? ma - eps_step : inf_proxy
#         @test f_ma isa Float64
#         @test f′_ma isa Float64
#         @test f_ma ≈ ev(ma, Order(0))
#         @test f_ma ≈ ev(ma_approx, Order(0)) atol = atol
#         @test f′_ma ≈ ev(ma, Order(1))
#         @test f′_ma ≈ ev(ma_approx, Order(1)) atol = atol
#     end
# end


# """
# $(SIGNATURES)

# Test derivatives of the `k ∈ ks` polynomials in `family` using `M` random points.
# """
# function test_derivatives(family, ks; M = 100)
#     generator = random_generator_in_domain(family)
#     for k in ks
#         for _ in 1:M
#             x = generator()
#             f, f′ = @inferred basis_function(family, x, OrdersTo(1), k)
#             @test f ≈ @inferred basis_function(family, x, Order(0), k)
#             @test f′ ≈ @inferred basis_function(family, x, Order(1), k)
#             @test f′ ≈ ForwardDiff.derivative(x -> basis_function(family, x, Order(0), k), x)
#         end
#     end
# end

# """
# $(SIGNATURES)

# Test linear combinations of `family` for `order`, using `K ∈ Ks` random coefficients.
# """
# function test_linear_combinations(family, order; Ks = 2:10, Mθ = 100, Mx = 10)
#     generator = random_generator_in_domain(family)
#     for _ in 1:Mθ
#         K = rand(Ks)
#         θ = randn(K)
#         for _ in 1:Mx
#             x = generator()
#             v = sum([basis_function(family, x, order, k) * θ for (k, θ) in enumerate(θ)])
#             @test v ≈ linear_combination(family, x, order, θ)
#         end
#     end
# end

# """
# $(SIGNATURES)

# Test multiple basis functions evaluated at once, with the given `Ks`, and `Mx` random `x`s.
# """
# function test_basis_many(family, order; Ks = 0:10, Mx = 100)
#     generator = random_generator_in_domain(family)
#     for K in Ks
#         for _ in Mx
#             x = generator()
#             b1 = basis_function.(family, x, order, 1:K)
#             b2 = @inferred basis_function(family, x, order, Val(K))
#             @test b2 isa SVector{K}
#             @test b1 ≈ Vector(b2)   # convert for comparison
#             b3 = collect(Iterators.take(basis_function(family, x, order), K))
#             @test b3 ≈ b1
#         end
#     end
# end
