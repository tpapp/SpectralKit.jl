#####
##### utility functions for tests
#####

"""
$(SIGNATURES)

Tests for function families.
"""
function test_is_function_family(family::F) where F
    @test is_function_family(family)
    @test is_function_family(F)
end

"""
$(SIGNATURES)

Test the `N` roots of `family`.
"""
function test_roots(family, N; atol = 1e-13)
    r = roots(family, N)
    @test r isa Vector{Float64}
    @test length(r) == N
    @test all(abs.(basis_function.(family, N + 1, r, Order(0))) .≤ atol)
end

"""
$(SIGNATURES)

Test the `N` augmented extrema of `family`.
"""
function test_augmented_extrema(family, N; atol = 1e-13)
    a = augmented_extrema(family, N)
    @test a isa Vector{Float64}
    @test length(a) == N
    @test all(abs.(basis_function.(family, N, a[2:(end-1)], Order(1))) .≤ atol)
    mi, ma = domain_extrema(family)
    â = sort(a)                 # non-increasing transformations switch signs
    @test mi ≈ â[1] atol = 1e-16
    @test ma ≈ â[end] atol = 1e-16
end

"""
$(SIGNATURES)

Test continuity of `family` at endpoints using a heuristic.

`expected_extrema` is the expected domain extrema. The function `k ∈ ks` is basis_functiond at the
extrema and tested for continuity:

1. finite extrema are adjusted by `eps_step`,
2. infinite extrema are proxied for by a signed `inf_proxy`.
"""
function test_endpoint_continuity(family, expected_extrema, ks;
                                  eps_step = eps(), inf_proxy = 1e6, atol = eps()^0.25)
    mi, ma = domain_extrema(family)
    @test mi == expected_extrema[1]
    @test ma == expected_extrema[2]
    for k in ks
        ev(x, order) = basis_function(family, k, x, order)

        # at domain minimum
        f_mi, f′_mi = @inferred ev(mi, OrdersTo(1))
        mi_approx = isfinite(mi) ? mi + eps_step : -inf_proxy
        @test f_mi isa Float64
        @test f′_mi isa Float64
        @test f_mi ≈ ev(mi, Order(0))
        @test f_mi ≈ ev(mi_approx, Order(0)) atol = atol
        @test f′_mi ≈ ev(mi, Order(1))
        @test f′_mi ≈ ev(mi_approx, Order(1)) atol = atol

        # at domain maximum
        f_ma, f′_ma = @inferred ev(ma, OrdersTo(1))
        ma_approx = isfinite(ma) ? ma - eps_step : inf_proxy
        @test f_ma isa Float64
        @test f′_ma isa Float64
        @test f_ma ≈ ev(ma, Order(0))
        @test f_ma ≈ ev(ma_approx, Order(0)) atol = atol
        @test f′_ma ≈ ev(ma, Order(1))
        @test f′_ma ≈ ev(ma_approx, Order(1)) atol = atol
    end
end

"""
$(SIGNATURES)

Return a closure that generates a random scalar in the domain.
"""
function random_generator_in_domain(family)
    mi, ma = domain_extrema(family)
    if isfinite(mi) && isfinite(ma) && mi ≤ ma
        Δ = ma - mi
        () -> rand() * Δ + mi
    elseif isfinite(mi) && ma == Inf
        () -> abs2(randn()) + mi
    elseif isfinite(ma) && mi == -Inf
        () -> ma - abs2(randn())
    elseif mi == -Inf && ma == Inf
        () -> randn()
    else
        error("Don't know how to generate random values in [$(mi),$(ma)]")
    end
end

"""
$(SIGNATURES)

Test derivatives of the `k ∈ ks` polynomials in `family` using `M` random points.
"""
function test_derivatives(family, ks; M = 100)
    generator = random_generator_in_domain(family)
    for k in ks
        for _ in 1:M
            x = generator()
            f, f′ = @inferred basis_function(family, k, x, OrdersTo(1))
            @test f ≈ @inferred basis_function(family, k, x, Order(0))
            @test f′ ≈ @inferred basis_function(family, k, x, Order(1))
            @test f′ ≈ ForwardDiff.derivative(x -> basis_function(family, k, x, Order(0)), x)
        end
    end
end

"""
$(SIGNATURES)

Test linear combinations of `family` for `order`, using `K ∈ Ks` random coefficients.
"""
function test_linear_combinations(family, order; Ks = 2:10, Mθ = 100, Mx = 10)
    generator = random_generator_in_domain(family)
    for _ in 1:Mθ
        K = rand(Ks)
        θ = randn(K)
        for _ in 1:Mx
            x = generator()
            v = sum([basis_function(family, k, x, order) * θ for (k, θ) in enumerate(θ)])
            @test v ≈ linear_combination(family, θ, x, order)
        end
    end
end

"""
$(SIGNATURES)

Test multiple basis functions evaluated at once, with the given `Ks`, and `Mx` random `x`s.
"""
function test_basis_many(family, order; Ks = 0:10, Mx = 100)
    generator = random_generator_in_domain(family)
    for K in Ks
        for _ in Mx
            x = generator()
            b1 = basis_function.(family, 1:K, x, order)
            b2 = @inferred basis_function(family, Val(K), x, order)
            @test b2 isa SVector{K}
            @test b1 ≈ Vector(b2)   # convert for comparison
            b3 = collect(Iterators.take(basis_iterator(family, x, order), K))
            @test b3 ≈ b1
        end
    end
end
