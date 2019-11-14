using SpectralKit
using Test
import ForwardDiff

####
#### utility functions for tests
####

function test_roots(family, N; atol = 1e-14)
    r = roots(family, N)
    @test r isa Vector{Float64}
    @test length(r) == N
    @test all(abs.(evaluate.(family, N, r, Val(0))) .≤ atol)
end

function test_augmented_extrema(family, N; atol = 1e-13)
    a = augmented_extrema(family, N)
    @test a isa Vector{Float64}
    @test length(a) == N
    @test all(abs.(evaluate.(family, N - 1, a[2:(end-1)], Val(1))) .≤ 1e-13)
    mi, ma = domain_extrema(family)
    â = sort(a)                 # non-increasing transformations switch signs
    @test mi ≈ â[1] atol = 1e-16
    @test ma ≈ â[end] atol = 1e-16
end

function test_endpoint_continuity(family, expected_extrema, ns;
                                  eps_step = eps(), inf_proxy = 1e6, atol = eps()^0.25)
    mi, ma = domain_extrema(family)
    @test mi == expected_extrema[1]
    @test ma == expected_extrema[2]
    for n in ns
        ev(x, order) = evaluate(family, n, x, order)
        # at domain minimum
        f_mi, f′_mi = @inferred ev(mi, Val(0:1))
        mi_approx = isfinite(mi) ? mi + eps() : -inf_proxy
        @test f_mi isa Float64
        @test f′_mi isa Float64
        @test f_mi ≈ ev(mi, Val(0))
        @test f_mi ≈ ev(mi_approx, Val(0)) atol = atol
        @test f′_mi ≈ ev(mi, Val(1))
        @test f′_mi ≈ ev(mi_approx, Val(1)) atol = atol

        # at domain maximum
        f_ma, f′_ma = @inferred ev(ma, Val(0:1))
        ma_approx = isfinite(ma) ? ma - eps() : inf_proxy
        @test f_ma isa Float64
        @test f′_ma isa Float64
        @test f_ma ≈ ev(ma, Val(0))
        @test f_ma ≈ ev(ma_approx, Val(0)) atol = atol
        @test f′_ma ≈ ev(ma, Val(1))
        @test f′_ma ≈ ev(ma_approx, Val(1)) atol = atol
    end
end

function test_derivatives(family, generator, ns; M = 100)
    for n in ns
        for _ in 1:M
            x = generator()
            f, f′ = @inferred evaluate(family, n, x, Val(0:1))
            @test f ≈ @inferred evaluate(family, n, x, Val(0))
            @test f′ ≈ @inferred evaluate(family, n, x, Val(1))
            @test f′ ≈ ForwardDiff.derivative(x -> evaluate(family, n, x, Val(0)), x)
        end
    end
end

####
#### tests
####

@testset "Chebyshev" begin
    F = Chebyshev()

    test_roots(F, 11)
    @test roots(F, 11)[6] == 0  # precise 0

    test_augmented_extrema(F, 11)
    @test augmented_extrema(F, 11)[6] == 0 # precise 0

    test_endpoint_continuity(F, (-1, 1), 0:10)

    test_derivatives(F, () -> rand() * 2 - 1, 0:10)
end

@testset "ChebyshevSemiInf" begin
    F = ChebyshevSemiInf(2.0, 4.7)

    test_roots(F, 9)

    test_augmented_extrema(F, 10)

    test_endpoint_continuity(F, (2.0, Inf), 0:10; atol = 1e-3)

    test_derivatives(F, () -> 2.0 + abs2(randn()), 0:10)

    F = ChebyshevSemiInf(3.0, -1.9)

    test_roots(F, 11; atol = 1e-13)

    test_augmented_extrema(F, 7; atol = 1e-10)

    test_endpoint_continuity(F, (-Inf, 3.0), 0:10; atol = 1e-3)

    test_derivatives(F, () -> 3.0 - abs2(randn()), 0:10)

    @test_throws ArgumentError ChebyshevSemiInf(0.0, 0.0)
end

@testset "ChebyshevInf" begin
    F = ChebyshevInf(0.0, 1.0)

    test_roots(F, 11)
    @test roots(F, 11)[6] == 0  # precise 0

    test_augmented_extrema(F, 11)
    @test augmented_extrema(F, 11)[6] == 0 # precise 0

    test_endpoint_continuity(F, (-Inf, Inf), 0:10)

    test_derivatives(F, randn, 0:10)

    @test_throws ArgumentError ChebyshevInf(0.0, -3.0)
    @test_throws ArgumentError ChebyshevInf(0.0, 0.0)
end
