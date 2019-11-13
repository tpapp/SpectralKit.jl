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
    @test r[(N + 1) ÷ 2] == 0   # precise 0
    @test all(abs.(evaluate.(family, r, N, Val(0))) .≤ atol)
end

function test_augmented_extrema(family, N; atol = 1e-13)
    a = augmented_extrema(family, N)
    @test a isa Vector{Float64}
    @test length(a) == N
    @test a[(N + 1) ÷ 2] == 0 # precise 0
    @test all(abs.(evaluate.(family, a[2:(end-1)], N - 1, Val(1))) .≤ 1e-13)
    mi, ma = domain_extrema(family)
    @test mi == a[1]
    @test ma == a[end]
end

function test_endpoint_continuity(family, expected_extrema, ns;
                                  eps_step = eps(), atol = eps()^0.25)
    mi, ma = domain_extrema(family)
    @test mi == expected_extrema[1]
    @test ma == expected_extrema[2]
    for n in ns
        # at domain minimum
        f_mi, f′_mi = @inferred evaluate(family, mi, n, Val(0:1))
        @test f_mi isa Float64
        @test f′_mi isa Float64
        @test f_mi ≈ evaluate(family, mi, n, Val(0))
        @test f_mi ≈ evaluate(family, mi + eps(), n, Val(0)) atol = atol
        @test f′_mi ≈ evaluate(family, mi, n, Val(1))
        @test f′_mi ≈ evaluate(family, mi + eps(), n, Val(1)) atol = atol

        # at domain maximum
        f_ma, f′_ma = @inferred evaluate(family, ma, n, Val(0:1))
        @test f_ma isa Float64
        @test f′_ma isa Float64
        @test f_ma ≈ evaluate(family, ma, n, Val(0))
        @test f_ma ≈ evaluate(family, ma - eps(), n, Val(0)) atol = atol
        @test f′_ma ≈ evaluate(family, ma, n, Val(1))
        @test f′_ma ≈ evaluate(family, ma - eps(), n, Val(1)) atol = atol
    end
end

function test_derivatives(family, generator, ns; M = 100)
    for n in ns
        for _ in 1:M
            x = generator()
            f, f′ = @inferred evaluate(family, x, n, Val(0:1))
            @test f ≈ @inferred evaluate(family, x, n, Val(0))
            @test f′ ≈ @inferred evaluate(family, x, n, Val(1))
            @test f′ ≈ ForwardDiff.derivative(x -> evaluate(family, x, n, Val(0)), x)
        end
    end
end

####
#### tests
####

@testset "Chebyshev" begin
    F = Chebyshev()
    test_roots(F, 11)
    test_augmented_extrema(F, 11)
    test_endpoint_continuity(F, (-1, 1), 0:10)
    test_derivatives(F, () -> rand() * 2 - 1, 0:10)
end
