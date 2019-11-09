using SpectralKit
using Test
import ForwardDiff

@testset "Chebyshev" begin
    F = Chebyshev()

    @testset "roots" begin
        r = roots(F, 11)
        @test r isa Vector{Float64}
        @test length(r) == 11
        @test r[6] == 0                 # precise 0
        @test all(abs.(evaluate.(F, r, 11, Val(0))) .≤ 1e-14)
    end

    @testset "augmented extrema" begin
        a = augmented_extrema(F, 11)
        @test a isa Vector{Float64}
        @test length(a) == 11
        @test a[6] == 0                 # precise 0
        @test all(abs.(evaluate.(F, a[2:(end-1)], 10, Val(1))) .≤ 1e-13)
    end

    @testset "endpoints" begin
        mi, ma = domain_extrema(F)
        @test mi == -1
        @test ma == 1
        for n in 0:10
            atol = eps()^0.25           # very crude continuity test

            # at domain minimum
            f_mi, f′_mi = @inferred evaluate(F, mi, n, Val(0:1))
            @test f_mi isa Float64
            @test f′_mi isa Float64
            @test f_mi ≈ evaluate(F, mi, n, Val(0))
            @test f_mi ≈ evaluate(F, mi + eps(), n, Val(0)) atol = atol
            @test f′_mi ≈ evaluate(F, mi, n, Val(1))
            @test f′_mi ≈ evaluate(F, mi + eps(), n, Val(1)) atol = atol

            # at domain maximum
            f_ma, f′_ma = @inferred evaluate(F, ma, n, Val(0:1))
            @test f_ma isa Float64
            @test f′_ma isa Float64
            @test f_ma ≈ evaluate(F, ma, n, Val(0))
            @test f_ma ≈ evaluate(F, ma - eps(), n, Val(0)) atol = atol
            @test f′_ma ≈ evaluate(F, ma, n, Val(1))
            @test f′_ma ≈ evaluate(F, ma - eps(), n, Val(1)) atol = atol
        end
    end

    @testset "random evaluations and AD tests" begin
        for _ in 1:10
            x = rand() * 2 - 1
            n = rand(0:10)
            f, f′ = @inferred evaluate(F, x, n, Val(0:1))
            @test f ≈ @inferred evaluate(F, x, n, Val(0))
            @test f′ ≈ @inferred evaluate(F, x, n, Val(1))
            @test f′ ≈ ForwardDiff.derivative(x -> evaluate(F, x, n, Val(0)), x)
        end
    end
end
