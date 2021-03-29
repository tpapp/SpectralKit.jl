@testset "Chebyshev bounded" begin
    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    A, B = 1, 5
    N = 11
    grid_dense = range(A, B, length = 100)

    basis0 = Chebyshev(N)
    trans = BoundedLinear(A, B)
    basis = univariate_basis(basis0, trans)
    @test dimension(basis) == N
    @test domain(basis) == (A, B)

    @testset "transformation" begin
        for i in 1:100
            x = rand_in_domain(i, A, B)
            @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
        end
    end

    @testset "interior grid" begin
        gi = grid(basis, InteriorGrid())
        @test length(gi) == N
        @test all(A .< gi .< B)
        C = collocation_matrix(basis, gi)
        θ = C \ f.(gi)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-5
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 5e-4
    end

    @testset "endpoints grid" begin
        ge = grid(basis, EndpointGrid())
        @test length(ge) == N
        @test all(A .≤ ge .≤ B)
        C = collocation_matrix(basis, ge)
        θ = C \ f.(ge)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-5
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 5e-4
    end
end

@testset "Chebyshev semi-infinite" begin
    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    grid_dense = range(3.1, 8, length = 100)

    A = 3.0
    L = 4.0
    N = 11
    basis0 = Chebyshev(N)
    trans = SemiInfRational(A, L)
    basis = univariate_basis(basis0, trans)
    @test dimension(basis) == N
    @test domain(basis) == (A, Inf)

    @testset "transformation" begin
        for i in 1:100
            x = rand_in_domain(i, A, Inf)
            @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
        end
    end

    @testset "interior grid" begin
        gi = grid(basis, InteriorGrid())
        @test length(gi) == N
        @test sum(gi .< A + L) == 5
        @test all(A .< gi .< Inf)
        C = collocation_matrix(basis, gi)
        θ = C \ f.(gi)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-8
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 1e-7
    end

    @testset "endpoints grid" begin
        ge = grid(basis, EndpointGrid())
        @test length(ge) == N
        @test sum(ge .< A + L) == 5
        @test all(A .≤ ge .≤ Inf)
        C = collocation_matrix(basis, ge)
        θ = C \ f.(ge)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-8
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 1e-7
    end
end

@testset "Chebyshev infinite" begin
    f(x) = exp(-4*abs2(x))
    f′(x) = -8*x*f(x)
    grid_dense = range(-1, 2, length = 100)

    A = 0.0
    L = 1.0
    N = 20
    basis0 = Chebyshev(N)
    trans = InfRational(A, L)
    basis = univariate_basis(basis0, trans)
    @test dimension(basis) == N
    @test domain(basis) == (-Inf, Inf)

    @testset "transformation" begin
        for i in 1:100
            x = rand_in_domain(i, -Inf, Inf)
            @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
        end
    end

    @testset "interior grid" begin
        gi = grid(basis, InteriorGrid())
        @test length(gi) == N
        @test sum(gi .< A) == 10
        @test all(-Inf .< gi .< Inf)
        C = collocation_matrix(basis, gi)
        θ = C \ f.(gi)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-4
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 2e-3
    end

    @testset "endpoints grid" begin
        ge = grid(basis, EndpointGrid())
        @test length(ge) == N
        @test sum(ge .< A) == 10
        @test all(-Inf .≤ ge .≤ Inf)
        C = collocation_matrix(basis, ge)
        θ = C \ f.(ge)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-4
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 2e-3
    end
end
