using SpectralKit: to_domain, from_domain

@testset "Chebyshev bounded" begin
    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    A, B = 1, 5
    N = 11
    grid_dense = range(A, B, length = 100)
    trans = BoundedLinear(A, B)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        basis0 = Chebyshev(grid_kind, N)
        basis = univariate_basis(Chebyshev, grid_kind, N, trans)
        @test dimension(basis) == N
        @test domain(basis) == (A, B)

        @testset "transformation" begin
            for i in 1:100
                x = rand_in_domain(i, A, B)
                @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
            end
        end

        g = @inferred grid(basis)
        @test length(g) == N
        C = collocation_matrix(basis, g)
        θ = C \ f.(g)

        if grid_kind ≡ InteriorGrid()
            @test all(A .< g .< B)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-5
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 5e-4
        else
            @test all(A .≤ g .≤ B)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-5
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 5e-4
        end
    end
end

@testset "Chebyshev semi-infinite" begin
    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    grid_dense = range(3.1, 8, length = 100)

    A = 3.0
    L = 4.0
    N = 11
    trans = SemiInfRational(A, L)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        basis0 = Chebyshev(grid_kind, N)
        basis = univariate_basis(Chebyshev, grid_kind, N, trans)
        @test dimension(basis) == N
        @test domain(basis) == (A, Inf)

        @testset "transformation" begin
            for i in 1:100
                x = rand_in_domain(i, A, Inf)
                @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
            end
        end

        g = grid(basis)
        @test length(g) == N
        C = collocation_matrix(basis, g)
        θ = C \ f.(g)

        if grid_kind ≡ InteriorGrid()
            @test sum(g .< A + L) == 5
            @test all(A .< g .< Inf)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-8
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 1e-7
        else

            @test sum(g .< A + L) == 5
            @test all(A .≤ g .≤ Inf)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-8
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 1e-7
        end
    end
end

@testset "Chebyshev infinite" begin
    f(x) = exp(-4*abs2(x))
    f′(x) = -8*x*f(x)
    grid_dense = range(-1, 2, length = 100)
    A = 0.0
    L = 1.0
    N = 20
    trans = InfRational(A, L)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        basis0 = Chebyshev(grid_kind, N)
        basis = univariate_basis(Chebyshev, grid_kind, N, trans)
        @test dimension(basis) == N
        @test domain(basis) == (-Inf, Inf)

        @testset "transformation" begin
            for i in 1:100
                x = rand_in_domain(i, -Inf, Inf)
                @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
            end
        end

        g = grid(basis)
        @test length(g) == N
        C = collocation_matrix(basis, g)
        θ = C \ f.(g)

        if grid_kind ≡ InteriorGrid()
            @test sum(g .< A) == 10
            @test all(-Inf .< g .< Inf)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-4
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 2e-3
        else
            @test sum(g .< A) == 10
            @test all(-Inf .≤ g .≤ Inf)
            @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)),
                          grid_dense) ≤ 1e-4
            @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                          grid_dense) ≤ 2e-3
        end
    end
end
