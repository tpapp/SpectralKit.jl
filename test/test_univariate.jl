using SpectralKit: to_domain, from_domain

@testset "Chebyshev bounded" begin
    @test_throws DomainError BoundedLinear(-1.0, Inf)
    @test_throws DomainError BoundedLinear(-1.0, -2.0)

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
    @test_throws DomainError SemiInfRational(-1.0, Inf)
    @test_throws DomainError SemiInfRational(-1.0, 0.0)
    @test_throws DomainError SemiInfRational(NaN, 2.0)

    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    grid_dense = range(3.1, 8, length = 100)

    A = 3.0
    L = 4.0
    N = 11
    trans = SemiInfRational(A, L)

    @testset "example mappings" begin
        basis0 = Chebyshev(InteriorGrid(), 10)
        @test from_domain(trans, basis0, 0) ≈ A + L
        @test from_domain(trans, basis0, -0.5) ≈ A + L / 3
        @test from_domain(trans, basis0, 0.5) ≈ A + 3 * L
    end

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
    @test_throws DomainError InfRational(1.0, Inf)
    @test_throws DomainError InfRational(1.0, 0.0)
    @test_throws DomainError InfRational(1.0, -2.0)
    @test_throws DomainError InfRational(NaN, 1)

    f(x) = exp(-4*abs2(x))
    f′(x) = -8*x*f(x)
    grid_dense = range(-1, 2, length = 100)
    A = 0.0
    L = 1.0
    N = 20
    trans = InfRational(A, L)

    @testset "example mappings" begin
        basis0 = Chebyshev(InteriorGrid(), 10)
        @test from_domain(trans, basis0, 0) ≈ A
        @test from_domain(trans, basis0, -0.5) ≈ A - L / √3
        @test from_domain(trans, basis0, 0.5) ≈ A + L / √3
    end

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

@testset "augmentation" begin
    grids = (InteriorGrid(), EndpointGrid())
    Ns = 4:5
    transformations = (BoundedLinear(2.0, 3.0), BoundedLinear(1.0, 2.0),
                       SemiInfRational(1.0, 3.0), InfRational(4.0, 5.0))
    for (grid1, grid2, N1, N2, t1, t2) in Iterators.product(grids, grids, Ns, Ns,
                                                            transformations, transformations)
        basis1 = univariate_basis(Chebyshev, grid1, N1, t1)
        basis2 = univariate_basis(Chebyshev, grid2, N2, t2)
        θ1 = randn(dimension(basis1))
        x = rand_in_domain(3, domain(basis1)...)
        if grid1 == grid2 && t1 == t2
            if N1 ≤ N2
                θ2 = @inferred augment_coefficients(basis1, basis2, θ1)
                @test is_subset_basis(basis1, basis2)
                @test linear_combination(basis1, θ1, x) ≈ linear_combination(basis2, θ2, x)
            else                # fewer
                @test !is_subset_basis(basis1, basis2)
                @test_throws ArgumentError augment_coefficients(basis1, basis2, θ1)
            end
        else                    # incompatible bases
            @test !is_subset_basis(basis1, basis2)
            @test_throws ArgumentError augment_coefficients(basis1, basis2, θ1)
        end
    end
end
