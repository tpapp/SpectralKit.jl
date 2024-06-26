####
#### Chebyshev on [-1,1]
####

@testset "Chebyshev" begin
    @test_throws ArgumentError Chebyshev(InteriorGrid(), 0)
    @test_throws ArgumentError Chebyshev(EndpointGrid(), 0)
    @test_throws TypeError Chebyshev(:invalid_grid, 10)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        for N in (grid_kind ≡ InteriorGrid() ? 1 : 2):10

            basis = Chebyshev(grid_kind, N)
            @test is_function_basis(basis)
            @test is_function_basis(typeof(basis))
            @test dimension(basis) == N

            # check linear combinations
            for _ in 1:100
                x = rand_in_domain(basis)
                bx = @inferred basis_at(basis, x)

                @test length(bx) == N
                @test eltype(bx) == Float64
                @test collect(bx) ≈ [chebyshev_cos(x, i) for i in 1:N]

                θ = rand(N)
                @test linear_combination(basis, θ, x) ≈
                    sum(chebyshev_cos(x, i) * θ for (i,θ) in enumerate(θ))
                @test linear_combination(basis, θ, 𝑑(x))[1] ≈
                    sum(chebyshev_cos_deriv(x, i) * θ for (i,θ) in enumerate(θ))
            end

            # check grid
            g = @inferred collect(grid(basis))
            @test length(g) == N
            if grid_kind ≡ InteriorGrid()
                @test all(x -> isapprox(chebyshev_cos(x, N + 1), 0, atol = 1e-14), g)
            else
                @test all(x -> isapprox(chebyshev_cos_deriv(x, N), 0, atol = 1e-13),
                          g[2:(end-1)])
                @test g[1] == -1
                @test g[end] == 1
            end

            # augmented coefficients
            for _ in 1:100
                x = rand_in_domain(basis)
                θ = rand(N)
                destination_basis = Chebyshev(grid_kind, N + 5)
                destination_θ = augment_coefficients(basis, destination_basis, θ)
                @test linear_combination(basis, θ, x) ≈
                    linear_combination(destination_basis, destination_θ, x)
            end

        end
    end

    # compatible and incompatible grids
    @testset "augment Chebyshev coefficients — errors" begin
        basis = Chebyshev(InteriorGrid(), 5)
        θ = randn(5)
        # different grids are compatible
        basis2_G = Chebyshev(EndpointGrid(), 6)
        @test is_subset_basis(basis, basis2_G)
        # fewer dimensions are not compatible
        basis2_N = Chebyshev(InteriorGrid(), 4)
        @test !is_subset_basis(basis, basis2_N)
        @test_throws ArgumentError augment_coefficients(basis, basis2_N, θ)
        # too few coefficients
        @test_throws ArgumentError augment_coefficients(basis, basis, randn(4))
    end
end

@testset "augmentation of transformed basis" begin
    N = 5
    M = N + 3
    t = SemiInfRational(0.3, 0.9)
    grid_kind = InteriorGrid()
    basis =  Chebyshev(grid_kind, N) ∘ t
    basis′ =  Chebyshev(grid_kind, M) ∘ t
    @test is_subset_basis(basis, basis′)
    for _ in 1:100
        x = rand_in_domain(basis)
        θ = rand(N)
        θ′ = augment_coefficients(basis, basis′, θ)
        @test linear_combination(basis, θ, x) ≈ linear_combination(basis′, θ′, x)
    end
end

@testset "univariate derivatives" begin
    basis = Chebyshev(InteriorGrid(), 5)
    for (transformation, N) in ((BoundedLinear(-2, 3), 5),
                                (SemiInfRational(0.7, 0.3), 1),
                                (InfRational(0.4, 0.9), 1))
        D = 𝑑^Val(N)
        transformed_basis = basis ∘ transformation
        f = linear_combination(transformed_basis, randn(dimension(transformed_basis)))
        for _ in 1:50
            x = transform_from(basis, transformation, rand_in_domain(basis))
            y = f(D(x))
            for i in 0:N
                @test y[i] ≈ DD(f, x, i) atol = 1e-6
            end
        end
    end
end

@testset "endpoint continuity for derivatives" begin
    N = 10
    basis = Chebyshev(InteriorGrid(), N)

    # NOTE here we are checking that in some sense, derivatives give the right limit at
    # endpoints for transformations to ∞. We use the analytical derivatives for
    # comparison, based on the chain rule.
    x_pinf = 𝑑(Inf)
    x_minf = 𝑑(-Inf)

    @testset "SemiInfRational endpoints continuity" begin
        trans = SemiInfRational(2.3, 0.7)

        for i in 1:N
            θ = e_i(basis ∘ trans, i)
            y_pinf = @inferred linear_combination(basis ∘ trans, θ, x_pinf)
            @test y_pinf[0] == 1
            @test y_pinf[1] == 0
            y_minf = @inferred linear_combination(basis ∘ trans, θ, x_minf)
            @test y_minf[0] == 1
            @test y_minf[1] == 0
        end
    end

    @testset "InfRational endpoints continuity" begin
        trans = InfRational(0.3, 3.0)

        for i in 1:N
            θ = e_i(basis ∘ trans, i)
            y_pinf = @inferred linear_combination(basis ∘ trans, θ, x_pinf)
            @test y_pinf[0] == 1
            @test y_pinf[1] == 0
            y_minf = @inferred linear_combination(basis ∘ trans, θ, x_minf)
            @test y_minf[0] == (-1)^(i+1)
            @test y_minf[1] == 0
        end
    end
end
