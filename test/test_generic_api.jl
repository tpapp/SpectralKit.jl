@testset "generic API sanity checks" begin
    @test !is_function_basis("a fish")
    @test_throws MethodError domain_kind("a fish")
end

@testset "collocation matrix with default grid" begin
    basis = Chebyshev(EndpointGrid(), 10)   # basis for approximation
    @test collocation_matrix(basis) == collocation_matrix(basis, collect(grid(basis)))
end

@testset "non-square collocation matrix" begin
    f = exp                                 # function for comparison
    basis = Chebyshev(EndpointGrid(), 10)   # basis for approximation
    g = grid(Chebyshev(EndpointGrid(), 20))
    iterator_sanity_checks(g)
    x = @inferred collect(g) # denser grid for approximation
    C = collocation_matrix(basis, x)
    @test all(isfinite, C)
    @test size(C) == (20, 10)
    φ = C \ f.(x)               # coefficients
    y = range(-1, 1; length = 50) # test grid
    @test maximum(abs, f.(y) .- linear_combination.(basis, Ref(φ), y)) ≤ 1e-9
end

@testset "length checks" begin
    basis = Chebyshev(InteriorGrid(), 5)
    bad_θ = zeros(dimension(basis) + 1)
    @test_throws ArgumentError linear_combination(basis, bad_θ, 0.0)
    @test_throws ArgumentError linear_combination(basis, bad_θ)
end

@testset "transformed bases and linear combinations (univariate)" begin
    N = 10
    basis = Chebyshev(EndpointGrid(), N)
    t = BoundedLinear(1.0, 2.0)
    @test domain(basis ∘ t) == domain(t)
    @test dimension(basis ∘ t) == dimension(basis)
    @test parent(basis ∘ t) ≡ basis
    @test transformation(basis ∘ t) ≡ t
    @test collect(grid(basis ∘ t)) ==
        [transform_from(domain(basis), t, x) for x in grid(basis)]

    θ = randn(10)
    l1 = linear_combination(basis, θ)
    l2 = linear_combination(basis ∘ t, θ)
    l3 = linear_combination(basis, θ) ∘ t
    for _ in 1:20
        x = rand() + 1.0
        @test l1(transform_to(domain(basis), t, x)) == l2(x) == l3(x)
    end
end

@testset "transform_to and transform_from univariate shortcuts" begin
    basis = Chebyshev(EndpointGrid(), 5)
    t = BoundedLinear(1.0, 2.0)
    y = rand_in_domain(basis)
    x = transform_to(domain(basis), t, y)
    @test transform_to(basis, t, y) == x
    @test transform_from(basis, t, x) == transform_from(domain(basis), t, x)
end

@testset "transformed bases and linear combinations (bivariate)" begin
    basis0 = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 2), Val(2))
    t = coordinate_transformations(BoundedLinear(1.0, 2.0), SemiInfRational(0, 1))
    basis = basis0 ∘ t
    @test domain(basis ∘ t) == domain(t)
    @test dimension(basis ∘ t) == dimension(basis)
    @test collect(grid(basis)) ==
        [transform_from(domain(basis0), t, x) for x in grid(basis0)]
    @test length(basis) == 2
    @test basis[2] == basis0.univariate_parent ∘ Tuple(t)[2]
    @test_throws BoundsError basis[3]

    θ = randn(dimension(basis))
    l1 = linear_combination(basis0, θ)
    l2 = linear_combination(basis, θ)
    l3 = linear_combination(basis0, θ) ∘ t
    for _ in 1:20
        x = rand(2) .+ 1.0
        @test l1(transform_to(domain(basis0), t, x)) == l2(x) == l3(x)
    end
end

@testset "transform_to and transform_from bivariate shortcuts" begin
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 2), Val(2))
    t = coordinate_transformations(BoundedLinear(1.0, 2.0), SemiInfRational(0, 1))
    for _ in 1:100
        y = rand_in_domain(basis ∘ t)
        x = transform_to(domain(basis), t, y)
        @test transform_to(basis, t, y) == x
        @test transform_from(basis, t, x) == transform_from(domain(basis), t, x)
    end
end

@testset "linear combination SVector passthrough" begin
    N = 10
    M = 3
    basis = Chebyshev(InteriorGrid(), N) ∘ SemiInfRational(0.0, 1.0)
    θ = rand(SVector{M,Float64}, N)
    x = 2.0
    @test linear_combination(basis, θ, x) ≡ SVector{M}(linear_combination(basis, map(x -> x[i], θ), x) for i in 1:M)
end

@testset "subset fallback" begin
    @test !is_subset_basis(Chebyshev(InteriorGrid(), 4), # just test the fallback method
                           smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 2), 2))
end
