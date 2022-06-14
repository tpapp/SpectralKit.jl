@testset "generic API sanity checks" begin
    @test !is_function_basis("a fish")
end

@testset "non-square collocation matrix" begin
    f = exp                                 # function for comparison
    basis = Chebyshev(EndpointGrid(), 10)   # basis for approximation
    x = grid(Chebyshev(EndpointGrid(), 20)) # denser grid for approximation
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
