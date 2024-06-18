#####
##### composite tests for applications
#####

@testset "Smolyak approximation" begin
    function f(x)               # cross-exp is particularly nasty for Smolyak
        x1, x2, x3, x4 = x
        exp(0.3 * (x1 - 2.0) * (x2 - 1.0)) + exp(-x3 * (x2 + 5)) - 1 / log(x4^2 + 10)
    end
    ct = coordinate_transformations(BoundedLinear(0, 4), BoundedLinear(-2.0, 2.0),
                                    SemiInfRational(1.0, 2.0), InfRational(0, 4))
    basis0 = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(5), 4)
    basis = basis0 ∘ ct
    @test domain(basis) == domain(ct)
    x = grid(basis)
    θ = collocation_matrix(basis, x) \ f.(x)
    for x in x                  # test on collocation grid
        @test linear_combination(basis, θ, x) ≈ f(x) atol = 1e-10
    end
    s = SobolSeq([0, -2, 1, -10], [4, 2, 5, 10])
    skip(s, 1 << 5)
    Δabs, Δrel = mapreduce((a, b) -> max.(a, b), Iterators.take(s, 10^3)) do y
        fy = f(y)
        Δabs = abs(fy - linear_combination(basis, θ, y))
        Δrel = Δabs / (1 + abs(fy))
        Δabs, Δrel
    end
    @test Δabs ≤ 1e-3
    @test Δrel ≤ 5e-4
end

@testset "Smolyak derivatives" begin
    function f(x)
        x1, x2 = x
        7.0 + x1 + 2*x2 + 3*x2*x1
    end
    ct = coordinate_transformations(BoundedLinear(0, 4), BoundedLinear(-5.0, 5.0))
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2), 2)
    x = grid(basis)
    θ = collocation_matrix(basis, x) \ f.(from_pm1.(ct, x))
    F = linear_combination(basis, θ)
    for x in x
        @test F(x) ≈ f(from_pm1(ct, x)) atol = 1e-10
        @test ForwardDiff.gradient(F, x) ≈ ForwardDiff.gradient(x -> f(from_pm1(ct, x)), x)
    end
end
