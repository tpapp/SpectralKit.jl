using SpectralKit: ∂Specification
using SpectralKit: _add                           # used to form a scalar for testing
using SpectralKit: Derivatives                    # test internals

@testset "partial derivatives interface" begin
    @test ∂(2, (1, 2), (2, 2)) == ∂Specification{(1, 2)}(((1, 1), (0, 2)))
    @test ∂(3, (), (1, 1, 2), (3, 2)) == ∂Specification{(2, 1, 1)}(((0, 0, 0),  (2, 1, 0), (0, 1, 1)))
    @test_throws ArgumentError ∂(3, (4, ))
    @test_throws ArgumentError ∂(3, (-1, ))
end

# @testset "univariate derivatives check" begin
#     z = 0.6
#     x = derivatives(z, Val(2))
#     @test x[0] == z
#     @test x[1] == 1
#     @test x[2] == 0
#     b = Chebyshev(EndpointGrid(), 4)
#     f(z) = sum(basis_at(b, z))
#     bz = basis_at(b, derivatives(z, Val(2)))
#     iterator_sanity_checks(bz)
#     ∑b = reduce(_add, bz)
#     @test ∑b[0] ≈ f(z)
#     @test ∑b[1] ≈ DD(f, z) atol = 1e-8
#     @test ∑b[2] ≈ DD(f, z, 2) atol = 1e-8
# end

@testset "univariate transformed derivatives" begin
    D = 1                       # derivatives up to this one
    b = Chebyshev(EndpointGrid(), 4)
    θ = randn(dimension(b))
    for t in (BoundedLinear(-2.0, 7.0), )
        for i in 1:100
            z = rand_pm1(i)
            x = transform_from(PM1(), t, z)
            ℓ = linear_combination(b, θ) ∘ t
            Dx = @inferred ℓ(derivatives(x, Val(D)))
            @test Dx[0] == ℓ(x)
            for d in 1:D
                @test DD(ℓ, x, d) ≈ Dx[d] atol = 1e-8
            end
        end
    end
end

@testset "Smolyak derivatives check" begin
    N = 3
    b = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), N)
    t = coordinate_transformations((BoundedLinear(1.0, 2.7),
                                    SemiInfRational(3.0, 0.7),
                                    InfRational(1.7, 0.3)))
    D = ∂(Val(3), (), (1,), (2, ), (3, ), (1, 2))
    D̃ = [(f, x) -> f(x),
         (f, x) -> DD(x1 -> f((x1, x[2], x[3])), x[1]),
         (f, x) -> DD(x2 -> f((x[1], x2, x[3])), x[2]),
         (f, x) -> DD(x3 -> f((x[1], x[2], x3)), x[3]),
         (f, x) -> DD(x1 -> DD(x2 -> f((x1, x2, x[3])), x[2]), x[1])]
    θ = randn(dimension(b))
    ℓ = linear_combination(b, θ) ∘ t
    d = domain(b)
    for i in 1:100
        z = [rand_pm1() for _ in 1:N]
        x = transform_from(d, t, z)
        if i ≤ 5                # just a few allocation tests
            @test @ballocated($ℓ(∂($D, $x))) == 0
        end
        ℓDx = @inferred ℓ(∂(D, x))
        for (i, a) in enumerate(Tuple(ℓDx))
            @test a ≈ D̃[i](ℓ, x) atol = 1e-3 # cross-derivatives: lower tolerance
        end
    end
end
