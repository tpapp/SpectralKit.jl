using SpectralKit: _add                           # used to form a scalar for testing
using SpectralKit: Derivatives, replace_zero_tags # test internals

@testset "univariate derivatives check" begin
    z = 0.6
    x = derivatives(z, Val(2))
    @test x[0] == z
    @test x[1] == 1
    @test x[2] == 0
    b = Chebyshev(EndpointGrid(), 4)
    f(z) = sum(basis_at(b, z))
    bz = basis_at(b, derivatives(z, Val(2)))
    iterator_sanity_checks(bz)
    ∑b = reduce(_add, )
    @test ∑b[0] ≈ f(z)
    @test ∑b[1] ≈ ddn(f, z, Val(1))
    @test ∑b[2] ≈ ddn(f, z, Val(2))
end

@testset "univariate transformed derivatives" begin
    # NOTE: currently only BoundedLinear is implemented
    x = 0.6
    t = to_pm1(BoundedLinear(0, 1))
    b = Chebyshev(EndpointGrid(), 4)
    f(x) = sum(basis_at(b, t(x)))
    bz = basis_at(b, t(derivatives(x, Val(2))))
    iterator_sanity_checks(bz)
    ∑b = reduce(_add, bz)
    @test ∑b[0] ≈ f(x)
    @test ∑b[1] ≈ ddn(f, x, Val(1))
    @test ∑b[2] ≈ ddn(f, x, Val(2))
end

@testset "zero tags" begin
    x1, x2, x3, x4 = range(1.0, 4.0; length = 4)
    _get(x::Derivatives{I}) where {I} = (I, x.derivatives[1])
    xs = @inferred replace_zero_tags((derivatives(x1), x2, derivatives(Val(3), x3), derivatives(x4)))
    @test _get(xs[1]) == (4, x1)
    @test xs[2] == x2
    @test _get(xs[3]) == (3, x3)
    @test _get(xs[4]) == (5, x4)
end

@testset "Smolyak derivatives check" begin
    b = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)
    x1 = 0.7
    x2 = 0.4
    z12 = derivatives(Val(2), x1) # exchange the order
    z21 = derivatives(Val(1), x2)
    f(x) =  sum(basis_at(b, x))
    F00 = f([x1, x2])
    F01 = ForwardDiff.derivative(x -> f([x1, x]), x2)
    F10 = ForwardDiff.derivative(x -> f([x, x2]), x1)
    F11 = ForwardDiff.derivative(x2 -> ForwardDiff.derivative(x -> f([x, x2]), x1), x2)
    bz = basis_at(b, (z12, z21))
    iterator_sanity_checks(bz)
    Z = @inferred reduce(_add, bz)
    @test Z[0][0] ≈ F00
    @test Z[1][0] ≈ F01
    @test Z[0][1] ≈ F10
    @test Z[1][1] ≈ F11
end
