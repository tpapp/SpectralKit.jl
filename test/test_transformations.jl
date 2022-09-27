@testset "BoundedLinear" begin
    @test_throws DomainError BoundedLinear(-1.0, Inf)
    @test_throws DomainError BoundedLinear(-1.0, -2.0)

    A, B = 1, 5
    trans = BoundedLinear(A, B)

    for i in 1:100
        x = rand_pm1(i)
        y = from_pm1(trans, x)
        i == 1 && @test y ≈ A
        i == 2 && @test y ≈ B
        i > 2 && @test A < y < B
        @test to_pm1(trans, y) ≈ x
    end
end

@testset "Chebyshev semi-infinite" begin
    @test_throws DomainError SemiInfRational(-1.0, Inf)
    @test_throws DomainError SemiInfRational(-1.0, 0.0)
    @test_throws DomainError SemiInfRational(NaN, 2.0)

    A = 3.0
    L = 4.0
    trans = SemiInfRational(A, L)

    for i in 1:100
        x = rand_pm1(i)
        y = from_pm1(trans, x)
        i == 1 && @test y ≈ A
        i == 2 && @test y ≈ Inf
        i > 2 && @test A < y < Inf
        @test to_pm1(trans, y) ≈ x
    end
end

@testset "Chebyshev infinite" begin
    @test_throws DomainError InfRational(1.0, Inf)
    @test_throws DomainError InfRational(1.0, 0.0)
    @test_throws DomainError InfRational(1.0, -2.0)
    @test_throws DomainError InfRational(NaN, 1)

    A = 0.0
    L = 1.0
    trans = InfRational(A, L)

    for i in 1:100
        x = rand_pm1(i)
        y = from_pm1(trans, x)
        i == 1 && @test y == -Inf
        i == 2 && @test y == Inf
        i > 2 && @test isfinite(y)
        @test to_pm1(trans, y) ≈ x
    end
end

@testset "coordinate transformations" begin
    t1 = BoundedLinear(2.0, 3.0)
    t2 = SemiInfRational(7.0, 1.0)
    ct = coordinate_transformations(t1, t2)
    x = SVector(rand_pm1(), rand_pm1())
    y = @inferred from_pm1(ct, x)
    @test y isa SVector{2,Float64}
    @test y == SVector(from_pm1(t1, x[1]), from_pm1(t2, x[2]))

    # handle generic inputs
    y2 = @inferred from_pm1(ct, Vector(x))
    @test y2 isa SVector{2,Float64} && y2 == y

    x2 = @inferred to_pm1(ct, [y...])
    @test x2 isa SVector{2,Float64} && all(x2 .≈ x)
end

@testset "partial application" begin
    t = SemiInfRational(7.0, 1.0)
    x = rand_pm1()
    y = from_pm1(t, x)
    @test from_pm1(t)(x) == y
    @test to_pm1(t)(y) == to_pm1(t, y)
end
