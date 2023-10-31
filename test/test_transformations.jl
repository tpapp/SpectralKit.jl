@testset "BoundedLinear" begin
    @test_throws DomainError BoundedLinear(-1.0, Inf)
    @test_throws DomainError BoundedLinear(-1.0, -2.0)

    A, B = 1, 5
    trans = BoundedLinear(A, B)

    @test extrema(domain(trans)) == (A, B)

    for i in 1:100
        x = rand_pm1(i)
        y = transform_from(PM1(), trans, x)
        i == 1 && @test y ≈ A
        i == 2 && @test y ≈ B
        i > 2 && @test A < y < B
        @test transform_to(PM1(), trans, y) ≈ x
    end
end

@testset "Chebyshev semi-infinite" begin
    @test_throws DomainError SemiInfRational(-1.0, Inf)
    @test_throws DomainError SemiInfRational(-1.0, 0.0)
    @test_throws DomainError SemiInfRational(NaN, 2.0)

    A = 3.0
    L = 4.0
    trans = SemiInfRational(A, L)

    @test extrema(domain(trans)) == (A, Inf)

    for i in 1:100
        x = rand_pm1(i)
        y = transform_from(PM1(), trans, x)
        i == 1 && @test y ≈ A
        i == 2 && @test y ≈ Inf
        i > 2 && @test A < y < Inf
        @test transform_to(PM1(), trans, y) ≈ x
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

    @test extrema(domain(trans)) == (-Inf, Inf)

    for i in 1:100
        x = rand_pm1(i)
        y = transform_from(PM1(), trans, x)
        i == 1 && @test y == -Inf
        i == 2 && @test y == Inf
        i > 2 && @test isfinite(y)
        @test transform_to(PM1(), trans, y) ≈ x
    end
end

@testset "coordinate transformations" begin
    t1 = BoundedLinear(2.0, 3.0)
    t2 = SemiInfRational(7.0, 1.0)
    ct = coordinate_transformations(t1, t2)
    md = coordinate_domains(Val(2), PM1())
    x = SVector(rand_pm1(), rand_pm1())
    y = @inferred transform_from(md, ct, x)
    @test y isa SVector{2,Float64}
    @test y == transform_from.(PM1(), Tuple(ct), x)

    # handle generic inputs
    y2 = @inferred transform_from(md, ct, Vector(x))
    @test y2 isa SVector{2,Float64} && y2 == y

    x2 = @inferred transform_to(md, ct, [y...])
    @test x2 isa SVector{2,Float64} && all(x2 .≈ x)
end

@testset "printing, promotion, broadcasting" begin
    v = [1.0, 2.0]
    t1 = BoundedLinear(2.0, 3)
    @test repr(t1) == "(2.0,3.0) ↔ domain [linear transformation]"
    @test transform_to.(PM1(), t1, v) isa Vector
    t2 = SemiInfRational(7.0, 1)
    @test repr(t2) == "(7.0,∞) ↔ domain [rational transformation with scale 1.0]"
    @test transform_to.(PM1(), t2, v) isa Vector
    t3 = InfRational(0.5, 1)
    @test repr(t3) == "(-∞,∞) ↔ domain [rational transformation with center 0.5, scale 1.0]"
    @test transform_to.(PM1(), t3, v) isa Vector
    ct = coordinate_transformations(t1, t2, t3)
    @test repr(ct) ==
        "coordinate transformations\n  " * repr(t1) * "\n  " * repr(t2) * "\n  " * repr(t3)
end
