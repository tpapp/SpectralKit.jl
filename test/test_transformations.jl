using SpectralKit: PM1

@testset "bounded linear domain transformations" begin
    @test_throws DomainError BoundedLinear(-1.0, Inf)
    @test_throws DomainError BoundedLinear(-1.0, -2.0)

    A, B = 1, 5
    trans = BoundedLinear(A, B)

    @test extrema(domain(trans)) == (A, B)

    for _ in 1:100
        x = rand_pm1()
        y = transform_from(PM1(), trans, x)
        if x == -1
            @test y ≈ A
        elseif x == 1
            @test y ≈ B
        else
            @test A < y < B
        end
        @test transform_to(PM1(), trans, y) ≈ x
    end
end

@testset "semi-infinite domain transformations" begin
    @test_throws DomainError SemiInfRational(; scale = Inf)
    @test_throws DomainError SemiInfRational(; scale = 0.0)
    @test_throws DomainError SemiInfRational(; endpoint = NaN)

    endpoint = 3.0
    scale = 4.0
    trans = SemiInfRational(; endpoint, scale)

    @test extrema(domain(trans)) == (endpoint, Inf)

    for _ in 1:100
        x = rand_pm1()
        y = transform_from(PM1(), trans, x)
        if x == -1
            @test y ≈ endpoint
        elseif x == 1
            @test y ≈ Inf
        else
            @test endpoint < y < Inf
        end
        @test transform_to(PM1(), trans, y) ≈ x
    end

    # compare to analytical limits NOTE extend when we add more derivatives
    y_pinf = @inferred transform_to(PM1(), trans, 𝑑(Inf))
    @test y_pinf[0] == 1 == @inferred transform_to(PM1(), trans, Inf)
    @test y_pinf[1] == 0

    y_minf = @inferred transform_to(PM1(), trans, 𝑑(-Inf))
    @test y_minf[0] == 1 == @inferred transform_to(PM1(), trans, -Inf)
    @test y_minf[1] == 0
end

@testset "infinite domain transformations" begin
    @test_throws DomainError InfRational(; scale = Inf)
    @test_throws DomainError InfRational(; scale = 0.0)
    @test_throws DomainError InfRational(; scale = -2.0)
    @test_throws DomainError InfRational(; center = NaN)

    trans = InfRational()

    @test extrema(domain(trans)) == (-Inf, Inf)

    for _ in 1:100
        x = rand_pm1()
        y = transform_from(PM1(), trans, x)
        if x == -1
            @test y == -Inf
        elseif x == 1
            @test y == Inf
        else
            @test isfinite(y)
        end
        @test transform_to(PM1(), trans, y) ≈ x
    end

    # compare to analytical limits NOTE extend when we add more derivatives
    y_pinf = @inferred transform_to(PM1(), trans, 𝑑(Inf))
    @test y_pinf[0] == 1 == transform_to(PM1(), trans, Inf)
    @test y_pinf[1] == 0

    y_minf = @inferred transform_to(PM1(), trans, 𝑑(-Inf))
    @test y_minf[0] == -1 == transform_to(PM1(), trans, -Inf)
    @test y_minf[1] == 0
end

@testset "printing, promotion" begin
    t1 = BoundedLinear(2.0, 3)
    @test repr(t1) == "BoundedLinear(2.0, 3.0)"
    t2 = SemiInfRational(; endpoint = 7.0)
    @test repr(t2) == "SemiInfRational(endpoint = 7.0, scale = 1.0)"
    t3 = InfRational(; center = 0.5, scale = 1)
    @test repr(t3) == "InfRational(; center = 0.5, scale = 1.0)"
end
