using SpectralKit, Test
# test internals
using SpectralKit: 𝑑Derivatives, _one, _add, _sub, _mul,
    Partials, _is_strict_subset, _partials_minimal_representation,
    _partials_canonical_expansion, ∂Derivatives, ∂CoordinateExpansion
using Random: randperm
using StaticArrays: SVector

@testset "𝑑Derivatives" begin
    @test_throws ArgumentError 𝑑^-1
    @test 𝑑^3 == 𝑑Derivatives{3}()
    @test 𝑑^3 * 𝑑 == 𝑑Derivatives{4}()
end

@testset "𝑑Expansion" begin
    N = 3
    d = 𝑑^3
    x = 0.5
    𝑑x = @inferred d(x)
    @test 𝑑x.coefficients == SVector(x, 1.0, ntuple(_ -> 0.0, N - 1)...)
    f(x) = _sub(_add(_one(typeof(x)), _mul(2.0, x)), _mul(x, x))
    f𝑑x = @inferred f(𝑑x)
    for i in 0:N
        @test f𝑑x[i] ≈ DD(f, x, i) atol = 1e-8
    end
end

function rand_Partials(max_length = 4, max_d = 5, zero_prob = 0.2)
    l = rand(0:max_length)
    if l == 0
        Partials(())
    else
        d = rand(0:max_d, l)
        for i in 1:l
            if rand() < zero_prob
                d[i] = 0
            end
        end
        if d[end] == 0
            d[end] += 1
        end
        Partials(Tuple(d))
    end
end

@testset "Partials sanity checks and subset tests" begin
    p123 = Partials((1, 2, 3))
    @test !_is_strict_subset(p123, p123)
    @test _is_strict_subset(Partials((1, 2)), p123)
    @test _is_strict_subset(Partials((1, 2, 2)), p123)
    @test _is_strict_subset(Partials((1, 0, 2)), p123)
    @test !_is_strict_subset(Partials((1, 0, 2, 1)), p123)
    @test _is_strict_subset(Partials(()), p123)
    @test !SpectralKit._is_strict_subset(Partials((1,3)), Partials((3,0,1)))
    @test_throws ArgumentError Partials((1, 2, -1))

    for _ in 1:100
        p1 = rand_Partials()
        p2 = rand_Partials()
        S1 = Set(_partials_canonical_expansion(Val(5), [p1]))
        S2 = Set(_partials_canonical_expansion(Val(5), [p2]))
        @test _is_strict_subset(p1, p2) == (S1 ⊆ S2 && S1 ≠ S2)
    end
end

@testset "Partials total order and strict subset" begin
    for _ in 1:1000
        p1 = rand_Partials()
        p2 = rand_Partials()
        @test isless(p1, p2) + isless(p2, p1) + isequal(p1, p2) == 1
        if _is_strict_subset(p1, p2)
            @test isless(p1, p2)
        end
    end
end

@testset "random partial derivatives minimal canonical roundtrip" begin
    for _ in 1:1000
        p_rand = [rand_Partials(3, 3, 0.9) for _ in 1:rand(1:20)]
        p_min = _partials_minimal_representation(p_rand)
        for _ in 1:50
            @test p_min ==  _partials_minimal_representation(p_min[randperm(length(p_min))])
            @test p_min ==  _partials_minimal_representation(p_rand[randperm(length(p_rand))])
        end
        p_exp = Set(_partials_canonical_expansion(Val(5), p_min))
        p_exp2 = mapreduce(p -> Set(_partials_canonical_expansion(Val(5), [p])), ∪, p_rand)
        @test p_exp == p_exp2
    end
end

@testset "partial derivatives API entry points" begin
    @test ∂() ≡ ∂Derivatives{Tuple{}}()
    @test ∂(2, 2) ≡ ∂Derivatives{Tuple{Partials(2, 2)}}()
    @test 𝑑^2 << Val(2) ≡ ∂Derivatives{Tuple{Partials(0, 2)}}()
    @test ∂(1, 2) ∪ ∂(2, 1) ≡
        ∂Derivatives{Tuple{_partials_minimal_representation([Partials(1, 2),
                                                             Partials(2, 1)])...}}()
    @test repr(∂()) == "∂()"
    @test repr(∂(1, 2)) == "∂(1, 2)"
end

@testset "partials derivatives expansions" begin
    D = ∂(2, 2)
    x = SVector(1.0, 2.0)
    Dx = @inferred(D(x))
    @test Dx isa ∂CoordinateExpansion{<:typeof(D)}
end


# @testset "partial derivatives interface" begin
#     @test ∂(2, (1, 2), (2, 2)) == ∂Specification{(1, 2)}(((1, 1), (0, 2)))
#     @test ∂(3, (), (1, 1, 2), (3, 2)) == ∂Specification{(2, 1, 1)}(((0, 0, 0),  (2, 1, 0), (0, 1, 1)))
#     @test_throws ArgumentError ∂(3, (4, ))
#     @test_throws ArgumentError ∂(3, (-1, ))
#     ∂spec = ∂(2, (1, 2), (2, 2))
#     @test repr(∂spec) == "partial derivatives\n[1] ∂²f/∂x₁∂x₂\n[2] ∂²f/∂²x₂"
#     @test repr(∂(∂spec, [1.0, 2.0])) ==
#         "partial derivatives\n[1] ∂²f/∂x₁∂x₂\n[2] ∂²f/∂²x₂\nat [1.0, 2.0]"
#     @test ∂((1.0, 2.0), (1, 2)) ≡ ∂(SVector(1.0, 2.0), (1, 2)) ≡ ∂(∂(Val(2), (1, 2)), (1.0, 2.0))
#     ∂o = ∂Output((1.0, 2.0))
#     @test repr(∂o) == "SpectralKit.∂Output(1.0, 2.0)"
#     @test ∂o[1] == 1.0
#     @test length(∂o) == 2
#     @test Tuple(∂o) == (1.0, 2.0)
# end

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

# @testset "univariate derivatives sanity checks" begin
#     d = derivatives(0.5, Val(2))
#     @test eltype(d) ≡ Float64
#     @test repr(d) == "0.5 + 1.0⋅Δ + 0.0⋅Δ²"
# end

# @testset "univariate transformed derivatives" begin
#     D = 1                       # derivatives up to this one
#     b = Chebyshev(EndpointGrid(), 4)
#     θ = randn(dimension(b))
#     for t in (BoundedLinear(-2.0, 7.0), )
#         for i in 1:100
#             z = rand_pm1(i)
#             x = transform_from(PM1(), t, z)
#             ℓ = linear_combination(b ∘ t, θ)
#             Dx = @inferred ℓ(derivatives(x, Val(D)))
#             @test Dx[0] == ℓ(x)
#             for d in 1:D
#                 @test DD(ℓ, x, d) ≈ Dx[d] atol = 1e-8
#             end
#         end
#     end
# end

# @testset "Smolyak derivatives check" begin
#     N = 3
#     b = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), N)
#     t = coordinate_transformations((BoundedLinear(1.0, 2.7),
#                                     SemiInfRational(3.0, 0.7),
#                                     InfRational(1.7, 0.3)))
#     D = ∂(Val(3), (), (1,), (2, ), (3, ), (1, 2))
#     D̃ = [(f, x) -> f(x),
#          (f, x) -> DD(x1 -> f((x1, x[2], x[3])), x[1]),
#          (f, x) -> DD(x2 -> f((x[1], x2, x[3])), x[2]),
#          (f, x) -> DD(x3 -> f((x[1], x[2], x3)), x[3]),
#          (f, x) -> DD(x1 -> DD(x2 -> f((x1, x2, x[3])), x[2]), x[1])]
#     bt = b ∘ t
#     θ = randn(dimension(bt))
#     ℓ = linear_combination(bt, θ)
#     d = domain(b)
#     for i in 1:100
#         z = [rand_pm1() for _ in 1:N]
#         x = transform_from(d, t, z)
#         if i ≤ 5                # just a few allocation tests
#             @test @ballocated($ℓ(∂($D, $x))) == 0
#         end
#         ℓDx = @inferred ℓ(∂(D, x))
#         for (i, a) in enumerate(Tuple(ℓDx))
#             @test a ≈ D̃[i](ℓ, x) atol = 1e-3 # cross-derivatives: lower tolerance
#         end
#     end
# end

# @testset "iteration for ∂Output" begin
#     o_src = (1.0, 2.0, 3.0)
#     ∂o = SpectralKit.∂Output(o_src)
#     for (o1, o2) in zip(o_src, ∂o)
#         @test o1 ≡ o2
#     end
#     function f(o)
#         o1, o2, o3 = o
#         o1 + o2 + o3
#     end
#     @test @inferred(f(∂o)) == sum(o_src)
#     @test @ballocated($f($∂o)) == 0
# end
