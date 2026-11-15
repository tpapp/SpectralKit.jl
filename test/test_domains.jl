using SpectralKit, Test
using SpectralKit: PM1, UnivariateDomain

@testset "domain API" begin
    d1 = PM1()
    @test repr(d1) == "[-1,1]"
    @test extrema(d1) == (-1, 1)
    @test (minimum(d1), maximum(d1)) == extrema(d1)
    @test domain_kind(d1) ≡ :univariate
end
