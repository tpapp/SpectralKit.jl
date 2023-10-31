@testset "unicode printing" begin
    i = 9876543210
    io = IOBuffer()
    print(io, SpectralKit.SubScript(i))
    print(io, SpectralKit.SuperScript(i))
    @test String(take!(io)) == "₉₈₇₆₅₄₃₂₁₀⁹⁸⁷⁶⁵⁴³²¹⁰"
end
