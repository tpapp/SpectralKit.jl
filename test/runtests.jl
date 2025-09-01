using SpectralKit
using Test, DocStringExtensions, StaticArrays, BenchmarkTools, FiniteDifferences

include("utilities.jl")

include("test_utilities.jl")
include("test_derivatives.jl")
include("test_domains.jl")
include("test_transformations.jl")
include("test_chebyshev.jl")
include("test_smolyak_traversal.jl")
include("test_smolyak.jl")
include("test_generic_api.jl")  # NOTE moved last as it used constructs from above
include("test_experimental.jl") # NOTE experimental code is not public API

using JET
@testset "static analysis with JET.jl" begin
    @test isempty(JET.get_reports(report_package(SpectralKit, target_modules=(SpectralKit,))))
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(SpectralKit)
end
