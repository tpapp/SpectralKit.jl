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

include("test_generic_api.jl")
