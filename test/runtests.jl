using SpectralKit
using SpectralKit: PM1
using Test, DocStringExtensions, StaticArrays, BenchmarkTools
# import ForwardDiff

include("utilities.jl")

include("test_domains.jl")
include("test_transformations.jl")
include("test_generic_api.jl")
include("test_chebyshev.jl")
include("test_smolyak.jl")
