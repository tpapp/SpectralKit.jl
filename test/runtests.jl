using SpectralKit
using Test, DocStringExtensions, StaticArrays, BenchmarkTools, Sobol
import ForwardDiff

include("utilities.jl")

include("test_generic_api.jl")
include("test_chebyshev.jl")
include("test_smolyak.jl")
include("test_transformations.jl")
