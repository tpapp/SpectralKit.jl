using SpectralKit
using Test, DocStringExtensions, StaticArrays, BenchmarkTools, Sobol
using ForwardDiff: derivative, gradient

include("utilities.jl")

include("test_generic_api.jl")
include("test_chebyshev.jl")
include("test_univariate.jl")
include("test_smolyak.jl")
