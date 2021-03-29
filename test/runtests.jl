using SpectralKit
using Test, DocStringExtensions, StaticArrays
using ForwardDiff: derivative

include("utilities.jl")

include("test_chebyshev.jl")
include("test_univariate.jl")
include("test_smolyak.jl")
