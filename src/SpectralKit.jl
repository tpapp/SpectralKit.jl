module SpectralKit

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector, sacollect
using UnPack: @unpack

include("generic_api.jl")
include("chebyshev.jl")
include("univariate.jl")
# include("smolyak.jl")

end # module
