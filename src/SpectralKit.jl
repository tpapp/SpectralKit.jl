module SpectralKit

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector, sacollect
using UnPack: @unpack

include("generic_api.jl")
include("chebyshev.jl")
include("univariate.jl")
include("smolyak_blocks.jl")
include("smolyak_traversal.jl")
include("smolyak_api.jl")

end # module
