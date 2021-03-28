module SpectralKit

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector, sacollect
using UnPack: @unpack

include("generic_api.jl")
include("chebyshev.jl")
# include("transformed_chebyshev.jl")
# include("smolyak.jl")

end # module
