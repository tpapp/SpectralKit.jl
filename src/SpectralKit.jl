module SpectralKit

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector, sacollect
using SimpleUnPack: @unpack

include("utilities.jl")
include("derivatives.jl")
include("transformations.jl")
# include("generic_api.jl")
# include("chebyshev.jl")
# include("smolyak_traversal.jl")
# include("smolyak_api.jl")

end # module
