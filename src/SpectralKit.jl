module SpectralKit

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using OrderedCollections: OrderedSet
using StaticArrays: MVector, SVector, sacollect

include("utilities.jl")
include("derivatives.jl")
include("domains.jl")
include("transformations.jl")
include("generic_api.jl")
include("chebyshev.jl")
include("smolyak_traversal.jl")
include("smolyak_api.jl")
include("experimental.jl") # experimental code is not part of the API, see its module docstring

end # module
