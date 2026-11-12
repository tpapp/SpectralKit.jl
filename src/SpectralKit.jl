"""
$(README)
"""
module SpectralKit

using ArgCheck: @argcheck
import ChainRulesCore
using ConcreteStructs: @concrete
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF, README
using OrderedCollections: OrderedSet
using StaticArrays: MVector, SVector, sacollect

include("utilities.jl")
include("derivatives.jl")
include("domains.jl")
include("transformations.jl")
include("generic_api.jl")
include("univariate.jl")
include("smolyak_traversal.jl")
include("smolyak.jl")

end # module
