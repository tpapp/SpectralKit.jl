module SpectralKit

export Order, OrdersTo, is_function_family, domain_extrema, roots, augmented_extrema,
    basis_iterator, basis_function, linear_combination, Chebyshev, ChebyshevSemiInf,
    ChebyshevInf, ChebyshevInterval

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector
using UnPack: @unpack

include("generic_api.jl")
include("chebyshev.jl")
include("transformed_chebyshev.jl")

end # module
