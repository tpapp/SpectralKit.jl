module SpectralKit

export
    # generic
    Order, OrdersTo, is_function_family, domain_extrema, roots, augmented_extrema,
    basis_function, linear_combination,
    # Chebyshev & transformed
    Chebyshev, ChebyshevSemiInf, ChebyshevInf, ChebyshevInterval

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF
using StaticArrays: MVector, SVector, sacollect
using UnPack: @unpack

include("generic_api.jl")
include("chebyshev.jl")
include("transformed_chebyshev.jl")
include("smolyak.jl")

end # module
