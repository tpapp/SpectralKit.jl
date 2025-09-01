"""
Experimental functionality. Officially not (yet) part of the API.

A best effort is made to

1. only change in breaking ways (including integration into the public API) with major or
   minor releases.
2. fix issues (users are encouraged to open them)
"""
module Experimental

using Compat: @compat

@compat public model_parameters_dimension, make_model_parameters, calculate_derived_quantities

using DocStringExtensions: FUNCTIONNAME, SIGNATURES

####
#### endpoint mappings
####

# TODO

####
#### modelling API
####

"""
$(FUNCTIONNAME)(model_family)

Dimension of the model parameters as a flat vector.
"""
function model_parameters_dimension end

"""
$(FUNCTIONNAME)(model_family, x::AbstractVector)

Convert a flat vector of dimension [`model_parameters_dimension`)(@ref) to the model
parameters, which can be an arbitrary value (eg a `NamedTuple`), which is not used for
dispatch.

!!! NOTE
    The method should accept all finite numbers in ``ℝ``, and transform them accordingly.
"""
function make_model_parameters end

"""
$(SIGNATURES)(model_family, model_parameters)

Calculate derived quantities (for use in determining the bases and calculating the residuals).
"""
function calculate_derived_quantities end

end
