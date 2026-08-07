"""
Experimental functionality. Officially not (yet) part of the API.

A best effort is made to

1. only change in breaking ways (including integration into the public API) with major or
   minor releases.
2. fix issues (users are encouraged to open them)
"""
module Experimental

# TODO
# - [ ] clean up API, too many functions with positional arguments
# - [ ] write up tutorial-like docs
# - [ ] docs: an ascii diagram of various calculations, with https://asciiflow.com/
# - [ ] solver functions
# - [ ] actually implement a solution and AD it

# migrate to generic API
export constant_coefficients

using Compat: @compat
@compat public model_parameters_dimension, make_model_parameters,
    calculate_derived_quantities, make_approximation_basis, describe_policy_transformations,
    policy_coefficients_dimension, make_policy_functions, constant_initial_guess,
    calculate_initial_guess, sum_of_squared_residuals, bridge

using ..SpectralKit: SpectralKit, Chebyshev, TransformedBasis, SmolyakBasis, PM1,
    AbstractUnivariateTransformation, dimension, linear_combination, grid, domain,
    transform_from, transform_to

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES
import InverseFunctions

####
#### utilities
####

"""
$(SIGNATURES)

FIXME docs
"""
function named_cumulative_ranges(lengths::NamedTuple{N}) where N
    # NOTE this approach  relies on Base.afoldl, so it unrolled only up to 31 elements
    ranges = accumulate(lengths, init = 0:0) do r, l
        a = last(r)
        (a + 1):(a + l)
    end
    NamedTuple{N}(ranges)
end

####
#### generic API additions
####

"""
$(FUNCTIONNAME)(basis, y)

Approximate a constant value `y` on `basis`.

Formally, return a set of coefficients `θ` such that `linear_combination(basis, θ, x) ≈
y` for all `x` in the domain.
"""
function constant_coefficients end

function constant_coefficients(basis::Chebyshev, y)
    θ = zeros(basis.N)
    θ[1] = y
    θ
end

function constant_coefficients(basis::TransformedBasis, y)
    constant_coefficients(parent(basis), y)
end

function constant_coefficients(basis::SmolyakBasis, y)
    θ = zeros(dimension(basis))
    θ[1] = y
    θ
end

####
#### modelling API
####

const USERNOTE = "**User should implement this method for the relevant `model_family`.**"

"""
$(FUNCTIONNAME)(model_family)

Dimension of the model parameters as a flat vector.

$(USERNOTE)
"""
function model_parameters_dimension end

"""
$(FUNCTIONNAME)(model_family, x::AbstractVector)

Convert a flat vector of dimension [`model_parameters_dimension`)(@ref) to the model
parameters.

$(USERNOTE)

The returned value can be an arbitrary (eg a `NamedTuple`), as it does not need to be
not used for dispatch.

!!! NOTE
    The method should accept all finite numbers in ``ℝ``, and transform them accordingly.
"""
function make_model_parameters end

"""
$(SIGNATURES)(model_family, model_parameters)

Calculate derived quantities (for use in determining the bases and calculating the residuals).

$(USERNOTE)
"""
function calculate_derived_quantities end

"""
$(FUNCTIONNAME)(model_family, derived_quantities, approximation_parameters)

Construct an approximation basis.

$(USERNOTE)
"""
function make_approximation_basis end

"""
$(FUNCTIONNAME)(model_family)

Return a `NamedTuple` is policy function approximation schemes. The values describe the
transformations.

$(USERNOTE)
"""
function describe_policy_transformations end

"""
$(FUNCTIONNAME)(model_family, model_parameters, policy_functions, gridpoint)

Calculate the residuals at the given gridpoint.

$(USERNOTE)
"""
function calculate_residuals end

function policy_coefficients_dimension(policy_transformations::NamedTuple, approximation_basis)
    dimension(approximation_basis) * length(policy_transformations)
end

function make_policy_functions(model_family, policy_transformations::NamedTuple,
                               approximation_basis, coefficients)
    d = dimension(approximation_basis)
    # QUESTION line below assumes all univariate, generalize?
    ranges = named_cumulative_ranges(map(_ -> d, policy_transformations))
    @argcheck firstindex(coefficients) == 1
    @argcheck lastindex(coefficients) == last(last(ranges))
    map(ranges, policy_transformations) do r, t
        t ∘ linear_combination(approximation_basis, @view coefficients[r])
    end
end

"""
$(FUNCTIONNAME)(model_family, model_parameters, derived_quantities)

Return initial guesses in a type that supports `getproperty` (eg a `NamedTuple`).

Should provide a scalar for each name in [`describe_policy_transformations`](@ref) (can
be in any arbitrary order, and contain other names).
"""
function constant_initial_guess end

"""
$(SIGNATURES)

Return an initial guess for the coefficients. Falls back to using
[`constant_initial_guess`](@ref), or the user may define a method in case that is not
sufficient.
"""
function calculate_initial_guess(model_family, model_parameters, derived_quantities,
                                 policy_transformations::NamedTuple{N},
                                 approximation_basis) where N
    constant_guess = constant_initial_guess(model_family, model_parameters, derived_quantities)
    d = dimension(approximation_basis)
    # QUESTION line below assumes all univariate, generalize?
    ranges = named_cumulative_ranges(map(_ -> d, policy_transformations))
    coefficients = zeros(last(last(ranges)))
    for (name, transformation) in pairs(policy_transformations)
        r = getproperty(ranges, name)
        transformed_y = getproperty(constant_guess, name)
        y = InverseFunctions.inverse(transformation)(transformed_y)
        # FIXME a constant_coefficient! API would have fewer allocations
        coefficients[r] .= constant_coefficients(approximation_basis, y)
    end
    coefficients
end

"""
$(SIGNATURES)

Return a grid for calculating the residuals. The default implementation just uses the
grid that corresponds to the approximation basis.
"""
function make_approximation_grid(model_family, model_parameters, approximation_parameters,
                                 derived_quantities, approximation_basis)
    grid(approximation_basis)
end

"""
$(SIGNATURES)

Sum of squares. Works for values returned by [`calculate_residuals`](@ref).
"""
sum_abs2(x::Real) = abs2(x)

sum_abs2(nt::NamedTuple) = sum(abs2, values(nt))

"""
$(SIGNATURES)

Calculate the sum of squared residuals.
"""
function sum_of_squared_residuals(model_family, model_parameters, policy_functions, grid)
    mapreduce(+, grid) do gridpoint
        sum_abs2(calculate_residuals(model_family, model_parameters, policy_functions,
                                     gridpoint))
    end
end

####
#### bridge — expose the univariate algebraic transformations
####

"""
Implementation of [`bridge`](@ref). Not part of the (experimental) API.
"""
struct Bridge{O<:AbstractUnivariateTransformation,I<:AbstractUnivariateTransformation}
    outer_transformation::O
    inner_transformation::I
end

"""
$(SIGNATURES)

Transform using the `outer_transformation⁻¹ ∘ inner_transformation`. Return a callable
that supports [`InverseFunctions.inverse`](@ref).
"""
function bridge(outer_transformation::AbstractUnivariateTransformation,
                inner_transformation::AbstractUnivariateTransformation)
    Bridge(outer_transformation, inner_transformation)
end

function (b::Bridge)(x)
    (; outer_transformation, inner_transformation) = b
    d = PM1()
    transform_from(d, outer_transformation, transform_to(d, inner_transformation, x))
end

SpectralKit.domain(b::Bridge) = domain(b.inner_transformation)

InverseFunctions.inverse(b::Bridge) = Bridge(b.inner_transformation, b.outer_transformation)

end
