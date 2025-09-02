import SpectralKit.Experimental as SKX
using LogExpFunctions: logistic
using Test
using SpectralKit

"""
A simple Ramsey model in discrete time.

### Setup

We document the model mainly to fix notation and make the code self-contained. Time is
discrete, but below time periods are `0` (“this”) and `1` (“next”) without loss of
generality.

The resource constraint is
```math
c_0 + k_1 = f(k_0)
```
where ``f`` includes production and depreciation. The value function is
```math
V(k) = max_{0 < c ≤ f(k)} u(c) + β V(f(k) - c)
```
which is solved by the *policy function* ``c(k)``.

The first order condition is
```math
u'(c(k)) = β V'(f(k) - c(k))
```
while the envelope condition is
```math
V'(k) = β V'(f(k) - c(k)) f'(k)
```
Combine to obtain
```math
V'(k) = u'(c(k)) f'(k)
```
and then
```math
u'(c(k)) f'(k) = β u'(c(f(k) - c(k))) f'(c(f(k) - c(k))) f'(k)
```
Then cancel ``f'(k)`` and rearrange as unitless
```math
1 = β u'(c(f(k) - c(k))) f'(c(f(k)) - c(k)) / u'(k)
```
From this it follows that the steady state ``k_s`` solves
```
f'(k_s) β = 1
```
and the resource constraint yields the steady state consumption as
```
c_s = f(k_s) - k_s
```
"""
struct RamseyDiscrete end

SKX.model_parameters_dimension(::RamseyDiscrete) = 3

function SKX.make_model_parameters(::RamseyDiscrete, x)
    pre_α, pre_β, pre_δ = x
    (; α = logistic(pre_α), β = logistic(pre_β), δ = logistic(pre_δ))
end

product(model_parameters, k) = k^model_parameters.α  - model_parameters.δ * k

function SKX.calculate_derived_quantities(::RamseyDiscrete, model_parameters)
    (; α, β, δ) = model_parameters
    # f'(k) β = (α k^{α-1} - δ) β = 1
    k_s = ((1 / β + δ) / α)^(α - 1)
    c_s = product(model_parameters, k_s) - k_s
    (; k_s, c_s)
end

function SKX.make_approximation_basis(::RamseyDiscrete, derived_quantities, approximation_scheme)
    (; policy_coefficients) = approximation_scheme
    (; k_s) = derived_quantities
    Chebyshev(InteriorGrid(), policy_coefficients) ∘ SemiInfRational(0.0, k_s)
end

SKX.describe_policy_transformations(::RamseyDiscrete) = (; c_share = logistic)

model_family = RamseyDiscrete()
model_parameters = SKX.make_model_parameters(model_family,
                                            randn(SKX.model_parameters_dimension(model_family)))
derived_quantities = SKX.calculate_derived_quantities(model_family, model_parameters)
approximation_scheme = (; policy_coefficients = 10)
approximation_basis = SKX.make_approximation_basis(model_family, derived_quantities,
                                                   approximation_scheme)
policy_transformations = SKX.describe_policy_transformations(model_family)
policy_functions = SKX.make_policy_functions(model_family, policy_transformations,
                                             approximation_basis,
                                             zeros(SKX.policy_coefficients_dimension(policy_transformations,
                                                                                     approximation_basis)))
