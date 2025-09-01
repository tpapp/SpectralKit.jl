import SpectralKit.Experimental as SK

using LogExpFunctions: logistic

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

SK.model_parameters_dimension(::RamseyDiscrete) = 3

function SK.make_model_parameters(::RamseyDiscrete, x)
    pre_α, pre_β, pre_δ = x
    (; a = logistic(pre_α), β = logistic(pre_β), δ = logistic(pre_δ))
end

function SK.calculate_derived_quantities(::RamseyDiscrete, model_parameters)
    (; α, β, δ) = model_parameters
    # f'(k) β = (α k^{α-1} - δ) β = 1
    k_s = ((1 / β + δ) / α)^(α - 1)
    c_s = k_s^α - δ_s * k_s - k_s
    (; k_s, c_s)
end
