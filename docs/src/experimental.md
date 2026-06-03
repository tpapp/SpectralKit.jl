# Experimental

```@docs
SpectralKit.Experimental
```

## Proposed additions to general API

```@docs
SpectralKit.Experimental.constant_coefficients
```

## modeling setup

```
          make_model_parameters       ┏━━━━━━━━━━━━━━━━━━━━┓
       ╭──────▷──────╭───────────────▶┃  model_parameters  ┃
       │  ┏━━━━━━━━━━┷━━━━━━━━━━━┓    ┗━━┯━━━━━━━━━━━━━━━━━┛
       │  ┃  model_coefficients  ┃       │
       │  ┃  (model_dimension)   ┃       │
       │  ┗━━━━━━━━━━┯━━━━━━━━━━━┛       │  calculate_derived_quantities   ┏━━━━━━━━━━━━━━━━━━━━━━┓
       │   ╭────▷────╰─────────▷─────────╰──────────────▷─────────────────▶┃  derived_quantities  ┃
       │   │                                                               ┗━━━┯━━━━━━━━━━━━━━━━━━┛
       │   │                                                                   │
       │   │                                                                   │
       │   │                          make_approximation                       │     ┏━━━━━━━━━━━━━━━━━┓
       │   │   ╭────────▷─────────╭────────────▷───────────╭─────▷────╭────▷───╰────▶┃  approximation  ┃
       │   │   │                  │                        │          │              ┗━━━━━━━━━━━━━━━━━┛
       │   │   │                  │                        │  ┏━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━┓
       │   │   │                  │                        │  ┃  approximation coefficients  ┃
       │   │   │                  │                        │  ┃   (approximation_dimension)  ┃
       │   │   │                  │                        │  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
  ┏━━━━┷━━━┷━━━┷━━━┓  ┏━━━━━━━━━━━┷━━━━━━━━━━━┓   ┏━━━━━━━━┷━━━━━━━━━━━━━━━━┓
  ┃  model_family  ┃  ┃  approximation_setup  ┃   ┃   approximation_level   ┃
  ┗━━━━━━━┯━━━━━━━━┛  ┗━━━━━━━━━━┯━━━━━━━━━━━━┛   ┃ (approximation_levels)  ┃
          │                      │                ┗━━━━━━━━━━┯━━━━━━━━━━━━━━┛
          │                      │                           │
          │                      │                           │      make_grid        ┏━━━━━━━━┓
          ╰─────────────▷────────╰─────────────▷─────────────╰───────────▷──────────▶┃  grid  ┃
                                                                                     ┗━━━━━━━━┛
```


```julia
augment_coefficients(model_family, approximation_setup, approximation_level, approximation_coefficients)

initial_guess(model_family, approximation_setup) => approximation_level, approximation_coefficients

residuals(model_family, model_parameters, approximation, gridpoint)

vec_residuals(model_family, approximation_level, model_coefficients, approximation_coefficients, gridpoint)
```
