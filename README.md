# SpectralKit.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/SpectralKit.jl/workflows/CI/badge.svg)](https://github.com/tpapp/SpectralKit.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/SpectralKit.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/SpectralKit.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/SpectralKit.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/SpectralKit.jl/dev)

Building blocks of spectral methods for Julia. Currently includes Chebyshev polynomials on univariate and Smolyak (multivariate) grids, with domain transformations to semi-infinite and infinite domains.

## Introduction

Mostly useful for algorithms along the lines of

> Boyd, John P. *Chebyshev and Fourier spectral methods*. 2001.

The aim is to provide simple, well-tested, robust, and fast *building blocks* for spectral algorithms, which can be easily combined into algorithms.

At the moment, the package API is experimental and subject to change.

## Help

Asking for help in [issues](https://github.com/tpapp/SpectralKit.jl/issues) is fine, you can also ping me as `@Tamas_Papp` on the [Discourse forum](https://discourse.julialang.org/)

## Pretty pictures

Some examples generated this library. **Circles mark values at the limit, shifted horizontally when this is needed to avoid overlap**. Infinite limits shown at finite values, so of course they don't match (this is a visual check of continuity, naturally it is unit tested).

### Chebyshev polynomials and their derivatives

<img src="docs/plots/chebyshev.png" width="50%">

<img src="docs/plots/chebyshev_deriv.png" width="50%">

### Chebyshev rational functions on [0,∞)

Up close, you can see the oscillation.

<img src="docs/plots/semiinf.png" width="50%">

Let's zoom out a bit to see convergence to 0 at ∞.

<img src="docs/plots/semiinf_birdseye.png" width="50%">

Derivatives die out faster.

<img src="docs/plots/semiinf_deriv.png" width="50%">

### Chebyshev rational functions on (-∞,∞)

Up close, you can see the oscillation.

<img src="docs/plots/inf.png" width="50%">

Let's zoom out a bit to see convergence at -∞ and ∞.

<img src="docs/plots/inf_birdseye.png" width="50%">

Derivatives die out slower than for the [0,∞) transformation.

<img src="docs/plots/inf_deriv.png" width="50%">

### A Smolyak grid

With `B = 3`.

<img src="docs/plots/smolyak_grid.png" width="50%">
