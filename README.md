# SpectralKit.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/tpapp/SpectralKit.jl.svg?branch=master)](https://travis-ci.com/tpapp/SpectralKit.jl)
[![codecov.io](http://codecov.io/github/tpapp/SpectralKit.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/SpectralKit.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/SpectralKit.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/SpectralKit.jl/latest)

Building blocks of spectral methods for Julia.

## Introduction

Mostly useful for algorithms along the lines of

> Boyd, John P. *Chebyshev and Fourier spectral methods*. 2001.

The aim is to provide simple, well-tested, robust, and fast *building blocks* for spectral algorithms, which can be easily combined into algorithms.

At the moment, the package API is experimental and subject to change.

## Pretty pictures

Some examples generated this library. **Circles mark values at the limit, shifted horizontally when this is needed to avoid overlap**. Infinite limits shown at finite values, so of course they don't match (this is a visual check of continuity, naturally it is unit tested).

### Chebyshev polynomials and their derivatives

<img src="scripts/chebyshev.png" width="50%">

<img src="scripts/chebyshev_deriv.png" width="50%">

### Chebyshev rational functions on [0,∞)

Up close, you can see the oscillation.

<img src="scripts/semiinf.png" width="50%">

Let's zoom out a bit to see convergence to 0 at ∞.

<img src="scripts/semiinf_birdseye.png" width="50%">

Derivatives die out faster.

<img src="scripts/semiinf_deriv.png" width="50%">

### Chebyshev rational functions on (-∞,∞)

Up close, you can see the oscillation.

<img src="scripts/inf.png" width="50%">

Let's zoom out a bit to see convergence at -∞ and ∞.

<img src="scripts/inf_birdseye.png" width="50%">

Derivatives die out slower than for the [0,∞) transformation.

<img src="scripts/inf_deriv.png" width="50%">
