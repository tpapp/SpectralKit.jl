# SpectralKit.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/SpectralKit.jl/workflows/CI/badge.svg)](https://github.com/tpapp/SpectralKit.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/tpapp/SpectralKit.jl/graph/badge.svg?token=OrJPt0v6yz)](https://codecov.io/gh/tpapp/SpectralKit.jl)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/SpectralKit.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/SpectralKit.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20527005.svg)](https://doi.org/10.5281/zenodo.20527005)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Building blocks of spectral methods for Julia. Currently includes Chebyshev polynomials on univariate and Smolyak (multivariate) grids, with domain transformations to semi-infinite and infinite domains.

## Introduction

Mostly useful for algorithms along the lines of

> Boyd, John P. *Chebyshev and Fourier spectral methods*. 2001.

The aim is to provide simple, well-tested, robust, and fast *building blocks* for spectral algorithms, which can be easily combined into algorithms.

At the moment, the package API is experimental and subject to change.

## Help

Asking for help in [issues](https://github.com/tpapp/SpectralKit.jl/issues) is fine, you can also ping me as `@Tamas_Papp` on the [Discourse forum](https://discourse.julialang.org/)

## Bibliography

- Boyd, J. P. (2001). Chebyshev and fourier spectral methods. Courier Corporation.

- Xu, K. (2016). The chebyshev points of the first kind. Applied Numerical Mathematics, 102, 17–30.
