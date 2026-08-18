####
#### Chebyshev polynomials on [-1,1]
####

export Chebyshev, Endpoints, Interior, univariate_basis

####
#### generic building blocks
####

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev end

####
#### basis function iterator
####

struct ChebyshevIterator{T}
    x::T
end

Base.eltype(::Type{<:ChebyshevIterator{T}}) where {T} = T

Base.IteratorSize(::Type{<:ChebyshevIterator}) = Base.IsInfinite()

function Base.iterate(itr::ChebyshevIterator{T}) where T
    (; x) = itr
    _one(T), (_one(T), x)
end

function Base.iterate(itr::ChebyshevIterator{T}, (fp, fpp)) where T
    (; x) = itr
    f = _sub(_mul(2, x, fp), fpp)
    f::T, (f, fp)
end

"""
$(SIGNATURES)

Helper function to calculate the extrema of the `N`th Chebyshev polynomial, indexed by
`1 ≤ i ≤ N` (not checked). `N==1` (the constant) is special cases to zero, for nesting.
"""
function _chebyshev_extremum(::Type{T}, i::Int, N::Int) where {T <: Real}
    if N == 1
        cospi(1/T(2))::T        # 0.0 as a practical fallback
    else
        cospi((N - i) ./ T(N - 1))::T
    end
end

struct ChebyshevShuffle
    N::Int
    endpoints::Bool
    ChebyshevShuffle(N::Int; endpoints) = new(N, endpoints)
end

function Base.length(shuffle::ChebyshevShuffle)
    (; N, endpoints) = shuffle
    endpoints ? N : max(1, N - 2)
end

Base.eltype(::Type{ChebyshevShuffle}) = Int

function Base.iterate(shuffle::ChebyshevShuffle, state = (0, -1))
    (; N, endpoints) = shuffle
    i, step = state
    if step == -1               # sentinel for first element
        i = (N + 1) ÷ 2
        if endpoints            # go to 1
            i′ = 1
            step′ = N - 1
        else                    # skip endpoints, go to next layer
            step′ = (N - 1) ÷ 2
            i′ = step′ ÷ 2 + 1
        end
        i, (i′, step′)
    elseif step ≤ 1             # N = 1, iteration is done
        nothing
    else
        i′ = i + step
        if i′ > N               # overrun, halve step and back
            step = step ÷ 2
            i′ = step ÷ 2 + 1
        end
        i, (i′, step)
    end
end

####
#### grids
####

"""
$(TYPEDEF)

Chebyshev-Lobatto grid. The extrema of Chebyshev polynomials, including endpoints of `[-1,1]`.

!!! note
    For small dimensions may fall back to a grid that does not contain endpoints.
"""
struct Endpoints end

function grid_length(::Chebyshev, ::Endpoints, level::Int)
    @argcheck level ≥ 1
    (level ≤ 2 ? (level - 1) * 2 : (1 << (level - 1))) + 1
end

function block_length(::Chebyshev, ::Endpoints, level::Int)
    @argcheck level ≥ 1
    level ≤ 2 ? level : 1 << (level - 2)
end

"""
$(TYPEDEF)

Like [`Endpoints`](@ref), but with endpoints dropped.
"""
struct Interior end

function grid_length(::Chebyshev, ::Interior, level::Int)
    @argcheck level ≥ 1
    (1 << level) - 1
end

function block_length(::Chebyshev, ::Interior, level::Int)
    @argcheck level ≥ 1
    1 << (level - 1)
end

"""
Implementation of univariate bases. Not part of the API.
"""
@concrete struct UnivariateBasis <: FunctionBasis
    family
    kind
    domain_transformation
    level
end

"""
$(SIGNATURES)

Univariate basis from `family`, using the given `kind`.

`level` is an integer, starting from `1`, specifying the number of *blocks* used to
build the grid, which in turn determine

"""
function univariate_basis(family, kind, domain_transformation, level)
    @argcheck level ≥ 1
    UnivariateBasis(family, kind, domain_transformation, level)
end

domain(U::UnivariateBasis) = domain(U.domain_transformation)

dimension(U::UnivariateBasis) = grid_length(U.family, U.kind, U.level)

function basis_at(U::UnivariateBasis{Chebyshev}, x::Scalar)
    Iterators.take(ChebyshevIterator(transform_to(PM1(), U.domain_transformation, x)), dimension(U))
end

function grid(::Type{T},
              U::UnivariateBasis{Chebyshev,K}) where {T <: AbstractFloat,
                                                      K <: Union{Interior,Endpoints}}
    (; family, kind, domain_transformation, level) = U
    N = grid_length(family, kind, level)
    if K ≡ Interior
        N += 2                  # account for dropped endpoints
        endpoints = false
    else
        endpoints = true
    end
    (transform_from(PM1(), U.domain_transformation, _chebyshev_extremum(T, i, N))
     for i in ChebyshevShuffle(N; endpoints))
end

function adjust_basis(U::UnivariateBasis, Δ::Int)
    level′ = U.level + Δ
    if level′ > 0
        UnivariateBasis(U.family, U.kind, U.domain_transformation, level′)
    else
        nothing
    end
end

function adjust_coefficients(θ1::AbstractVector, U1::UnivariateBasis{Chebyshev}, U2::UnivariateBasis{Chebyshev})
    d1 = dimension(U1)
    d2 = dimension(U2)
    @argcheck length(θ1) == d1 "coefficients are not compatible with the first basis"
    @argcheck(U1.domain_transformation == U2.domain_transformation,
              "incompatible domain transformations")
    if d2 ≤ d1                  # truncate
        θ1[1:d2]
    else                        # pad with zeros
        θ2 = zeros(d2)
        θ2[1:d1] .= θ1
        θ2
    end
end
