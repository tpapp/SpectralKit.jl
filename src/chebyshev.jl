####
#### Chebyshev polynomials on [-1,1]
####

export Chebyshev

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev{K<:AbstractGrid} <: FunctionBasis
    "Grid specification."
    grid_kind::K
    "The number of basis functions."
    N::Int
    @doc """
    $(SIGNATURES)
    `N` Chebyshev polynomials (of the first kind) on ``[-1, 1]``, with the associated
    grid of `grid_kind`.

    # Example

    ```julia
    basis = Chebyshev(InteriorGrid(), 10)
    ```
    """
    function Chebyshev(grid_kind::K, N::Int) where K
        @argcheck N ≥ 1
        new{K}(grid_kind, N)
    end
end

@inline dimension(basis::Chebyshev) = basis.N

@inline domain(::Chebyshev) = (-1, 1)

function Base.show(io::IO, chebyshev::Chebyshev)
    @unpack grid_kind, N = chebyshev
    print(io, "Chebyshev polynomials (1st kind), ", grid_kind, ", dimension: ", N)
end

####
#### basis function iterator
####

struct ChebyshevIterator{T}
    x::T
    N::Int
end

Base.eltype(::Type{<:ChebyshevIterator{T}}) where {T} = T

Base.length(itr::ChebyshevIterator) = itr.N

function basis_at(basis::Chebyshev, x::Scalar)
    ChebyshevIterator(x, basis.N)
end

function Base.iterate(itr::ChebyshevIterator)
    @unpack x = itr
    _one(x), (2, _one(x), x)
end

function Base.iterate(itr::ChebyshevIterator{T}, (i, fp, fpp)) where T
    @unpack x, N = itr
    i > N && return nothing
    f = _sub(_mul(2, x, fp), fpp)
    f::T, (i + 1, f, fp)
end

####
#### grids
####

"""
$(SIGNATURES)

Return a gridpoint for collocation, with `1 ≤ i ≤ dimension(basis)`.

`T` is used *as a hint* for the element type of grid coordinates, and defaults to `Float64`.
The actual type can be broadened as required. Methods are type stable.

!!! note
    Not all grids have this method defined, especially if it is impractical. See
    [`grid`](@ref), which is part of the API, this function isn't.
"""
function gridpoint(::Type{T}, basis::Chebyshev{InteriorGrid}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N                  # FIXME use boundscheck
    sinpi((N - 2 * i + 1) / T(2 * N))::T # use formula Xu (2016)
end

function gridpoint(::Type{T}, basis::Chebyshev{EndpointGrid}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    if N == 1
        cospi(1/T(2))::T        # 0.0 as a fallback, even though it does not have endpoints
    else
        cospi((N - i) ./ T(N - 1))::T
    end
end

function gridpoint(::Type{T}, basis::Chebyshev{InteriorGrid2}, i::Integer) where {T <: Real}
    @unpack N = basis
    @argcheck 1 ≤ i ≤ N         # FIXME use boundscheck
    cospi(((N - i + 1) ./ T(N + 1)))::T
end

struct ChebyshevGridIterator{T,B}
    basis::B
end

Base.eltype(::Type{<:ChebyshevGridIterator{T}}) where {T} = T

Base.length(itr::ChebyshevGridIterator) = dimension(itr.basis)

function Base.iterate(itr::ChebyshevGridIterator{T}, i = 1) where {T}
    @unpack basis = itr
    if i ≤ dimension(basis)
        gridpoint(T, basis, i), i + 1
    else
        nothing
    end
end

grid(::Type{T}, basis::B) where {T<:Real,B<:Chebyshev} = ChebyshevGridIterator{T,B}(basis)

####
#### augmenting
####

function augment_coefficients(basis1::Chebyshev{K1}, basis2::Chebyshev{K2},
                              θ1::AbstractVector) where {K1,K2}
    @argcheck is_subset_basis(basis1, basis2)
    @argcheck length(θ1) == dimension(basis1)
    vcat(θ1, zeros(eltype(θ1), basis2.N - basis1.N))
end

function is_subset_basis(basis1::Chebyshev{K1}, basis2::Chebyshev{K2}) where {K1,K2}
    K1 == K2 && basis2.N ≥ basis1.N
end
