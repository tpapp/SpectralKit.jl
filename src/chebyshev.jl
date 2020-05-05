####
#### Chebyshev on [-1,1]
####

"""
$(TYPEDEF)

Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev <: FunctionFamily end

@inline domain_extrema(::Chebyshev) = (-1, 1)

one_like(::Type{T}) where {T} = cos(zero(T))

"""
$(SIGNATURES)

Evaluate the `k`th Chebyshev polynomial at `-1` for the given orders.
"""
function chebyshev_min(::Type{T}, k::Integer, ::Order{0}) where {T}
    z = one_like(T)
    iseven(k) ? -z : z
end

"""
$(SIGNATURES)

Evaluate the `k`th Chebyshev polynomial at `1` for the given orders.
"""
chebyshev_max(::Type{T}, k::Integer, ::Order{0}) where {T} = one_like(T)

function chebyshev_min(::Type{T}, k::Integer, ::Order{1}) where {T}
    z = one_like(T) * abs2(k - 1)
    isodd(k) ? -z : z
end

chebyshev_max(::Type{T}, k::Integer, ::Order{1}) where {T} = one_like(T) * abs2(k - 1)

function chebyshev_min(::Type{T}, k::Integer, ::OrdersTo{1}) where {T}
    SVector(chebyshev_min(T, k, Order(0)), chebyshev_min(T, k, Order(1)))
end

function chebyshev_max(::Type{T}, k::Integer, ::OrdersTo{1}) where {T}
    SVector(chebyshev_max(T, k, Order(0)), chebyshev_max(T, k, Order(1)))
end

"""
$(SIGNATURES)

Evaluate the `k`th Chebyshev polynomial at `-1 < x < 1` for the given orders.
"""
chebyshev_interior(k::Integer, x::Real, ::Order{0}) = cos((k - 1) * acos(x))

function chebyshev_interior(k::Integer, x::Real, ::Order{1})
    n = k - 1
    t = acos(x)
    n * sin(n * t) / sin(t)
end

function chebyshev_interior(k::Integer, x::Real, ::OrdersTo{1})
    n = k - 1
    t = acos(x)
    t′ = 1 / sin(t)
    s, c = sincos(n * t)
    SVector(c, s * n * t′)
end

####
#### basis function iterator
####

struct ChebyshevIterator{T,O}
    x::T
    order::O
end

Base.IteratorEltype(::Type{<:ChebyshevIterator}) = Base.HasEltype()

Base.eltype(::Type{<:ChebyshevIterator{T,<:Order}}) where {T} = T

Base.eltype(::Type{<:ChebyshevIterator{T,<:OrdersTo{K}}}) where {T,K} = SVector{K+1,T}

Base.IteratorSize(::Type{<:ChebyshevIterator}) = Base.IsInfinite()

function basis_function(::Chebyshev, x::Real, order)
    ChebyshevIterator(x, order)
end

function Base.iterate(itr::ChebyshevIterator)
    @unpack x = itr
    one(x), (one(x), x)
end

function Base.iterate(itr::ChebyshevIterator, state)
    @unpack x = itr
    fp, fpp = state
    f = 2 * x * fp - fpp
    f, (f, fp)
end

function Base.iterate(itr::ChebyshevIterator{T,O}) where {T,O<:Union{Order{1},OrdersTo{1}}}
    @unpack x, order = itr
    fp = SVector(one(x), zero(x))
    fpp = SVector(x, one(x))
    O ≡ Order{1} ? last(fp) : fp, (fp, fpp)
end

function Base.iterate(itr::ChebyshevIterator{T,O},
                      state) where {T,O<:Union{Order{1},OrdersTo{1}}}
    @unpack x = itr
    fp, fpp = state
    f = SVector(2 * x * fp[1] - fpp[1], 2 * fp[1] + 2 * x * fp[2] - fpp[2])
    O ≡ Order{1} ? last(f) : f, (f, fp)
end

function basis_function(family::Chebyshev, x::T, order, k::Integer) where {T <: Real}
    @argcheck k > 0
    if x == -1
        chebyshev_min(T, k, order)
    elseif x == 1
        chebyshev_max(T, k, order)
    else
        chebyshev_interior(k, x, order)
    end
end

function basis_function(family::Chebyshev, x::T, ::Order{0}, ::Val{K}) where {K, T <: Real}
    # FIXME this is a somewhat naive implementation, could be improved (eg unrolling for
    # small K, etc)
    @argcheck K ≥ 0
    z = MVector{K,T}(undef)
    if K ≥ 1
        z[1] = one(x)
    end
    if K ≥ 2
        z[2] = x
    end
    for k in 3:K
        z[k] = 2 * x * z[k - 1] - z[k - 2]
    end
    SVector{K,T}(z)
end

function basis_function(family::Chebyshev, x::T, ::OrdersTo{1},
                        ::Val{K}) where {K, T <: Real}
    @argcheck K ≥ 0
    S = SVector{2,T}
    z = MVector{K,S}(undef)
    if K ≥ 1
        z[1] = SVector(one(x), zero(x))
    end
    if K ≥ 2
        z[2] = SVector(x, one(x))
    end
    for k in 3:K
        p0, p1 = z[k - 1]
        pp0, pp1 = z[k - 2]
        z[k] = SVector(2 * x * p0 - pp0, 2 * p0 + 2 * x * p1 - pp1)
    end
    SVector{K,S}(z)
end

function basis_function(family::Chebyshev, x::Real, ::Order{1}, k::Val)
    # FIXME we punt here, should be possible using recursion directly
    map(last, basis_function(family, x, OrdersTo(1), k))
end

function roots(::Type{T}, ::Chebyshev, N) where {T <: Real}
    cospi.(((2 * N - 1):-2:1) ./ T(2 * N))
end

function augmented_extrema(::Type{T}, family::Chebyshev, N) where {T}
    cospi.(((N-1):-1:0) ./ T(N - 1))
end
