#####
##### Smolyak implementation details
#####

"""
$(SIGNATURES) → accum, slack, remainders, states, levels

Initialize the state for [`__smolyak_step`](@ref), states are documented there.
"""
function __smolyak_init(family, kind, total::Int, f::F, itrs::NTuple{N,Any}) where {F,N}
    slack = total
    r = block_length(family, kind, 0)
    remainders = ntuple(_ -> r - 1, Val(N))
    itr_results = map(iterate, itrs)
    states = map(last, itr_results)
    levels = ntuple(_ -> 0, Val(N))
    accum = foldr(f, map(first, itr_results); init = ())
    accum, slack, remainders, states, levels
end

"""
$(SIGNATURES) → accum, Δ, remainders′, states′, levels′

Step through Smolyak indices of iterators.

`itrs` yield the `xs`. **The iterators should be stateless.**

`accum` is equivalent to `foldr(f, xs)`. `f` should map a value and a tuple to a tuple
of one more element.

`slack` is the `total - sum(levels)`. `Δ` is the change in `slack`. Design note: easier
to apply recursively than `slack`, caller should make the adjustment.

`remainders` contains the count of elements left in each level before we move to a
different combination.

`states` are states of iterators. `levels` are the levels currently visited.
"""
function __smolyak_step(family, kind, each::Int, f::F, itrs::NTuple{N,Any},
                        accum::NTuple{N,Any}, slack::Int, remainders::NTuple{N,Any},
                        states::NTuple{N,Any}, levels::NTuple{N,Any}) where {F,N}
    I1, Iτ... = itrs
    a1, aτ... = accum
    r1, rτ... = remainders
    s1, sτ... = states
    l1, lτ... = levels
    if r1 > 0                   # step within block
        x1, s1′ = iterate(I1, s1)
        (f(x1, aτ),
         0,                     # no change in slack
         (r1 - 1, rτ...),       # one less element in 1
         (s1′, sτ...),          # step iterator
         levels)
    elseif l1 < each && slack > 0 # next block, same tail
        x1, s1′ = iterate(I1, s1)
        (f(x1, aτ),
         -1,                    # decrease slack
         (block_length(family, kind, l1 + 1) - 1, rτ...), # remaining elements: all in block
         (s1′, sτ...),          # step state 1
         (l1+1, lτ...))         # next level
    elseif N == 1
        nothing                 # done with iteration
    else                        # go into tail
        next = __smolyak_step(family, kind, each, f, Iτ, aτ, slack + l1, rτ, sτ, lτ)
        next ≡ nothing && return nothing
        aτ′, Δτ, rτ′, sτ′, lτ′ = next
        x1, s1 = iterate(I1)
        (f(x1, aτ′),
         l1 + Δτ,                                     # more slack as we reset 1
         (block_length(family, kind, 0) - 1, rτ′...), # all remaining in block 0
         (s1, sτ′...),                                # states with tail
         (0, lτ′...))                                 # back to level 0 here
    end
end

####
#### index traversal
####

# function __inc_init(family, kind, ::Val{N}, total) where N
#     indices = ntuple(_ -> 1, Val(N))
#     levels = ntuple(_ -> 0, Val(N))
#     b0 = block_length(family, kind, 0)
#     limits = ntuple(_ -> b0, Val(N))
#     slack = total
#     slack, indices, levels, limits
# end

# """
# $(SIGNATURES) → valid, Δ, indices, levels, limits

# Internal implementation of the Smolyak indexing iterator.

# # Arguments

# - `slack`: `total - sum(blocks)`, cached

# - `indices`: current indices

# - `levels`: level along each dimension

# - `limits`: limit for each index (for column-major reset)

# # Return values

# - `valid::Bool`: `false` iff there is no next element, in which case the following values
#   should be ignored

# - `Δ::Int`: change in `slack`

# - `indices′`, `levels′, `limits′`: next values for corresponding arguments above, each an
#   `::NTuple{N,Int}`
# """
# @inline function __inc(family, kind, each::Int, # unchanged
#                        slack::Int, indices::NTuple{N,Int}, levels::NTuple{N,Int},
#                        limits::NTuple{N,Int}) where N
#     i1, iτ... = indices
#     l1, lτ... = levels
#     L1, Lτ... = limits
#     if i1 < L1                  # increment i1, same block
#         true, 0, (i1 + 1, iτ...), levels, limits
#     elseif l1 < each && slack > 0  # increment i1, next block
#         l1′ = l1 + 1
#         true, -1, (i1 + 1, iτ...), (l1′, lτ...), (grid_length(family, kind, l1′), Lτ...)
#     else
#         if N == 1               # end of iteration, arbitrary value since !valid
#             false, 0, indices, levels, limits
#         else                    # i1 = 1, increment tail if applicable
#             Δ1 = l1
#             valid, Δτ, iτ′, lτ′, Lτ′ = __inc(family, kind, each, slack + Δ1, iτ, lτ, Lτ)
#             valid, Δ1 + Δτ, (1, iτ′...), (0, lτ′...), (grid_length(family, kind, 0), Lτ′...)
#         end
#     end
# end

"""
$(SIGNATURES)

Calculate the length of a [`SmolyakIndices`](@ref) iterator. Argument as in the latter.
"""
function __smolyak_length(family, kind, N::Int, total::Int, each::Int)
    # implicit assumption: each ≤ total, enforced by the SmolyakParameters constructor
    _bl(b) = block_length(family, kind, b)
    c = zeros(Int, total + 1) # indexed as 0, …, total
    each = min(each, total)
    for b in 0:each
        c[b + 1] = _bl(b)
    end
    for n in 2:N
        for b in total:(-1):0            # blocks with indices that sum to b
            s = 0
            for a in 0:min(b, each)
                s += _bl(a) * c[b - a + 1]
            end
            # can safely overwrite since they will not be used again for n + 1
            c[b + 1] = s
        end
    end
    sum(c)
end

# """
# $(TYPEDEF)

# Indexing specification in a Smolyak basis/interpolation.

# # Type parameters

# - `N`: the dimension of indices

# - `H`: highest index visited for all dimensions

# - `B ≥ 0`: sum of block indices, starting from `0` (ie `B = 0` has just one element),

# - `M`: upper bound on each block index

# # Constructor

# Takes the dimension `N` as a parameter, `kind`, and a `SmolyakParameters` object,
# calculating everything else.

# # Details

# Consider positive integer indices `(i1, …, iN)`, each starting at one.

# Let `ℓ(b) = nesting_total_length(Chebyshev, grid_knid, kind, b)`, and `l1` denote the
# smallest integer such that `i1 ≤ ℓ(l1)`, and similarly for `i2, …, iN`. Extend this with
# `ℓ(-1) = 0` for the purposes of notation.

# An index `(i1, …, iN)` is visited iff all of the following hold:

# 1. `1 ≤ i1 ≤ ℓ(M)`, …, `1 ≤ iN ≤ ℓ(M)`,
# 2. `0 ≤ l1 ≤ M`, …, `1 ≤ bN ≤ M`,
# 3. `l1 + … + bN ≤ B`

# Visited indexes are in *column-major* order.
# """
# struct SmolyakIndices{N,H,B,M,Mp1}
#     "number of coefficients (cached)"
#     len::Int
#     "nesting total lengths (cached)"
#     nesting_total_lengths::NTuple{Mp1,Int}
#     function SmolyakIndices{N}(kind, smolyak_parameters::SmolyakParameters{B,M}) where {N,B,M}
#         @argcheck N ≥ 1
#         Mp1 = M + 1
#         len = __smolyak_length(kind, Val(N), Val(B), M)
#         first_block_length = nesting_total_length(Chebyshev, kind, 0)
#         nesting_total_lengths = ntuple(bp1 -> nesting_total_length(Chebyshev, kind, bp1 - 1),
#                                        Val(Mp1))
#         H = last(nesting_total_lengths)
#         new{N,H,B,M,Mp1}(len, nesting_total_lengths)
#     end
# end

# function Base.show(io::IO, smolyak_indices::SmolyakIndices{N,H,B,M}) where {N,H,B,M}
#     (; len) = smolyak_indices
#     print(io, "Smolyak indexing, ∑bᵢ ≤ $(B), all bᵢ ≤ $(M), dimension $(len)")
# end

# @inline highest_visited_index(::SmolyakIndices{N,H}) where {N,H} = H

# Base.eltype(::Type{<:SmolyakIndices{N}}) where N = NTuple{N,Int}

# @inline Base.length(ι::SmolyakIndices) = ι.len

# @inline function Base.iterate(ι::SmolyakIndices{N,H,B}) where {N,H,B}
#     slack, indices, blocks, limits = __inc_init(ι.nesting_total_lengths, Val(N), Val(B))
#     indices, (slack, indices, blocks, limits)
# end

# @inline function Base.iterate(ι::SmolyakIndices, (slack, indices, blocks, limits))
#     valid, Δ, indices′, blocks′, limits′ = __inc(ι.nesting_total_lengths, slack, indices,
#                                                  blocks, limits)
#     valid || return nothing
#     slack′ = slack + Δ
#     indices′, (slack′, indices′, blocks′, limits′)
# end

# ####
# #### product traversal
# ####

# struct SmolyakProduct{I<:SmolyakIndices,S<:Tuple,P}
#     smolyak_indices::I
#     sources::S
#     product_kind::P
#     @doc """
#     $(SIGNATURES)

#     An iterator conceptually equivalent to

#     ```
#     [prod(getindex.(sources, indices)) for indices in smolyak_indices]
#     ```

#     using [`_product`](@ref) instead to account for derivatives. Detailed docs of the
#     arguments are in [`SmolyakIndices`](@ref).

#     Caller should arrange the elements of `sources` in the correct order, see
#     [`nested_extrema_indices`](@ref). Each element in `sources` should have at least
#     `H` elements (cf type parameters of [`SmolyakIndices`](@ref)), this is not checked.
#     """
#     function SmolyakProduct(smolyak_indices::I, sources::S,
#                             product_kind::P) where {N,I<:SmolyakIndices{N},S,P}
#         @argcheck length(sources) == N
#         new{I,S,P}(smolyak_indices, sources, product_kind)
#     end
# end

# Base.length(smolyak_product::SmolyakProduct) = length(smolyak_product.smolyak_indices)

# function Base.eltype(::Type{SmolyakProduct{I,S,P}}) where {I,S,P}
#     _product_type(P, fieldtypes(S))
# end

# @inline function Base.iterate(smolyak_product::SmolyakProduct, state...)
#     (; smolyak_indices, sources, product_kind) = smolyak_product
#     itr = iterate(smolyak_indices, state...)
#     itr ≡ nothing && return nothing
#     indices, state′ = itr
#     _product(product_kind, map(getindex, sources, indices)), state′
# end
