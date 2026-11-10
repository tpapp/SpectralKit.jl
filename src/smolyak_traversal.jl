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
    itr_results = map(_start, itrs)
    states = map(last, itr_results)
    levels = ntuple(_ -> 0, Val(N))
    accum = foldr(f, map(first, itr_results); init = ())
    accum, slack, remainders, states, levels
end

"""
Sentinel value for having finished iteration in `__smolyak_step`.
"""
const Δ_DONE = typemax(Int)

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
        x1, s1′ = _next(I1, s1)
        (f(x1, aτ),
         0,                     # no change in slack
         (r1 - 1, rτ...),       # one less element in 1
         (s1′, sτ...),          # step iterator
         levels)
    elseif l1 < each && slack > 0 # next block, same tail
        x1, s1′ = _next(I1, s1)
        (f(x1, aτ),
         -1,                    # decrease slack
         (block_length(family, kind, l1 + 1) - 1, rτ...), # remaining elements: all in block
         (s1′, sτ...),          # step state 1
         (l1+1, lτ...))         # next level
    elseif N == 1
        accum, Δ_DONE, remainders, states, levels # done with iteration
    else                        # go into tail
        aτ′, Δτ, rτ′, sτ′, lτ′ = __smolyak_step(family, kind, each, f, Iτ, aτ, slack + l1, rτ, sτ, lτ)
        if Δτ == Δ_DONE
            accum, Δ_DONE, remainders, states, levels # tail is done with iteration
        else
            x1, s1 = _start(I1)
            (f(x1, aτ′),
             l1 + Δτ,                                     # more slack as we reset 1
             (block_length(family, kind, 0) - 1, rτ′...), # all remaining in block 0
             (s1, sτ′...),                                # states with tail
             (0, lτ′...))                                 # back to level 0 here
        end
    end
end

struct NonIncreasingSmolyakLevels{N}
    total::Int
    each::Int
    @doc """
    $(SIGNATURES) → itr

    An iterable which yields `ℓ::Ntuple{N,Int}`, with the following properties:

    1. each tuple `ℓ` is non-increasing (weakly decreasing),
    2. each element of each tuple is between `0` and `each` (inclusive),
    3. the sum of all elements in each tuple is not larger than total.
    """
    function NonIncreasingSmolyakLevels{N}(total, each) where N
        @argcheck N isa Integer && N ≥ 1
        @argcheck each ≤ total
        new{N}(total, each)
    end
end

Base.eltype(::Type{NonIncreasingSmolyakLevels{N}}) where N = NTuple{N,Int}

Base.IteratorSize(::Type{<:NonIncreasingSmolyakLevels}) = Base.SizeUnknown()

function __step_noninc(total::Int, each::Int, Σ_and_indices::Int...)
    Σ, i1, iτ... = Σ_and_indices
    if Σ < total && isempty(iτ)
        # single index, increment if possible
        (i1 + 1, i1 + 1)
    elseif Σ < total && i1 > last(iτ)
        # room to increment indices in the tail
        Σ′, iτ′... = __step_noninc(total - i1, i1, Σ - i1, iτ...)
        (i1 + Σ′, i1, iτ′...)
    elseif !isempty(iτ) && first(iτ) < min(i1, each) && first(iτ) + i1 < total
        # can increment next index, zero out tail of tail
        i2 = first(iτ)
        (i1 + i2 + 1, i1, i2 + 1, ntuple(_ -> 0, Val(length(iτ) - 1))...)
    else
        # increment first index; stop iteration if this is > each
        (i1 + 1, i1 + 1, ntuple(_ -> 0, Val(length(iτ)))...)
    end
end

function Base.iterate(itr::NonIncreasingSmolyakLevels{N}, state = nothing) where N
    (; total, each) = itr
    if state ≡ nothing
        indices = ntuple(_ -> 0, N)
        indices, (0, indices...)
    else
        state′ = __step_noninc(total, each, state...)
        Σ′, indices′... = state′
        if first(indices′) ≤ each
            indices′, state′
        else
            nothing
        end
    end
end

"""
$(SIGNATURES)

Calculate the dimension of Smolyak basis.
"""
function __smolyak_length(family, kind, ::Val{N}, total::Int, each::Int) where N
    L = 0
    P = factorial(N)            # permutations
    for ℓ in NonIncreasingSmolyakLevels{N}(total, each)
        p = P                   # combinations, accounting for repetitions
        C = 1                   # will contain product of block lengths after loop below
        l_prev = -1             # previous value (sentinel)
        r = 1                   # run counter for repeated values
        for l in ℓ
            C *= block_length(family, kind, l)
            if l == l_prev
                r += 1
                p ÷= r          # account for repetitions
            else
                l_prev = l
                r = 1           # reset counter
            end
        end
        L += p * C
    end
    L
end
