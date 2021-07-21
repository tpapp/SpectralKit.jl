#####
##### Block sizes and shuffling
#####

## NOTE
## Smolyak grids use “blocks” of polynomials, each indexed by ``b == 0, …, B`.
## Currently block calculations are provided for `InteriorGrid`, FIXME `EndpointGrid` should
## be added.

"""
$(SIGNATURES)

Length of each block `b`.
"""
block_length(kind::InteriorGrid, b::Int) = b ≤ 1 ? b + 1 : 2^(b - 1)

"""
$(SIGNATURES)

Cumulative block length at block `b`.
"""
cumulative_block_length(kind::InteriorGrid, b::Int) =  b == 0 ? 1 : (2^b + 1)

struct InteriorGridShuffle
    len::Int
end

Base.length(ι::InteriorGridShuffle) = ι.len

Base.eltype(::Type{InteriorGridShuffle}) = Int

function Base.iterate(ι::InteriorGridShuffle)
    @unpack len = ι
    i = (len + 1) ÷ 2
    i, (0, 0)                   # step = 0 is special-cased
end

function Base.iterate(ι::InteriorGridShuffle, (i, step))
    @unpack len = ι
    i == 0 && return len > 1 ? (1, (1, len - 1)) : nothing
    i′ = i + step
    if i′ ≤ len
        i′, (i′, step)
    else
        step′ = step ÷ 2
        if step′ ≥ 2
            i′ = step′ ÷ 2 + 1
            i′, (i′, step′)
        else
            nothing
        end
    end
end

"""
$(SIGNATURES)

Return an iterator of indices for picking elements from a grid of length `H`, which should
be a valid cumulative block length.
"""
function block_shuffle(kind::InteriorGrid, H::Int)
    @argcheck H ≥ 0
    InteriorGridShuffle(H)
end
