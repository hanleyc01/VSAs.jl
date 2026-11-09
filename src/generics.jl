#= Generic method implementations for VSA =#

Base.length(x::AbstractVSA) = length(x.data)

Base.:(==)(x::AbstractVSA, y::AbstractVSA) = typeof(x) == typeof(y) && x.data == y.data

Base.show(io::IO, x::AbstractVSA) = print(io, typeof(x), "(dim=", length(x), ")")

unpermute(x::AbstractVSA, k::Integer=1) = permute(x, -k)

bind(xs::AbstractVector{<:AbstractVSA}) = reduce(bind, xs)

bundle(xs::AbstractVector{<:AbstractVSA}) = reduce(bundle, xs)

unbind(xs::AbstractVector{<:AbstractVSA}) = reduce(unbind, xs)

normal(::Type{T}, size::Integer) where {T<:AbstractVSA} = normal(Random.default_rng(), T, size)

unitary(::Type{T}, size::Integer) where {T<:AbstractVSA} = unitary(Random.default_rng(), T, size)

#= Cleanup Memory =#

"""
    simplecleanup(x::T, codebook::AbstractVector{T})::Vector{Real} where {T<:AbstractVSA}

Simple cleanup memory

### Inputs

- `x` -- A VSA vector symbol
- `codebook` -- A list of vector symbols to function as a codebook


### Output

Returns the list of similarities between `x` and the codebook values.

### Notes

The simplest way to retrieve the value is via `argmax`.
"""
function simplecleanup(x::T, codebook::AbstractVector{T})::Vector{Real} where {T<:AbstractVSA}
    sims = map(y -> similarity(x, y), codebook)
    sims
end
