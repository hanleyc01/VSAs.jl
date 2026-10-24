module VSAs

using Distributions
using Random
using FFTW
using LinearAlgebra

export AbstractVSA, HRR, circular_convolution, circular_correlation, ⊗, ⨸, ⊕, ⊖, similarity, permute, normal, unpermute, unitary, simplecleanup
public bind, unbind, bundle, unbundle

#= Abstract Supertype =#

"""
    AbstractVSA{T}

Abstract type for all Vector-Symbolic Architectures.

### Notes

All subtypes of AbstractVSA must implement:
- [`bind`](@ref bind) -- An association of two vectors to create a third dissimilar vector
- [`unbind`](@ref unbind) -- A disassociation of a two bound vectors and one of the members to retrieve the other
- [`bundle`](@ref bundle) -- An association of two vectors to create a third similar vector
- [`similarity`](@ref similarity) -- A similarity function
- [`permute`](@ref permute) -- A one place permutation operator
"""
abstract type AbstractVSA{T} end

"""
    bind

An association of two vectors in a given VSA which produces a third vector
that is dissimilar to the two component vectors of which it is made up of.

``c = \\mathbf{x} \\otimes \\mathbf{y} ``

``|\\operatorname{sim}(c, \\mathbf x)| < \\theta_D``

``|\\operatorname{sim}(c, \\mathbf y)| < \\theta_D``

Where:
- ``\\mathbf x, \\mathbf y`` are to ``D``-dimensional vector symbols;
- ``\\operatorname{sim}`` is the [similarity](@ref similarity) function; and,
- ``\\theta_D`` is a threshold for a VSA given dimension ``D``.

"""
function bind end

"""
    unbind

Given an binding of two vector symbols, and a vector symbol component,
produce a vector which is similar to the other component.

``c = \\mathbf{x} \\otimes \\mathbf{y} ``

``a = c \\otimes^{-1} \\mathbf x``

``b = c \\otimes^{-1} \\mathbf y``

``\\operatorname{sim}(a, \\mathbf y) > \\theta_D``

``\\operatorname{sim}(b, \\mathbf x) > \\theta_D``

Where:
- ``\\mathbf x, \\mathbf y`` are to ``D``-dimensional vector symbols;
- ``\\operatorname{sim}`` is the [similarity](@ref similarity) function; and,
- ``\\theta_D`` is a threshold for a VSA given dimension ``D``.
"""
function unbind end

"""
    bundle

Associate two vector symbols so that the product of the association
is similar to the component parts.

``c = \\mathbf{x} \\oplus \\mathbf{y} ``

``\\operatorname{sim}(c, \\mathbf y) > \\theta_D``

``\\operatorname{sim}(c, \\mathbf x) > \\theta_D``

Where:
- ``\\mathbf x, \\mathbf y`` are to ``D``-dimensional vector symbols;
- ``\\operatorname{sim}`` is the [similarity](@ref similarity) function; and,
- ``\\theta_D`` is a threshold for a VSA given dimension ``D``.
"""
function bundle end

"""
    similarity

Compute the similarity between two vectors. Usually, when
``\\operatorname{sim}(x, y) > \\theta_D``, with ``\\theta_D`` being
some threshold value, then ``x`` is judged to be nearly equal to ``y``.
"""
function similarity end

"""
    permute

One place permutation of a vector symbol to produce a dissimilar vector symbol.
"""
function permute end

#= Vector Utilities =#

"""
    circular_convolution(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})

The circular convolution of two vectors.

### Inputs

- `x, y` -- Two vectors of the same size.

### Outputs

The circular convolution of two vectors.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.
"""
function circular_convolution(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Vector{<:AbstractFloat}
    length(x) == length(y) || throw(DimensionMismatch("vectors must have the same length"))
    irfft(rfft(x) .* rfft(y), length(x))
end

"""
    circular_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})

The circular correlation of two vectors.

### Inputs

- `x, y` -- Two vectors of the same size.

### Outputs

The circular correlation of two vectors.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.
"""
function circular_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})::Vector{<:AbstractFloat}
    length(x) == length(y) || throw(DimensionMismatch("vectors must have the same length"))
    irfft(rfft(x) .* conj.(rfft(y)), length(x))
end

"""
    involution(x::AbstractVector)::Vector

### Inputs

- `x`: Vector to involute


### Outputs

Involuted vector

### Throws

Throws `BoundsError` when `x` is empty.
"""
function involution(x::AbstractVector)::Vector
    x[[1; length(x):-1:2]]
end

#= HRRs =#

"""
    HRR{T<:Real} <: AbstractVSA{T}

Holographic Reduced Representations. Uses circular convolution
for binding, circular correlation for unbinding, element-wise addition
for superposition, and cosine similarity as the similarity function.

### Fields

- `data` -- `D`-dimensional real vector.

### Throws

Throws `ArgumentError` when the data vector provided is empty.
"""
struct HRR{T<:Real} <: AbstractVSA{T}
    data::Vector{T}

    function HRR(data::AbstractVector{T}) where {T<:Real}
        if isempty(data) throw(ArgumentError("data must be non-empty")) end
        new{T}(collect(data))
    end
end

"""
    HRR([rng::AbstractRNG,] dist::Sampleable, dim::Int)::HRR

Using the random number generator and distribution, sample a `dim`-sized
vector.

### Input

- `rng` -- Random number generator
- `dist` -- Any type that produces random values
- `size` -- Desired size the vector

### Output

Vector containing randomly sampled values from the provided distribution.

### Throws

Throws `ArgumentError` when `size` is less than `1`.
"""
function HRR(rng::AbstractRNG, dist::Sampleable, size::Int)::HRR
    data = rand(rng, dist, size)
    HRR(data)
end

function HRR(dist::Sampleable, size::Int)::HRR
    data = rand(dist, size)
    HRR(data)
end

"""
    normal([rng::AbstractRNG,] size::Int)::HRR

Sample a new vector from the ``\\mathcal{N}(0, \\frac{1}{\\sqrt{D}})``.
"""
function normal(rng::AbstractRNG, size::Int)::HRR
    dist = Normal(0, 1 / sqrt(size))
    HRR(rng, dist, size)
end

function normal(size::Int)::HRR
    dist = Normal(0, 1 / sqrt(size))
    HRR(dist, size)
end

function Base.length(x::HRR)::Int
    return length(x.data)
end

function Base.:(==)(x::HRR, y::HRR)::Bool
    x.data == y.data
end

function Base.show(io::IO, x::HRR)
    print(io, typeof(x), "(dim=", length(x), ")")
end

#= HRR Binding =#

"""
    bind(x::HRR, y:HRR)

Perform circular convolution on two vectors.

### Inputs

- `x, y` -- HRR vectors that share the same dimension.

### Output

Circular convolution of the arguments.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> xarr, yarr = rand(rng, Uniform(), 10), rand(rng, Uniform(), 10);

julia> VSAs.bind(HRR(xarr), HRR(yarr)).data == circular_convolution(xarr, yarr)
true

julia> (HRR(xarr) ⊗ HRR(yarr)).data == circular_convolution(xarr, yarr)
true
```
"""
function bind(x::HRR, y::HRR)::HRR
    HRR(circular_convolution(x.data, y.data))
end

function bind(xs::AbstractVector{<:HRR})::HRR
    reduce(⊗, xs)
end

"""
    unbind(x::HRR, y:HRR)

Perform circular correlation on two vectors.

### Inputs

- `x, y` -- HRR vectors that share the same dimension.

### Output

Circular correlation of the arguments.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> xarr, yarr = rand(rng, Uniform(), 10), rand(rng, Uniform(), 10);

julia> VSAs.unbind(HRR(xarr), HRR(yarr)).data == circular_correlation(xarr, yarr)
true

julia> (HRR(xarr) ⨸ HRR(yarr)).data == circular_correlation(xarr, yarr)
true
```
"""
function unbind(x::HRR, y::HRR)::HRR
    HRR(circular_correlation(x.data, y.data))
end

function unbind(xs::AbstractVector{<:AbstractVSA})::AbstractVSA
    reduce(⨸, xs)
end

#= HRR Bundling =#

"""
    bundle(x::HRR, y::HRR)::HRR

Superposition of two vectors.

### Inputs

- `x, y` -- HRR vectors that share the same dimension.

### Output

Superposition of the arguments.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> xarr, yarr = rand(rng, Uniform(), 10), rand(rng, Uniform(), 10);

julia> VSAs.bundle(HRR(xarr), HRR(yarr)).data == xarr .+ yarr
true

julia> (HRR(xarr) ⊕ HRR(yarr)).data == xarr .+ yarr
true
```
"""
function bundle(x::HRR, y::HRR)::HRR
    length(x) == length(y) || throw(DimensionMismatch("vectors must have the same length"))
    HRR(x.data .+ y.data)
end

function bundle(xs::AbstractVector{<:AbstractVSA})::AbstractVSA
    reduce(⊕, xs)
end

"""
    unbundle(x::HRR, y::HRR)::HRR

Element-wise subtraction of two vectors

### Inputs

- `x, y` -- HRR vectors that share the same dimension.

### Output

Element-wise difference of the arguments.

### Throws

Throws `DimensionMismatch` when `x` and `y` have different lengths.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> xarr, yarr = rand(rng, Uniform(), 10), rand(rng, Uniform(), 10);

julia> VSAs.unbundle(HRR(xarr), HRR(yarr)).data == xarr .- yarr
true

julia> (HRR(xarr) ⊖ HRR(yarr)).data == xarr .- yarr
true
```
"""
function unbundle(x::HRR, y::HRR)::HRR
    length(x) == length(y) || throw(DimensionMismatch("vectors must have the same length"))
    HRR(x.data .- y.data)
end

#= HRR Permutation =#

"""
    permute(x::HRR, k::Integer=1)::HRR

Circular shift of the vector to create a new uncorrelated vector by a set amount of shifts.

### Inputs

- `x` -- A vector
- `k` -- Optional shift value, defaults to `1`.

### Output

A circularly shifted HRR value.
"""
function permute(x::HRR, k::Integer=1)::HRR
    HRR(circshift(x.data, k))
end

"""
    unpermute(x::HRR, k::Integer=-1)::HRR

Circular shifting of a vector back to its original form by a set amount of shifts.

### Inputs

- `x` -- A vector
- `k` -- Optional shift value, defaults to `-1`.

### Notes

If one uses [`permute`](@ref permute), make sure that the `k` used for unpermuting
a vector is the additive inverse of the one used for permuting.
"""
function unpermute(x::HRR, k::Integer=-1)::HRR
    HRR(circshift(x.data, k))
end

#= HRR Similarity =#

"""
    similarity(x::HRR, y::HRR)::Real

Computes the cosine similarity between two vectors.

### Inputs

- `x, y` -- HRRs of the same dimension.

### Output

Value between `[-1, 1]`.

### Throws

`DimensionMismatch` when the vectors do not have the same length.

### Notes

```jldoctest
julia> d = 1400;

julia> theta = 4/sqrt(d);

julia> rng = Xoshiro(123);

julia> x, y = HRR(rand(rng, Normal(0, 1/sqrt(d)), d)), HRR(rand(rng, Normal(0, 1/sqrt(d)), d));

julia> similarity(x, x) > theta
true

julia> similarity(y, y) > theta
true

julia> abs(similarity(x, y)) < theta
true
```
"""
function similarity(x::HRR, y::HRR)::Real
    length(x) == length(y) || throw(DimensionMismatch("vectors must have the same length"))

    norms = norm(x.data) * norm(y.data) + 1.1e-8
    dot(x.data, y.data) / norms
end

#= Unitary Vectors =#

"""
    unitary([rng::AbstractRNG,] size::Integer)::HRR

Sample a unitary HRR vector, which makes the circular correlation exact.
"""
function unitary(rng::AbstractRNG, size::Integer)::HRR
    dist = Normal(0, 1)
    spectrum = rfft(rand(rng, dist, size))
    HRR(irfft(spectrum ./ abs(spectrum), size))
end

function unitary(size::Integer)::HRR
    dist = Normal(0, 1)
    spectrum = rfft(rand(dist, size))
    HRR(irfft(spectrum ./ abs(spectrum), size))
end

#= Cleanup Memory =#

"""
    simplecleanup(x::T, codebook::AbstractVector{T})::Vector{T} where {T<:Real}

Simple cleanup memory

### Inputs

- `x` -- A VSA vector symbol
- `codebook` -- A list of vector symbols to function as a codebook


### Output

Returns the list of similarities between `x` and the codebook values.

### Notes

The simplest way to retrieve the value is via [`argmax`](@ref argmax).
"""
function simplecleanup(x::T, codebook::AbstractVector{T})::Vector{Real} where {T<:AbstractVSA}
    sims = map(y -> similarity(x, y), codebook)
    sims
end

#= Constants =#

const ⊗ = bind
const ⨸ = unbind
const ⊕ = bundle
const ⊖ = unbundle

end # module VSAs
