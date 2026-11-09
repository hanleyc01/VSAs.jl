#= Holographic Reduced Representations =#

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
    normal([rng::AbstractRNG,] ::Type{HRR{T}}, size::Integer)::HRR{T} where {T<:AbstractFloat}

Sample a new HRR from ``\\mathcal{N}(0, \\frac{1}{\\sqrt{D}})``, with elements
of type `T`. Passing plain `HRR` samples an `HRR{Float64}`.

### Notes

```jldoctest
julia> x = normal(Xoshiro(123), HRR, 512);

julia> length(x)
512

julia> normal(Xoshiro(123), HRR{Float32}, 512) isa HRR{Float32}
true
```
"""
normal(rng::AbstractRNG, ::Type{HRR}, size::Integer) = normal(rng, HRR{Float64}, size)

function normal(rng::AbstractRNG, ::Type{HRR{T}}, size::Integer)::HRR{T} where {T<:AbstractFloat}
    σ = one(T) / sqrt(T(size))
    HRR(randn(rng, T, size) .* σ)
end

"""
    unitary([rng::AbstractRNG,] ::Type{HRR{T}}, size::Integer)::HRR{T} where {T<:Union{Float32,Float64}}

Sample a unitary HRR with elements of type `T`, which makes the circular
correlation exact. Passing plain `HRR` samples an `HRR{Float64}`.

Only `Float32` and `Float64` are supported, because those are the only
element types FFTW handles.

### Notes

```jldoctest
julia> rng = Xoshiro(123);

julia> x, y = unitary(rng, HRR, 512), normal(rng, HRR, 512);

julia> similarity((x ⊗ y) ⨸ x, y) > 0.99
true
```
"""
unitary(rng::AbstractRNG, ::Type{HRR}, size::Integer) = unitary(rng, HRR{Float64}, size)

function unitary(rng::AbstractRNG, ::Type{HRR{T}}, size::Integer)::HRR{T} where {T<:Union{Float32,Float64}}
    spectrum = rfft(randn(rng, T, size))
    HRR(irfft(spectrum ./ abs.(spectrum), size))
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

julia> VSAs.bind(HRR(xarr), HRR(yarr)).data == VSAs.circular_convolution(xarr, yarr)
true

julia> (HRR(xarr) ⊗ HRR(yarr)).data == VSAs.circular_convolution(xarr, yarr)
true
```
"""
function bind(x::HRR, y::HRR)::HRR
    HRR(circular_convolution(x.data, y.data))
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

julia> VSAs.unbind(HRR(xarr), HRR(yarr)).data == VSAs.circular_correlation(xarr, yarr)
true

julia> (HRR(xarr) ⨸ HRR(yarr)).data == VSAs.circular_correlation(xarr, yarr)
true
```
"""
function unbind(x::HRR, y::HRR)::HRR
    HRR(circular_correlation(x.data, y.data))
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
Undo it with [`unpermute`](@ref unpermute) using the same `k`.

### Inputs

- `x` -- A vector
- `k` -- Optional shift value, defaults to `1`.

### Output

A circularly shifted HRR value.
"""
function permute(x::HRR, k::Integer=1)::HRR
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

    # Keep the result in the vectors' float type (e.g. Float32 stays Float32).
    T = float(promote_type(eltype(x.data), eltype(y.data)))
    norms = norm(x.data) * norm(y.data) + T(1.1e-8)
    dot(x.data, y.data) / norms
end
