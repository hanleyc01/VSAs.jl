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
