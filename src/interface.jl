#= interface.jl - the contract every VSA model implements =#

#= Abstract Sypertype =#

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
    unbundle

Remove a vector symbol from a bundle, so that the result is no longer
similar to the removed component.

``c = (\\mathbf{x} \\oplus \\mathbf{y}) \\ominus \\mathbf y``

``\\operatorname{sim}(c, \\mathbf x) > \\theta_D``
"""
function unbundle end

"""
    permute

One place permutation of a vector symbol to produce a dissimilar vector symbol.
"""
function permute end

"""
    unpermute(x, k=1)

Inverse of [`permute`](@ref permute): `unpermute(permute(x, k), k) == x`.
"""
function unpermute end

#= Sampling =#

"""
    normal([rng::AbstractRNG,] T::Type{<:AbstractVSA}, size::Integer)

Sample a random vector symbol of model `T` with `size` dimensions, using the
model's standard random distribution.

Models implement the `rng` form, `normal(rng, ::Type{T}, size)`. The form
without `rng` falls back to `Random.default_rng()`.
"""
function normal end

"""
    unitary([rng::AbstractRNG,] T::Type{<:AbstractVSA}, size::Integer)

Sample a unitary vector symbol of model `T`, for which unbinding is the exact
inverse of binding.

Models implement the `rng` form, `unitary(rng, ::Type{T}, size)`. The form
without `rng` falls back to `Random.default_rng()`.
"""
function unitary end

#= Operator aliases =#

const ⊗ = bind
const ⨸ = unbind
const ⊕ = bundle
const ⊖ = unbundle