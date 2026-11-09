module VSAs

using Distributions
using Random
using FFTW
using LinearAlgebra

#= Interface =#

include("interface.jl")
export AbstractVSA
export ⊗, ⨸, ⊕, ⊖
export similarity, permute, unpermute
export normal, unitary
public bind, unbind, bundle, unbundle

#= Vector Utilities =#

include("utils.jl")
public circular_convolution, circular_correlation, involution

#= Generic Implementations =#

include("generics.jl")
export simplecleanup

#= Models =#

include("models/hrr.jl")
export HRR

end # module VSAs
