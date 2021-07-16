module LinearMixingModels

using AbstractGPs
using Distributions
using KernelFunctions
using LinearAlgebra
using Random
using FillArrays

using AbstractGPs: AbstractGP, FiniteGP

include("independent_mogp.jl")
include("orthogonal_matrix.jl")
include("ilmm.jl")
include("oilmm.jl")

export ILMM
export IndependentMOGP, independent_mogp
export Orthogonal, OILMM

end
