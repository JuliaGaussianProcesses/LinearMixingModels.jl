module LinearMixingModels

using AbstractGPs
using BlockDiagonals: BlockDiagonal
using ChainRulesCore: @non_differentiable
using Distributions
using KernelFunctions
using LinearAlgebra
using Random
using Statistics
using FillArrays

using AbstractGPs: AbstractGP, FiniteGP
using KernelFunctions: MOInputIsotopicByOutputs, MOInputIsotopicByFeatures

include("independent_mogp.jl")
include("iid_mogp.jl")
include("orthogonal_matrix.jl")
include("ilmm.jl")
include("oilmm.jl")

export ILMM
export IndependentMOGP, independent_mogp
export Orthogonal, OILMM
export get_latent_gp

end
