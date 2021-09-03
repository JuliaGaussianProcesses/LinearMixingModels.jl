module LinearMixingModels

using AbstractGPs
using BlockDiagonals: BlockDiagonal
using ChainRulesCore: @non_differentiable
using Distributions
using FillArrays
using KernelFunctions
using LinearAlgebra
using PDMatsExtras
using Random
using Statistics

using AbstractGPs: AbstractGP, FiniteGP
using KernelFunctions: MOInputIsotopicByOutputs

include("independent_mogp.jl")
include("orthogonal_matrix.jl")
include("ilmm.jl")
include("oilmm.jl")

export ILMM
export IndependentMOGP, independent_mogp
export Orthogonal, OILMM
export get_latent_gp

end
