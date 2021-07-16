"""
    ILMM(fs, H, D)
An Instantaneous Linear Mixing Model (ILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then
# Arguments:
- fs: a length-`m` vector of Gaussian process objects from AbstractGPs.jl.
- H: a `p x m` matrix representing a fixed basis of our p-dim target: h_1,...,h_m
- D: an `m x m` `Diagonal` matrix, variance of noise on each latent process. Positive entries.

"""
struct ILMM{
    Tfs<:AbstractGP,
    TH<:AbstractMatrix{<:Real},
    TD<:Diagonal{<:Real},
} <: AbstractGP
    f::Tfs
    H::TH
    D::TD
end

# Σ := pxp heterogeneous noise deriving from latent processes
function get_sigma(H::AbstractMatrix{<:Real}, σ², D::Diagonal{<:Real},)
    Σ = σ²I + H*D*H'
    return Σ
end

# Implement AbstractGPs API, for ILMM and OILMM
# posterior, rand, marginals, logpdf, mean, var, cov, mean_and_var,

noise_var(Σ::Diagonal{<:Real, <:Fill}) = FillArrays.getindex_value(Σ.diag)

reshape_y(y::AbstractVector{<:Real}, N::Int) = ColVecs(reshape(y, N, :)')
