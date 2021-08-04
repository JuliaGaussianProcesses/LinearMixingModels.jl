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
    Tf<:AbstractGP,
    TH<:AbstractMatrix{<:Real}
} <: AbstractGP
    f::Tf
    H::TH
end

# Get the latent space multi-output GP belonging to the ILMM
get_latent_gp(gp::ILMM) = gp.f

function unpack(fx::FiniteGP{<:ILMM, <:MOInput})
    f = fx.f.f
    H = fx.f.H
    σ² = noise_var(fx.Σy)
    x = fx.x.x

    # Check that the number of outputs requested agrees with the model.
    fx.x.out_dim == size(H, 1) || throw(error("out dim of x != out dim of f."))
    return f, H, σ², x
end

# Implement AbstractGPs API, for ILMM
# posterior, rand, marginals, logpdf, mean, var, cov, mean_and_var,

function project(
    H::AbstractMatrix{T},
    Y::ColVecs{T},
    σ²::T,
) where {T<:Real}

    Σ = σ²*I

    # Compute the projected noise, which is a matrix of size `size(Yproj)`.
    ΣT = inv(H' * inv(Σ) * H)

    # Compute the projection of the data.
    Yproj = ΣT * H' * inv(Σ) * Y.X

    return Yproj, ΣT
end

function project(
    H::AbstractMatrix{T},
    Y::RowVecs{T},
    σ²::T,
    D::Diagonal{T},
) where {T<:Real}
    return project(H, ColVecs(Y.X'), σ²)
end

# Compute the regularisation term in the log marginal likelihood. See e.g. appendix A.4.
function regulariser(fx, Y::ColVecs{<:Real})
    fs, H, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    Yproj, ΣT = project(H, Y, σ²)

    Σ = σ²I

    n = length(Y)
    p, m = size(H)
    Ip = Matrix(I, p, p)

    # Wrong norm?
    return -((p - m) * log(2π) + log(det(Σ) / det(ΣT)) +
        sum((y.X .- T'*y.X)' * inv(Σ) * (y.X .- T'*y.X))) / 2
end

function regulariser(fx, Y::RowVecs{<:Real})
    return regulariser(fx, ColVecs(Y.X'))
end

"""
    rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM})
Sample from the ILMM, including the observation noise.
Follows generative structure of model 2 from [1].
Follows the AbstractGPs.jl API.
See also `rand_latent`.
[1] - Bruinsma et al 2020.
"""
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM})
    f_latent, H, σ², x = unpack(fx)

    x_mo_input = MOInput(x, size(H,2))

    return rand(rng, f_latent(x_mo_input))
    # Generate iid noise and add to each output.
    # return F .+ sqrt(noise_var(fx.Σy)) .* randn(rng, size(F))
end

"""
    denoised_marginals(fx::FiniteGP{<:OILMM})
Returns the marginal distribution over the OILMM without the IID noise components.
See also `marginals`.
"""
function marginals(fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    marginals = marginals(fx)
    M_latent = mean(marginals)
    V_latent = var(marginals)

    # Compute the means.
    M = H * M_latent

    # Compute the variances.
    V = abs2.(H) * V_latent

    # Package everything into independent Normal distributions.
    return Normal.(vec(M'), sqrt.(vec(V')))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:ILMM})
    f_latent, H, σ², x = unpack(fx)

    x_mo_input = MOInput(x, size(H,2))

    return mean_and_var(f_latent(x_mo_input))
end

AbstractGPs.mean(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[1]

AbstractGPs.var(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[2]

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    fs, H, σ², x = unpack(fx)

    # # Projection step.
    Y = reshape_y(y, length(x))
    Ty, ΣT = project(H, Y, σ²)
    Xproj, Yproj = prepare_isotopic_multi_output_data(x.x, Y)
    ΣT = BlockDiagonal([ΣT for _ in 1:m])

    return logpdf(fx.f(Xproj, ΣT), Yproj) + regulariser(fx, Y)
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    f, H, σ², x = unpack(fx)
    p, m = size(H)

    # # Projection step.
    Y = reshape_y(y, length(x))
    Ty, ΣT = project(H, Y, σ²)
    Xproj, Yproj = prepare_isotopic_multi_output_data(x.x, Y)
    ΣT = BlockDiagonal([ΣT for _ in 1:m])

    # # Condition each latent process on the projected observations.
    # y_rows = collect(eachrow(Yproj))
    # ΣT_rows = collect(eachrow(ΣT))
    f_posterior = posterior(fx.f(Xproj, ΣT), Yproj)
    return ILMM(f_posterior, H)
end


# Helper functions
noise_var(Σ::Diagonal{<:Real, <:Fill}) = FillArrays.getindex_value(Σ.diag)

reshape_y(y::AbstractVector{<:Real}, N::Int) = ColVecs(reshape(y, N, :)')
