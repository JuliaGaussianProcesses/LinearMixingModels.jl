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
    TH<:AbstractMatrix{<:Real},
    TD<:Diagonal{<:Real},
} <: AbstractGP
    f::Tf
    H::TH
    D::TD
end

# Σ := pxp heterogeneous noise deriving from latent processes
function get_sigma(H::AbstractMatrix{<:Real}, σ², D::Diagonal{<:Real},)
    Σ = σ²I + H*D*H'
    return Σ
end

function unpack(fx::FiniteGP{<:ILMM, <:MOInput, <:Diagonal{<:Real, <:Fill}})
    fs = fx.f.f.fs
    H = fx.f.H
    D = fx.f.D
    σ² = noise_var(fx.Σy)
    x = fx.x.x

    # Check that the number of outputs requested agrees with the model.
    fx.x.out_dim == size(H, 1) || throw(error("out dim of x != out dim of f."))
    return fs, H, D, σ², x
end

# Implement AbstractGPs API, for ILMM
# posterior, rand, marginals, logpdf, mean, var, cov, mean_and_var,

function project(
    H::AbstractMatrix{T},
    Y::ColVecs{T},
    σ²::T,
    D::Diagonal{T},
) where {T<:Real}

    Σ = get_sigma(H, σ², D)

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
    return project(H, ColVecs(Y.X'), σ², D)
end

# Compute the regularisation term in the log marginal likelihood. See e.g. appendix A.4.
function regulariser(fx, Y::ColVecs{<:Real})
    fs, H, D, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    Yproj, ΣT = project(H, Y, σ², D)

    Σ = get_sigma(H, σ², D)

    n = length(Y)
    p, m = size(H)
    Ip = Matrix(I, p, p)

    # Wrong norm?
    return -((p - m) * log(2π * σ²) + log(det(Σ) / det(ΣT)) +
        norm((Ip - H * ΣT * H' * inv(Σ)) * Y.X)) / 2
end

function regulariser(fx, Y::RowVecs{<:Real})
    return regulariser(fx, ColVecs(Y.X'))
end

"""
    rand_latent(rng::AbstractRNG, fx::FiniteGP{<:ILMM})
Sample from the latent (noiseless) process.
See also `rand`.
"""
function rand_latent(rng::AbstractRNG, fx::FiniteGP{<:ILMM})
    fs, H, D, σ², x = unpack(fx)

    # Generate from the latent processes.
    X = hcat(map((f, d) -> rand(rng, f(x, d)), fs, D.diag)...)

    # Transform latents into observed space.
    return vec(H * X')
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

    # Sample from the latent process.
    F = rand_latent(rng, fx)

    # Generate iid noise and add to each output.
    return F .+ sqrt(noise_var(fx.Σy)) .* randn(rng, size(F))
end

"""
    denoised_marginals(fx::FiniteGP{<:OILMM})
Returns the marginal distribution over the OILMM without the IID noise components.
See also `marginals`.
"""
function denoised_marginals(fx::FiniteGP{<:ILMM})
    fs, H, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = reduce(hcat, map(f -> marginals(f(x)), fs))
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # Compute the means.
    M = H * M_latent

    # Compute the variances.
    V = abs2.(H) * V_latent

    # Package everything into independent Normal distributions.
    return Normal.(vec(M'), sqrt.(vec(V')))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:ILMM})
    fs, H, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = hcat(map(f -> marginals(f(x)), fs)...)
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # # Compute the latent -> observed transform.
    # H = U * cholesky(S).U

    # Compute the means.
    M = H * M_latent

    # Compute the variances.
    V = abs2.(H) * (V_latent .+ D.diag) .+ σ²

    # Package everything into independent Normal distributions.
    return vec(M'), vec(V')
end

AbstractGPs.mean(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[1]

AbstractGPs.var(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[2]

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    fs, H, D, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    Yproj, ΣT = project(H, Y, σ², D)

    # Latent process log marginal likelihood calculation.
    y_rows = collect(eachrow(Yproj))
    ΣT_rows = collect(eachrow(ΣT))
    lmls_latents = map((f, s, y) -> logpdf(f(x, collect(s)), collect(y)), fs, ΣT_rows, y_rows)

    return regulariser(H, σ², Y) + sum(lmls_latents)
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    fs, H, D, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    Yproj, ΣT = project(H, Y, σ², D)

    # Condition each latent process on the projected observations.
    y_rows = collect(eachrow(Yproj))
    ΣT_rows = collect(eachrow(ΣT))
    fs_posterior = map((f, s, y) -> posterior(f(x, collect(s)), collect(y)), fs, ΣT_rows, y_rows)
    return ILMM(IndependentMOGP(fs_posterior), H, D)
end


#########################################

noise_var(Σ::Diagonal{<:Real, <:Fill}) = FillArrays.getindex_value(Σ.diag)

reshape_y(y::AbstractVector{<:Real}, N::Int) = ColVecs(reshape(y, N, :)')
