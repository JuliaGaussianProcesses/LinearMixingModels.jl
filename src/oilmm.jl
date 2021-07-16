"""
    OILMM(fs, H, D)
An Orthogonal Instantaneous Linear Mixing Model (OILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then
# Arguments:
- fs: a length-`m` vector of Gaussian process objects as an IndependentMOGP.
- H: a `p x m` orthogonal matrix representing a fixed basis of our p-dim target: h_1,...,h_m
- D: an `m x m` `Diagonal` matrix, variance of noise on each latent process. Positive entries.
"""
const OILMM = ILMM{<:IndependentMOGP, <:Orthogonal}

function unpack(fx::FiniteGP{<:OILMM, <:MOInput, <:Diagonal{<:Real, <:Fill}})
    fs = fx.f.f.fs
    H = fx.f.H
    D = fx.f.D
    σ² = noise_var(fx.Σy)
    x = fx.x.x

    # Check that the number of outputs requested agrees with the model.
    fx.x.out_dim == size(H, 1) || throw(error("out dim of x != out dim of f."))
    return fs, H, D, σ², x
end

# Note that `cholesky` exploits the diagonal structure of `S`.
function project(
    H::Orthogonal{T},
    Y::ColVecs{T},
    σ²::T,
    D::Diagonal{T},
) where {T<:Real}
    # Obtain U and S
    U, S = H.U, H.S

    # Compute the projection of the data.
    Yproj = cholesky(S).U \ U' * Y.X

    # Compute the projected noise, which is a matrix of size `size(Yproj)`.
    ΣT = repeat(diag(σ² * inv(S) + D), 1, size(Yproj, 2))

    return Yproj, ΣT
end

# Compute the regularisation term in the log marginal likelihood. See e.g. appendix A.4.
function regulariser(
    H::Orthogonal{T},
    σ²::T,
    Y::ColVecs{T},
) where {T<:Real}
    # Obtain U and S
    U, S = H.U, H.S

    n = length(Y)
    p, m = size(U)
    return -(n * (logdet(cholesky(S)) + (p - m) * log(2π * σ²)) +
        sum(abs2, (I - U * U') * Y.X) / σ²) / 2
end

"""
    rand_latent(rng::AbstractRNG, fx::FiniteGP{<:OILMM})
Sample from the latent (noiseless) process.
See also `rand`.
"""
function rand_latent(rng::AbstractRNG, fx::FiniteGP{<:OILMM})
    fs, H, D, σ², x = unpack(fx)

    # Obtain U and S
    U, S = H.U, H.S

    # Generate from the latent processes.
    X = hcat(map((f, d) -> rand(rng, f(x, d)), fs, D.diag)...)

    # Transform latents into observed space.
    return vec(U * cholesky(S).U * X')
end

"""
    rand(rng::AbstractRNG, fx::FiniteGP{<:OILMM})
Sample from the OILMM, including the observation noise.
Follows generative structure of model 2 from [1].
Follows the AbstractGPs.jl API.
See also `rand_latent`.
[1] - Bruinsma et al 2020.
"""
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteGP{<:OILMM})

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
function denoised_marginals(fx::FiniteGP{<:OILMM})
    fs, H, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = reduce(hcat, map(f -> marginals(f(x)), fs))
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # # Compute the latent -> observed transform.
    # H = U * cholesky(S).U

    # Compute the means.
    M = H.U * H.S * M_latent

    # Compute the variances.
    V = abs2.(H.U * H.S) * V_latent

    # Package everything into independent Normal distributions.
    return Normal.(vec(M'), sqrt.(vec(V')))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:OILMM})
    fs, H, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = hcat(map(f -> marginals(f(x)), fs)...)
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # # Compute the latent -> observed transform.
    # H = U * cholesky(S).U

    # Compute the means.
    M = H.U * H.S * M_latent

    # Compute the variances.
    V = abs2.(H.U * H.S) * (V_latent .+ D.diag) .+ σ²

    # Package everything into independent Normal distributions.
    return vec(M'), vec(V')
end

AbstractGPs.mean(fx::FiniteGP{<:OILMM}) = mean_and_var(fx)[1]

AbstractGPs.var(fx::FiniteGP{<:OILMM}) = mean_and_var(fx)[2]

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:OILMM}, y::AbstractVector{<:Real})
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
function AbstractGPs.posterior(fx::FiniteGP{<:OILMM}, y::AbstractVector{<:Real})
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
