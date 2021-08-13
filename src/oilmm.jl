"""
    OILMM(fs, H)
An Orthogonal Instantaneous Linear Mixing Model (OILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then
# Arguments:
- fs: a length-`m` vector of Gaussian process objects as an IndependentMOGP.
- H: a `p x m` orthogonal matrix representing a fixed basis of our p-dim target: h_1,...,h_m
"""
const OILMM = ILMM{<:IndependentMOGP, <:Orthogonal}

"""
    project(H, σ²)

Computes the projection `T` and `ΣT` given the mixing matrix and noise.
"""
function project(H::Orthogonal{Z}, σ²::Z) where {Z<:Real}
    U, S = H.U, H.S

    # Compute the projection of the data.
    T = sqrt(S) \ U'

    # Compute the projected noise, which is a matrix of size `size(Yproj)`.
    ΣT = diag(σ² * inv(S))

    return T, ΣT
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
    fs, H, σ², x = unpack(fx)

    # Obtain U and S
    U, S = H.U, H.S

    # Generate from the latent processes.
    X = hcat(map(f -> rand(rng, f(x)), fs.fs)...)

    # Transform latents into observed space.
    F = vec(U * sqrt(S) * X')

    # Generate iid noise and add to each output.
    return F .+ sqrt(noise_var(fx.Σy)) .* randn(rng, size(F))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:OILMM})
    fs, H, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = hcat(map(f -> AbstractGPs.marginals(f(x)), fs.fs)...)
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # Obtain U and S
    U, S = H.U, H.S

    # Compute the means.
    M = U * sqrt(S) * M_latent

    # Compute the variances.
    V = abs2.(U * sqrt(S)) * (V_latent) .+ σ²

    # Package everything into independent Normal distributions.
    return collect(vec(M')), collect(vec(V'))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:OILMM}, y::AbstractVector{<:Real})
    fs, H, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = T*Y

    # Latent process log marginal likelihood calculation.
    y_rows = collect(eachrow(Ty))
    ΣT_rows = collect(eachrow(repeat(ΣT, 1, size(Ty, 2))))
    lmls_latents = map((f, s, y) -> logpdf(f(x, collect(s)), collect(y)), fs.fs, ΣT_rows, y_rows)

    return sum(lmls_latents) + regulariser(H, σ², ColVecs(Y))
end

"""
    regulariser(fx, y)

Computes the regularisation term of the logpdf.
See e.g. appendix A.4 of [1] - Bruinsma et al 2020.
"""
function regulariser(
    H::Orthogonal{T},
    σ²::T,
    Y::ColVecs{T},
) where {T<:Real}
    U, S = H.U, H.S

    n = length(Y)
    p, m = size(U)
    return -(n * (logdet(S) + (p - m) * log(2π * σ²)) +
        sum(abs2, (I - U * U') * Y.X) / σ²) / 2
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(fx::FiniteGP{<:OILMM}, y::AbstractVector{<:Real})
    fs, H, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = T*Y

    # Condition each latent process on the projected observations.
    y_rows = collect(eachrow(Ty))
    ΣT_rows = collect(eachrow(repeat(ΣT, 1, size(Ty, 2))))
    fs_posterior = map((f, s, y) -> AbstractGPs.posterior(f(x, collect(s)), collect(y)), fs.fs, ΣT_rows, y_rows)
    return ILMM(IndependentMOGP(fs_posterior), H)
end
