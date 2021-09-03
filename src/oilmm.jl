"""
    OILMM(fs, H)
An Orthogonal Instantaneous Linear Mixing Model (OILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then `H`, also known as the mixing matrix, is a `p x m` orthogonal matrix whose
column space spans the output space. The latent processes are represented an `Independent_MOGP`
as the latent processes remain decoupled.

# Arguments:
- f: a length-`m` vector of Gaussian process objects as an IndependentMOGP.
- H: a `p x m` orthogonal matrix representing a fixed basis of our p-dim target: h_1,...,h_m
"""
const OILMM = ILMM{<:IndependentMOGP, <:Orthogonal}

"""
    OILMMNoiseCovarianceMatrix

A covariance matrix specifically designed to work with an OILMM.
Please construct using [`oilmm_noise_covariance`](@ref) to ensure correctness.
"""
struct OILMMNoiseCovarianceMatrix{
    T, TH<:Orthogonal{T}, TD<:Diagonal{T},
} <: AbstractMatrix{T}
    H::TH
    D::TD
    σ²::T
    num_features::Int
end

function Base.size(X::OILMMNoiseCovarianceMatrix)
    num_variables = X.num_features * size(X.H, 1)
    return (num_variables, num_variables)
end

function Base.collect(X::OILMMNoiseCovarianceMatrix{T}) where {T}
    return kron(X.H * X.D * X.H' + X.σ² * I, Matrix{T}(I, X.num_features, X.num_features))
end



# Doing all of this is harder than it initially seems, because it requires that you also do
# it for the ILMM. This creates a number of additional possibilities that seem reasonable
# to account for (e.g. not using the same matrix in both cases).
# This means that the complexity of the possibilities increases substantially, and more
# general solutions must be accounted for.
"""
    oilmm_noise_covariance(
        f::OILMM, x::MOInputIsotopicByOutputs, D::Diagonal, σ²::Real,
    )

Build the observation noise associated with an `f` at a particular set of inputs.
"""
function oilmm_noise_covariance(
    f::OILMM, x::MOInputIsotopicByOutputs, D::Diagonal, σ²::Real,
)
    return OILMMNoiseCovarianceMatrix(f.H, D, σ², length(x.x))
end

"""
    oilmm_noise_covariance(f::OILMM, x::MOInputIsotopicByOutputs, σ²::Real)

Build the observation noise associated with an `f` at a particular set of inputs, with
`D` set to zero.
"""
function oilmm_noise_covariance(f::OILMM, x::MOInputIsotopicByOutputs, σ²::Real)
    D = Diagonal(Fill(zero(σ²), size(f.H, 2)))
    return oilmm_noise_covariance(f, x, D, σ²)
end

(f::OILMM)(x::AbstractVector, σ²::Real) = f(x, oilmm_noise_covariance(f, x, σ²))

const FiniteOILMM = FiniteGP{<:OILMM, <:AbstractVector, <:OILMMNoiseCovarianceMatrix}

function unpack(fx::FiniteOILMM)
    f = fx.f.f
    H = fx.f.H
    D = fx.Σy.D
    σ² = fx.Σy.σ²
    x = fx.x.x

    # Check that the number of outputs requested agrees with the model.
    fx.x.out_dim == size(H, 1) || throw(error("out dim of x != out dim of f."))
    return f, H, D, σ², x
end

"""
    project(H, σ²)

Computes the projection `T` and `ΣT` given the mixing matrix and noise.
"""
function project(H::Orthogonal{V}, D::Diagonal{V}, σ²::V) where {V<:Real}
    U, S = H.U, H.S

    # Compute the projection of the data.
    T = sqrt(S) \ U'

    # Compute the projected noise, which is a matrix of size `size(Yproj)`.
    ΣT = diag(σ² * inv(S)) + diag(D)

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
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteOILMM)
    fs, H, D, σ², x = unpack(fx)

    # Generate from the latent processes.
    latent_samples = map((f, d) -> rand(rng, f(x, d)), fs.fs, diag(D))
    X = reshape(reduce(vcat, latent_samples), length(x), :)

    # Transform latents into observed space.
    F = vec((H.U * sqrt(H.S) * X')')

    # Generate iid noise and add to each output.
    return F .+ sqrt(σ²) .* randn(rng, size(F))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteOILMM)
    fs, H, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    latent_marginals = map((f, d) -> AbstractGPs.marginals(f(x, d)), fs.fs, diag(D))
    fs_marginals = reduce(hcat, latent_marginals)
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
function AbstractGPs.logpdf(fx::FiniteOILMM, y::AbstractVector{<:Real})
    fs, H, D, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, D, σ²)
    Ty = T*Y

    # Latent process log marginal likelihood calculation.
    y_rows = collect(eachrow(Ty))
    ΣT_rows = collect(eachrow(repeat(ΣT, 1, size(Ty, 2))))
    lmls_latents = map((f, s, y) -> logpdf(f(x, collect(s)), collect(y)), fs.fs, ΣT_rows, y_rows)

    return sum(lmls_latents) + regulariser(H, σ², Y)
end

"""
    regulariser(fx, y)

Computes the regularisation term of the logpdf.
See e.g. appendix A.4 of [1] - Bruinsma et al 2020.
"""
function regulariser(
    H::Orthogonal{T},
    σ²::T,
    Y::AbstractMatrix{T},
) where {T<:Real}
    U, S = H.U, H.S

    n = size(Y, 2)
    p, m = size(U)

    return -(n * (logdet(S) + (p - m) * log(2π * σ²)) +
        sum(abs2, (I - U * U') * Y) / σ²) / 2
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(
    fx::FiniteOILMM, y::AbstractVector{<:Real},
)
    fs, H, D, σ², x = unpack(fx)

    # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, D, σ²)
    Ty = T*Y

    # Condition each latent process on the projected observations.
    y_rows = collect(eachrow(Ty))
    ΣT_rows = collect(eachrow(repeat(ΣT, 1, size(Ty, 2))))
    fs_posterior = map(
        (f, s, y) -> AbstractGPs.posterior(f(x, collect(s)), collect(y)),
        fs.fs,
        ΣT_rows,
        y_rows,
    )
    return ILMM(independent_mogp(fs_posterior), H)
end
