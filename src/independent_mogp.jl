"""
    IndependentMOGP
"""
struct IndependentMOGP{Tg<:Vector{<:AbstractGPs.AbstractGP}} <: AbstractGPs.AbstractGP
    fs::Tg
end

"""
    independent_mogp

Returns an IndependentMOGP given a list of AbstractGPs.
"""
function independent_mogp(fs::Vector{<:AbstractGPs.AbstractGP})
    return IndependentMOGP(fs)
end

"""
A constant to represent all isotopic input types.
"""
const isotopic_inputs = Union{
    KernelFunctions.MOInputIsotopicByFeatures, KernelFunctions.MOInputIsotopicByOutputs
}

"""
    finite_gps()

Returns a list of of the finite GPs for all latent processes, given isotopic inputs.
"""
function finite_gps(ft::FiniteGP{<:IndependentMOGP,<:isotopic_inputs})
    return [f(ft.x.x, ft.Σy[1:length(ft.x.x),1:length(ft.x.x)]) for f in ft.f.fs]
end

"""
    finite_gps()

Returns a list of of the finite GPs for all latent processes for some input x.
"""
function finite_gps(ft::FiniteGP{<:IndependentMOGP,<:isotopic_inputs}, x::AbstractVector)
    return [f(x, ft.Σy[1:length(ft.x.x),1:length(ft.x.x)]) for f in ft.f.fs]
end

# Implement AbstractGPs API

# Marginals implementation
function AbstractGPs.marginals(ft::FiniteGP{<:IndependentMOGP})
    finiteGPs = finite_gps(ft)
    return reduce(vcat, map(AbstractGPs.marginals, finiteGPs))
end

# Mean and Variance implementation
function AbstractGPs.mean_and_var(ft::FiniteGP{<:IndependentMOGP})
    ms = AbstractGPs.marginals(ft)
    return reshape(map(mean, ms), length(ft.x)), map(var, ms)
end

# Mean and Covariance implementation
function AbstractGPs.mean_and_cov(ft::FiniteGP{<:IndependentMOGP})
    return mean(ft), cov(ft)
end

# Variance implementation
AbstractGPs.var(ft::FiniteGP{<:IndependentMOGP}) = mean_and_var(ft)[2]

# Cov implementation
function Statistics.cov(
    f::IndependentMOGP,
    x::AbstractVector,
    y::AbstractVector
)
    n = length(x)
    m = length(y)
    Σ = zeros(n, m)
    for i in 1:n
        for j in i:m
            if x[i][2]==y[j][2]
                p = x[i][2]
                Σ[i,j] = f.fs[p].kernel(x[i][1], y[j][1])
                Σ[j,i] = Σ[i,j]
            end
        end
    end
    return Σ
end

Statistics.cov(ft::FiniteGP{<:IndependentMOGP}) = cov(ft.f, ft.x, ft.x)

function Statistics.cov(ft::FiniteGP{<:IndependentMOGP}, gt::FiniteGP{<:IndependentMOGP})
    return cov(ft.f, ft.x, gt.x)
end

# Mean implementation
AbstractGPs.mean(ft::FiniteGP{<:IndependentMOGP}) = mean_and_var(ft)[1]

#Logpdf implementation
function AbstractGPs.logpdf(ft::FiniteGP{<:IndependentMOGP}, y::AbstractVector)
    finiteGPs = finite_gps(ft)
    ys = collect(eachcol(reshape(y, (length(ft.x.x),:))))
    return sum([logpdf(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)])
end

# Random sampling implementation
function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteGP{<:IndependentMOGP})
    finiteGPs = finite_gps(ft)
    return vcat(map(fx -> rand(rng, fx), finiteGPs)...)
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}) = rand(Random.GLOBAL_RNG, ft)

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteGP{<:IndependentMOGP}, N::Int)
    return reduce(hcat, [rand(rng, ft) for _ in 1:N])
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

# Posterior implementation for isotopic inputs, given diagonal Σy (OILMM)
function AbstractGPs.posterior(
    ft::FiniteGP{<:IndependentMOGP},
    y::AbstractVector{<:Real}
)
    finiteGPs = finite_gps(ft)
    ys = collect(eachcol(reshape(y, (length(ft.x.x),:))))
    ind_posts = [posterior(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)]
    return IndependentMOGP(ind_posts)
end

