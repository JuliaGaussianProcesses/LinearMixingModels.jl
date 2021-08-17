"""
    IndependentMOGP(fs)

A multi-output GP with independent outputs where output `i` is modelled by the
single-output GP fs[i].

# Arguments:
- fs: a vector of `p` single-output GPs where `p` is the dimension of the output.
"""
struct IndependentMOGP{Tg<:Vector{<:AbstractGPs.AbstractGP}} <: AbstractGPs.AbstractGP
    fs::Tg
end

"""
    independent_mogp(fs)

Returns an IndependentMOGP given a list of single output GPs `fs`.

```jldoctest
julia> independent_mogp([GP(SEKernel())]) == IndependentMOGP([GP(SEKernel())])
true
```
"""
function independent_mogp(fs::Vector{<:AbstractGPs.AbstractGP})
    return IndependentMOGP(fs)
end

"""
A constant to represent all isotopic input types.
"""
const isotopic_inputs = Union{
    MOInputIsotopicByFeatures, MOInputIsotopicByOutputs
}

"""
    finite_gps(fx)

Returns a list of of the finite GPs for all latent processes, given a finite
IndependentMOGP and *isotopic inputs*.
"""
function finite_gps(fx::FiniteGP{<:IndependentMOGP, <:isotopic_inputs}, σ²::Real)
    return [f(fx.x.x, σ²) for f in fx.f.fs]
end

"""
    finite_gps(fx)

Returns a list of of the finite GPs for all latent processes, given a finite
IndependentMOGP, *isotopic inputs* and .
"""
# function finite_gps(fx::FiniteGP{<:IndependentMOGP,<:isotopic_inputs}, x::AbstractVector)
#     return [f(x, fx.Σy[1:length(fx.x.x),1:length(fx.x.x)]) for f in fx.f.fs]
# end

const IsotropicFiniteIndependentMOGP = FiniteGP{
    <:IndependentMOGP, <:MOInputIsotopicByOutputs, <:Diagonal{<:Real, <:Fill},
}

# Implement AbstractGPs API

# # See AbstractGPs.jl API docs.
# function AbstractGPs.mean(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs})
#     finiteGPs = finite_gps(ft, 0)
#     return reduce(vcat, map(mean, finiteGPs))
# end

# # See AbstractGPs.jl API docs.
# function AbstractGPs.var(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs})
#     finiteGPs = finite_gps(ft, 0)
#     return reduce(vcat, map(var, finiteGPs)) + diag(ft.Σy)
# end

# # See AbstractGPs.jl API docs.
# function AbstractGPs.cov(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs})
#     finiteGPs = finite_gps(ft, 0)
#     return collect(BlockDiagonal(cov.(finiteGPs)) + ft.Σy)
# end

# # See AbstractGPs.jl API docs.
# function AbstractGPs.mean_and_var(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs})
#     return mean(ft), var(ft)
# end

# # See AbstractGPs.jl API docs.
# function AbstractGPs.mean_and_cov(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs})
#     return mean(ft), cov(ft)
# end

# # See AbstractGPs.jl API docs.
# function AbstractGPs.marginals(ft::IsotropicFiniteIndependentMOGP)
#     finiteGPs = finite_gps(ft, ft.Σy[1])
#     return reduce(vcat, map(AbstractGPs.marginals, finiteGPs))
# end

function AbstractGPs.mean(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    return reduce(vcat, map(f -> mean(f, x.x), f.fs))
end

function AbstractGPs.var(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    return reduce(vcat, map(f -> var(f, x.x), f.fs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    Cs = map(f -> cov(f, x.x), f.fs)
    return collect(BlockDiagonal(Cs))
    # n = length(x)
    # m = length(y)
    # Σ = zeros(n, m)
    # for i in 1:n
    #     for j in i:m
    #         if x[i][2]==y[j][2]
    #             p = x[i][2]
    #             Σ[i,j] = f.fs[p].kernel(x[i][1], y[j][1])
    #             Σ[j,i] = Σ[i,j]
    #         end
    #     end
    # end
    # return Σ
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(
    f::IndependentMOGP,
    x::MOInputIsotopicByOutputs,
    y::MOInputIsotopicByOutputs,
)
    Cs = map(f -> cov(f, x.x, y.x), f.fs)
    return collect(BlockDiagonal(Cs))
    # n = length(x)
    # m = length(y)
    # Σ = zeros(n, m)
    # for i in 1:n
    #     for j in i:m
    #         if x[i][2]==y[j][2]
    #             p = x[i][2]
    #             Σ[i,j] = f.fs[p].kernel(x[i][1], y[j][1])
    #             Σ[j,i] = Σ[i,j]
    #         end
    #     end
    # end
    # return Σ
end


# function AbstractGPs.cov(ft::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByFeatures, <:Any})
#     x = ft.x.x
#     n = length(x)
#     p = ft.x.out_dim
#     X_by_outputs = MOInputIsotopicByOutputs(x, p)
#     cov_matrix_by_outputs = cov(ft.f(X_by_outputs, ft.Σy))
#     cov_matrix = 1.0 * Matrix(I, n*p, n*p)
#     for i in 1:n*p
#         for j in 1:i-1
#             cov_matrix[i,j] = cov_matrix_by_outputs[i,j]
#             cov_matrix[i,j] = cov_matrix[i*p-1,j]
#             # cov_matrix[i*p-1,j*p-1] = cov_matrix[i,j]
#         end
#     end
#     return cov_matrix
# end

# function Statistics.cov(ft::FiniteGP{<:IndependentMOGP}, gt::FiniteGP{<:IndependentMOGP})
#     return cov(ft.f, ft.x, gt.x)
# end

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(ft::IsotropicFiniteIndependentMOGP, y::AbstractVector)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    return sum([logpdf(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)])
end

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicFiniteIndependentMOGP)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    return reduce(vcat, map(fx -> rand(rng, fx), finiteGPs))
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}) = rand(Random.GLOBAL_RNG, ft)

function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicFiniteIndependentMOGP, N::Int)
    return reduce(hcat, [rand(rng, ft) for _ in 1:N])
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

function Distributions._rand!(
    rng::AbstractRNG,
    fx::FiniteGP{<:IndependentMOGP},
    y::AbstractVecOrMat{<:Real}
)
    N = size(y, 2)
    if N == 1
        y .= AbstractGPs.rand(rng, fx)
    else
        y .= AbstractGPs.rand(rng, fx, N)
    end
end

"""
Posterior implementation for isotopic inputs, given diagonal Σy (OILMM).
See AbstractGPs.jl API docs.
"""
function AbstractGPs.posterior(
    ft::IsotropicFiniteIndependentMOGP,
    y::AbstractVector{<:Real},
)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x),:))))
    ind_posts = [AbstractGPs.posterior(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)]
    return IndependentMOGP(ind_posts)
end
