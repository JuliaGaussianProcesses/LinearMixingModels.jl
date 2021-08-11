"""
    Orthogonal(U, S)

An AbstractMatrix `H` that takes the form `H = U * sqrt(S)` with `U` a matrix
with orthonormal columns and `S` a diagonal matrix with positive entries.

# Arguments:
- U: a `p x m` matrix with mutually orthonormal columns.
- S: an `m x m` `Diagonal` matrix with positive entries.
"""
struct Orthogonal{T<:Real, TU<:AbstractMatrix{T}, TS<:Diagonal{T}} <: AbstractMatrix{T}
    U::TU
    S::TS
end

function Base.getindex(h::Orthogonal, args...)
    H = h.U * sqrt(h.S)
    return getindex(H, args...)
end

function Base.size(h::Orthogonal)
    return size(h.U)
end
