# OrthogonalMatrix
struct Orthogonal{T<:Real, TU<:AbstractMatrix{T}, TS<:Diagonal{T}} <: AbstractMatrix{T}
    U::TU
    S::TS
end

function Base.getindex(h::Orthogonal, args...)
    H = h.U * h.S
    return getindex(H, args...)
end

function Base.size(h::Orthogonal)
    return size(h.U)
end
