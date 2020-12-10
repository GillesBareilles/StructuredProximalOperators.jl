
struct PlaneParabola <: Manifold{ℝ}
end

normalvec(::PlaneParabola, x) = [2*x[1], -1]
tangentvec(::PlaneParabola, x) = [1, 2*x[1]]


==(M::PlaneParabola, N::PlaneParabola) = true
<(M::PlaneParabola, N::PlaneParabola) = false

function check_manifold_point(::PlaneParabola, x; kwargs...)
    if size(x) != (2,)
        return DomainError(size(x), "x should be vector of size 2.")
    end
    atol = haskey(kwargs, :atol) ? kwargs[:atol] : 1e-13
    if abs(x[1]^2-x[2]) > atol
        return DomainError(
            "x²-y = $(x[1]^2-x[2]).",
        )
    end
    return nothing
end

function check_tangent_vector(M::PlaneParabola, x, ξ; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, x; kwargs...)
        mpe === nothing || return mpe
    end
    atol = haskey(kwargs, :atol) ? kwargs[:atol] : 1e-13
    if abs(dot(ξ, normalvec(M, x))) > atol
        return DomainError(
            "⟨ξ, nₓ⟩ = $(dot(ξ, normalvec(M, x)))",
        )
    end
    return nothing
end


copy(M::PlaneParabola) = PlaneParabola()
# distance(::PlaneParabola, p, q) = norm(p - q)


# Euclidean to riemannian gradients, hessians at vectors.
function ehess_to_rhess!(M::PlaneParabola, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
    return project!(M, Hessf_xξ, x, ∇²f_ξ - [2ξ[1], 0] * inner(M, x, ∇f_x, [2x[1], -1])/(1+4x[1]^2) )
end


function ehess_to_rhess!(M::Euclidean, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
    return Hessf_xξ .= ∇²f_ξ
end

embed(::PlaneParabola, x) = x
embed(::PlaneParabola, x, ξ) = ξ
embedding_dimension(M::PlaneParabola) = 2

# exp!(::PlaneParabola, y, x, ξ) = (@. y = x + ξ)

@inline inner(::PlaneParabola, x, ξ, η) = dot(ξ, η)

# @inline log!(M::PlaneParabola, ξ, x, y) = (ξ .= y .- x)

manifold_dimension(M::PlaneParabola) = 1

name(M::PlaneParabola; short = true) = "PlaneParabola"

norm(::PlaneParabola, x, ξ) = norm(ξ)

function project!(M::PlaneParabola, ξ, x, X)
    tvec = tangentvec(M, x)
    ξ .= dot(X, tvec) * tvec / norm(tvec)^2
    return ξ
end

function project(M::PlaneParabola, x, X)
    res = zeros(2)
    return project!(M, res, x, X)
end

# function project(M::PlaneParabola, X)
#     nnz_inds = get_nnz_indices(M)
#     return sparsevec(nnz_inds, X[nnz_inds], n)
# end


function randomMPoint(M::PlaneParabola)
    x = rand()*2-1
    return [x, x^2]
end

function randomTVector(M::PlaneParabola, x)
    t = rand(0:1)*2-1
    tvec = tangentvec(M, x)
    return t * tvec / norm(tvec)
end

representation_size(::PlaneParabola) = (2)

function retract!(M::PlaneParabola, res, xy, ξ)
    x, y = xy

    ## Projection retraction
    ps = roots([
        -(x+ξ[1]),
        1-2*(y+ξ[2]),
        0.0,
        2
    ])
    ps = filter(c -> abs(imag(c)) < 1e-12, ps)
    length(ps) == 0 && @warn "Retraction on PlaneParabola troubled" ps, xy, ξ
    ps = map(c -> real(c), ps)
    # length(ps) != 1 && @warn "Retraction on PlaneParabola troubled" ps, xy, ξ

    # NOTE: we pick the p closest to x+ξₓ for now, kinda heuristic.
    val, indmin = findmin(@. abs(ps-x-ξ[1]))
    p = ps[indmin]

    res[1] = p
    res[2] = p^2
    return res
end


function Base.show(io::IO, M::PlaneParabola)
    return print(io, name(M))
end

zero_tangent_vector(M::PlaneParabola, ::Any) = zeros(2)
zero_tangent_vector!(M::PlaneParabola, v, p) = fill!(v, 0)
