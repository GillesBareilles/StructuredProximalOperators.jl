
struct l1Manifold{n} <: Manifold{ℝ}
    nnz_coords::BitArray{1}
end
function l1Manifold(nnz_coords)
    return l1Manifold{length(nnz_coords)}(convert(BitArray, nnz_coords))
end


==(M::l1Manifold, N::l1Manifold) = (M.nnz_coords == N.nnz_coords)

function <(M::l1Manifold, N::l1Manifold)
    return M.nnz_coords < N.nnz_coords
end

function check_manifold_point(M::l1Manifold{n}, x; kwargs...) where {n}
    if size(x) != (n,)
        return DomainError(size(x), "x should be vector of size $n.")
    end
    norm_zerocoords = norm(x .* (1 .- M.nnz_coords))
    if norm_zerocoords > 0
        return DomainError(
            norm_zerocoords,
            "Coordinates of x supposed to be 0 have norm $norm_zerocoords.",
        )
    end
    return nothing
end

function check_tangent_vector(M::l1Manifold, x, ξ; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, x; kwargs...)
        mpe === nothing || return mpe
    end
    check_manifold_point(M, ξ; kwargs...)
    return nothing
end

# decorated_manifold(M::l1Manifold) = Euclidean(representation_size(M)...; field = ℝ)


distance(::l1Manifold, p, q) = norm(p - q)


# Euclidean to riemannian gradients, hessians at vectors.
egrad_to_rgrad!(M::l1Manifold, gradf_x, x, ∇f_x) = project!(M, gradf_x, x, ∇f_x)  # ! this should factor out
egrad_to_rgrad(M::l1Manifold, x, ∇f_x) = project(M, x, ∇f_x)                      # ! this should factor out

function ehess_to_rhess!(M::l1Manifold, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
    return project!(M, Hessf_xξ, x, ∇²f_ξ)
end
function ehess_to_rhess(M::l1Manifold, x, ∇f_x, ∇²f_ξ, ξ)              # ! this should factor out
    Hessf_xξ = zeros(size(x))
    return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
end


# embed!(::l1Manifold, X, x) = copyto!(X, x)
# embed!(::l1Manifold, Xi, x, ξ) = copyto!(Xi, ξ)

embed(::l1Manifold, x) = x
embed(::l1Manifold, x, ξ) = ξ

exp!(::l1Manifold, y, x, ξ) = (@. y = x + ξ)

@inline inner(::l1Manifold, x, ξ, η) = dot(ξ, η)

@inline log!(M::l1Manifold, ξ, x, y) = (ξ .= y .- x)

manifold_dimension(::l1Manifold{n}) where {n} = n

function name(M::l1Manifold; short = true)
    return short ? "l1-$(sum(M.nnz_coords))/$(length(M.nnz_coords))" :
           "l1Manifold with $(sum(M.nnz_coords)) nnz"
end

norm(::l1Manifold, x, ξ) = norm(ξ)

function project!(M::l1Manifold, ξ, x, X)
    (@. ξ = X * M.nnz_coords)
    return ξ
end
project!(M::l1Manifold, x, X) = (@. x = X * M.nnz_coords)

function randomMPoint(M::l1Manifold)
    n = length(M.nnz_coords)
    return (2 * rand(n) .- 1) .* M.nnz_coords
end

function randomTVector(M::l1Manifold, x)
    pt = randomMPoint(M)
    pt /= norm(pt)
    return pt
end

representation_size(::l1Manifold{n}) where {n} = (n)

retract!(::l1Manifold, y, x, ξ) = (@. y = x + ξ)


function Base.show(io::IO, M::l1Manifold{n}) where {n}
    return print(io, name(M))
end

# zero_tangent_vector(::l1Manifold, ::Any)
zero_tangent_vector!(M::l1Manifold, v, p) = fill!(v, 0)




# const l1MPoint = Vector
# const l1TVector = Vector

# copy(m::l1Manifold) = l1Manifold(m.nnz_coords)

# ==(m1::l1Manifold, m2::l1Manifold) = m1.nnz_coords == m2.nnz_coords


# # Functions
# # ---
# manifold_dimension(M::l1Manifold) = sum(M.nnz_coords)



# # metric
# inner(M::l1Manifold, x::l1MPoint, ξ::l1TVector, ν::l1TVector) = dot(ξ, ν)
# norm(M::l1Manifold, x::l1MPoint, ξ::l1TVector) = inner(M, x, ξ, ξ)

# # Tangent space
# function project!(M::l1Manifold, ξ::l1TVector, x::l1MPoint, q::AbstractArray)
#     @. ξ = q * M.nnz_coords
#     return ξ
# end

# function project(M::l1Manifold, x::l1MPoint, q::AbstractArray)                  # ! This should factor out, with `zro_tangent_vector`
#     ξ = zeros(size(x))
#     return project!(M, ξ, x, q)
# end



# function retract!(M::l1Manifold, y::l1MPoint, x::l1MPoint, ξ::l1TVector)
#     return @. y = x + ξ
# end
# function retract(M::l1Manifold, x::l1MPoint, ξ::l1TVector)
#     y = zeros(size(x))
#     return retract!(M, y, x, ξ)
# end








# # Display
# # ---
# show(io::IO, M::l1Manifold) = print(io, M.abbreviation)
