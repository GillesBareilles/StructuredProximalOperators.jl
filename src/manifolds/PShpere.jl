
"""
    PShere

Set of points x ∈ ℝ^n in such that ||x||_p = r.
"""
struct PSphere <: Manifold{ℝ}
    p::Float64
    r::Float64
    n::Int64
end


==(M::PSphere, N::PSphere) = M.p == N.p && M.r == N.r && M.n == N.n

function <(::PSphere, ::PSphere)
    return false
end

function check_manifold_point(M::PSphere, x; kwargs...)
    if size(x) != (M.n,)
        return DomainError(size(x), "x should be vector of size $(M.n).")
    end
    norm_x = norm(x, M.p)
    if !isapprox(norm_x, M.r)
        return DomainError(norm_x - M.r, "p-norm of x supposed to be r, residual is $norm_x.")
    end
    return nothing
end

function check_tangent_vector(M::PSphere, x, ξ; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, x; kwargs...)
        mpe === nothing || return mpe
    end

    normalvector = x.^(M.p-1) ./ norm(x, M.p)^(1-M.p)
    tangent_component = dot(x, ξ)
    if !isapprox(tangent_component, 0.0)
        return DomainError(tangent_component, "p-norm of x supposed to be r, residual is $tangent_component.")
    end
    return nothing
end


# distance(::PSphere, p, q) = norm(p - q)


# # Euclidean to riemannian gradients, hessians at vectors.
# egrad_to_rgrad!(M::PSphere, gradf_x, x, ∇f_x) = project!(M, gradf_x, x, ∇f_x)  # ! this should factor out
# egrad_to_rgrad(M::PSphere, x, ∇f_x) = project(M, x, ∇f_x)                      # ! this should factor out

# function ehess_to_rhess!(M::PSphere, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
#     return project!(M, Hessf_xξ, x, ∇²f_ξ)
# end
# function ehess_to_rhess(M::PSphere, x, ∇f_x, ∇²f_ξ, ξ)              # ! this should factor out
#     Hessf_xξ = zeros(size(x))
#     return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
# end


# # embed!(::PSphere, X, x) = copyto!(X, x)
# # embed!(::PSphere, Xi, x, ξ) = copyto!(Xi, ξ)

# embed(::PSphere, x) = x
# embed(::PSphere, x, ξ) = ξ

# exp!(::PSphere, y, x, ξ) = (@. y = x + ξ)

# @inline inner(::PSphere, x, ξ, η) = dot(ξ, η)

# @inline log!(M::PSphere, ξ, x, y) = (ξ .= y .- x)

# manifold_dimension(::PSphere{n}) where {n} = n

# function name(M::PSphere; short = true)
#     return short ? "l1-$(sum(M.nnz_coords))/$(length(M.nnz_coords))" :
#            "PSphere with $(sum(M.nnz_coords)) nnz"
# end

# norm(::PSphere, x, ξ) = norm(ξ)

# function project!(M::PSphere, ξ, x, X)
#     (@. ξ = X * M.nnz_coords)
#     return ξ
# end
# project!(M::PSphere, x, X) = (@. x = X * M.nnz_coords)

# function randomMPoint(M::PSphere)
#     n = length(M.nnz_coords)
#     return (2 * rand(n) .- 1) .* M.nnz_coords
# end

# function randomTVector(M::PSphere, x)
#     pt = randomMPoint(M)
#     pt /= norm(pt)
#     return pt
# end

# representation_size(::PSphere{n}) where {n} = (n)

# retract!(::PSphere, y, x, ξ) = (@. y = x + ξ)


# function Base.show(io::IO, M::PSphere{n}) where {n}
#     return print(io, name(M))
# end

# # zero_tangent_vector(::PSphere, ::Any)
# zero_tangent_vector!(M::PSphere, v, p) = fill!(v, 0)




# # const l1MPoint = Vector
# # const l1TVector = Vector

# # copy(m::PSphere) = PSphere(m.nnz_coords)

# # ==(m1::PSphere, m2::PSphere) = m1.nnz_coords == m2.nnz_coords


# # # Functions
# # # ---
# # manifold_dimension(M::PSphere) = sum(M.nnz_coords)



# # # metric
# # inner(M::PSphere, x::l1MPoint, ξ::l1TVector, ν::l1TVector) = dot(ξ, ν)
# # norm(M::PSphere, x::l1MPoint, ξ::l1TVector) = inner(M, x, ξ, ξ)

# # # Tangent space
# # function project!(M::PSphere, ξ::l1TVector, x::l1MPoint, q::AbstractArray)
# #     @. ξ = q * M.nnz_coords
# #     return ξ
# # end

# # function project(M::PSphere, x::l1MPoint, q::AbstractArray)                  # ! This should factor out, with `zro_tangent_vector`
# #     ξ = zeros(size(x))
# #     return project!(M, ξ, x, q)
# # end



# # function retract!(M::PSphere, y::l1MPoint, x::l1MPoint, ξ::l1TVector)
# #     return @. y = x + ξ
# # end
# # function retract(M::PSphere, x::l1MPoint, ξ::l1TVector)
# #     y = zeros(size(x))
# #     return retract!(M, y, x, ξ)
# # end








# # # Display
# # # ---
# # show(io::IO, M::PSphere) = print(io, M.abbreviation)
