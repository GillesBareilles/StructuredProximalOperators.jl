# Types
# ---
struct l1Manifold <: AbstractManifold
    name::String
    nnz_coords::BitArray{1}
    abbreviation::String
    l1Manifold(nnz_coords) = new(
        "l1Manifold with $(sum(nnz_coords)) nnz",
        convert(BitArray, nnz_coords),
        "l1-$(sum(nnz_coords))/$(length(nnz_coords))",
    )
end

const l1MPoint = Vector
const l1TVector = Vector

copy(m::l1Manifold) = l1Manifold(m.nnz_coords)

==(m1::l1Manifold, m2::l1Manifold) = m1.nnz_coords == m2.nnz_coords


# Functions
# ---
manifold_dimension(M::l1Manifold) = sum(M.nnz_coords)

function randomMPoint(M::l1Manifold)
    n = length(M.nnz_coords)
    return (2 * rand(n) .- 1) .* M.nnz_coords
end

function randomTVector(M::l1Manifold, x::l1MPoint)
    pt = randomMPoint(M)
    pt /= norm(pt)
    return pt
end


# metric
inner(M::l1Manifold, x::l1MPoint, ξ::l1TVector, ν::l1TVector) = dot(ξ, ν)
norm(M::l1Manifold, x::l1MPoint, ξ::l1TVector) = inner(M, x, ξ, ξ)

# Tangent space
function project!(M::l1Manifold, ξ::l1TVector, x::l1MPoint, q::AbstractArray)
    @. ξ = q * M.nnz_coords
    return ξ
end

function project(M::l1Manifold, x::l1MPoint, q::AbstractArray)                  # ! This should factor out, with `zro_tangent_vector`
    ξ = zeros(size(x))
    return project!(M, ξ, x, q)
end



function retract!(M::l1Manifold, y::l1MPoint, x::l1MPoint, ξ::l1TVector)
    @. y = x + ξ
end
function retract(M::l1Manifold, x::l1MPoint, ξ::l1TVector)
    y = zeros(size(x))
    return retract!(M, y, x, ξ)
end


# Euclidean to riemannian gradients, hessians at vectors.
egrad_to_rgrad!(M::l1Manifold, gradf_x, x::l1MPoint, ∇f_x) = project!(M, gradf_x, x, ∇f_x)  # ! this should factor out
egrad_to_rgrad(M::l1Manifold, x::l1MPoint, ∇f_x) = project(M, x, ∇f_x)                      # ! this should factor out

ehess_to_rhess!(M::l1Manifold, Hessf_xξ, x::l1MPoint, ∇f_x, ∇²f_ξ, ξ::l1TVector) =
    project!(M, Hessf_xξ, x, ∇²f_ξ)
function ehess_to_rhess(M::l1Manifold, x::l1MPoint, ∇f_x, ∇²f_ξ, ξ::l1TVector)              # ! this should factor out
    Hessf_xξ = zeros(size(x))
    return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
end


# Validation
# ---
function check_manifold_point(M::l1Manifold, x::l1MPoint)
    if length(x) != length(M.nnz_coords)
        return DomainError(
            size(x),
            "The dimension of x must be $(size(M.nnz_coords)) but it is $(size(x))",
        )
    end
    norm_zerocoords = norm(x .* (1 .- M.nnz_coords))
    if norm_zerocoords > 1e-14
        return DomainError(
            norm_zerocoords,
            "Coordinates of x supposed to be 0 have norm $norm_zerocoords.",
        )
    end
    return nothing
end

function check_tangent_vector(
    M::l1Manifold,
    x::l1MPoint,
    ξ::l1TVector;
    check_base_point = true,
)
    if check_base_point
        check_manifold_point(M, x)
    end
    check_manifold_point(M, ξ)
    return nothing
end



# Display
# ---
show(io::IO, M::l1Manifold) = print(io, M.abbreviation)
