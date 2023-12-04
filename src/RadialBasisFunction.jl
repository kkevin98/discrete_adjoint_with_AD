module RadialBasisFunction
using LinearAlgebra

export interpolation_matrix, global_matrix, nearest_neighbour_search, solve_RBF



#TODO: a utils module would be better ??
function nearest_neighbour_search(x0, x, n_neighbours)
    # (inefficient) n_neighbours nearest neighbour search centered in x0
    # returns vector of indices of first n_neighbours nearest neighbours from x0
    return sortperm( abs.(x .- x0) )[1:n_neighbours]
end



# interpolation matrix M given vector xn of n coordinates of interpolation nodes and center x0 (required for the polynomial part)
function interpolation_matrix(xn, x0, RBF)
    D = abs.( xn .- xn' )       # matrix of distances between each couple of nodes
    Φ = RBF(D)                  # RBF matrix, RBF evaluated for each entry of D
    P = ( xn .- x0 ) .^ [0 1 2] # polynomial part (degree 2) of interpolation matrix
    M = [ Φ P ; P' zeros(3,3) ] # interpolation matrix
    return M
end



function global_matrix(X, RBF, ddRBF, num_of_points_in_stencil)

    C = zeros(length(X), length(X))
    
    for i ∈ eachindex(X)

        xi = X[i]                                                               # coordinate of node i (actual stencil's center)
        neighbours = nearest_neighbour_search(xi, X, num_of_points_in_stencil)  # vector of indices of n nearest neighours of node i
        x_neighbours = X[neighbours]                                            # vector of coordinates of neighbours
        
        C[i,neighbours] = global_matrix_row_from_stencil(xi, x_neighbours, RBF, ddRBF)
    end

    return C

end


function global_matrix_row_from_stencil(central_point, nearby_points, RBF, ddRBF)
    
    r_neighbours = abs.(nearby_points .- central_point)  # vector of distances from stencil's center to neighbours

    M = interpolation_matrix(nearby_points, central_point, RBF)
    Ψ = ddRBF(r_neighbours)
    Π = [0, 0 ,2]
    c = (M') \ [Ψ; Π]

    return c[1:length(nearby_points)]  # Discardin the indexes corresponding to the polynomial part since multiplied by zero

end



function solve_RBF(C, X, internal_points, boundary_points, boundary_conditions, q)
    #=

    =#
    
    C_internal = C[internal_points, internal_points]
    C_cols_of_bouondary_points = C[internal_points, boundary_points]
    
    q_num = q(X[internal_points])
    f = C_cols_of_bouondary_points * boundary_conditions

    return C_internal \ (q_num-f)

end

end