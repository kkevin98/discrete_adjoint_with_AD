using LinearAlgebra
using SparseArrays

# solution of 1D poisson equation u" = d²u/dx² = q(x) on [a,b]
# q(x) is a known function (see later)
# u(x) is the unknown function to solve for
# Dirichlet boundary conditions (BCs):
# u(a) = u_a, u(b) = u_b (u_a, u_b are known)

# Mathematical problem settings:
# 1D domain: [a,b]
a = 0
b = 2π

# let's define an analytical solution to be used for comparison, errors, etc.
# u(x) = sin(x)
u(x) = sin.(x)

# let's define the known function q(x), obtained from the analytical solution u(x) = sin(x)
# => q(x) = u"(x) = -sin(x)
q(x) = -sin.(x)

# BCs, taken from the chosen analytical solution u
u_a = u(a)
u_b = u(b)

# Numerical part:
# Node distribution with N nodes on [a,b]: x1=a,..., xN=b
N = 40
X = range(a, b, length=N) # uniform, but not necessary

# (inefficient) n_neighbours nearest neighbour search centered in x0
# returns vector of indices of first n_neighbours nearest neighbours from x0
function nearest_neighbour_search(x0, x, n_neighbours)
    return sortperm( abs.(x .- x0) )[1:n_neighbours]
end

# RBF definitions
RBF(r)   = r .^ 3 # example with cubic polyharmonics (r^3). NB: r = |x-x0| ≥ 0
ddRBF(r) = 6*r    # second derivative d²/dx² (because the differential problem deals with u" = d²u/dx²). Again here NB: r = |x-x0| ≥ 0

# interpolation matrix M given vector xn of n coordinates of interpolation nodes and center x0 (required for the polynomial part)
function interpolation_matrix(xn, x0)
    D = abs.( xn .- xn' )       # matrix of distances between each couple of nodes
    R = RBF(D)                  # RBF matrix, RBF evaluated for each entry of D
    P = ( xn .- x0 ) .^ [0 1 2] # polynomial part (degree 2) of interpolation matrix
    M = [ R P ; P' zeros(3,3) ] # interpolation matrix
    return M
end

# Construction of global matrix
n  = 7           # number of nearest neighbours for each node
A  = zeros(N, N) # global differential matrix to build node by node = row by row
qA = zeros(N)    # global RHS (right hand side) vector, derived from known function q
for i ∈ eachindex(X) # loop over each of the N nodes of the node distribution
    xi = X[i]                                       # coordinate of node i (center)
    neighbours = nearest_neighbour_search(xi, X, n) # vector of indices of n nearest neighours of node i
    x_neighbours = X[neighbours]                    # vector of coordinates of neighbours
    M = interpolation_matrix(x_neighbours, xi)      # interpolation matrix based on neighbours
    r_neighbours = abs.( x_neighbours .- xi )       # vector of distances from center (node i) to neighbours
    bM = [ ddRBF(r_neighbours); 0; 0; 2 ]           # vector of sought differential operator apllied to basis functions (RBF & poly), evaluated at center (node i)
    aM = (M')\bM                                    # vector of RBF-FD coefficients: solution of local square linear system M' * aM = bM
    A[i,neighbours] = aM[1:n]                       # RBF-FD coefficients aM in row i (referred to node i => i-th equation, row i of A)
                                                    # note that column indices are exactly the indices of neighbours
    qA[i] = q(xi)                                   # global RHS at center (node i) (could be computed as a single vector instruction outside the loop)
end

# OK, now we have written our (approximated) differential problem for each node, getting A * uA = qA
# A has N rows and N columns, uA is the vector of approximated values u at each node
# But we know u_a = u(a) (first node) and u_b = u(b) (last node), and these values must be imposed as BCs in order to have a unique solution
# => we don't use first and last equation, centered at first and last node (first and last row of A)
# => and move first and last value of uA (u_a and u_b) to the right side => move first and last column of A to the right side (inverted sign)
internal = 2:(N-1) # vector of indices of internal nodes, from 2nd to the (N-1)-th
boundary = [1,N]   # vector of indices of 2 boundary nodes (first and last)
A_internal  = A[internal,internal]           # (N-2)x(N-2) square matrix for internal nodes: we keep internal rows and columns
A_RHS       = A[internal,boundary]           # (N-2)x2 (two columns matrix) part multiplying first and last (known) values u_a and u_b (boundary values)
qA_internal = qA[internal] - A_RHS*[u_a,u_b] # global RHS for internal nodes - (minus) boundary contribution

# FINAL SOLUTION! A_internal * uh_internal = qA_internal => solve for vector uh_internal
uh_internal = A_internal\qA_internal

# Exact solution & error
u_internal  = u(X[internal])
err = norm( uh_internal - u_internal ) / norm( u_internal )
print("")
println("Error: $(err)")



# ------------------------------------------------------------

# Define J (obj function) ~> We want to reach a designed u_target
u_target(x) = (x .- a) .* (b .- x)
u_target_num = u_target(X[internal])

J(u_num) = norm(u_num - u_target_num)^2
g(u_num) = 2 * (u_num - u_target_num)

# Define ideal parameters
h_target = A_internal * u_target_num 
q_target_num = h_target + A_RHS*[u_a,u_b]
error_h(h) = norm(h-h_target)^2



function BB_geom_mean_step(x_k, x_k_plus_one, g_k, g_k_plus_one)

    return √(BB_long_step(x_k, x_k_plus_one, g_k, g_k_plus_one)*BB_short_step(x_k, x_k_plus_one, g_k, g_k_plus_one))
    
end

function adaptive_BB_step(x_k, x_k_plus_one, g_k, g_k_plus_one, number_of_optimizations, opt_cycle)  # my heuristic
    return ( (number_of_optimizations - opt_cycle)*BB_long_step(x_k, x_k_plus_one, g_k, g_k_plus_one) + (opt_cycle)*BB_short_step(x_k, x_k_plus_one, g_k, g_k_plus_one) ) / number_of_optimizations
end

function BB_long_step(x_k, x_k_plus_one, g_k, g_k_plus_one)
    #? should check g_k ≈ 0
    dx = x_k_plus_one - x_k
    dg = g_k_plus_one - g_k

    return (dx' * dx) / (dx' * dg)
end

function BB_short_step(x_k, x_k_plus_one, g_k, g_k_plus_one)
    #? should check g_k ≈ 0
    dx = x_k_plus_one - x_k
    dg = g_k_plus_one - g_k

    return (dx' * dg) / (dg' * dg)
end



# function adjoint_optimization(A, h0, J, g, number_of_optimization, gain)
#     #=
#         Use adjoint method to min J(u)= g'+u subject to Au=h,
#         starting from initial parameters h0
#     =# 

#     h = h0  # in our RBF case h = q - f

#     u_num = q(X[internal])
    
#     for opt_cycle ∈ range(1, number_of_optimization)

#         u_num = A \ h  # primal varibales from constraints
#                        # in our case: - primal variables = field
#                        #              - constraints = RBF equation

#         # Sensitivity analysis
#         g_num = g(u_num)
#         v = A' \ g_num

#         # Parameters update
#         step_size = gain * opt_cycle^(-0.5)
#         gradient_direction = v / norm(v)
#         h -= step_size * gradient_direction

#         println("J$(opt_cycle) = $(J(u_num)), error_h = $(error_h(h))")

#     end

#     h_opt = h
#     u_opt_num = u_num

#     return h_opt, u_opt_num

# end

function adjoint_optimization(A, h0, J, g, number_of_optimization)
    #=
        Use adjoint method to min J(u)= g'+u subject to Au=h,
        starting from initial parameters h0
    =#

    h_k = h0                              # in our RBF case h = q - f
    u_num_k = A \ h_k                     # field associated to current params
    
    println("J1 = $(J(u_num_k)), error_h = $(error_h(h_k))")
    
    # 1st update with simple gradient descent
    g_num_k = g(u_num_k)                  # sensitivity analysis
    v_k = A' \ g_num_k
    h_k_plus_one = h_k - v_k / norm(v_k)  # params update
    

    for opt_cycle ∈ range(2, number_of_optimization)

        u_num_k_plus_one = A \ h_k_plus_one

        println("J$(opt_cycle) = $(J(u_num_k_plus_one)), error_h = $(error_h(h_k_plus_one))")

        g_num_k_plus_one = g(u_num_k_plus_one)
        v_k_plus_one = A' \ g_num_k_plus_one
        
        # Step size
        γ_k = BB_long_step(h_k, h_k_plus_one, v_k, v_k_plus_one)

        h_k = h_k_plus_one
        v_k = v_k_plus_one

        h_k_plus_one = h_k - γ_k*v_k

    end

    h_opt = h_k_plus_one
    u_opt_num = A \ h_opt

    return h_opt, u_opt_num

end

h0 = qA_internal  # h0 = q0 - f
N = 1e4
h_opt, u_opt_num = adjoint_optimization(A_internal, h0, J, g, N)
q_opt_num = h_opt + A_RHS*[u_a,u_b]

println("Fitness value is: $(J(u_opt_num))")
println("Errors on the obtained q: $(q_opt_num - q_target_num)")
println("Errors on the obtained u: $(u_opt_num - u_target_num)")
println("Relative error on u: $(norm(u_opt_num - u_target_num) / norm(u_target_num))")
println("Relative error on q: $(norm(q_opt_num - q_target_num) / norm(q_target_num))")