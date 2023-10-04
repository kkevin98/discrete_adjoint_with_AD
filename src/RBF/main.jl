using LinearAlgebra
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
N = 20
X = range(a, b, length=N) # uniform, but not necessary

# (inefficient) n_neighbours nearest neighbour search centered in x0
# returns vector of indices of first n_neighbours nearest neighbours from x0
function nearest_neighbour_search(x0, x, n_neighbours)
    return sortperm( abs.(x .- x0) )[1:n_neighbours]
end

# RBF definitions
RBF(r)   = r .^ 3 # example with cubic polyharmonics (r^3). NB: r = |x-x0| ≥ 0
ddRBF(r) = 6*r    # second derivative d²/dx² (because the differential problem deals with u" = d²u/dx²). Again here NB: r = |x-x0| ≥ 0

# interpolation matrix M given vector xn of n coordinates of interpolation nodes
function interpolation_matrix(xn)
    D = abs.( xn .- xn' ) # matrix of distances between each couple of nodes
    M = RBF(D)            # RBF matrix, RBF evaluated for each entry of D
    # polynomial part can be added here M = [ RBF(D) p ; p' Z ] where p = polynomial matrix and Z=matrix of zeros...
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
    M = interpolation_matrix(x_neighbours)          # interpolation matrix based on neighbours
    r_neighbours = abs.( x_neighbours .- xi )       # vector of distances from center (node i) to neighbours
    bM = ddRBF(r_neighbours)                        # vector of sought differential operator of basis functions, evaluated at center (node i)
    aM = (M')\bM                                    # vector of RBF-FD coefficients: solution of local square linear system M' * aM = bM
    A[i,neighbours] = aM                            # RBF-FD coefficients aM in row i (referred to node i => i-th equation, row i of A)
                                                    # note that column indices are exactly the indices of neighbours
    qA[i] = q(xi)                                   # global RHS at center (node i)
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

# FINAL SOLUTION! A_internal * u_internal = qA_internal => solve for vector u_internal
u_internal = A_internal\qA_internal
err = norm(u_internal - u(X[internal])) / norm(u_internal)
println(err)