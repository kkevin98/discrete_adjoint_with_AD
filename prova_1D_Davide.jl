# include("src/include_macro.jl")

# using .RBF_1D_utils
# using LinearAlgebra,Plots,Zygote

#node generation
a = 0.0
b = 1.0
N = 41
X = range(a,b, length=N)
boundary_idx = [1, N]
X_boundary   = X[boundary_idx]
internal_idx = 2:(N-1)
X_internal   = X[internal_idx]

#exact solution
u(x)  = -x.^2 .+ b*x   # Not known in advance
q(x)  = -2 .+ 0*x     # Known in advance (param)

boundary_conditions = u(X_boundary)

#RBF basic function
φ(r)   = r .^ 3  # r = |x-x_i| ≥ 0
ddφ(r) = 6*r 

n_points_in_stencil = 7

C = global_matrix(X, φ, ddφ, n_points_in_stencil)
approximated_u = solve_RBF(C, X, internal_idx, boundary_idx, boundary_conditions, q)

rel_error_on_u = norm(approximated_u - u(X_internal)) / norm(u(X_internal))

#cost function
approx_integral(u_vals,b) = (b/(N-1))*sum(u_vals)
cost(u_vals,b,E) = 0.5*( approx_integral(u_vals,b) - E)^2

#valori
u0 = approximated_u;
b0 = b;
approx_integral(u0,b0)
E0 = 2.0;
cost(u0,b0,E0)

# primi test gradiente
# grad_test = gradient((u,b) -> cost(u,b,E0), u0, b0)
grad_test = gradient((u,b,E) -> cost(u,b,E), u0, b0, E0)
grad_test[1]
grad_test[2]

