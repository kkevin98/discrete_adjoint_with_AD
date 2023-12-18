using LinearAlgebra, Zygote, Plots

load_local_pkg = true #IMPORTANTE: mettere false dopo la prima esecuzione
if load_local_pkg 
    include("src/RadialBasisFunction.jl")
    include("src/DiscreteAdjoint.jl")
    include("src/GradientDescentTools.jl")

    using .RadialBasisFunction
    using .DiscreteAdjoint
    using .GradientDescentTools
end

a = 0
b = Float64(π)
N = 41

# Points will change each time b will change
X = range(a,b, length=N)
boundary_idxs = [1, N]
internal_idxs = 2:(N-1)

ω = 1
u(x) = sin.(ω*x)
q(x) = -ω^2*u(x)  # Known in advance (param)
boundary_conditions = [0, 0]   # Decided by us <- occhio che quì le boundary conditions sono dettate dalla funzione q()

# Defininig J (obj function)
E = 2.5
J(u_num, b) = 1/2 * ( b/(N-1) * sum(u_num) - E)^2  #* u_num is so small as if it does not exist

#funzioni RBF
n_points_in_stencil = 7
φ(r)   = r .^ 3  # r = |x-x_i| ≥ 0
ddφ(r) = 6*r   #second derivative d²/dx²

#ridefinizione funzioni con multiple dispatch (sarebbe da definirle così già all'inizio e quindi riscrivere le funzioni RBF: pensare a un modo intelligente)
x(i,b)     = (i-1)*b/(N-1)
r(i,j,b)   = sqrt((x(i,b)-x(j,b))^2)
φ(i,j,b)   = r(i,j,b)^3     # r = |x-x_i| ≥ 0 #poi sarà da derivare anche rispetto a x
∇²φ(i,j,b) = 6*r(i,j,b)     # second derivative d²/dx²

q(i,b)     = -ω^2*u(x(i,b))

function dφ_db(i,j,b)
    if i != j 
        return (gradient(b -> φ(i,j,b), b)[1])
    else
        return 0
    end
end
function d∇²φ_db(i,j,b) #serve per dΨ_db
    if i != j
        return (gradient(b -> ∇²φ(i,j,b), b)[1])
    else 
        return 0
    end
end
function cᵢ_from_stencil(central_point, nearby_points, RBF, ddRBF)
    
    r_neighbours = abs.(nearby_points .- central_point)  # vector of distances from stencil's center to neighbours

    M = interpolation_matrix(nearby_points, central_point, RBF)
    Ψ = ddRBF(r_neighbours)
    Π = [0, 0 ,2]
    c = (M') \ [Ψ; Π]

    return c #without discarding poly terms

end

dq_db  = zeros(length(internal_idxs))
λ₂     = zeros(length(internal_idxs))
dΨ_db  = zeros(n_points_in_stencil+3) #derivata del rhs del sistemino locale: M'c = L 
dM_db  = zeros(n_points_in_stencil+3, n_points_in_stencil+3)

u_approximated_total = zeros(N)

cost_function_log = []
b_log = [b]

N_iterations = 30
for iteration = 1:N_iterations #ciclo ottimizzazione gradient descend

    C = global_matrix(X, φ, ddφ, n_points_in_stencil) #allocazione inefficiente
    u_approximated = solve_RBF(C, X, internal_idxs, boundary_idxs, boundary_conditions, q); #allocazione inefficiente
    u_approximated_total[boundary_idxs] .= boundary_conditions
    u_approximated_total[internal_idxs] .= u_approximated

    push!(cost_function_log, J(u_approximated, b))

    #gradienti cost function
    dJ_du,∂J_∂b = gradient((u_num,b) -> J(u_num,b), u_approximated,b) #allocazione inefficiente
    dJ_du       = Array(dJ_du) #mi assicuro che sia in formato Array

    #first adjoint problem
    Cᵢ = C[internal_idxs,internal_idxs] #just internal nodes
    λ₁ = Cᵢ\dJ_du

    int_count = [0]
    for i in internal_idxs #ciclo sui nodi interni
        int_count[1] += 1

        xi = X[i]
        neighbours_idxs = nearest_neighbour_search(xi, X, n_points_in_stencil) #allocazione inefficiente

        dq_db[int_count[1]] = gradient(b -> q(i,b), b)[1]

        dΨ_db[1:n_points_in_stencil] .= [d∇²φ_db(i,j,b) for j in neighbours_idxs] 
        #i termini restanti di dΨ_db dovrebbero essere tutti zero con polinomio di grado 3

        cᵢ  = cᵢ_from_stencil(xi, X[neighbours_idxs], φ, ddφ)    #allocazione inefficiente

        M   = interpolation_matrix(X[neighbours_idxs], xi, φ)    #allocazione inefficiente
        α_β = M\[u_approximated_total[neighbours_idxs]; [0,0,0]] #allocazione inefficiente

        #costruzione di dM_db
        for ngb_i in eachindex(neighbours_idxs)
            ix_i = neighbours_idxs[ngb_i]
            for ngb_j in eachindex(neighbours_idxs)
                ix_j = neighbours_idxs[ngb_j]

                dM_db[ngb_i,ngb_j] = dφ_db(ix_i,ix_j,b)
            end
            dM_db[ngb_i, n_points_in_stencil+2] = gradient(b -> x(ix_i,b), b)[1]
            dM_db[n_points_in_stencil+2, ngb_i] = dM_db[ngb_i, n_points_in_stencil+2]
            dM_db[ngb_i, n_points_in_stencil+3] = gradient(b -> x(ix_i,b)^2, b)[1]
            dM_db[n_points_in_stencil+3, ngb_i] = dM_db[ngb_i, n_points_in_stencil+3]
        end

        λ₂[int_count[1]] = (dΨ_db - dM_db'*cᵢ)' * α_β
    end

    dJ_db = ∂J_∂b' + λ₁'*(dq_db - λ₂)

    global b = b - 0.1*dJ_db #minimizzazione della cost function
    push!(b_log, b)

    global X = range(a,b, length=N) #aggiornamento delle coordinate (sarebbe da parametrizzare tutto un po' meglio)

end


