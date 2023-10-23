module DiscreteAdjoint
include("GradientDescentTools.jl")
using .GradientDescentTools, LinearAlgebra

export adjoint_opt_BB

function adjoint_opt_BB(A, h0, J, g, num_opt_cycles, BB_step, eps=10^(-14))
    #=
        Use adjoint method to min J(u)= g'+u subject to Au=h,
        starting from initial parameters h0 and using
        Barzilai-Borwein method for gradient descent.
        Note that: - A, h, u are numerical
                   - J, g; BB_step are functions 
    =#

    # Iniital parameters and associated variable values
    h_k_minus_one = zeros(size(h0))
    v_k_minus_one = zeros(size(h0))
    h_k = h0
    u_k = A \ h_k

    for opt_cycle ∈ range(1, num_opt_cycles)

        # Sensitivity analysis
        g_k = g(u_k)
        v_k = A' \ g_k

        if norm(v_k) < eps
            println("Optimum was found before the end of the iterations")
            break
        elseif opt_cycle == 1
            step_size = 1/norm(v_k) 
        else
            step_size = BB_step(InfoToStep(h_k, h_k_minus_one, v_k, v_k_minus_one))
        end

        # store previous
        v_k_minus_one = v_k
        h_k_minus_one = h_k

        # update param
        h_k = h_k - step_size*v_k
        u_k = A \ h_k

    end

    return h_k, u_k

end



end