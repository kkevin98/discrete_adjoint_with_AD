module DiscreteAdjoint
include("GradientDescentTools.jl")
using .GradientDescentTools, LinearAlgebra

export adjoint_opt_BB, adjoint_opt, RBFEquation



function adjoint_opt_BB(dJdu, dJdq, RBFEquation, num_opt_cycles::Int64, eval_freq::Int64,  BB_step, eps=10^(-10))
    #=
        Use adjoint method to min J(u)= g'+u subject to Au=h,
        starting from initial parameters h0 and using
        Barzilai-Borwein method for gradient descent.
        Note that: - C, h, u are numerical
                   - J, g; BB_step are functions 
    =#

    # Initial parameters...
    C, u_k, q_k, f = RBFEquation.C, RBFEquation.u, RBFEquation.q, RBFEquation.f
    q_k_minus_one = zeros(size(q_k))
    v_k_minus_one = zeros(size(q_k))

    # Arrays to store the results
    steps_to_eval = 0:eval_freq:num_opt_cycles
    q_results = zeros(size(q_k)..., length(steps_to_eval))
    u_results = zeros(size(u_k)..., length(steps_to_eval))
    q_results[:,1] = q_k
    u_results[:,1] = u_k

    for opt_cycle ∈ range(1, num_opt_cycles)

        # Sensitivity analysis
        dJdu_k = dJdu(u_k, q_k)
        dJdq_k = dJdq(u_k, q_k)
        v_k = C' \ dJdu_k + dJdq_k

        if norm(v_k) < eps
            last_eval = opt_cycle ÷ eval_freq
            u_results[:, last_eval:end] .= NaN
            u_results[:, last_eval:end] .= NaN
            println("Optimum was found at iterations $(opt_cycle), before the end")
            break
        elseif opt_cycle == 1
            step_size = 1/norm(v_k)
        else
            step_size = BB_step(InfoToStep(q_k, q_k_minus_one, v_k, v_k_minus_one))
        end

        # store previous
        v_k_minus_one = v_k
        q_k_minus_one = q_k

        # update param
        q_k = q_k - step_size*v_k
        u_k = C \ (q_k - f)

        # Store results
        if opt_cycle in steps_to_eval
            idx = opt_cycle ÷ eval_freq + 1
            q_results[:,idx] = q_k
            u_results[:,idx] = u_k
        end

    end

    return q_k, u_k, q_results, u_results, steps_to_eval

end


function adjoint_opt(dJdu, dJdq, RBFEquation, num_opt_cycles::Int64, eval_freq::Int64, α=(∇, n)->1, eps=10^(-10))
    #=

    =#

    # Initial parameters...
    C, u_k, q_k, f = RBFEquation.C, RBFEquation.u, RBFEquation.q, RBFEquation.f

    # Arrays to store the results
    steps_to_eval = 0:eval_freq:num_opt_cycles
    q_results = zeros(size(q_k)..., length(steps_to_eval))
    u_results = zeros(size(u_k)..., length(steps_to_eval))
    q_results[:,1] = q_k
    u_results[:,1] = u_k

    for opt_cycle ∈ range(1, num_opt_cycles)

        # Sensitivity analysis
        dJdu_k = dJdu(u_k, q_k)
        dJdq_k = dJdq(u_k, q_k)
        v_k = C' \ dJdu_k + dJdq_k

        if norm(v_k) < eps
            last_eval = opt_cycle ÷ eval_freq + 1
            u_results = hcat(u_results[:, 1:last_eval], u_k)
            q_results = hcat(q_results[:, 1:last_eval], q_k)
            println("Optimum was found at iterations $(opt_cycle), before the end")
            break
        else
            step_size = α(v_k, opt_cycle)
        end

        # update param
        q_k = q_k - step_size*v_k
        u_k = C \ (q_k - f)

        # Store results
        if opt_cycle in steps_to_eval
            idx = opt_cycle ÷ eval_freq + 1
            q_results[:,idx] = q_k
            u_results[:,idx] = u_k
        end

    end

    return q_k, u_k, q_results, u_results, steps_to_eval

end



struct RBFEquation
    C
    u
    q
    f
end


end