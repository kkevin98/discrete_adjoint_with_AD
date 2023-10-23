module GradientDescentTools

export InfoToStep, BB_short_step, BB_long_step, BB_geom_mean_step

const machine_error::Float64 = 10^(-14)


struct InfoToStep

    k
    k_minus_one
    k_grad
    k_minus_one_grad

end



function BB_short_step(x::InfoToStep)
    #? Should check grad_k close to 0
    dx = x.k .- x.k_minus_one
    dg = x.k_grad .- x.k_minus_one_grad

    return (dx' * dg) / (dg' * dg)
end



function BB_long_step(x)
    dx = x.k .- x.k_minus_one
    dg = x.k_grad .- x.k_minus_one_grad

    return (dx' * dx) / (dx' * dg)
end



function BB_geom_mean_step(x::InfoToStep)

    return √(BB_long_step(x)*BB_short_step(x))
    
end


end