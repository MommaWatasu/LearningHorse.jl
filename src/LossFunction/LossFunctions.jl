using LinearAlgebra
function MSE(y, t; mean_f = true) # this is Mean Square Error
    if size(y) != size(t)
        y = y'
    end
    mse = sum((y - t) .^ 2)
    if mean_f
        mse /= length(y) #mse is Mean Square error 
    end
    return mse
end
function CEE(p, t; mean_f = true, t_f = false) #this is Cross entropy Error
    function safe_log(x, miniv = 0.00000000001)
        return map(log, clamp.(x, miniv, Inf))
    end
    if t_f
        loss = -sum(t .* safe_log(p) + (1 .- t) .* safe_log(1 .- p))
    else
        loss = -sum(t .* safe_log(p))
    end
    if mean_f
        loss = loss / length(t)
    end
    return loss
end