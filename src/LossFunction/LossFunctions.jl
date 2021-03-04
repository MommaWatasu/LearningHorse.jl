using LinearAlgebra
function MSE(x, t, w; b = "", m = nothing, s = nothing, mean_f = true) # this is Mean Square Error
    function gauss_func(x, m, s)
        return exp.(-(x .- m) .^ 2 ./ (2 * s ^ 2))
    end 
    function polynomial_func(x, i)
        return x .^ i
    end
    if length(x) != length(t)
        throw("The sizes of the arguments x and t you passed do not match.")
    end
    if b != ""
        w_len = length(w)
        y = fill!(copy(x), 0)
        if b == "gauss"
            for i in 1 : w_len - 1
                y += w[i] * gauss_func(x, m[i], s)
            end
        elseif b == "polynomial"
            for i in 1 : w_len - 1
                t += (w[i] * polynomial_func(x, i - 1))'
            end
        end
        y .+= w[w_len]
        if size(t)[1] != size(y)[1]
            y = y'
        end
    elseif length(size(x)) < 2
        y = w[1] * x .+ w[2]
    else x2 = ones(1, size(x)[2])
        x = vcat(x2, x)
        y = w' * x
    end
    mse = sum((y - t) .^ 2)
    if mean_f
        mse /= length(y) #mse is Mean Square error 
    end
    return mse
end
function CEE(x, t, w; mean_f = true, sigmoid_f = false, t_f = false) #this is Cross entropy Error
    function softmax(a)
        if ndims(a) == 1
            grad =  exp.(a) ./ sum(exp.(a))
        else
            grad =  exp.(a) ./ sum(exp.(a), dims = 2)
        end
        return grad
    end
    function sigmoid(a)
        return 1.0 / (1 + exp(-a))
    end
    function safe_log(x, miniv = 0.00000000001)
        return map(log, clamp.(x, miniv, Inf))
    end
    
    if size(x)[2] + 1 != size(w)[1]#Processing when the matrix is ​​organized by dependent variable
        x = x'
    end
    x = hcat(ones(size(x)[1], 1), x)
    if sigmoid_f
        p = []
        for i in 1 : size(x)[1]
            tp = [sigmoid(dot(w[j, :][1], x[i, :])) for j in 1:size(w)[1]]
            tp = tp'
            if p == []
                p = tp
            else
                p = vcat(p, tp)
            end
        end
    else
        p = softmax(x * w)
    end
    if t_f
        loss = -sum(t .* safe_log(p) + (1 .- p) .* safe_log(1 .- p))
    else
        loss = -sum(t .* safe_log(p))
    end
    if mean_f
        loss = loss / length(x[1, :])
    end
    return loss
end