using LinearAlgebra
function softmax(a)
    if ndims(a) == 1
        return exp.(a) ./ sum(exp.(a))
    else 
        return exp.(a) ./ sum(exp.(a), dims = 2)
    end
end
function fit(x, t; alpha = 0.01, tau_max = 1000) 
    function CEE(w, x, t) #this is Cross entropy Error
        p = softmax(x * w)
        grad = -(x' * (t - p))
        return grad / length(x[:, 1]) 
    end
    if size(x)[1] != size(t)[1]#Processing when the matrix is ​​organized by dependent variable
        x = x'
    end
    x = hcat(ones(size(x)[1], 1), x)
    w = ones(size(x)[2], size(t)[2])
    for tau in 1 : tau_max
        grad = CEE(w, x, t)
        w -= alpha * grad
    end
    return w
end

function forecast(x, w)
    x = hcat(ones(size(x)[1], 1), x)
    return softmax(x * w)
end

#Pass so that the row is each data sample and the column is each feature.
function predict(x, w)
    x = hcat(ones(size(x)[1], 1), x)
    s = softmax(x * w)
    p = [findfirst(s[i, :] .== maximum(s[i, :])) for i in 1:size(s)[1]]
end
