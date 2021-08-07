using LinearAlgebra
function sigmoid(a)
    return 1.0 / (1 + exp(-a))
end
function fit(x, t; alpha = 0.01, tau_max = 1000)
    function CEE(w, x, t)
        grad = zeros(size(w))
        for i in 1:length(t)
            ti = (t[i] > 0) ? 1 : 0
            h = sigmoid(dot(w, x[i, :]))
            grad += ((h - ti) * x[i, :])'
        end
        return grad / length(t)
    end
    c = size(t)[2]
    w = []
    x = hcat(ones(size(x)[1], 1), x)
    for i in 1:c
        if size(x)[1] != size(t[:, i])[1]#Processing when the matrix is ​​organized by dependent variable
            x = x'
        end
        w0 = ones(1, length(x[1, :]))
        for j in 1:tau_max
            grad = CEE(w0, x, t[:, i])
            w0 -= alpha * grad
        end
        push!(w, w0)
    end
    return w
end

function forecast(x, w)
    x = hcat(ones(size(x)[1], 1), x)
    p = nothing
    for i in 1 : size(x)[1]
        tp = [sigmoid(dot(w[j, :][1], x[i, :])) for j in 1:size(w)[1]]
        tp = tp'
        (p == nothing) ? p = tp : p = vcat(p, tp)
    end
    return p
end

function predict(x, w)
    x = hcat(ones(size(x)[1], 1), x)
    p = []
    for i in 1 : size(x)[1]
        tp = [sigmoid(dot(w[j, :][1], x[i, :])) for j in 1:size(w)[1]]
        tp = argmax(tp)
        push!(p, tp)
    end
    return p
end
