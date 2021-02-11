using LinearAlgebra
using Statistics
module LearningHorse

    export LinearRegression
    module LinearRegression
        module SGD() #this is Slope Gradient Descent
            function fit(x, t; alpha = 0.001, tau_max = 100000, eps = 0.1)
                function DMSE(x,t,w)
                    if length(x) != length(t)
                        throw("The sizes of the arguments x and t you passed do not match.")
                    end
                    y = w[1] * x .+ w[2]
                    d_w0 = 2 * sum((y - t) .* x) / length(y)
                    d_w1 = 2 * sum((y - t)) / length(y)
                    return d_w0, d_w1
                end
                t1 = maximum(t) - minimum(t) / maximum(x) - minimum(x)
                t2 = minimum(t)
                w_init = [t1, t2]
                w_hist = zeros(tau_max, 2)
                w_hist[1, :] = w_init
                for tau = 1:tau_max
                    global a = tau
                    dmse = DMSE(x, t, w_hist[tau,:])
                    w_hist[tau + 1, 1] = w_hist[tau, 1] - alpha * dmse[1]
                    w_hist[tau + 1, 2] = w_hist[tau, 2] - alpha * dmse[2]
                    if maximum(broadcast(abs, dmse)) < eps
                        break
                    end
                end
                w0 = w_hist[a, 1]
                w1 = w_hist[a, 2]
                return w0, w1
            end
            function predict(x, w)
                t = w[1] * x .+ w[2]
                return t
            end
        end
        module MR#this is Multiple Regression
            function fit(x, t)
                try
                    w = nothing
                    if size(x) != size(t)
                        x = x'
                        t = t'
                    end
                    x2 = ones(size(x)[1],1)
                    x = hcat(x2, x)
                    w = inv(x' * x) * x' * t
                    return w
                catch
                    println("Perhaps the matrix x you passed does not have an inverse matrix.
In that case, use ridge regression.")
                end
            end
            function predict(x, w)
                x2 = ones(1, size(x)[2])
                x = vcat(x2, x)
                return w' * x
            end
        end
        module RR()#this is Ridge Regression
            using LinearAlgebra
            function fit(x, t; alpha = 0.1)
                w = nothing
                if size(x)[1] != size(t)[1]#Processing when the matrix is ​​organized by dependent variable
                    x = x'
                    t = t'
                end
                x2 = ones(size(x)[1], 1)
                x = hcat(x2, x)
                i = Matrix{Float64}(I, size(x)[2], size(x)[2])
                w = inv(x' * x + alpha * i) * x' * t
                return w
            end
            function predict(x, w)
                x2 = ones(1, size(x)[2])
                x = vcat(x2, x)
                return w' * x
            end
        end
        module LR()#this is Lasso Regression
            using LinearAlgebra
            function sfvf(x, y)
                sign(x) * max(abs(x) - y, 0)
            end
            function fit(x, t; alpha = 0.1, tol = 0.0001, mi = 1000000)
                function update(n, d, x, t, w, alpha)
                    l = length(w)
                    w[1] = sum(t - x * w[2:l]) / n
                    wvec = ones(n) * w[1]
                    for k = 1:d
                        ww = w[2:l]
                        ww[k] = 0
                        q = dot(vec(t) - wvec - x * ww, x[:, k])
                        r = dot(x[:, k], x[:, k])
                        w[k+1] = sfvf(q / r, alpha)
                    end
                end
                if size(x)[1] != size(t)[1]
                    x = x'
                    t = t'
                end
                n, d = size(x)
                w = zeros(d + 1)
                e = 0.0
                for i = 1:mi
                    eb = e
                    update(n, d, x, t, w, alpha)
                    e = sum(broadcast(abs, w)) / size(w)[1]
                    if abs(e - eb) <= tol
                        return w
                    end
                end
            end
            function predict(x, w)
                x2 = ones(1, size(x)[2])
                x = vcat(x2, x)
                return w' * x
            end
        end
        module BFM() #this use Basis Function Model for regression
            using LinearAlgebra
            function gauss(x, m, s)
                return exp.(-(x .- m) .^ 2 ./ (2 * s ^ 2))
            end
            function polynomial(x, i)
                return x .^ i
            end
            function fit(x, t, M; m = [], s = nothing, alpha = 0.0, f = "gauss")
                if m == []
                    ma, mi = round.(Int, (maximum(x), minimum(x)))
                    inter = (ma - mi) / (M - 1)
                    for j in 1 : M
                        push!(m, mi)
                        mi += inter
                    end
                end
                if s == nothing
                    s = inter
                end
                if size(x)[1] != size(t)[1]#Processing when the matrix is ​​organized by dependent variable
                    x = x'
                    t = t'
                end
                i = Matrix{Float64}(I, M + 1, M + 1)
                try
                    phi = ones(size(x)[2], M + 1)
                    if f == "gauss"
                        for j in 1 : M
                            phi[:, j] = gauss(x, m[j], s)
                        end
                    elseif f == "polynomial"
                        for j in 1 : M
                            phi[:, j] = polynomial(x, j - 1)
                        end
                    end
                    phit = phi'
                    b = inv(phit * phi + i * alpha)
                    c = b * phit
                    w = c * t
                    return w, m, s
                catch
                    println("Perhaps the matrix x you passed does not have an inverse matrix. In that case, you should add the 'alpha' to the arguments('alpha' is a regularization term)")
                end
            end
            function predict(x, w, m, s; f = "gauss")
                w_len = length(w)
                t = fill!(copy(x), 0)
                if f == "gauss"
                    for i in 1 : w_len - 1
                        t += w[i] * gauss(x, m[i], s)
                    end
                elseif f == "polynomial"
                    for i in 1 : w_len - 1
                        t += w[i] * polynomial(x, i - 1)
                    end
                end
                t .+= w[w_len]
                return t
            end
        end
    end

    export Classification
    module Classification #this is Classification
        function LE(t)#this is Label Encoder
            n = 1
            le = length(t)
            target = ones(1, le)
            s = [v for v in Set(t)]
            for i in 1:le
                for ii in 1:length(s)
                    if t[i] == s[ii]
                        target[i] = ii
                        break
                    end
                end
            end
            return target
        end
        function CTOH(ts) #This is Convert to one hot
            try
                ts = broadcast(Int, ts)
            catch
                throw("Perhaps the array of objective variables you passed has a non-zero decimal point in the name of the class (as in 1.1).Please change it to an integer type.")
            end
            if 0 in ts
                ts .+= 1
            end
            c = maximum(ts)
            oh = zeros(size(ts)[2], c)
            i = 1
            for t in ts
                oh[i, t] = 1.0
                i += 1
            end
            return oh
        end
        module MS #This is Multiclass softmax
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
            #Pass so that the row is each data sample and the column is each feature.
            function predict(x, w)
                x2 = ones(size(x)[1], 1)
                x = hcat(x2, x)
                s = softmax(x * w)
                p = [findfirst(s[i, :] .== maximum(s[i, :])) for i in 1:size(s)[1]]
            end
        end
        module OVR #this is One-vs-Rest
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
            function predict(x, w)
                x = hcat(ones(size(x)[1], 1), x)
                p = []
                for i in 1 : size(x)[1]
                    tp =  [sigmoid(dot(w[j, :][1], x[i, :])) for j in 1:size(w)[1]]
                    tp = argmax(tp)
                    push!(p, tp)
                end
                return p
            end
        end
    end

　　export Preprocessing
    module Preprocessing
        module SS() #this is Standard Scaler
            using Statistics
            p = []
            function fit(x; axis = 1) #this function dose only performs calculations.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [mean(x), std(x)]
                else
                    p = []
                    for i in 1:size(x)[2]
                        push!(p, [mean(x[:, i]), std(x[:, i])])
                    end
                end
                return p
            end
            function transform(x, p; axis = 1) #Only transform is performed based on the calculation results of the'fit'function and'fit_transform'function.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t = (x .- p[1]) / p[2]
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i] .- p[i][1]) / p[i][2])
                    end
                end
                return t
            end
            function inverse_transform(x, p; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t = (x * p[2]) .+ p[1]
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i][1] * p[i][2]) .+ p[i][1])
                    end
                end
                return t
            end
            function fit_transform(x; axis = 1) #This function handles both the fit function and the transform function.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [mean(x), std(x)]
                    t = (x .- p[1]) / p[2]
                else
                    t, p = [], []
                    for i in 1:size(x)[2]
                        push!(p, [mean(x[:, i]), std(x[:, i])])
                        push!(t, (x[:, i] .- p[i][1]) / p[i][2])
                    end
                end
                return t, p
            end
        end
        module MM() #this is MinMax Scaler
            function fit(x; axis = 1)　#this function dose only performs calculations.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [maximum(x), minimum(x)]
                else
                    p = []
                    for i in 1:size(x)[2]
                        push!(p, [maximum(x[:, i]), minimum(x[:, i])])
                    end
                end
                return p
            end
            function transform(x, p; axis = 1)　#Only transform is performed based on the calculation results of the'fit'function and'fit_transform'function.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t =  (x .- p[2]) / (p[1] - p[2])
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i] .- p[i][2]) / (p[i][1] - p[i][2]))
                    end
                end
                return t
            end
            function inverse_transform(x, p; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t = (x * (p[1] - p[2])) .+ p[2]
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i][1] * (p[i][1] - p[i][2])) .+ p[i][2])
                    end
                end
                return t
            end
            function fit_transform(x; axis = 1)　#This function handles both the fit function and the transform function.
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [maximum(x), minimum(x)]
                    t = (x .- p[2]) / (p[1] - p[2])
                else
                    p, t = [], []
                    for i in 1:size(x)[2]
                        push!(p, [maximum(x[:, i]), minimum(x[:, i])])
                        push!(t, (x[:, i] .- p[i][2]) / (p[i][1] - p[i][2]))
                    end
                end
                return t, p
            end
        end
        module RS() #this is Robust Scaler
            using Statistics
            function fit(x; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [quantile(x, 1/4), quantile(x, 1/2), quantile(x, 3/4)]
                else
                    p = []
                    for i in 1:size(x)[2]
                        push!(p, [quantile(x[:, i], 1/4), quantile(x[:, i], 1/2), quantile(x[:, i], 3/4)])
                    end
                end
                return p
            end
            function transform(x, p; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t = (x .- p[2]) / (p[3] - p[1])
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i] .- p[i][2]) / (p[i][3] - p[i][1]))
                    end
                end
                return t
            end
            function inverse_transform(x, p; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    t = (x * (p[3] - p[1])) .+ p[2]
                else
                    t = []
                    for i in 1:size(x)[2]
                        push!(t, (x[:, i][1] * (p[i][3] - p[i][1])) .+ p[i][2])
                    end
                end
                return t
            end
            function fit_transform(x; axis = 1)
                if axis == 2
                    x = x'
                end
                if ndims(x) == 1
                    p = [quantile(x, 1/4), quantile(x, 1/2), quantile(x, 3/4)]
                    t = (x .- p[2]) / (p[3] - p[1])
                else
                    t, p = [], []
                    for i in 1:size(x)[2]
                        push!(p, [quantile(x[:, i], 1/4), quantile(x[:, i], 1/2), quantile(x[:, i], 3/4)])
                        push!(t,  (x[:, i] .- p[i][2]) / (p[i][3] - p[i][1]))
                    end
                end
                return t, p
            end
        end
    end
    export LossFunction
    module LossFunction #Loss Fuunctions 
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
            else
                x2 = ones(1, size(x)[2])
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
    end

end
