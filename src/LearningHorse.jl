using LinearAlgebra
module LearningHorse

    export Linear_Regression
    module Linear_Regression
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
                x2 = ones(size(x)[1],1)
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
    end

    export Classification
    module Classification #this is Classification
        module SGD #This uses SGD for classification
            function fit(x, t, c; alpha = 0.001, tau_max = 100000, eps = 0.1)
                function softmax(a)
                    if ndims(a) == 1
                        return exp.(a) / sum(exp.(a))
                    else
                        s = sum(exp.(a), dims = 1)
                        for i in 2 : size(a)[1]
                            s = vcat(s, s)
                        end
                        return exp.(a) / s
                function CEL(w, x, t) #this is Cross entropy Loss
                    p = softmax(x * w)
                    grad = -(x' * (y - p))
                    return grad / length(x)
                end
                w = ones(length(x[0]) + 1, c)
                x = hcat(ones(len(x), 1), x)
                for tau in 1 : tau_max
                    grad = self.CEL(w, x, t)
                    w -= alpha * grad
                end
            end
        end
    end

    export Loss_Function
    module Loss_Function #Loss Fuunctions 
        export MSE
        function MSE(x, t, w) # this is Mean Square Error
        if length(x) != length(t)
            throw("The sizes of the arguments x and t you passed do not match.")
        end
        if length(size(x)) < 2
            y = w[1] * x .+ w[2]
        else
            x2 = ones(1, size(x)[2])
            x = vcat(x2, x)
            y = w' * x
        end
        mse = sum((y - t) .^ 2) / length(y) #mse is Mean Square error
        return mse
        end
    end

end
