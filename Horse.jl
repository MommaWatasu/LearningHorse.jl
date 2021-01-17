using LinearAlgebra
module Horse

    export Linear_Regression
    module Linear_Regression
        module SGD() #this is Slope Gradient Descent
            function fit(x, t, alpha = 0.001, tau_max = 100000, eps = 0.1)
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
                    x2 = zeros(size(x)[1],1)
                    x = hcat(x2, x)
                    w = inv(x' * x) * x' * t
                    return w
                catch
                    println("Perhaps the matrix x you passed does not have an inverse matrix.
In that case, use ridge regression.")
                end
            end
            function predict(x, w)
                x2 = zeros(size(x)[1],1)
                x = hcat(x2, x)
                return x * w
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
                x2 = zeros(size(x)[1], 1)
                x = hcat(x2, x)
                i = Matrix{Float64}(I, size(x)[2], size(x)[2])
                w = inv(x' * x + alpha * i) * x' * t
                return w
            end
            function predict(x, w)
                x2 = zeros(size(x)[1], 1)
                x = hcat(x2 * x)
                return x * w
            end
        end
        module LR()#this is Lasso Regression
            using LinearAlgebra
            function sfvf(x, y)
                sign(x) * maximum(abs(x) - y, 0)
            end
            function fit(x, t; alpha = 0.1, tol = 0.0001, mi = 1000000)
                function update(n, d, x, t, w, alpha)
                    l = length(w)
                    w[1] = sum(t - x * w[2:l]) / n#this line has problem
                    wvec = ones(n) * w[1]
                    for k = 1:d
                        ww = w[2:l]
                        ww[k] = 0
                        println(size(vec(t) - wvec - x * ww), size(x[:, k]))
                        q = dot(vec(t) - wvec - x * ww, x[:, k])
                        r = dot(x[:, k], x[:, k]) #this line has problem
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
                    e = sum(abs(w)) / size(w)[1]
                    if abs(e - eb) <= tol
                        break
                    end
                end
            end
        end
    end

    export Loss_Function
    module Loss_Function # this is Mean Square Error
        export MSE
        function MSE(x, t, w)
        if length(x) != length(t)
            throw("The sizes of the arguments x and t you passed do not match.")
        end
        y = w[1] * x .+ w[2]
        mse = sum((y - t) .^ 2) / length(y) #mse is Mean Square error
        return mse
        end
    end
# Write your package code here.

end
