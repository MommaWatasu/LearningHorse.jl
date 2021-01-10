module Horse

    export Simple_Regression
    module Simple_Regression
        export GSM
        function GSM(x, t, alpha = 0.001, tau_max = 100000, eps = 0.1) #this is Gradient steepest method
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
    end

  export Loss_Function
  module Loss_Function # this is Mean Square error
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