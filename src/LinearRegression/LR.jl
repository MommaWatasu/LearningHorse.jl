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
