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