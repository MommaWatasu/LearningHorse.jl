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
