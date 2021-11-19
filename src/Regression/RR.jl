mutable struct Ridge
    w::Array
    α::Float64
    Ridge(alpha = 0.1) = new(Array{Float64}(undef, 0), alpha)
end

function fit!(model::Ridge, x, t)
    check_size(x, t)
    x = hcat(x, ones(size(x, 1), 1))
    n = size(x, 2)
    _I = Matrix{Float64}(I, n, n)
    model.w = inv(x' * x .+ model.α * _I) * x' * t
end

predict(model::Ridge, x) = hcat(x, ones(size(x, 1), 1)) * model.w