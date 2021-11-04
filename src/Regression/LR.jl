mutable struct Lasso
    w::Array
    α::Float64
    tol::Float64
    mi::Int64
    Lasso(; alpha = 0.1, tol = 1e-4, mi = 1e+8) = new(Array{Float64}(undef, 0), alpha, tol, mi)
end

sfvf(x, y) = sign(x) * max(abs(x) - y, 0)

function fit!(model::Lasso, x, t)
    function update(n, d, x, t, w, α)
        w[1] = mean(t - x * w[2:end])
        wvec = fill!(Array{Float64}(undef, d), w[1])
        for k in 1 : n
            ww = w[2:end]
            ww[k] = 0
            q = (t - wvec - x * ww) ⋅ x[:, k]
            r = x[:, k] ⋅ x[:, k]
            w[k+1] = sfvf(q / r, α)
        end
    end
    α, tol, mi = model.α, model.tol, model.mi
    check_size(x, t)
    if ndims(x) == 1
        d, n = (size(x, 1), 1)
        x = x[:, :]
    else
        d, n = size(x)
    end
    w = zeros(n + 1)
    e = 0.0
    for _ in 1 : mi
        eb = e
        update(n, d, x, t, w, α)
        e = sum(broadcast(abs, w)) / size(w, 1)
        if abs(e - eb) <= tol
            break
        end
    end
    model.w = w[end:-1:1]
end

predict(model::Lasso, x) = hcat(x, ones(size(x, 1), 1)) * model.w