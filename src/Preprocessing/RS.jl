mutable struct Robust
    p::AbstractVecOrMat
    Robust() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::Robust, x; dims=1)
    if dims == 1
        scaler.p = hcat([quantile(x[:, i], [0.25, 0.5, 0.75]) for i in 1 : size(x, 2)]...)
    elseif dims == 2
        scaler.p = hcat([quantile(x[i, :], [0.25, 0.5, 0.75]) for i in 1 : size(x, 1)]...)
    end
end

rs(x, q1, q2, q3) = @. (x-q2) / (q3-q1)

function transform!(scaler::Robust, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = rs(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = rs(x[i, :], p[:, i]...)
        end
    end
    return x
end

function fit_transform!(scaler::Robust, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x; dims=dims)
end

irs(x, q1, q2, q3) = @. x*(q3-q1)+q2

function inv_transform!(scaler::Robust, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = irs(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = irs(x[i, :], p[:, i]...)
        end
    end
    return x
end