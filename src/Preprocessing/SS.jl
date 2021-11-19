mutable struct Standard
    p::AbstractVecOrMat
    Standard() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::Standard, x; dims=1)
    scaler.p = vcat(mean(x, dims=dims), std(x, dims=dims))
end

ss(x, m, s) = @. (x-m)/s

function transform!(scaler::Standard, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = ss(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = ss(x[i, :], p[:, i]...)
        end
    end
    return x
end

function fit_transform!(scaler::Standard, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x, dims=dims)
end

iss(x, m, s) = @. x*s+m

function inv_transform!(scaler::Standard, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = iss(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = iss(x[i, :], p[:, i]...)
        end
    end
    return x
end