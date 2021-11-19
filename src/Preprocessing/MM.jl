mutable struct MinMax
    p::AbstractVecOrMat
    MinMax() = new(Array{Float64}(undef, 0))
end

function fit!(scaler::MinMax, x; dims=1)
    scaler.p = vcat(maximum(x, dims=dims), minimum(x, dims=dims))
end 

mms(x, ma, mi) = @. (x-mi) / (ma - mi)

function transform!(scaler::MinMax, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = mms(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = mms(x[i, :], p[:, i]...)
        end
    end
    return x
end

imms(x, ma, mi) = @. x*(ma-mi)+mi

function inv_transform!(scaler::MinMax, x; dims=1)
    p = scaler.p
    check_size(x, p, dims)
    if dims == 1
        for i in 1 : size(x, 2)
            x[:, i] = imms(x[:, i], p[:, i]...)
        end
    elseif dims == 2
        for i in 1 : size(x, 1)
            x[i, :] = imms(x[i, :], p[:, i]...)
        end
    end
    return x
end

function fit_transform!(scaler::MinMax, x; dims=1)
    fit!(scaler, x, dims=dims)
    transform!(scaler, x, dims=dims)
end