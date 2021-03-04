using Statistics
function fit(x; axis = 1)
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        p = [quantile(x, 1/4), quantile(x, 1/2), quantile(x, 3/4)]
    else
        p = []
        for i in 1:size(x)[2]
            push!(p, [quantile(x[:, i], 1/4), quantile(x[:, i], 1/2), quantile(x[:, i], 3/4)])
        end
    end 
    return p
end
function transform(x, p; axis = 1)
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        t = (x .- p[2]) / (p[3] - p[1])
    else
        t = []
        for i in 1:size(x)[2]
            push!(t, (x[:, i] .- p[i][2]) / (p[i][3] - p[i][1]))
        end
    end
    return t
end
function inverse_transform(x, p; axis = 1)
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        t = (x * (p[3] - p[1])) .+ p[2]
    else 
        t = []
        for i in 1:size(x)[2]
            push!(t, (x[:, i][1] * (p[i][3] - p[i][1])) .+ p[i][2])
        end
    end 
    return t
end
function fit_transform(x; axis = 1)
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        p = [quantile(x, 1/4), quantile(x, 1/2), quantile(x, 3/4)]
        t = (x .- p[2]) / (p[3] - p[1])
    else
        t, p = [], []
        for i in 1:size(x)[2]
            push!(p, [quantile(x[:, i], 1/4), quantile(x[:, i], 1/2), quantile(x[:, i], 3/4)])
            push!(t, (x[:, i] .- p[i][2]) / (p[i][3] - p[i][1]))
        end
    end
    return t, p
end
