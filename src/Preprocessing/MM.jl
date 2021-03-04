function fit(x; axis = 1)　#this function dose only performs calculations.
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        p = [maximum(x), minimum(x)]
    else
        p = []
        for i in 1:size(x)[2]
            push!(p, [maximum(x[:, i]), minimum(x[:, i])])
        end
    end
    return p
end 
function transform(x, p; axis = 1)　#Only transform is performed based on the calculation results of the'fit'function and'fit_transform'function.
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        t = (x .- p[2]) / (p[1] - p[2])
    else
        t = []
        for i in 1:size(x)[2]
            push!(t, (x[:, i] .- p[i][2]) / (p[i][1] - p[i][2]))
        end
    end
    return t
end
function inverse_transform(x, p; axis = 1)
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        t = (x * (p[1] - p[2])) .+ p[2]
    else 
        t = []
        for i in 1:size(x)[2]
            push!(t, (x[:, i][1] * (p[i][1] - p[i][2])) .+ p[i][2])
        end
    end 
    return t
end
function fit_transform(x; axis = 1)　#This function handles both the fit function and the transform function.
    if axis == 2
        x = x'
    end
    if ndims(x) == 1
        p = [maximum(x), minimum(x)]
        t = (x .- p[2]) / (p[1] - p[2])
    else
        p, t = [], []
        for i in 1:size(x)[2]
            push!(p, [maximum(x[:, i]), minimum(x[:, i])])
            push!(t, (x[:, i] .- p[i][2]) / (p[i][1] - p[i][2]))
        end
    end
    return t, p
end
