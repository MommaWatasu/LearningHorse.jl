function fit(x, t)
    try w = nothing
        if size(x) != size(t)
            x = x'
            t = t'
        end
        x2 = ones(size(x)[1],1)
        x = hcat(x2, x)
        w = inv(x' * x) * x' * t
        return w
    catch
        println("Perhaps the matrix x you passed does not have an inverse matrix.
In that case, use ridge regression.")
    end
end
function predict(x, w)
    x2 = ones(1, size(x)[2]) 
    x = vcat(x2, x)
    return w' * x
end
