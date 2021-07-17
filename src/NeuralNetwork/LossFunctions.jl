function mse(y, t; back = false)
    if back == true
        return 2(y .- t)
    end
    return sum((y .- t) .^ 2)
end