using Statistics

"""
    mse(y, t; back = false)
Return the mean square error:
    mean((y .- t) .^ 2)
# Example
```
julia> loss = mse
mse (generic function with 1 method)

julia> y, t = [1, 3, 2, 5], [1, 2, 3, 4]
([1, 3, 2, 5], [1, 2, 3, 4])

julia> loss(y, t)
0.75
```
"""
mse(y, t) = mean(y.-t).^2
mse(y, t, back::Bool) = @. 2(y-t)