module LossFunction

using LinearAlgebra
using Statistics

export mse, cee

safe_log(x, miniv = 1e-8) = map(log, clamp.(x, miniv, Inf))

"""
    mse(y, t)
Return the mean square error:`mean((y .- t) .^ 2)`

# Example
```jldoctest
julia> loss = mse
mse (generic function with 1 method)

julia> y, t = [1, 3, 2, 5], [1, 2, 3, 4]
([1, 3, 2, 5], [1, 2, 3, 4])

julia> loss(y, t)
0.75
```
"""
mse(y, t) = mean((y.-t).^2)

"""
    cee(y, t, dims = 1)
Return the cross entropy error:`mean(-sum(t.*log(y)))`

# Example
```jldoctest
julia> loss = cee
cee (generic function with 1 method)

julia> y, t = [1, 3, 2, 5], [1, 2, 3, 4]
([1, 3, 2, 5], [1, 2, 3, 4])

julia> loss(y, t)
-10.714417768752456
```
"""
cee(y, t) = mean(-sum(t.*safe_log(y)))

end