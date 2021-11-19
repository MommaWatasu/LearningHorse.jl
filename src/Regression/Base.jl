"""
    LinearRegression()
Classic regression model. This struct has no parameter.
If you want to use polynomial model, use `Regression.make_design_matrix()`.

see also: [`make_design_matrix`](@ref)

# Example
```
julia> 
```
"""
mutable struct LinearRegression
    w::Array
    LinearRegression() = new(Array{Float64}(undef, 0))
end

function fit!(model::LinearRegression, x, t)
    check_size(x, t)
    x = hcat(x, ones(size(x, 1), 1))
    model.w = inv(x' * x) * x' * t
end

predict(model::LinearRegression, x) = hcat(x, ones(size(x, 1), 1)) * model.w