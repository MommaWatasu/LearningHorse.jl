"""
    LinearRegression()
Classic regression model. This struct has no parameter.
If you want to use polynomial model, use `Regression.make_design_matrix()`.

see also: [`make_design_matrix`](@ref)

# Example
```jldoctest regression
julia> x = [
    16.862463771320925 68.10823385851712
    15.382965696961577 65.4313485700859
    8.916228406218375 53.92034559524475
    10.560285659132695 59.17305391117168
    12.142253214135884 62.28708207525656
    5.362107221163482 43.604947901567414
    13.893239446341777 62.44348617377496
    11.871357065173395 60.28433066289655
    29.83792267802442 69.22281924803998
    21.327107214235483 70.15810991597944
    23.852372696012498 69.81780163668844
    26.269031430914108 67.61037566099782
    22.78907104644012 67.78105545358633
    26.73342178134947 68.59263965946904
    9.107259141706415 56.565383817343495
    29.38551885863976 68.1005579469209
    7.935966787763017 53.76264777936664
    29.01677894379809 68.69484161138638
    6.839609488194577 49.69794758177567
    13.95215840314148 62.058116579899085]; #These data are also used to explanations of other functions.

julia> t = [169.80980778351542, 167.9081124078835, 152.30845618985222, 160.3110300206261, 161.96826472170756, 136.02842285615077, 163.98131131382686, 160.117817321485, 172.22758529098235, 172.21342437006865, 171.8939175591617, 169.83018083884602, 171.3878062674257, 170.52487535026015, 156.40282783981309, 170.6488327896672, 151.69267899906185, 172.32478221316322, 145.14365314788827, 163.79383292080666];

julia> model = LinearRegression()
LinearRegression(Float64[])

julia> fit!(model, x, t)
3-element Vector{Float64}:
 -0.04772448076255398
  1.395963968616736
 76.7817095600793

julia> model(x)
20-element Vector{Float64}:
 171.05359766482795
 167.38737053144575
 151.62704681535598
 158.88117658330424
 163.15274911747872
 137.3967419011542
 163.28751869479999
 160.3699086857777
 171.99027166957023
 173.70207799107243
 173.10650291105486
 169.9096820022986
 170.31402414534642
 171.25872436348817
 155.31030802635905
 170.44522606721017
 151.45368882321284
 171.29242257091374
 145.83183688699864
 162.74674475052848
```
"""
mutable struct LinearRegression
    w::Array
    LinearRegression() = new(Array{Float64}(undef, 0))
end

"""
    fit!(model, x, t)
`x` must be the number of features in the first dimension and the second dimension must be the number of data.
"""
function fit!(model::LinearRegression, x, t)
    check_size(x, t)
    x = expand(x)
    model.w = inv(x * x') * x * t
end

(model::LinearRegression)(x) = expand(x)' * model.w