# The First Step

## First Model
First, let's try the simplest model, the linear regression model.
The code is:
```
using LearningHorse.Regression
using LearningHorse.Regression: fit!, predict

#data
x = 5 .+ 25 .* rand(20)
t = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)
x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
train_data = hcat(x, x1)

#model
model = LinearRegression()
fit!(model, train_data, t)
```
Just this, you can train the simple model. Let's make predictions too.
```
#generate new data
x = 5 .+ 25 .* rand(20)
x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
test_data = hcat(x, x1)

predict(model test_data)
```
Congratulations! You was able to train your first model using LearningHorse and use it to predict it.

## Other Regression Models
Let's build another regression model.

### Polynomial Regression
In many cases, the data is nonlinear. Let's use polunomial regression. With LearningHorse, you can do this by using `make_design_matrix`, which converts data to designed matrix.

The code of training is this:
```
using LearningHorse.Regression
using LearningHorse.Regression: fit!, predict

#model
model = LinearRegression()
fit!(model, make_design_matrix(train_data), t)
```
Just pass the training data to `make_design_matrix`. Isn't it easy?

predicting is easy too:
```
predict(model, make_design_matrix(test_data))
```

### Ridge Regression, Lasso Regression
When building these models, you need to use different structures. But other than that, it's the same. Let's try it.
```
#Ridge Regression
model = Ridge()

fit!(model, make_design_matrix(train_data))

#Lasso Regression
model = Lasso()

fit!(model, make_design_matrix(train_data))
```
Predicting is the same as polynomial regression.

## Saveing and Loading
There's no point in just learning the model you made, you need to save it and load it when you need it.
LearningHorse itself doesn't implement this function, but it is very easy by using [`BSON.jl`](https://github.com/JuliaIO/BSON.jl).
after learning:
```
using BSON
using BSON: @save

@save "/home/ubuntu/model.bson" model
```
and loading:
```
using BSON
using BSON: @load

@load "/home/ubuntu/model.bson" model
```