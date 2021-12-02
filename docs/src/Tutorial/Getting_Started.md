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
x = hcat(x, x1)

#model
model = LinearRegression()
fit!(model, x, t)
```
Just this, you can train the simple model. Let's make predictions too.
```
#generate new data
x = 5 .+ 25 .* rand(20)
x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
x = hcat(x, x1)

predict(model, x)
```
Congratulations! You was able to train your first model using LearningHorse and use it to predict it.

## Other Regression Models
Let's build another regression model.