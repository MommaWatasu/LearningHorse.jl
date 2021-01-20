# Horse.jl Manual

Horse provides an easy-to-use machine learning library 

## Installation

Documenter can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run
However, it can't be installed by name.
```@example
pkg > https://github.com/QGMW22/Horse/Horse.jl
```

## How to use

### How to use SGD
Stochastic Gradient Descent(SGD) can be used as follows.

```@example
using Horse
x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.1 26.95 5.68 21.76]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.SGD.fit(x, t)
println(w) #[1.539947 136.176160] The first element is tilted and the second element is the intercept.
```

The arguments are as follows. However, other than x and t, they are keyword arguments. The values ​​in parentheses are the default values.
- Linear_Regression.SGD.fit(x=[Dependent variable], t = [Objective variable], alpha = (0.001)[Learning rate], tau_max = (100000)[Maximum number of iterations], eps = (0.1)[Absolute threshold of gradient to stop repeating])
The return value is an Array type.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.SGD.predict(x = [Dependent variable], w = [Array of w obtained by fitting])

### How to use multiple regression
Multiple regression can be used as follows.

```@example
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.MR.fit(x, t) #Here, because there is no inverse matrix, I get an error.
println(w)
```

In multiple regression, if the inverse matrix cannot be generated in the program, the following error is returned.
```@example
Perhaps the matrix x you passed does not have an inverse matrix. In that case, use ridge regression.
```

As shown in the error statement, regression can be performed by using ridge regression or lasso regression, but the eigenvalue is close to 0, which makes the model unstable. Therefore, it is one way to perform processing such as deleting a row of highly correlated data.

The arguments are as follows. However, x and t are not keyword arguments.
- Linear_Regression.MR.fit(x = [Dependent variable], t = [Objective variable])
The return value is an Array type, the first element is the intercept of the model, and the subsequent elements are the coefficients of each dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.MR.predict(x = [Dependent variable], w = [Array of w obtained by fitting])
The return value is an Array type.

### How to use Ridge Regression
Ridge regression can be used as follows.

```@example
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.RR.fit(x, t)
#[65.47458454192515; 0.1042034860770639; 1.5756199878279644] The first element represents the intercept and the second and subsequent elements represent the coefficients of the dependent variable.
println(w)
```

The arguments are as follows. However, other than x and t, they are keyword arguments. The values ​​in parentheses are the default values.
- Linear_Regression.RR.fit(x = [Dependent variable], t = [Objective variable], alpha = (0.1)[learning rate])
The return value is an Array type, the first element represents the intercept of the model, and the second and subsequent elements represent the coefficients of the dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.RR.predict(x = [Dependent variable], w = [Array of obtained by fitting])
The return value is an Array type.

### How to use Lasso Regression
Lasso regression can be used as follows.

```@example
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.LR.fit(x, t)
# [144.61846209854855, 0.951158307335192, 0.0] The first element represents the intercept and the second and subsequent elements represent the coefficients of the dependent variable.
println(w)
```

The arguments are as follows. However, other than x and t, they are keyword arguments. The values ​​in parentheses are the default values.
- Linear_Regression.LR.fit(x = [Dependent variable], t = [Objective variable], alpha = (0.1)[learning rate], tol = (0.0001)[The absolute value of the error to stop repeating], mi = (1000000)[Maximum number of iterations] )
The return value is of type Array, the first element represents the intercept of the model, and the second and subsequent elements represent the coefficients of the dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.LR.predict(x = [Dependent variable], w = [Array of obtained by fitting])
The return value is of type LinearAlgebra.Adjoint.

### How to use MSE(Mean Square Error)
Mean Square Error can be used as follows.

```@example
using Horse
mean = Loss_Function.MSE([1, 2, 3, 4], [1, 2, 3, 4], [1, 2])
println(mean) #4
```

The arguments are as follows. However, x, t, w are not keyword arguments.
Loss_Function.MSE(x = [従属変数], t = [目的変数], w = [モデル])
The return value is a Float type. In addition, model w is passed so that the first element is the intercept and the second and subsequent elements are the coefficients of the dependent variable.