## Regression

### How to use SGD
Stochastic Gradient Descent(SGD) can be used as follows.

```
using Horse
x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.1 26.95 5.68 21.76]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.SGD.fit(x, t)
println(w) #[1.539947 136.176160] The first element is tilted and the second element is the intercept.
```

The arguments are as follows.
- Linear_Regression.SGD.fit()
    - x : Dependent variable
    - t : Objective variable
The following are keyword arguments.
    - alpha : Learning rate. The default value is 0.001
	- tau_max : Maximum number of iterations. The default variable is 100000
	- eps : Absolute threshold of gradient to stop repeating. The default value is 0.1
The return value is an Array type.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.SGD.predict()
	- x:Dependent variable
	- w:Array of obtained by fitting

### How to use multiple regression
Multiple regression can be used as follows.

```
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.MR.fit(x, t) #Here, because there is no inverse matrix, I get an error.
println(w)
```

In multiple regression, if the inverse matrix cannot be generated in the program, the following error is returned.
```
Perhaps the matrix x you passed does not have an inverse matrix. In that case, use ridge regression.
```

As shown in the error statement, regression can be performed by using ridge regression or lasso regression, but the eigenvalue is close to 0, which makes the model unstable. Therefore, it is one way to perform processing such as deleting a row of highly correlated data.

The arguments are as follows. However, x and t are not keyword arguments.
- Linear_Regression.MR.fit()
	- x : Dependent variable
	- t : Objective variable
The return value is an Array type, the first element is the intercept of the model, and the subsequent elements are the coefficients of each dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.MR.predict()
	- x : Dependent variable
	- w : Array of w obtained by fitting
The return value is an Array type.

### How to use Ridge Regression
Ridge regression can be used as follows.

```
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.RR.fit(x, t)
#[65.47458454192515; 0.1042034860770639; 1.5756199878279644] The first element represents the intercept and the second and subsequent elements represent the coefficients of the dependent variable.
println(w)
```

The arguments are as follows.
- Linear_Regression.RR.fit()
	- x : Dependent variable
	- t : Objective variable
The following are keyword arguments.
	- alpha : learning rate. The deault value is 0.1
The return value is an Array type, the first element represents the intercept of the model, and the second and subsequent elements represent the coefficients of the dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.RR.predict()
	- x : Dependent variable
	- w : Array of obtained by fitting
The return value is an Array type.

### How to use Lasso Regression
Lasso regression can be used as follows.

```
using Horse
x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
w = Linear_Regression.LR.fit(x, t)
# [144.61846209854855, 0.951158307335192, 0.0] The first element represents the intercept and the second and subsequent elements represent the coefficients of the dependent variable.
println(w)
```

The arguments are as follows.
- Linear_Regression.LR.fit()
	- x : Dependent variable
	- t : Objective variable
The following are keyword arguments.
	- alpha : learning rate. The deault value is 0.1
	- tol : The absolute value of the error to stop repeating. The deault value is 0.0001
	- mi : Maximum number of iterations. The default value is 1000000
The return value is of type Array, the first element represents the intercept of the model, and the second and subsequent elements represent the coefficients of the dependent variable.

Prediction using the built model can be done with the following function. However, x and w are not keyword arguments.
- Linear_Regression.LR.predict()
	- x : Dependent variable
	- w : Array of abstained fitting
The return value is of type LinearAlgebra.Adjoint.

### How to use basis function model
The basis function model can be used as follows.

```
using LearningHorse
 x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]

#Use Gaussian basis functions
w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4)
println("w:", w, "m:", m, "s:", s)
#w:[-10.6733; 65.4475; -17.7623; 66.6557; 104.918]m:Any[5, 12.3333, 19.6667, 27.0]s:7.333333333333333

p = LinearRegression.BFM.predict(x[1, :], w, m, s)
println(p)
#[165.07, 168.55, 132.278, 162.546, 150.433, 144.04, 154.507, 164.011, 164.902, 165.417, 165.082, 167.712, 156.153, 169.583, 135.749, 167.366]

#Use polynomial basis functions
w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4, alpha = 0.1, f = "polynomial")
println("w, m, s(polynomial):", w, m, s)
#w, m, s(polynomial):[30.074721512969795; 17.649472753936088; -0.9665443355191238; 0.01733639801805678; 30.074721512971067]Any[5, 12.333333333333332, 19.666666666666664, 26.999999999999996]7.333333333333333

p = LinearRegression.BFM.predict(x[1, :], w, m, s, f = "polynomial")
println("p(polynomial):", p)
#p(polynomial):[166.04905485192677, 165.72435351634755, 126.40024815990033, 163.70105505929791, 151.81467689141562, 144.29063632133014, 156.07721189214902, 165.05792844782184, 165.89978336872304, 165.6418324683704, 166.05887282906252, 165.26960817342376,157.70790163159944, 173.13926201582598, 132.39231072417851, 165.16779149696566]
```
The only functions are predict and fit, but you can use two models, the polynomial regression and the Gaussian basis function model, by specifying the keyword argument f. The arguments are as follows.

Function to learn.
- LinearRegression.BFM.fit()
	- x : Dependent variable vector (matrix cannot be included) 
	- t : Objective variable vector (matrix cannot be included)
	- M : Set the number of dimensions to fit, for example M = 2 for 2 dimensions.
The following are keyword arguments.
	- m : It can be specified for Gaussian basis functions. By default it uses the center vector of each Gaussian function
	- s : Like m, it can be specified for Gaussian functions. The default is to use a scalar that represents the spacing of each Gaussian function.
	- alpha : The coefficient applied to the regularization term that is added when calculating the design matrix. The default value is 0.0
	- f : Specify the type of basis function, pass the string "gauss" for Gaussian functions, or "polynomial" for polynomial regression. Note that the default value is "gauss"

Function to predict
- LinearRegression.BFM.predict()
	- x : The vector of the dependent variable used for the prediction
	- w : Pass w of Array of the model obtained by fitting