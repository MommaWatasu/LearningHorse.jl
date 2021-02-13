## LossFunction


### How to use MSE(Mean Square Error)
Mean Square Error can be used as follows.

```@example
using LearningHorse
mean = Loss_Function.MSE([1, 2, 3, 4], [1, 2, 3, 4], [1, 2])
println(mean) #4

x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76]
t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]

#Mean square error of Gaussian basis function
w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4)
mse = LossFunction.MSE(x[1, :], t, w, b = "gauss", m = m, s = s)
println("mse(gauss):", mse)
#mse(gauss):16.777305554499097

#Mean square error of polynomial regression
w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4, alpha = 0.1, f = "polynomial")
mse = LossFunction.MSE(x[1, :], t, w, b = "polynomial", m = m, s = s)
println("mse(polynomial):", mse)
#mse(polynomial):18.491924125591847
```
By specifying the argument b, the MSE of the basis function model can be calculated.
The arguments are as follows.

- Loss_Function.MSE()
	- x : Dependent variable
	- t : Objective variable
	- w : Array of obtained by fitting
The following are keyword arguments
	- b : Specify when calculating the MSE of the basis function. "Gauss" for Gaussian basis functions, "polynomial" for polynomial regression
	- m : It can be specified when calculating the MSE of a Gaussian basis function. By default, it is calculated using the vector at the center of each Gaussian function.
	- s : It can be specified when calculating the MSE of a Gaussian function. Calculated using the interval of each Gaussian function by default
	- mean_f : Whether to divide the sum of the squared and added errors to get the mean squared error, the default value is true

### How to use CEE(Cross Entropy Error)
The cross entropy error is used as follows.
```@example
using LearningHorse
#CEE of MS function
x = [5.1 3.5; 4.9 3.0; 4.7 3.2; 4.6 3.1; 5.0 3.6; 5.4 3.9; 4.6 3.4; 5.0 3.4; 4.4 2.9; 4.9 3.1; 5.4 3.7; 4.8 3.4; 4.8 3.0; 4.3 3.0; 5.8 4.0; 5.7 4.4; 5.4 3.9; 5.1 3.5; 5.7 3.8; 5.1 3.8; 5.4 3.4; 5.1 3.7; 4.6 3.6; 5.1 3.3; 4.8 3.4; 5.0 3.0; 5.0 3.4; 5.2 3.5; 5.2 3.4; 4.7 3.2; 4.8 3.1; 5.4 3.4; 5.2 4.1; 5.5 4.2; 4.9 3.1; 5.0 3.2; 5.5 3.5; 4.9 3.6; 4.4 3.0; 5.1 3.4; 5.0 3.5; 4.5 2.3; 4.4 3.2; 5.0 3.5; 5.1 3.8; 4.8 3.0; 5.1 3.8; 4.6 3.2; 5.3 3.7; 5.0 3.3; 7.0 3.2; 6.4 3.2; 6.9 3.1; 5.5 2.3; 6.5 2.8; 5.7 2.8; 6.3 3.3; 4.9 2.4; 6.6 2.9; 5.2 2.7; 5.0 2.0; 5.9 3.0; 6.0 2.2; 6.1 2.9; 5.6 2.9; 6.7 3.1; 5.6 3.0; 5.8 2.7; 6.2 2.2; 5.6 2.5; 5.9 3.2; 6.1 2.8; 6.3 2.5; 6.1 2.8; 6.4 2.9; 6.6 3.0; 6.8 2.8; 6.7 3.0; 6.0 2.9; 5.7 2.6; 5.5 2.4; 5.5 2.4; 5.8 2.7; 6.0 2.7; 5.4 3.0; 6.0 3.4; 6.7 3.1; 6.3 2.3; 5.6 3.0; 5.5 2.5; 5.5 2.6; 6.1 3.0; 5.8 2.6; 5.0 2.3; 5.6 2.7; 5.7 3.0; 5.7 2.9; 6.2 2.9; 5.1 2.5; 5.7 2.8; 6.3 3.3; 5.8 2.7; 7.1 3.0; 6.3 2.9; 6.5 3.0; 7.6 3.0; 4.9 2.5; 7.3 2.9; 6.7 2.5; 7.2 3.6; 6.5 3.2; 6.4 2.7; 6.8 3.0; 5.7 2.5; 5.8 2.8; 6.4 3.2; 6.5 3.0; 7.7 3.8; 7.7 2.6; 6.0 2.2; 6.9 3.2; 5.6 2.8; 7.7 2.8; 6.3 2.7; 6.7 3.3; 7.2 3.2; 6.2 2.8; 6.1 3.0; 6.4 2.8; 7.2 3.0; 7.4 2.8; 7.9 3.8; 6.4 2.8; 6.3 2.8; 6.1 2.6; 7.7 3.0; 6.3 3.4; 6.4 3.1; 6.0 3.0; 6.9 3.1; 6.7 3.1; 6.9 3.1; 5.8 2.7; 6.8 3.2; 6.7 3.3; 6.7 3.0; 6.3 2.5; 6.5 3.0; 6.2 3.4; 5.9 3.0]
t = [0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]
t = Classification.CTOH(t)
w = Classification.MS.fit(x, t, alpha = 0.1)
cee = LossFunction.CEE(x, t, w, t_f = true)
println("cee:", cee)
#52.20173706758349

#CEE of OVR function
w = Classification.OVR.fit(x, t; alpha = 0.1)
cee = LossFunction.CEE(x, t, w, sigmoid_f = true)
println("cee:", cee)
#28.702950289767305
```
The arguments are as follows.

- LossFunction.CEE()
	- x : Matrix of dependent variables
	-t : Matrix of Objective variables
	- w : Model parameters obtained by fitting
The following are keyword arguments.
	- mean_f : Whether to divide the sum of the errors to get the average cross entropy error, true by default
	- sigmoid_f : This argument must be true if you want the CEE of the OVR function. The default value is false
	- t_f : Specify the formula required by CEE. If true, it will be (Equation-1), if false, it will be (Equation-2). The default value is false

(N is the number of elements, K is the number of dimensions of the input)
(Equation-1) ``-\,\frac{1}{N}\,\sum_{n=0}^{N-1} {t_n\,\log\,y_n\,+\,(1\,-\,t_n)\log(1\,-\,y_n)}``

(Equation-2) ``-\,\frac{1}{N}\,\sum_{n=0}^{N-1}\sum_{k=0}^{K-1}\,t_{nk}\,\log\,y_{nk}``