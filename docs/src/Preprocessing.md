## Preprocessing

All Preprocessing functions have the same arguments and specifications, so write only Standard Scaler.

### How to use Standard Scaler
Use the standard scaler (and other preprocessing functions) as follows.
```@example
x = [1 2 3 4 5; 2 3 4 5 6]]
sd = Preprocessing.SS.fit_transform(x, axis = 2)
println("sd:", sd)
#sd:(Any[[-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518], [-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518]], Any[[3.0, 1.5811388300841898], [4.0, 1.5811388300841898]])
inv = Preprocessing.SS.inverse_transform(sd[1], sd[2], axis = 2)
println("inv:", inv)
#inv:Any[[1.0 2.0 3.0 4.0 5.0], [2.0 3.0 4.0 5.0 6.0]]
p = Preprocessing.SS.fit(x, axis = 2)
println("fit:", p)
#fit:Any[[3.0, 1.5811388300841898], [4.0, 1.5811388300841898]]
p = Preprocessing.SS.transform(x, p, axis = 2)
println("p:", p)
#p:Any[[-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518], [-1.2649110640673518, -0.6324555320336759, 0.0, 0.6324555320336759, 1.2649110640673518]]
```
Using the fit function and then the transform function is completely equivalent to using the fit_transform function.
The fit function is a function that only calculates and returns the numerical value required for scaling, and the transform function scales using the numerical value obtained by fit. The fit_transform function does all this at once.
On the other hand, the inverse_transform function is a function that restores the scaled numerical value or the value predicted by the model to the original value.

- Preprocessing.SS.fit_transform()
	- x : Matrix to scale
	- axis : Specify the axis to be scaled, axis = 1 if each feature is lined up in each column, and axis = 2 if each feature is lined up in each row. The default value is 1. This argument is a keyword argument

- Preprocessing.SS.fit()
	- x : Matrix to scale
	- axis : Specify the axis to be scaled, axis = 1 if each feature is lined up in each column, and axis = 2 if each feature is lined up in each row. The default value is 1. This argument is a keyword argument
	
- Preprocessing.SS.transform()
	- x : Matrix tp scale
	- p : Pass the number p obtained by the fit function and fit_transform function
	- axis :  Specify the axis to be scaled, axis = 1 if each feature is lined up in each column, and axis = 2 if each feature is lined up in each row. The default value is 1. This argument is a keyword argument
- Preprocessing.SS.invers_transform()
	- x : Matrix to scale in reverse
	- p : Pass the number p obtained when scaling
	- axis : On the contrary, specify the axis to be scaled, axis = 1 if each feature is lined up in each column, and axis = 2 if each feature is lined up in each row. The default value is 1

### Other functions

MinMaxScaler can be used in Preprocessing.MM.
MM scales based on the following formula:
``\tilde{\boldsymbol{x}} = \frac{\boldsymbol{x}-min(\boldsymbol{x})}{max(\boldsymbol{x})-min(\boldsymbol{x})}``

RobustScaler can be used with Preprocessing.RS.
RS scales based on the following formula:
``\tilde{\boldsymbol{x}} = \frac{\boldsymbol{x}-Q2}{Q3 - Q1}``