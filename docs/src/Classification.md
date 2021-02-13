## Classification


### How to use Label Encoder
Label Encoder can used as follows.

```@example
using LearningHorse
label = ["Apple", "Apple", "Pear", "Pear", "Lemon", "Apple", "Pear", "Lemon"]
l, d = Classification.LE(label)
println(l)
#[3.0 3.0 1.0 1.0 2.0 3.0 1.0 2.0]
println("d:", d)
#d:Dict{Any, Any}("Pear" => 1, "Lemon" => 2, "Apple" => 3)
```
Since the label encoder is implemented using the Set type, the order of the elements is not guaranteed, so a Dict is returned indicating which class corresponds to which label.
The arguments are as follows.

- Classification.LE()
	- t : Array of labels to convert

### Conversion to One-hot format
The CTOH function that converts to One-hot format is used as follows. (However, this code is a continuation of the previous code.)

```@example
t = Classification.CTOH(l)
println("t:", t)
#t:[0.0 0.0 1.0; 0.0 0.0 1.0; 1.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
```
This function can convert the class of the passed vector if it starts from 0, but if you pass a vector that represents the class with a negative number, an error will occur.
The arguments are as follows.

- Classification.CTOH()
	- t : A numerical vector representing the class to be converted.

### Multi-class classification using Multi Class Softmax
Classification using multi-class softmax using MS functions is performed as follows.

```@example
x = [5.1 3.5; 4.9 3.0; 4.7 3.2; 4.6 3.1; 5.0 3.6; 5.4 3.9; 4.6 3.4; 5.0 3.4; 4.4 2.9; 4.9 3.1; 5.4 3.7; 4.8 3.4; 4.8 3.0; 4.3 3.0; 5.8 4.0; 5.7 4.4; 5.4 3.9; 5.1 3.5; 5.7 3.8; 5.1 3.8; 5.4 3.4; 5.1 3.7; 4.6 3.6; 5.1 3.3; 4.8 3.4; 5.0 3.0; 5.0 3.4; 5.2 3.5; 5.2 3.4; 4.7 3.2; 4.8 3.1; 5.4 3.4; 5.2 4.1; 5.5 4.2; 4.9 3.1; 5.0 3.2; 5.5 3.5; 4.9 3.6; 4.4 3.0; 5.1 3.4; 5.0 3.5; 4.5 2.3; 4.4 3.2; 5.0 3.5; 5.1 3.8; 4.8 3.0; 5.1 3.8; 4.6 3.2; 5.3 3.7; 5.0 3.3; 7.0 3.2; 6.4 3.2; 6.9 3.1; 5.5 2.3; 6.5 2.8; 5.7 2.8; 6.3 3.3; 4.9 2.4; 6.6 2.9; 5.2 2.7; 5.0 2.0; 5.9 3.0; 6.0 2.2; 6.1 2.9; 5.6 2.9; 6.7 3.1; 5.6 3.0; 5.8 2.7; 6.2 2.2; 5.6 2.5; 5.9 3.2; 6.1 2.8; 6.3 2.5; 6.1 2.8; 6.4 2.9; 6.6 3.0; 6.8 2.8; 6.7 3.0; 6.0 2.9; 5.7 2.6; 5.5 2.4; 5.5 2.4; 5.8 2.7; 6.0 2.7; 5.4 3.0; 6.0 3.4; 6.7 3.1; 6.3 2.3; 5.6 3.0; 5.5 2.5; 5.5 2.6; 6.1 3.0; 5.8 2.6; 5.0 2.3; 5.6 2.7; 5.7 3.0; 5.7 2.9; 6.2 2.9; 5.1 2.5; 5.7 2.8; 6.3 3.3; 5.8 2.7; 7.1 3.0; 6.3 2.9; 6.5 3.0; 7.6 3.0; 4.9 2.5; 7.3 2.9; 6.7 2.5; 7.2 3.6; 6.5 3.2; 6.4 2.7; 6.8 3.0; 5.7 2.5; 5.8 2.8; 6.4 3.2; 6.5 3.0; 7.7 3.8; 7.7 2.6; 6.0 2.2; 6.9 3.2; 5.6 2.8; 7.7 2.8; 6.3 2.7; 6.7 3.3; 7.2 3.2; 6.2 2.8; 6.1 3.0; 6.4 2.8; 7.2 3.0; 7.4 2.8; 7.9 3.8; 6.4 2.8; 6.3 2.8; 6.1 2.6; 7.7 3.0; 6.3 3.4; 6.4 3.1; 6.0 3.0; 6.9 3.1; 6.7 3.1; 6.9 3.1; 5.8 2.7; 6.8 3.2; 6.7 3.3; 6.7 3.0; 6.3 2.5; 6.5 3.0; 6.2 3.4; 5.9 3.0]
t = [0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]
t = Classification.CTOH(t)
w = Classification.MS.fit(x, t, alpha = 0.1)
println(w)
#[1.7209083498295872 1.8953943405297293 -0.6163026903593205; -1.256172465412238 1.8287641755675748 2.4274082898446574; 4.812985176776389 -0.7076002068908029 -1.1053849698855869]
p = Classification.MS.predict(x, w)
println(p)
#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 2, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2]
```
The arguments are as follows.

Functions that perform learning.
- Classification.MS.fit()
	- x : Matrix of dependent variables
	- t : Matrix of objective variables of a class formatted using LE or CTOH (Array)
The following are keyword arguments.
	- alpha : Learning rate when searching for the optimum parameter by the steepest gradient method. The default value is 0.01
	- tau_max : Maximum number of iterations when optimizing parameters. The default value is 1000

Function to make a prediction.
- Classification.MS.predict()
	- x : Matrix of dependent variables used for prediction
	- w : Array of parameters w obtained by fitting

### Multiclass classification using One-vs-Rest
One-vs-Rest can be used as follows. (This code is a continuation of the previous Multiclass Softmax.)
```@example
w = Classification.OVR.fit(x, t; alpha = 0.1)
println("w:", w)
#w:Any[[1.4720211645121106 -3.049021804351106 4.768579857703322], [1.496138559080458 0.5690372319250491 -1.8729783593510938], [-1.2013066350855492 1.258936673691226 -2.3333697507254283]]
p = Classification.OVR.predict(x, w)
println("predict:", p)
#predict:Any[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 1, 1, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2]
```
The arguments are as follows.

Function that preform learning.
- Classification.OVR.fit()
	- x : Dependent variable matrix (Array)
	- t : Matrix of objective variables of a class formatted using LE or CTOH (Array)
The following are keyword arguments.
	- alpha : Learning rate when searching for the optimum parameter by the steepest gradient method. The default value is 0.01
	- tau_max : Maximum number of iterations when optimizing parameters. The default value is 1000

Function to make a prediction.
- Classification.OVR.predict()
	- x : Matrix of dependent variables used for prediction
	- w : Array of parameters w obtained by fitting