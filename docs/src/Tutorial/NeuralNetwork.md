# NeuralNetwork
Let's finally build the most powerful model, NeuralNetwork!

## Basic NetWork
In anything, it's important to first learn the basics.　Let's create a networkconsisting only of [`Dense`](@ref) layer.
```
using LearningHorse.Preprocessing
using LearningHorse.NeuralNetwork
using LearningHorse.LossFunction

data = Matrix(dataloader("iris"))
DS = DataSplitter(150, test_size = 0.3)
train_data, test_data = DS(data)
train_x, train_t = train_data[:, 1:4], train_data[:, 5]
train_data = zip(train_x, train_t)
test_x, test_t = test_data[:, 1:4], test_data[:, 5]

model = NetWork(Dense(4=>2, relu), Dense(2=>3, sigmoid))
loss(x, y) = mse(model(x), y)
opt = Adam()
train!(model, loss, train_data, opt)
loss(test_x, test_y)
```
I wrote it all at once, but I'm doing is preparing the data, then definig and training the model.

## Advanced Model
Now that we know hot to build a network, let's build a model for image processing using the convolution layer.
!!! warning
    THIS DOCS IS INCOMPLETE!
```
using LearningHorse.Preprocessing
using LearningHorse.NeuralNetwork
using LearningHorse.LossFunction

data = Matrix(dataloader("MNIST"))
DS = DataSplitter(150, test_size = 0.3)
train_data, test_data = DS(data)

model = NetWork(Convolution())
loss(x, y) = mse(model(x), y)
opt = Adam()
train!(model, loss, train_data, opt)
loss(test_x, test_y)
```