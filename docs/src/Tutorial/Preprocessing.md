# Preprocessing

## Data Loader
With Preprocessing module, you can load the data. Also, some famous data sets can be loaded by specifying a name.
First, let's load the MNIST data.
```
using LearningHorse.Preprocessing

train_data, test_dara = dataloder("MNIST")
```
If it is your first time to load a data set. you wil be asked if you want to download the data like this:
```
Can I download the mnist_train.csv(110MB)? (y/n)
```

!!! note
    The data downloaded here is in `/home_directory/learningdatasets/` by default. for more details, see [`make_design_matrix`](@ref).

Let's also read the local data. 
```
#Please create train.csv directly under the /home_directory/learningdatasets/ as follows:
#0.202027,0.246752,0.608301,0.406351,0.0260402,0.747845
#0.735292,0.332892,0.438458,0.0787028,0.797796,0.294831
#0.710725,0.213594,0.527118,0.579191,0.298599,0.23684
#0.288168,0.787194,0.809412,0.464031,0.960465,0.655897
df = dataloader("train.csv", header = false)
```

## Data Preprocessing
Even if you use the data as it is, you won't be able to make a model with high accuracy, let's normalizethe data.
The following three scalers are available in LearningHorse:
- Standard Scaler : ``z_{i} = \\frac{x_{i}-\\mu}{\\sigma}``
- MinMax Scaler : ``\\tilde{\\boldsymbol{x}} = \\frac{\\boldsymbol{x}-min(\\boldsymbol{x})}{max(\\boldsymbol{x})-min(\\boldsymbol{x})}``
- Robust Scaler : ``\\tilde{\\boldsymbol{x}} = \\frac{\\boldsymbol{x}-Q2}{Q3 - Q1}``
For example, if you normalize iris data set with Standard Scaler,
```
using LearningHorse.Preprocessing: fit!
train_data = Matrix(dataloader("iris"))[1:4]

scaler = Standard()
fit_transform!(scaler, train_data)
```
Once the scaler is fittted, sclaes using the same value unless it fits again.