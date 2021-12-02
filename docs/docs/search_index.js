var documenterSearchIndex = {"docs":
[{"location":"Manual/Classification/#Classification","page":"Classification","title":"Classification","text":"","category":"section"},{"location":"Manual/Classification/#Encoders","page":"Classification","title":"Encoders","text":"","category":"section"},{"location":"Manual/Classification/","page":"Classification","title":"Classification","text":"Classification.LabelEncoder\nClassification.OneHotEncoder","category":"page"},{"location":"Manual/Classification/#LearningHorse.Classification.LabelEncoder","page":"Classification","title":"LearningHorse.Classification.LabelEncoder","text":"LabelEncoder()\n\nLabelEncoder structure. LE(label; count=false, decode=false) Convert labels(like string) to class numbers(encode), and convert class numbers to labels(decode).\n\nExample\n\njulia> label = [\"Apple\", \"Apple\", \"Pear\", \"Pear\", \"Lemon\", \"Apple\", \"Pear\", \"Lemon\"]\n8-element Vector{String}\n \"Apple\"\n \"Apple\"\n \"Pear\"\n \"Pear\"\n \"Lemon\"\n \"Apple\"\n \"Pear\"\n \"Lemon\"\n\njulia> LE = LabelEncoder()\nLabelEncoder(Dict{Any, Any}())\n\njulia> classes, count = LE(label, count=true) #Encode\n([3.0 3.0 … 1.0 2.0], [3.0 2.0 3.0])\n\njulia> LE(classes, decode=true) #Decode\n1×18 Matrix{String}:\n \"Apple\" \"Apple\" \"Pear\" \"Pear\" \"Lemon\" \"Apple\" \"Pear\" \"Lemon\"\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#LearningHorse.Classification.OneHotEncoder","page":"Classification","title":"LearningHorse.Classification.OneHotEncoder","text":"OneHotEncoder()\n\nconvert data to Ont-Hot format. If you specified decode, data will be decoded.\n\nExample\n\njulia> x = [4.9 3.0; 4.6 3.1; 4.4 2.9; 4.8 3.4; 5.1 3.8; 5.4 3.4; 4.8 3.4; 5.2 4.1; 5.5 4.2; 5.5 3.5; 4.8 3.0; 5.1 3.8; 5.0 3.3; 6.4 3.2; 5.7 2.8; 6.1 2.9; 6.7 3.1; 5.6 2.5; 6.3 2.5; 5.6 3.0; 5.6 2.7; 7.6 3.0; 6.4 2.7; 6.4 3.2; 6.5 3.0; 7.7 3.8; 7.2 3.2; 7.2 3.0; 6.3 2.8; 6.1 2.6; 6.3 3.4; 6.0 3.0; 6.9 3.1; 6.7 3.1; 5.8 2.7; 6.8 3.2; 6.3 2.5];#These data are also used to explanations of other functions.\n\njulia> t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];\n\njulia> OHE = OneHotEncoder()\nOneHotEncoder()\n\njulia> ct = OHE(t)\n37×3 Matrix{Float64}:\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 1.0 0.0 0.0\n 0.0 1.0 0.0\n 0.0 1.0 0.0\n 0.0 1.0 0.0\n 0.0 1.0 0.0\n 0.0 1.0 0.0\n ⋮\n 0.0 1.0 0.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n 0.0 0.0 1.0\n\njulia> OHE(ct, decode = true)\n37-element Vector{Int64}:\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 1\n 2\n 2\n 2\n 2\n ⋮\n 2\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n 3\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#Models","page":"Classification","title":"Models","text":"","category":"section"},{"location":"Manual/Classification/","page":"Classification","title":"Classification","text":"Classification.Logistic\nClassification.SVC","category":"page"},{"location":"Manual/Classification/#LearningHorse.Classification.Logistic","page":"Classification","title":"LearningHorse.Classification.Logistic","text":"Logistic(; alpha = 0.01, ni = 1000)\n\nLogistic Regression classifier.\n\nThis struct learns classifiers using multi class softmax. Parameter α indicates the learning rate, and ni indicates the number of learnings.\n\nExample\n\njulia> model = Logistic(alpha = 0.1)\nLogistic(0.1, 1000, Matrix{Float64}(undef, 0, 0))\n\njulia> fit!(model, x, ct)\n3×3 Matrix{Float64}:\n  1.80736  1.64037  -0.447735\n -1.27053  1.70026   2.57027\n  4.84966 -0.473835 -1.37582\n\njulia> println(predict(model, x))\n[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#LearningHorse.Classification.SVC","page":"Classification","title":"LearningHorse.Classification.SVC","text":"SVC(; alpha=0.01, ni=1000)\n\nSupport Vector Machine Classifier.\n\nThis struct learns classifiers using One-Vs-Rest. One-Vs-Rest generates two-class classifiers divided into one class and the other classes using Logistic Regression, adopting the most likely one among all classifiers.\n\nParameter α indicates the learning rate, and ni indicates the number of learnings.\n\nExample\n\njulia> model = SVC()\nSVC(0.01, 1000, Logistic[])\n\njulia> fit!(model, ct)\n3-element Vector{Logistic}:\n Logistic(0.01, 1000, [0.8116709490679518 1.188329050932049; 1.7228257190036231 0.2771742809963788; -0.1519960725403138 2.1519960725403116])\n Logistic(0.01, 1000, [0.9863693439936144 1.0136306560063886; 0.8838433946106077 1.11615660538939; 1.4431044559203794 0.5568955440796174])\n Logistic(0.01, 1000, [1.262510641510418 0.7374893584895849; 0.5242383002319192 1.4757616997680822; 1.864635796779504 0.135364203220495])\n\njulia> println(predict(model, x))\n[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#DecisionTree","page":"Classification","title":"DecisionTree","text":"","category":"section"},{"location":"Manual/Classification/#Models-2","page":"Classification","title":"Models","text":"","category":"section"},{"location":"Manual/Classification/","page":"Classification","title":"Classification","text":"Tree.DecisionTree\nTree.RandomForest","category":"page"},{"location":"Manual/Classification/#LearningHorse.Tree.DecisionTree","page":"Classification","title":"LearningHorse.Tree.DecisionTree","text":"DecisionTree(; alpha = 0.01)\n\nNormal DecisionTree. alpha specify the complexity of the model. If it's small, it's complicated, and if it's big, it's simple.\n\nExample\n\njulia> tree = DecisionTree()\nDecisionTree(0.01, Dict{Any, Any}(), Any[])\n\njulia> fit!(tree, x, t)\nDict{String, Any} with 5 entries:\n  \"left\"        => Dict{String, Any}(\"left\"=>Dict{String, Union{Nothing, Vector…\n  \"class_count\" => [8, 13, 16]\n  \"threshold\"   => 5.7\n  \"right\"       => Dict{String, Any}(\"left\"=>Dict{String, Union{Nothing, Vector…\n  \"feature_id\"  => 1\n\njulia> println(predict(tree, x))\n[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#LearningHorse.Tree.RandomForest","page":"Classification","title":"LearningHorse.Tree.RandomForest","text":"RandomForest(nt; alpha = 0.01)\n\nRandomForest Model. nt is the number of trees, and alpha is the same as alpha in DecisionTree.\n\nExample\n\njulia> model = RandomForest(10)\nRandomForest(0.01, 10, DecisionTree[], Vector{Any}[], #undef)\n\njulia> fit!(model, x, t)\n10×1 Matrix{Int64}:\n 1\n 2\n 2\n 2\n 2\n 1\n 1\n 1\n 1\n 1\n\njulia> println(predict(model, x))\nAny[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]\n\n\n\n\n\n","category":"type"},{"location":"Manual/Classification/#VisualTool","page":"Classification","title":"VisualTool","text":"","category":"section"},{"location":"Manual/Classification/","page":"Classification","title":"Classification","text":"Tree.MV","category":"page"},{"location":"Manual/Classification/#LearningHorse.Tree.MV","page":"Classification","title":"LearningHorse.Tree.MV","text":"MV(path, forest; rounded=false, bg=\"#ffffff\", fc=\"#000000\", label=\"Tree\", fs=\"18\")\n\nMake DicisionTree and RandomForest Visual(make a dot file, see also Graphviz). The arguments are as follows:\n\npath : The full path of the dot file. The suffix must be .dot.\nforest : The model.\nrounded : If rounded is true, the nodes will be rounded.\nbg : Background color, type of this must be String.\nfc : Font color, type of this must be String.\nlabel : The label of the graph.\nfs : Font size, type of this must be String.\n\nExample\n\njulia> MV(\"/home/ubuntu/test.dot\", model, rounded = true)\n\n\n\n\n\n","category":"function"},{"location":"Manual/Classification/","page":"Classification","title":"Classification","text":"If you make the model created in DecisionTree Example visualized, it'll be like this: (Image: Tree Visualized)","category":"page"},{"location":"Tutorial/Getting_Started/#The-First-Step","page":"Getting Started","title":"The First Step","text":"","category":"section"},{"location":"Tutorial/Getting_Started/#First-Model","page":"Getting Started","title":"First Model","text":"","category":"section"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"First, let's try the simplest model, the linear regression model. The code is:","category":"page"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"using LearningHorse.Regression\nusing LearningHorse.Regression: fit!, predict\n\n#data\nx = 5 .+ 25 .* rand(20)\nt = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)\nx1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)\nx = hcat(x, x1)\n\n#model\nmodel = LinearRegression()\nfit!(model, x, t)","category":"page"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"Just this, you can train the simple model. Let's make predictions too.","category":"page"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"#generate new data\nx = 5 .+ 25 .* rand(20)\nx1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)\nx = hcat(x, x1)\n\npredict(model, x)","category":"page"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"Congratulations! You was able to train your first model using LearningHorse and use it to predict it.","category":"page"},{"location":"Tutorial/Getting_Started/#Other-Regression-Models","page":"Getting Started","title":"Other Regression Models","text":"","category":"section"},{"location":"Tutorial/Getting_Started/","page":"Getting Started","title":"Getting Started","text":"Let's build another regression model.","category":"page"},{"location":"Manual/Regression/#Regression","page":"Regression","title":"Regression","text":"","category":"section"},{"location":"Manual/Regression/#Models","page":"Regression","title":"Models","text":"","category":"section"},{"location":"Manual/Regression/","page":"Regression","title":"Regression","text":"Regression.LinearRegression\nRegression.Lasso\nRegression.Ridge","category":"page"},{"location":"Manual/Regression/#LearningHorse.Regression.LinearRegression","page":"Regression","title":"LearningHorse.Regression.LinearRegression","text":"LinearRegression()\n\nClassic regression model. This struct has no parameter. If you want to use polynomial model, use Regression.make_design_matrix().\n\nsee also: make_design_matrix\n\nExample\n\njulia> x = [\n    16.862463771320925 68.10823385851712\n    15.382965696961577 65.4313485700859\n    8.916228406218375 53.92034559524475\n    10.560285659132695 59.17305391117168\n    12.142253214135884 62.28708207525656\n    5.362107221163482 43.604947901567414\n    13.893239446341777 62.44348617377496\n    11.871357065173395 60.28433066289655\n    29.83792267802442 69.22281924803998\n    21.327107214235483 70.15810991597944\n    23.852372696012498 69.81780163668844\n    26.269031430914108 67.61037566099782\n    22.78907104644012 67.78105545358633\n    26.73342178134947 68.59263965946904\n    9.107259141706415 56.565383817343495\n    29.38551885863976 68.1005579469209\n    7.935966787763017 53.76264777936664\n    29.01677894379809 68.69484161138638\n    6.839609488194577 49.69794758177567\n    13.95215840314148 62.058116579899085]; #These data are also used to explanations of other functions.\n\njulia> t = [169.80980778351542, 167.9081124078835, 152.30845618985222, 160.3110300206261, 161.96826472170756, 136.02842285615077, 163.98131131382686, 160.117817321485, 172.22758529098235, 172.21342437006865, 171.8939175591617, 169.83018083884602, 171.3878062674257, 170.52487535026015, 156.40282783981309, 170.6488327896672, 151.69267899906185, 172.32478221316322, 145.14365314788827, 163.79383292080666];\n\njulia> model = LinearRegression()\nLinearRegression(Float64[])\n\njulia> fit!(model, x, t)\n3-element Vector{Float64}:\n -0.04772448076255398\n  1.395963968616736\n 76.7817095600793\n\njulia> predict(model, x)\n20-element Vector{Float64}:\n 171.05359766482795\n 167.38737053144575\n 151.62704681535598\n 158.88117658330424\n 163.15274911747872\n 137.3967419011542\n 163.28751869479999\n 160.3699086857777\n 171.99027166957023\n 173.70207799107243\n 173.10650291105486\n 169.9096820022986\n 170.31402414534642\n 171.25872436348817\n 155.31030802635905\n 170.44522606721017\n 151.45368882321284\n 171.29242257091374\n 145.83183688699864\n 162.74674475052848\n\n\n\n\n\n","category":"type"},{"location":"Manual/Regression/#LearningHorse.Regression.Lasso","page":"Regression","title":"LearningHorse.Regression.Lasso","text":"Lasso(; alpha = 0.1, tol = 1e-4, mi = 1e+8)\n\nLasso Regression structure. eEach parameters are as follows:\n\nalpha : leaarning rate.\ntol : Allowable error.\nmi : Maximum number of learning.\n\nExample\n\njulia> model = Lasso()\nLasso(Float64[], 0.1, 0.0001, 100000000)\n\njulia> fit!(model, x, t)\n3-element Vector{Float64}:\n   0.0\n   0.5022766549841176\n 154.43624186616267\n\njulia> predict(model, x, t)\n20-element Vector{Float64}:\n 188.64541774549468\n 187.30088075704523\n 181.5191726873298\n 184.15748544986084\n 185.72158909964372\n 176.33798923891868\n 185.80014722707335\n 184.71565381947883\n 189.20524796663838\n 189.67502263476888\n 189.50409373058318\n 188.39535519538825\n 188.481083670683\n 188.88872347085172\n 182.8477136378307\n 188.64156231429416\n 181.43996475587224\n 188.9400571253936\n 179.39836073711297\n 185.6065850765288\n\n\n\n\n\n","category":"type"},{"location":"Manual/Regression/#LearningHorse.Regression.Ridge","page":"Regression","title":"LearningHorse.Regression.Ridge","text":"Ridge(alpha = 0.1)\n\nRidge Regression. alpha is the value multiplied by regularization term.\n\nExample\n\njulia> model = Ridge()\nRidge(Float64[], 0.1)\n\njulia> fit!(model, x, t)\n3-element Vector{Float64}:\n -0.5635468573581848\n  2.1185952951687614\n 40.334109796666425\n\njulia> predict(model, x)\n20-element Vector{Float64}:\n 175.12510514593038\n 170.28763505842625\n 149.54478779081344\n 159.7466476176333\n 165.4525001910219\n 129.6935485937018\n 164.79709438985097\n 161.3621431448216\n 170.17418141858434\n 176.95192713546982\n 174.8078461898064\n 168.76930346791391\n 171.09202561187362\n 170.58861763111338\n 155.04089855305028\n 168.05151465675456\n 149.76311329450505\n 169.5183634524783\n 141.7695072903308\n 163.94744858842117\n\n\n\n\n\n","category":"type"},{"location":"Manual/Regression/#Other","page":"Regression","title":"Other","text":"","category":"section"},{"location":"Manual/Regression/","page":"Regression","title":"Regression","text":"Regression.make_design_matrix","category":"page"},{"location":"Manual/Regression/#LearningHorse.Regression.make_design_matrix","page":"Regression","title":"LearningHorse.Regression.make_design_matrix","text":"make_design_matrix(x, dims)\n\nThis function return the design matrix.\n\nExample\n\njulia> make_design_matrix(x, dims = 2) |> size\n(20, 5)\n\n\n\n\n\n","category":"function"},{"location":"Tutorial/Welcome/#Welcome-to-LearningHorse!","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse!","text":"","category":"section"},{"location":"Tutorial/Welcome/#Easy-to-use-and-very-functional-ML-Library","page":"Welcome to LearningHorse","title":"Easy-to-use and very functional ML Library","text":"","category":"section"},{"location":"Tutorial/Welcome/","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse","text":"LearningHorse is the ML Library for JuliaLang. With this package, you can build various models such as regression, classification, and decition tree, etc. ","category":"page"},{"location":"Tutorial/Welcome/","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse","text":"In addition, each function is designed to be easy to use,  so it's good for those who want to learn machine learning with Julia!","category":"page"},{"location":"Tutorial/Welcome/","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse","text":"Please try using LearningHorse!","category":"page"},{"location":"Tutorial/Welcome/#Installation","page":"Welcome to LearningHorse","title":"Installation","text":"","category":"section"},{"location":"Tutorial/Welcome/","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse","text":"From the Julia REPL, type ] to enter the Pkg REPL mode and run.","category":"page"},{"location":"Tutorial/Welcome/","page":"Welcome to LearningHorse","title":"Welcome to LearningHorse","text":"pkg> add LearningHorse","category":"page"},{"location":"Manual/LossFunction/#LossFunction","page":"LossFunction","title":"LossFunction","text":"","category":"section"},{"location":"Manual/LossFunction/","page":"LossFunction","title":"LossFunction","text":"LossFunction.mse\nLossFunction.cee","category":"page"},{"location":"Manual/LossFunction/#LearningHorse.LossFunction.mse","page":"LossFunction","title":"LearningHorse.LossFunction.mse","text":"mse(y, t)\n\nReturn the mean square error:mean((y .- t) .^ 2)\n\nExample\n\njulia> loss = mse\nmse (generic function with 1 method)\n\njulia> y, t = [1, 3, 2, 5], [1, 2, 3, 4]\n([1, 3, 2, 5], [1, 2, 3, 4])\n\njulia> loss(y, t)\n0.75\n\n\n\n\n\n","category":"function"},{"location":"Manual/LossFunction/#LearningHorse.LossFunction.cee","page":"LossFunction","title":"LearningHorse.LossFunction.cee","text":"cee(y, t, dims = 1)\n\nReturn the cross entropy error:mean(-sum(t.*log(y)))\n\nExample\n\njulia> loss = cee\ncee (generic function with 1 method)\n\njulia> y, t = [1, 3, 2, 5], [1, 2, 3, 4]\n([1, 3, 2, 5], [1, 2, 3, 4])\n\njulia> loss(y, t)\n-10.714417768752456\n\n\n\n\n\n","category":"function"},{"location":"Manual/Preprocessing/#Preprocessing","page":"Preprocessing","title":"Preprocessing","text":"","category":"section"},{"location":"Manual/Preprocessing/#Data-Preprocessing","page":"Preprocessing","title":"Data Preprocessing","text":"","category":"section"},{"location":"Manual/Preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"Preprocessing.dataloader\nPreprocessing.DataSplitter","category":"page"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.dataloader","page":"Preprocessing","title":"LearningHorse.Preprocessing.dataloader","text":"dataloader(name; header=true, dir=learninghorsedatasets)\n\nLoad a data for Machine Learning. name is either the name of the datasets or the full path of the data file to be loaded. The following three can be specified as the name of the dataset in name.\n\nMNIST : The MNIST Datasets\niris : The iris Datasets\nBostonHousing : Boston Housing DataSets\n\nAnd these datasets are downloaded and saved by creating a dir folder under the home directly(i.e. it is saved in the {homedirectly}/learninghorsedatasets by default). When importing a data file, you can specify whether to read the header with header.\n\nExample\n\njulia> dataloader(\"MNIST\");\n\njulia> dataloader(\"/home/ubuntu/data/data.csv\", header = false)\n\n\n\n\n\n","category":"function"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.DataSplitter","page":"Preprocessing","title":"LearningHorse.Preprocessing.DataSplitter","text":"DataSplitter(ndata; test_size=nothing, train_size=nothing)\n\nSplit the data into test data and training data. ndata is the number of the data, and you must specify either test_size or train_size. thease parameter can be proprtional or number of data.\n\nnote: Note\nIf both test_size and train_size are specified, test_size takes precedence.\n\n#Example\n\njulia> x = rand(20, 2);\n\njulia> DS = DataSplitter(50, train_size = 0.3);\n\njulia> DS(x, dims = 2) |> size\n(14, 2)\n\n\n\n\n\n","category":"type"},{"location":"Manual/Preprocessing/#Scaler","page":"Preprocessing","title":"Scaler","text":"","category":"section"},{"location":"Manual/Preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"Preprocessing.Standard\nPreprocessing.MinMax\nPreprocessing.Robust\nPreprocessing.fit_transform!\nPreprocessing.inv_transform!","category":"page"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.Standard","page":"Preprocessing","title":"LearningHorse.Preprocessing.Standard","text":"Standard()\n\nStandard Scaler. This scaler scale data as: z_i = fracx_i-musigma\n\nExample\n\njulia> x = [\n    16.862463771320925 68.10823385851712\n    15.382965696961577 65.4313485700859\n    8.916228406218375 53.92034559524475\n    10.560285659132695 59.17305391117168\n    12.142253214135884 62.28708207525656\n    5.362107221163482 43.604947901567414\n    13.893239446341777 62.44348617377496\n    11.871357065173395 60.28433066289655\n    29.83792267802442 69.22281924803998\n    21.327107214235483 70.15810991597944\n    23.852372696012498 69.81780163668844\n    26.269031430914108 67.61037566099782\n    22.78907104644012 67.78105545358633\n    26.73342178134947 68.59263965946904\n    9.107259141706415 56.565383817343495\n    29.38551885863976 68.1005579469209\n    7.935966787763017 53.76264777936664\n    29.01677894379809 68.69484161138638\n    6.839609488194577 49.69794758177567\n    13.95215840314148 62.058116579899085]; #These data are also used to explanations of other functions.\n\njulia> t = [169.80980778351542, 167.9081124078835, 152.30845618985222, 160.3110300206261, 161.96826472170756, 136.02842285615077, 163.98131131382686, 160.117817321485, 172.22758529098235, 172.21342437006865, 171.8939175591617, 169.83018083884602, 171.3878062674257, 170.52487535026015, 156.40282783981309, 170.6488327896672, 151.69267899906185, 172.32478221316322, 145.14365314788827, 163.79383292080666];\n\njulia> scaler = Standard()\nStandard(Float64[])\n\njulia> fit!(scaler, x)\n2×2 Matrix{Float64}:\n 19.0591   64.467\n  6.95818   6.68467\n\njulia> transform!(scaler, x)\n20×2 Matrix{Float64}:\n  0.0310374   0.44579\n  0.0104337   0.218714\n  1.01139     0.710027\n  1.37285     0.954091\n -0.895893   -0.270589\n  0.983361    0.47397\n  1.39855     0.645624\n -1.31861    -0.901482\n  0.0147702   0.241708\n  0.29675     0.483076\n -0.338824   -0.104812\n  0.432442    0.387922\n  0.860418    0.567095\n -0.495306    0.140871\n  0.963084    0.552767\n -1.38901    -1.05926\n  0.469323    0.719196\n  0.0669475   0.512023\n -1.47583    -1.44017\n -1.99789    -3.27656\n\n\n\n\n\n","category":"type"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.MinMax","page":"Preprocessing","title":"LearningHorse.Preprocessing.MinMax","text":"MinMax()\n\nMinMax Scaler. This scaler scale data as: tildeboldsymbolx = fracboldsymbolx-min(boldsymbolx)max(boldsymbolx)-min(boldsymbolx)\n\nExample\n\njulia> scaler = MinMax()\nMinMax(Float64[])\n\njulia> fit!(scaler, x)\n2×2 Matrix{Float64}:\n  1.39855   0.954091\n -1.99789  -3.27656\n\njulia> transform!(scaler, x)\n20×2 Matrix{Float64}:\n 0.597368  0.879853\n 0.591301  0.826179\n 0.88601   0.942311\n 0.992432  1.0\n 0.324455  0.710522\n 0.877757  0.886514\n 1.0       0.927088\n 0.199996  0.561398\n 0.592578  0.831614\n 0.675601  0.888666\n 0.488471  0.749707\n 0.715552  0.866175\n 0.841559  0.908526\n 0.442398  0.807779\n 0.871787  0.905139\n 0.179268  0.524104\n 0.72641   0.944478\n 0.607941  0.895508\n 0.153707  0.43407\n 0.0       0.0\n\n\n\n\n\n","category":"type"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.Robust","page":"Preprocessing","title":"LearningHorse.Preprocessing.Robust","text":"Robust()\n\nRobust Scaler. This scaler scale data as: tildeboldsymbolx = fracboldsymbolx-Q2Q3 - Q1\n\nExample\n\njulia> scaler = Robust()\nRobust(Float64[])\n\njulia> fit!(scaler, x)\n3×2 Matrix{Float64}:\n 0.412913  0.739911\n 0.602654  0.873014\n 0.849116  0.905986\n\njulia> transform!(scaler, x)\n20×2 Matrix{Float64}:\n -0.0121192   0.041181\n -0.0260262  -0.28201\n  0.649595    0.417263\n  0.89357     0.764633\n -0.637774   -0.978423\n  0.630675    0.0812893\n  0.910919    0.3256\n -0.923097   -1.87636\n -0.0230992  -0.249283\n  0.16723     0.0942497\n -0.261766   -0.742477\n  0.258819   -0.041181\n  0.547691    0.213832\n -0.367388   -0.392802\n  0.616988    0.193438\n -0.970618   -2.10092\n  0.283712    0.430314\n  0.0121192   0.135449\n -1.02922    -2.64305\n -1.38159    -5.25675\n\n\n\n\n\n","category":"type"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.fit_transform!","page":"Preprocessing","title":"LearningHorse.Preprocessing.fit_transform!","text":"fit_transform!(scaler, x; dims=1)\n\nfit scaler with x, and transform x.\n\n\n\n\n\n","category":"function"},{"location":"Manual/Preprocessing/#LearningHorse.Preprocessing.inv_transform!","page":"Preprocessing","title":"LearningHorse.Preprocessing.inv_transform!","text":"inv_transform!(scaler, x; dims=1)\n\nConvert x in reverse.\n\n\n\n\n\n","category":"function"},{"location":"#LearningHorse.jl","page":"Home","title":"LearningHorse.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"LearningHorse provides an easy-to-use machine learning library","category":"page"},{"location":"#Resources","page":"Home","title":"Resources","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation\nSource code","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"LearningHorse can be installed using the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run.","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add LearningHorse","category":"page"},{"location":"Manual/NeuralNetwork/#NeuralNetwork","page":"NeuralNetwork","title":"NeuralNetwork","text":"","category":"section"},{"location":"Manual/NeuralNetwork/#Basics","page":"NeuralNetwork","title":"Basics","text":"","category":"section"},{"location":"Manual/NeuralNetwork/","page":"NeuralNetwork","title":"NeuralNetwork","text":"To build a neural network with LearningHorse, use the NetWork type.","category":"page"},{"location":"Manual/NeuralNetwork/","page":"NeuralNetwork","title":"NeuralNetwork","text":"NetWork","category":"page"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.NetWork","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.NetWork","text":"NetWork(layers...)\n\nConnect multiple layers, and build a NeuralNetwork. NetWork also supports index. You can also add layers later using the add_layer!() Function.\n\nExample\n\njulia> N = NetWork(Dense(10=>5, relu), Dense(5=>1, relu))\n\njulia> N[1]\n\nDense(IO:10=>5, σ:relu)\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#Layers","page":"NeuralNetwork","title":"Layers","text":"","category":"section"},{"location":"Manual/NeuralNetwork/","page":"NeuralNetwork","title":"NeuralNetwork","text":"Conv\n\nDense\n\nDropout\n\nFlatten","category":"page"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Conv","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Conv","text":"Conv(kernel, in=>out, σ; stride = 1, pading = 0, set_w = \"Xavier\")\n\nThis is the traditional convolution layer. kernel is a tuple of integers that specifies the kernel size, it must have one or two elements. And, in and out specifies number of input and out channels.\n\nThe input data must have a dimensions WHCB(weight, width, channel, batch). If you want to use a data which has a dimentions WHC, you must be add a dimentioins of B.\n\nstride and padding are single integers or tuple(stride is tuple of 2 elements, padding is tuple of 2 elements), and if you specifies KeepSize to padding, we adjust sizes of input and return a matrix which has the same sizes. set_w is Xavier or He, it decide a method to create a first parameter. This parameter is the same as Dense().\n\nExample\n\njulia> C = Conv((2, 2), 2=>2, relu)\nConvolution(k:(2, 2), IO:2 => 2, σ:relu)\n\njulia> C(rand(10, 10, 2, 5)) |> size\n(9, 9, 2, 5)\n\nwarning: Warning\nWhen you specidies same to padding, in some cases, it will be returned one size smaller. Because of its expression.\n\njulia> C = Conv((2, 2), 2=>2, relu, padding = KeepSize)\nConvolution(k:(2, 2), IO:2 => 2, σ:relu\n\njulia> C(rand(10, 10, 2, 5)) |> size\n(9, 9, 2, 5)\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Dense","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Dense","text":"Dense(in=>out, σ; set_w = \"Xavier\", set_b = zeros)\n\nCrate a traditinal Dense layer, whose forward propagation is given by:     y = σ.(W * x .+ b) The input of x should be a Vactor of length in, (Sorry for you can't learn using batch. I'll implement)\n\nExample\n\njulia> D = Dense(5=>2, relu)\nDense(IO:5=>2, σ:relu)\n\njulia> D(rand(Float64, 5)) |> size\n(2,)\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Dropout","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Dropout","text":"Dropout(p)\n\nThis layer dropout the input data.\n\nExample\n\njulia> D = Dropout(0.25) Dropout(0.25)\n\njulia> D(rand(10)) 10-element Array{Float64,1}:  0.0  0.3955865029078952  0.8157710047424143  1.0129613533211907  0.8060508293474877  1.1067504108970596  0.1461289547292684  0.0  0.04581776023870532  1.2794087133638332\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Flatten","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Flatten","text":"Flatten()\n\nThis layer change the dimentions Image to Vector.\n\nExample\n\njulia> F = Flatten()\nFlatten(())\n\njulia> F(rand(10, 10, 2, 5)) |> size\n(1000, )\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#Optimizers","page":"NeuralNetwork","title":"Optimizers","text":"","category":"section"},{"location":"Manual/NeuralNetwork/","page":"NeuralNetwork","title":"NeuralNetwork","text":"Descent\n\nMomentum\n\nAdaGrad\n\nAdam","category":"page"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Descent","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Descent","text":"Descent(η=0.1)\n\nBasic gradient descent optimizer with learning rate η.\n\nParameters\n\nlearning rate : η\n\nExample\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Momentum","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Momentum","text":"Momentum(η=0.01, α=0.9, velocity)\n\nMomentum gradient descent optimizer with learning rate η and parameter of velocity α.\n\nParameters\n\nlearning rate : η\nparameter of velocity : α\n\nExample\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.AdaGrad","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.AdaGrad","text":"AdaGrad(η = 0.01)\n\nGradient descent optimizer with learning rate attenuation.\n\nParameters\n\nη : initial learning rate\n\nExamples\n\n\n\n\n\n","category":"type"},{"location":"Manual/NeuralNetwork/#LearningHorse.NeuralNetwork.Adam","page":"NeuralNetwork","title":"LearningHorse.NeuralNetwork.Adam","text":"Adam(η=0.01, β=(0.9, 0.99))\n\nGradient descent adaptive moment estimation optimizer.\n\nParameters\n\nη : learning rate\nβ : Decay of momentums\n\nExamples\n\n\n\n\n\n","category":"type"}]
}
