using LearningHorse.Classification
using LearningHorse.Tree
using CSV
import DataFrames

@testset "Classification" begin
    #Read test datas
    x = Matrix(CSV.read("./Classification_1.csv", DataFrames.DataFrame, header = false))
    t = Matrix(CSV.read("./Classification_2.csv", DataFrames.DataFrame, header = false))
    
    tt = dropdims(t', dims = 2)
    #The test for Decision Tree
    model = DecisionTree()
    @test_nowarn Tree.fit!(model, x, tt)
    @test_nowarn Tree.predict(model, x)
    @test_nowarn MV("./test.dot", model)
    
    #The test for Random forest
    model = RandomForest(10)
    @test_nowarn Tree.fit!(model, x, tt)
    @test_nowarn Tree.predict(model, x)
    @test_nowarn MV("./test2.dot", model)
    
    #The test for Encoders
    label = ["Apple", "Apple", "Pear", "Pear", "Lemon", "Apple", "Pear", "Lemon"]
    LE = LabelEncoder()
    OHE = OneHotEncoder()
    @test_nowarn LE(label)
    @test_nowarn OHE(t)
    t = OHE(t)
    
    #The test for Logistic Regression
    model = Logistic(alpha = 0.1)
    @test_nowarn Classification.fit!(model, x, t)
    @test_nowarn Classification.predict(model, x)
    
    #The test for Support Vector machine Classification(One-Vs-Rest)
    model = SVC()
    @test_nowarn Classification.fit!(model, x, t)
    @test_nowarn Classification.predict(model, x)
end