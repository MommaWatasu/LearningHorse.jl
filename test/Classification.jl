@testset "Tree" begin #Tree, Ensamble, Classification module test
    x = [5.1 3.5; 4.9 3.0; 4.7 3.2; 4.6 3.1; 5.0 3.6; 5.4 3.9; 4.6 3.4; 5.0 3.4; 4.4 2.9; 4.9 3.1; 5.4 3.7; 4.8 3.4; 4.8 3.0; 4.3 3.0; 5.8 4.0; 5.7 4.4; 5.4 3.9; 5.1 3.5; 5.7 3.8; 5.1 3.8; 5.4 3.4; 5.1 3.7; 4.6 3.6; 5.1 3.3; 4.8 3.4; 5.0 3.0; 5.0 3.4; 5.2 3.5; 5.2 3.4; 4.7 3.2; 4.8 3.1; 5.4 3.4; 5.2 4.1; 5.5 4.2; 4.9 3.1; 5.0 3.2; 5.5 3.5; 4.9 3.6; 4.4 3.0; 5.1 3.4; 5.0 3.5; 4.5 2.3; 4.4 3.2; 5.0 3.5; 5.1 3.8; 4.8 3.0; 5.1 3.8; 4.6 3.2; 5.3 3.7; 5.0 3.3; 7.0 3.2; 6.4 3.2; 6.9 3.1; 5.5 2.3; 6.5 2.8; 5.7 2.8; 6.3 3.3; 4.9 2.4; 6.6 2.9; 5.2 2.7; 5.0 2.0; 5.9 3.0; 6.0 2.2; 6.1 2.9; 5.6 2.9; 6.7 3.1; 5.6 3.0; 5.8 2.7; 6.2 2.2; 5.6 2.5; 5.9 3.2; 6.1 2.8; 6.3 2.5; 6.1 2.8; 6.4 2.9; 6.6 3.0; 6.8 2.8; 6.7 3.0; 6.0 2.9; 5.7 2.6; 5.5 2.4; 5.5 2.4; 5.8 2.7; 6.0 2.7; 5.4 3.0; 6.0 3.4; 6.7 3.1; 6.3 2.3; 5.6 3.0; 5.5 2.5; 5.5 2.6; 6.1 3.0; 5.8 2.6; 5.0 2.3; 5.6 2.7; 5.7 3.0; 5.7 2.9; 6.2 2.9; 5.1 2.5; 5.7 2.8; 6.3 3.3; 5.8 2.7; 7.1 3.0; 6.3 2.9; 6.5 3.0; 7.6 3.0; 4.9 2.5; 7.3 2.9; 6.7 2.5; 7.2 3.6; 6.5 3.2; 6.4 2.7; 6.8 3.0; 5.7 2.5; 5.8 2.8; 6.4 3.2; 6.5 3.0; 7.7 3.8; 7.7 2.6; 6.0 2.2; 6.9 3.2; 5.6 2.8; 7.7 2.8; 6.3 2.7; 6.7 3.3; 7.2 3.2; 6.2 2.8; 6.1 3.0; 6.4 2.8; 7.2 3.0; 7.4 2.8; 7.9 3.8; 6.4 2.8; 6.3 2.8; 6.1 2.6; 7.7 3.0; 6.3 3.4; 6.4 3.1; 6.0 3.0; 6.9 3.1; 6.7 3.1; 6.9 3.1; 5.8 2.7; 6.8 3.2; 6.7 3.3; 6.7 3.0; 6.3 2.5; 6.5 3.0; 6.2 3.4; 5.9 3.0]
    t = [0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]
    tree, classes = Tree.DT.fit(x, t)
    p = Tree.DT.predict(x, tree, classes)
    println(p)
    Tree.MV("./test.dot", tree)
    forest, classes, using_feature = Ensemble.RF.fit(x, t)
    p = Ensemble.RF.predict(x, forest, classes, using_feature)
    println(p)
    label = ["Apple", "Apple", "Pear", "Pear", "Lemon", "Apple", "Pear", "Lemon"]
    l = Classification.LE(label)
    println(l)
    t = Classification.CTOH(t)
    w = Classification.MS.fit(x, t, alpha = 0.1)
    println(w)
    p = Classification.MS.predict(x, w)
    println(p)
    cee = LossFunction.CEE(p, t, t_f = true)
    println(cee)
    w = Classification.OVR.fit(x, t; alpha = 0.1)
    println("w:", w)
    p = Classification.OVR.predict(x, w)
    println("predict:", p)
    cee = LossFunction.CEE(p, t)
    println("cee:", cee)
end