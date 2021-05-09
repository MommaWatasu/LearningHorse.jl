@testset "Preprocessing" begin
    for (a, x) in zip(1:2, [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1 2 3 4 5; 2 3 4 5 6]])
        sd = Preprocessing.SS.fit_transform(x, axis = a)
        inv = Preprocessing.SS.inverse_transform(sd[1], sd[2], axis = a)
        p = Preprocessing.SS.fit(x, axis = a)
        @test Preprocessing.SS.transform(x, p, axis = a) == sd[1]
        sd = Preprocessing.MM.fit_transform(x, axis = a)
        inv = Preprocessing.MM.inverse_transform(sd[1], sd[2], axis = a)
        p = Preprocessing.MM.fit(x, axis = a)
        @test Preprocessing.MM.transform(x, p, axis = a) == sd[1]
        sd = Preprocessing.RS.fit_transform(x, axis = a)
        inv = Preprocessing.RS.inverse_transform(sd[1], sd[2], axis = a)
        p = Preprocessing.RS.fit(x, axis = a)
        @test Preprocessing.RS.transform(x, p, axis = a) == sd[1]
    end
end