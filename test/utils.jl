@testset "utils" begin
    data = (LearningHorse.sample(1:100, 30).<=100)
    @test ones(size(data)...) == (data .<= 100)
end