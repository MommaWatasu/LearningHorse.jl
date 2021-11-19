import LearningHorse.LossFunction

@testset "LossFunction" begin
    y, t = rand(20), rand(20)
    @test LossFunction.mse(y, t) == mean(y.-t, dims=1).^2
    @test LossFunction.mse(y, t, true) == @. 2(y-t)
    _safe_log = LossFunction.safe_log
    @test LossFunction.cee(y, t) == mean(-sum(t.*_safe_log(y), dims=1), dims=1)
    @test LossFunction.cee(y, t, true) == -(t ./ y)
end