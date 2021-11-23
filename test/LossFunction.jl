import LearningHorse.LossFunction

@testset "LossFunction" begin
    y, t = rand(20), rand(20)
    @test LossFunction.mse(y, t) == mean(y.-t).^2
    _safe_log = LossFunction.safe_log
    @test LossFunction.cee(y, t) == mean(-sum(t.*_safe_log(y)))
end