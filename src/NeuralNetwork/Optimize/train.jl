include("./optimize.jl")
include("./grad.jl")

c_t(x::Tuple) = x
c_t(x) = Tuple(x)

function train!(m, loss, data, opt)
    ps = Params(m)
    try
        for d in data
            l = loss(c_t(d)...)
            g = grad(m, d, l, ps)
        end
        #update!(opt, ps, g)
    end
end