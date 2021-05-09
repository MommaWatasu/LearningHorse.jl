include("./optimize.jl")
include("./grad.jl")

function update!(opt, xs::Params, g::Grads)
    for x in xs
        update!(opt, x, g[x])
    end
end

function update!(opt, x, g)
    x .-= aplly!(opt, x, g)
end

function train!(m::NetWork, loss, data, opt)
    ps = Params(m)
    for d in data
        try
            l = loss(c_t(d)...)
            g = grad(m, d, l, ps)
            update!(opt, ps, g)
        catch
            @warn "you can't train by one data because something wrong with the data"
        end
    end
end