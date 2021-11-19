include("./optimizers.jl")
include("./grad.jl")

function update!(opt, x, g)
    x .-= apply!(opt, x, g)
end

function update!(opt, G::Grads)
    ps = G.ps
    g = G.g
    for i in 1 : length(ps)
        x = ps[i]
        isnothing(x) && continue
        update!(opt, x, g[i])
    end
end

function update!(opt, gs::Zygote.Grads, ps::Zygote.Params)
    for p in ps
        isnothing(gs[p]) && continue
        update!(opt, p, gs[p])
    end
end

function train!(m::NetWork, loss, data, opt)
    ps = Params(m)
    for d in data
        try
            g = grad(m, d, loss, ps)
            update!(opt, g)
        catch
            @warn "you can't train by one data because something wrong with the data!"
        end
    end
end

make_d_tuple(d::Tuple) = d
make_d_tuple(d) = tuple(d)

function train!(m::NetWork, loss, data, opt, AD::Bool)
    ps = zygote_params(m)
    for d in data
        try
            g = Zygote.gradient(ps) do
                loss(make_d_tuple(d)...)
            end
            update!(opt, g, ps)
        catch
            @warn "you can't train by one data because something wrong with the data!"
        end
    end
end