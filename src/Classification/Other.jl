function LE(t; count = false)#this is Label Encoder
    n = 1
    le = length(t)
    target = ones(1, le)
    s = [v for v in Set(t)]
    c = zeros(1, length(s))
    for i in 1:le
        for ii in 1:length(s)
            if t[i] == s[ii]
                target[i] = ii
                c[ii] += 1
                break
            end
        end
    end 
    d = Dict()
    for i in 1:length(s)
        d[s[i]] = i
    end
    if count
        return target, d, c
    else
        return target, d
    end
end
function CTOH(ts) #This is Convert to one hot
    try
        ts = broadcast(Int, ts)
    catch 
        throw("Perhaps the array of objective variables you passed has a non-zero decimal point in the name of the class (as in 1.1). Please change it to an integer type.")
    end
    if 0 in ts
        ts .+= 1
    end
    c = maximum(ts)
    oh = zeros(size(ts)[2], c)
    i = 1
    for t in ts
        oh[i, t] = 1.0
        i += 1
    end
    return oh
end