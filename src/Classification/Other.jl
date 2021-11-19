mutable struct LabelEncoder
    d::Dict
    LabelEncoder() = new(Dict())
end

function (LE::LabelEncoder)(data; count = false, decode = false)
    if !decode
        target = ones(1, length(data))
        s = [v for v in Set(data)]
        if count
            c = zeros(1, length(s))
        end
        for i in 1 : length(data)
            j = findfirst(isequal(data[i]), s)
            target[i] = j
            if count
                c[j]+=1
            end
        end
        d = Dict()
        for i in 1 : length(s)
            d[i] = s[i]
        end
        LE.d = d
        if count
            return target, c
        end
        return target
    else
        return map(x->LE.d[x], data)
    end
end

mutable struct OneHotEncoder end

function (OHE::OneHotEncoder)(data; decode = false)
    if !decode
        try
            data = Int.(data)
        catch
            throw(ArgumentError("the array of objective variables you passed has a non-zero decimal point in the name of the class (as in 1.1). you must change it to integer type."))
        end
        0 in data && data.+=1
        OH = zeros(length(data), maximum(data))
        for (i, d) in zip(1:length(data), data)
            OH[i, d] = 1.0
        end
        return OH
    else
        return [argmax(data[i, :]) for i in 1 : size(data, 1)]
    end
end