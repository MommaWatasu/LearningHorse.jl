"""
    LabelEncoder()
LabelEncoder structure. `LE(label; count=false, decode=false)` Convert labels(like string) to class numbers(encode), and convert class numbers to labels(decode).

# Example
```jldoctest
julia> label = ["Apple", "Apple", "Pear", "Pear", "Lemon", "Apple", "Pear", "Lemon"]
8-element Vector{String}
 "Apple"
 "Apple"
 "Pear"
 "Pear"
 "Lemon"
 "Apple"
 "Pear"
 "Lemon"

julia> LE = LabelEncoder()
LabelEncoder(Dict{Any, Any}())

julia> classes, count = LE(label, count=true) #Encode
([3.0 3.0 … 1.0 2.0], [3.0 2.0 3.0])

julia> LE(classes, decode=true) #Decode
1×18 Matrix{String}:
 "Apple" "Apple" "Pear" "Pear" "Lemon" "Apple" "Pear" "Lemon"
```
"""
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

"""
    OneHotEncoder()
convert data to Ont-Hot format. If you specified `decode`, `data` will be decoded.

# Example
```jldoctest classification
julia> x = [4.9 3.0; 4.6 3.1; 4.4 2.9; 4.8 3.4; 5.1 3.8; 5.4 3.4; 4.8 3.4; 5.2 4.1; 5.5 4.2; 5.5 3.5; 4.8 3.0; 5.1 3.8; 5.0 3.3; 6.4 3.2; 5.7 2.8; 6.1 2.9; 6.7 3.1; 5.6 2.5; 6.3 2.5; 5.6 3.0; 5.6 2.7; 7.6 3.0; 6.4 2.7; 6.4 3.2; 6.5 3.0; 7.7 3.8; 7.2 3.2; 7.2 3.0; 6.3 2.8; 6.1 2.6; 6.3 3.4; 6.0 3.0; 6.9 3.1; 6.7 3.1; 5.8 2.7; 6.8 3.2; 6.3 2.5];#These data are also used to explanations of other functions.

julia> t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

julia> OHE = OneHotEncoder()
OneHotEncoder()

julia> ct = OHE(t)
37×3 Matrix{Float64}:
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 1.0 0.0 0.0
 0.0 1.0 0.0
 0.0 1.0 0.0
 0.0 1.0 0.0
 0.0 1.0 0.0
 0.0 1.0 0.0
 ⋮
 0.0 1.0 0.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0
 0.0 0.0 1.0

julia> OHE(ct, decode = true)
37-element Vector{Int64}:
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 1
 2
 2
 2
 2
 ⋮
 2
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
 3
```
"""
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