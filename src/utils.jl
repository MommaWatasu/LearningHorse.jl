function sample(range::UnitRange{Int64}, n)
    N = range.stop - range.start + 2
    result = Array{Int64}(undef, n)
    m = 1
    t = 1
    while m <= n
        if (N-t) * rand() >= (n - m+1)
            t+=1
        else
            result[m] = t
            t+=1
            m+=1
        end
    end
    
    return result
end