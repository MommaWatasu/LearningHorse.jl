function Same(x, k, stride)
    Ih, Iw = size(x)[1:2]
    padh = Int(div(ceil((stride[1]-1)Ih-stride[1]+k[1]), 2))
    padw = Int(div(ceil((stride[2]-1)Iw-stride[2]+k[2]), 2))
    return (padh, padh, padw, padw)
end

struct Im2Col
    x::AbstractArray
    Oh::Int
    Ow::Int
    b::Int
    c::Int
    function Im2Col(x, k, stride, padding; trans = "Cov")
        if padding == "same"
            padding = Same(x, k, stride)
        end
        i, j = 1, 1
        while j <= 2
            c = d = collect(size(x))
            c[j], d[j] = padding[i], padding[i + 1]
            x = cat(zeros(Tuple(c)), x, zeros(Tuple(d)), dims = j)
            i += 2
            j += 1
        end
        Xh, Xw, C, B = size(x)
        Fh, Fw = k
        Sh, Sw = stride[1], stride[2]
        Oh = Int(div((Xh - Fh), Sh) + 1) # ceil
        Ow = Int(div((Xw - Fw), Sw) + 1)
        y = zeros(Oh, Ow, Fh, Fw, C, B)
        for h in 1 : k[1]
            for w in 1 : k[2]
                y[:, :, h, w, :, :] = x[h : Sh : h+Oh*Sh-1, w : Sw : w+Ow*Sw-1, :, :]
            end
        end
        y = permutedims(y, [1, 5, 6, 2, 3, 4])
        if trans == "Cov"
            new(reshape(y, C*Fh*Fw, B*Oh*Ow), Oh, Ow, B, C)
        else
            new(reshape(y, B*C*Oh*Ow, Fh*Fw), Oh, Ow, B, C)
        end
    end
end

struct Col2Im
    x::Array
    function Col2Im(col, k, shape::IOShape, stride, padding)
        Fh, Fw = k
        Ih, Iw = shape.IShape
        Oh, Ow = shape.OShape
        B, C = Int(size(col)[2]/(Oh*Ow)), Int(size(col)[1]/(Fh*Fw))
        gis(f, o, stride, pad) = stride * (Oh - 1) + Fh - pad
        Ih, Iw = gis(Fh, Oh, stride[1], sum(padding[1:2])), gis(Fw, Ow, stride[2], sum(padding[3:4]))
        padding = ceil.(padding)
        padding = [sum(padding[1:2]), sum(padding[3:4])]
        col = permutedims(reshape(col, C, Fh, Fw, B, Oh, Ow), [4, 1, 2, 3, 5, 6])
        images = zeros(B, C, Ih+padding[1]+stride[1]-1, Iw+padding[2]+stride[2]-1)
        for h in 1 : Fh
            h_lim = h + stride[1] * Oh-1
            for w in 1 : Fw
                w_lim = w + stride[2] * Ow-1
                #println("image : ", size(images[:, :, h:stride[1]:h_lim, w:stride[2]:w_lim]))
                #println("col : ", size(col[:, :, h, w, :, :]))
                images[:, :, h:stride[1]:h_lim, w:stride[2]:w_lim] += col[:, :, h, w, :, :]
            end
        end
        padding .+= 1
        new(images[:, :, padding[1]:Ih+padding[1]-1, padding[2]:Iw+padding[2]-1])
    end
end