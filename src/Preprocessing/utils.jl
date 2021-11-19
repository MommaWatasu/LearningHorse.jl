function check_size(x, p)
    x_s, p_s = size(x, 2), size(p[:, :], 2)
    if x_s != p_s
        throw(DimensionMismatch("second dimension of the target data is $x_s, and first dimension of scaler's parameter $p_s. these dimensions must match(i.e. Number of features of fitting data and number of features of target data must match)."))
    end
end