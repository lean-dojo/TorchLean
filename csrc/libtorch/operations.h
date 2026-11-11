// Shared operation list for the LibTorch exports and builds without LibTorch.
// Intentionally included more than once, with the signature macros defined by the caller.

TORCHLEAN_UNARY_EXPORT(abs, aten<"abs">(x))
TORCHLEAN_UNARY_EXPORT(sqrt, selected_sqrt(x))
TORCHLEAN_UNARY_EXPORT(sqrt_raw, aten<"sqrt">(x))
TORCHLEAN_UNARY_EXPORT(exp, aten<"exp">(x))
TORCHLEAN_UNARY_EXPORT(expm1, aten<"expm1">(x))
TORCHLEAN_UNARY_EXPORT(sin, aten<"sin">(x))
TORCHLEAN_UNARY_EXPORT(cos, aten<"cos">(x))
TORCHLEAN_UNARY_EXPORT(log, aten<"log">(x))
TORCHLEAN_UNARY_EXPORT(log1p, aten<"log1p">(x))
TORCHLEAN_UNARY_EXPORT(inv, aten<"reciprocal">(x))
TORCHLEAN_UNARY_EXPORT(relu, selected_relu(x))
TORCHLEAN_UNARY_EXPORT(sigmoid, stable_sigmoid(x))
TORCHLEAN_UNARY_EXPORT(silu, staged_silu(x))
TORCHLEAN_UNARY_EXPORT(tanh, aten<"tanh">(x))
TORCHLEAN_UNARY_EXPORT(gelu, staged_gelu(x))

TORCHLEAN_BINARY_EXPORT(max, ieee_maximum(a, b))
TORCHLEAN_BINARY_EXPORT(min, ieee_minimum(a, b))
TORCHLEAN_BINARY_EXPORT(div, aten<"div.Tensor">(a, b))
TORCHLEAN_UNARY_EXPORT(neg, aten<"neg">(x))
TORCHLEAN_BINARY_EXPORT(add, add(a, b))
TORCHLEAN_BINARY_EXPORT(sub, sub(a, b))
TORCHLEAN_BINARY_EXPORT(mul, aten<"mul.Tensor">(a, b))
TORCHLEAN_BINARY_EXPORT(mask, aten<"where.self">(aten<"ne.Tensor">(b, scalar_for(b, 0.0)), a,
                                                 scalar_for(a, 0.0)))

TORCHLEAN_UNARY_SCALAR_EXPORT(scale, aten<"mul.Tensor">(x, scalar_for(x, scalar)))
TORCHLEAN_BINARY_SCALAR_EXPORT(axpy, axpy(a, b, scalar))
TORCHLEAN_BINARY_SCALAR_EXPORT(mean_update, mean_update(a, b, scalar))
// Preserve left association and use the same exp operation as the composed path.
TORCHLEAN_BINARY_SCALAR_EXPORT(scaled_prod_exp,
    aten<"exp">(aten<"mul.Tensor">(aten<"mul.Tensor">(a, scalar_for(a, scalar)), b)))
// Huber.term at each prediction and target, and the slope of Tensor.huberLossBackward at each
// difference, with the threshold rounded to the buffers' dtype.
TORCHLEAN_BINARY_SCALAR_EXPORT(huber_terms, huber_terms(a, b, scalar))
TORCHLEAN_UNARY_SCALAR_EXPORT(huber_slope, huber_slope(x, scalar))

// The zero factor of Activation.reluDeriv x * g gives -0 at a negative g and a NaN at a
// nonfinite one.
TORCHLEAN_VJP_EXPORT(relu,
    aten<"mul.Tensor">(aten<"where.self">(aten<"gt.Tensor">(x, scalar_for(x, 0.0)),
                                          aten<"ones_like">(g), aten<"zeros_like">(g)), g))
TORCHLEAN_VJP_EXPORT(sigmoid, activation_backward(x, g, false))
TORCHLEAN_VJP_EXPORT(tanh, activation_backward(x, g, true))
TORCHLEAN_VJP_EXPORT(gelu, staged_gelu_backward(x, g))
TORCHLEAN_VJP_EXPORT(silu, staged_silu_backward(x, g))
// g / (2 |sqrt x|) in one division: +inf times g at either zero, a NaN below zero and at a NaN.
TORCHLEAN_VJP_EXPORT(sqrt_raw, aten<"div.Tensor">(
    g, aten<"mul.Tensor">(aten<"abs">(aten<"sqrt">(x)), scalar_for(x, 2.0))))

TORCHLEAN_UNARY_EXPORT(reduce_sum, reduce_sum(x))
TORCHLEAN_UNARY_EXPORT(reduce_mean, reduce_mean(x))
