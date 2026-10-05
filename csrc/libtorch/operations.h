// Shared operation list for the LibTorch exports and builds without LibTorch.
// Intentionally included more than once, with the signature macros defined by the caller.

TORCHLEAN_UNARY_EXPORT(abs, at::abs(x))
TORCHLEAN_UNARY_EXPORT(floor, at::floor(x))
TORCHLEAN_UNARY_EXPORT(sqrt, selected_sqrt(x))
TORCHLEAN_UNARY_EXPORT(exp, at::exp(x))
TORCHLEAN_UNARY_EXPORT(sin, at::sin(x))
TORCHLEAN_UNARY_EXPORT(cos, at::cos(x))
TORCHLEAN_UNARY_EXPORT(log, at::log(x))
TORCHLEAN_UNARY_EXPORT(inv, at::reciprocal(x))
TORCHLEAN_UNARY_EXPORT(relu, selected_relu(x))
TORCHLEAN_UNARY_EXPORT(sigmoid, at::sigmoid(x))
TORCHLEAN_UNARY_EXPORT(tanh, at::tanh(x))
TORCHLEAN_UNARY_EXPORT(gelu, staged_gelu(x))

TORCHLEAN_BINARY_EXPORT(max, at::fmax(a, b))
TORCHLEAN_BINARY_EXPORT(min, at::fmin(a, b))
TORCHLEAN_BINARY_EXPORT(div, at::div(a, b))
TORCHLEAN_BINARY_EXPORT(add, at::add(a, b))
TORCHLEAN_BINARY_EXPORT(sub, at::sub(a, b))
TORCHLEAN_BINARY_EXPORT(mul, at::mul(a, b))
TORCHLEAN_BINARY_EXPORT(mask, at::where(at::ne(b, 0.0f), a, 0.0f))

TORCHLEAN_UNARY_SCALAR_EXPORT(scale, at::mul(x, scalar))
TORCHLEAN_BINARY_SCALAR_EXPORT(axpy, axpy(a, b, scalar))
// Preserve left association and use the same exp operation as the composed path.
TORCHLEAN_BINARY_SCALAR_EXPORT(scaled_prod_exp, at::exp(at::mul(at::mul(a, scalar), b)))

TORCHLEAN_VJP_EXPORT(relu, at::where(at::gt(x, 0.0f), g, 0.0f))
TORCHLEAN_VJP_EXPORT(gelu, staged_gelu_backward(x, g))

TORCHLEAN_UNARY_EXPORT(reduce_sum, reduce_sum(x))
TORCHLEAN_UNARY_EXPORT(reduce_mean, reduce_mean(x))
