// LibTorch SDPA forward/backward (g++; nvcc cannot parse torch headers).
#include <lean/lean.h>
#include <torch/torch.h>
#include "torchlean_cuda_buffer.h"
#include <cuda_runtime.h>

static at::Tensor view3(b_lean_obj_arg o, uint32_t batch, int64_t n, int64_t d) {
  return at::from_blob(torchlean_cuda_buffer_unbox(o)->data, {batch, n, d}, at::kFloat).cuda();
}

static c10::optional<at::Tensor> attnMask(b_lean_obj_arg M, uint32_t hasMask, uint32_t batch,
                                        uint32_t n) {
  if (!hasMask) return c10::nullopt;
  return c10::optional<at::Tensor>((1 - view3(M, batch, n, n)) * -1000.f);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_fwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  auto mask = attnMask(M, hasMask, batch, n);
  auto y = at::scaled_dot_product_attention(view3(Q, batch, n, d), view3(K, batch, n, d),
                                            view3(V, batch, n, d), mask, 0., false, scale)
               .contiguous();
  auto* out = torchlean_cuda_buffer_alloc(y.numel());
  cudaMemcpy(out->data, y.data_ptr<float>(), y.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
  return torchlean_cuda_buffer_box(out);
}

extern "C" LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_bwd(
    b_lean_obj_arg Q, b_lean_obj_arg K, b_lean_obj_arg V, b_lean_obj_arg M, b_lean_obj_arg DOut,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scale) {
  at::AutoGradMode enable_grad(true);
  auto mask = attnMask(M, hasMask, batch, n);
  auto q = view3(Q, batch, n, d).detach().requires_grad_(true);
  auto k = view3(K, batch, n, d).detach().requires_grad_(true);
  auto v = view3(V, batch, n, d).detach().requires_grad_(true);
  auto grad_out = view3(DOut, batch, n, d);
  auto out = at::scaled_dot_product_attention(q, k, v, mask, 0., false, scale);
  out.backward(grad_out);
  const size_t numel = (size_t)batch * (size_t)n * (size_t)d;
  at::Tensor grads[3] = {q.grad(), k.grad(), v.grad()};
  torchlean_cuda_buffer* bufs[3];
  for (int i = 0; i < 3; ++i) bufs[i] = torchlean_cuda_buffer_alloc(numel);
  for (int i = 0; i < 3; ++i) {
    auto t = grads[i].contiguous();
    cudaMemcpy(bufs[i]->data, t.data_ptr<float>(), numel * sizeof(float), cudaMemcpyDeviceToDevice);
  }
  return torchlean_cuda_box_three_buffers(bufs[0], bufs[1], bufs[2]);
}
