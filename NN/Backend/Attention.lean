/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Attention Backend Capsules

Attention is the first place where the backend-contract distinction matters in practice.

The proof-facing FlashAttention spec and every registered runtime provider use hard-mask semantics:
blocked entries have exactly zero softmax numerator. Additive attention biases are a separate
operation and are never used to encode a boolean mask.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Attention

/--
Composed TorchLean attention path.

TorchLean owns the tape and local VJP. CUDA batched matrix multiplication evaluates the two dense
contractions, while TorchLean's hard-masked softmax supplies the attention weights. Unlike the
direct reference kernel, this implementation has the expected quadratic dependence on sequence
length.
-/
def torchLeanComposed : KernelCapsule :=
  { name := "torchlean.composed_attention"
    op := .scaledDotProductAttention
    provider := .torchLean
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
        ("Q/K/V have logical shape (batch, heads, n, headDim), with batch optional; the runtime " ++
          "folds (batch, heads) for BMM and broadcasts the mask over that folded axis.")
        "Runtime.Autograd.Cuda.requireValue plus checked UInt32 dimensions"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility .scaledDotProductAttention .flatRowMajor)
        "Flat row-major buffers; the folded (batch, head) axis is the BMM batch axis."
        "Buffer.swapAdjacentAtDepth and bmm shape checks"
    valueContract :=
      ContractDescriptor.tested
        (.valueRefinement .scaledDotProductAttention)
        "Composed bmm, hard-masked row softmax, and bmm."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement .scaledDotProductAttention .torchLeanTape)
        ("TorchLean tape VJP through the composed expression, summing shared weight gradients " ++
          "over the leading batch.")
        "Runtime autograd attention tests"
    numericalPolicy := { reduction := .implementationDefined } }

/--
Direct native CUDA reference path.

The implementation computes attention without materializing the score matrix, but it is not the
IO-tiled FlashAttention algorithm. Its backward kernel recomputes row statistics and is intended
for parity checks and small inputs rather than large-model training.
-/
def nativeDirectAttention : KernelCapsule :=
  { name := "native_cuda.direct_attention"
    op := .scaledDotProductAttention
    provider := .nativeCuda
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .backendVJP
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
        ("Q/K/V use a folded (batch, head, n, headDim) layout; the optional mask broadcasts " ++
          "over the folded batch-head axis.")
        "torchlean_cuda_buffer_flash_attention_* size checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility .scaledDotProductAttention .flatRowMajor)
        "Flat row-major buffers; the folded batch-head axis is the kernel batch axis."
        "Cuda.Buffer shape/size FFI checks"
    valueContract :=
      ContractDescriptor.tested (.valueRefinement .scaledDotProductAttention)
        "Direct CUDA fused attention with hard-mask zero-numerator semantics."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement .scaledDotProductAttention .backendVJP)
        "CUDA VJP kernels return dQ, dK, and dV for the fused operator."
        "NN.Tests.Runtime.Cuda.Attention"
    numericalPolicy := { reduction := .implementationDefined } }

/-- LibTorch SDPA forward provider while TorchLean keeps the graph/tape boundary. -/
def libTorchSDPAForward : KernelCapsule :=
  { name := "libtorch.sdpa_forward"
    op := .scaledDotProductAttention
    provider := .libTorch
    device := .cuda
    trustLevel := .trustedExternal
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
        ("Q/K/V are checked after folding batch and head axes; the optional mask is broadcast " ++
          "over that folded axis.")
        "torchlean_libtorch_sdpa_fwd size checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility .scaledDotProductAttention .libTorchCudaView)
        "TorchLean CUDA buffers are wrapped by LibTorch from_blob and copied back contiguous."
        "contiguous CUDA tensor views"
        #[.nativeSymbol
          { path := "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp"
            symbol := "torchlean_libtorch_sdpa_fwd"
            buildTarget? := some "torchlean_libtorch_sdpa_so" }]
    valueContract :=
      ContractDescriptor.trusted
        (.valueRefinement .scaledDotProductAttention)
        "LibTorch scaled_dot_product_attention forward."
        "LibTorch/CUDA SDPA implementation"
    vjpContract :=
      ContractDescriptor.guarded
        (.vjpRefinement .scaledDotProductAttention .torchLeanTape)
        "TorchLean records the node and keeps the backward boundary inside TorchLean."
        "backend profile requires vjpMode=torchLeanTape"
    numericalPolicy := { reduction := .implementationDefined } }

/-- Built-in attention capsules in default planner order. Optional external providers register
their own capsules in their provider modules. -/
def capsules : Array KernelCapsule :=
  #[nativeDirectAttention, torchLeanComposed]

end Attention
end Backend
end NN
