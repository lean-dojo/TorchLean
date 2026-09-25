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

TorchLean owns the tape and local VJP. LibTorch batched matrix multiplication evaluates the two
dense contractions, and the LibTorch bridge implements TorchLean's hard-masked softmax convention.
The provider is `torchLean` because Lean composes these primitive calls and their selected VJP.
The primitives do not record a LibTorch autograd graph.
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
        (.layoutCompatibility .scaledDotProductAttention .libTorchCudaView)
        "Row-major LibTorch tensors; the folded (batch, head) axis is the BMM batch axis."
        "LibTorch bridge dtype, device, contiguity, and bmm shape checks"
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
Direct LibTorch attention bridge.

The native bridge evaluates forward and the selected local VJP using ATen operations. TorchLean
still owns the global tape. Neither call records a LibTorch autograd graph, and the capsule makes
no promise about an IO-tiled algorithm or a particular reduction schedule.
-/
def libTorchDirectAttention : KernelCapsule :=
  { name := "libtorch.direct_attention"
    op := .scaledDotProductAttention
    provider := .libTorch
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .backendVJP
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
        ("Q/K/V use a folded (batch, head, n, headDim) layout; the optional mask broadcasts " ++
          "over the folded batch-head axis.")
        "torchlean_libtorch_attention_fwd/bwd size and saved-context checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility .scaledDotProductAttention .libTorchCudaView)
        "Row-major LibTorch tensors; the folded batch-head axis is the kernel batch axis."
        "LibTorch bridge dtype, device, contiguity, and element-count checks"
    valueContract :=
      ContractDescriptor.tested (.valueRefinement .scaledDotProductAttention)
        "LibTorch attention with hard-mask zero numerators and zero fully blocked rows."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement .scaledDotProductAttention .backendVJP)
        "ATen operations evaluate the selected local VJP and return dQ, dK, and dV."
        "NN.Tests.Runtime.Cuda.Attention"
    numericalPolicy := { reduction := .implementationDefined } }

/--
Maintained attention choices; the checked CUDA profile prefers the direct LibTorch bridge.
The TorchLean composition remains selectable through an explicit provider preference.
-/
def capsules : Array KernelCapsule :=
  #[libTorchDirectAttention, torchLeanComposed]

end Attention
end Backend
end NN
