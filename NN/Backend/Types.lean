/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Backend Types

Small vocabulary for backend selection and trust boundaries.

TorchLean owns the spec, graph, and proof-facing contracts. Backends are execution providers for
parts of that graph: a Lean reference path, the TorchLean runtime, native CUDA kernels, or LibTorch.
This file deliberately contains only data. It should stay cheap to import from specs, runtime
wrappers, docs generators, and tests.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Hardware or execution target visible to the planner. -/
inductive Device where
  | cpu
  | cuda
  | rocm
  | metal
  | wasm
  | tpu
  | trainium
  | custom
  | external
  deriving DecidableEq, Repr

namespace Device

/-- Stable spelling used in profile names, reports, and CLI bridges. -/
def cliName : Device → String
  | .cpu => "cpu"
  | .cuda => "cuda"
  | .rocm => "rocm"
  | .metal => "metal"
  | .wasm => "wasm"
  | .tpu => "tpu"
  | .trainium => "trainium"
  | .custom => "custom"
  | .external => "external"

/-- Parse an explicit backend device name. CLI layers may resolve policy names such as `auto`
before calling this function. -/
def parse? : String → Option Device
  | "cpu" => some .cpu
  | "cuda" => some .cuda
  | "rocm" => some .rocm
  | "metal" => some .metal
  | "wasm" => some .wasm
  | "tpu" => some .tpu
  | "trainium" => some .trainium
  | "custom" => some .custom
  | "external" => some .external
  | _ => none

/-- Parse an explicit backend device name or return a diagnostic suitable for command-line use. -/
def parse (value : String) : Except String Device :=
  match parse? value with
  | some device => pure device
  | none =>
      throw <| s!"unknown device {value} (known targets: cpu | cuda | rocm | metal | wasm | tpu " ++
        "| trainium | custom | external)"

end Device

/-- Concrete provider family used to execute a kernel capsule. -/
inductive Provider where
  | reference
  | torchLean
  | nativeCuda
  | libTorch
  | aten
  | mps
  | webGpu
  | cuBLAS
  | cuDNN
  | cuFFT
  | xla
  | neuron
  | customChip
  | external
  deriving DecidableEq, Repr

/-- Stable operation vocabulary used by backend capsules, graph planning, and runtime guards.

This is deliberately a closed vocabulary. New backend-visible operations should be added here and
then wired through the IR adapter and capsule registry. Runtime tape/debug labels may still be
strings, but the backend planner should not accept arbitrary stringly-typed operation names.
-/
inductive BackendOp where
  | randUniform
  | bernoulliMask
  | add
  | sub
  | mul
  | scale
  | abs
  | sqrt
  | clamp
  | max
  | min
  | relu
  | gelu
  | sigmoid
  | tanh
  | softplus
  | exp
  | log
  | sin
  | cos
  | inv
  | safeLog
  | logSoftmax
  | softmax
  | hardMaskedSoftmax
  | reduceSum
  | reduceMean
  | reshape
  | permute
  | broadcast
  | concat
  | slice
  | gather
  | scatterAdd
  | matmul
  | linear
  | mseLoss
  | layerNorm
  | batchNorm
  | conv
  | convTranspose
  | maxPool
  | smoothMaxPool
  | avgPool
  | fftFno
  | selectiveScan
  | scaledDotProductAttention
  deriving DecidableEq, BEq, Repr

namespace BackendOp

/-- Stable spelling used in reports, capsule names, and CLI diagnostics. -/
def name : BackendOp → String
  | .randUniform => "rand_uniform"
  | .bernoulliMask => "bernoulli_mask"
  | .add => "add"
  | .sub => "sub"
  | .mul => "mul"
  | .scale => "scale"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .clamp => "clamp"
  | .max => "max"
  | .min => "min"
  | .relu => "relu"
  | .gelu => "gelu"
  | .sigmoid => "sigmoid"
  | .tanh => "tanh"
  | .softplus => "softplus"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .inv => "inv"
  | .safeLog => "safe_log"
  | .logSoftmax => "log_softmax"
  | .softmax => "softmax"
  | .hardMaskedSoftmax => "hard_masked_softmax"
  | .reduceSum => "reduce_sum"
  | .reduceMean => "reduce_mean"
  | .reshape => "reshape"
  | .permute => "permute"
  | .broadcast => "broadcast"
  | .concat => "concat"
  | .slice => "slice"
  | .gather => "gather"
  | .scatterAdd => "scatter_add"
  | .matmul => "matmul"
  | .linear => "linear"
  | .mseLoss => "mse_loss"
  | .layerNorm => "layer_norm"
  | .batchNorm => "batch_norm"
  | .conv => "conv"
  | .convTranspose => "conv_transpose"
  | .maxPool => "max_pool"
  | .smoothMaxPool => "smooth_max_pool"
  | .avgPool => "avg_pool"
  | .fftFno => "fft_fno"
  | .selectiveScan => "selective_scan"
  | .scaledDotProductAttention => "scaled_dot_product_attention"

instance : ToString BackendOp where
  toString op := op.name

/-- Whether training through this operation requires a registered local VJP.

Random sources create values but are not themselves differentiated. Every other backend-visible
operation must provide a compatible VJP whenever gradient tracking is requested.
-/
def requiresVJP : BackendOp → Bool
  | .randUniform | .bernoulliMask => false
  | _ => true

end BackendOp

/--
How much TorchLean knows about an implementation.

`trustedExternal` is allowed, but it is intentionally loud: the contract names the boundary instead
of silently treating an industrial kernel as though Lean had verified its source.
-/
inductive TrustLevel where
  | checked
  | trustedExternal
  deriving DecidableEq, Repr

/--
Which trust boundaries a kernel plan may cross.

The same record decides both which capsules the planner may select (by trust level) and which
contract evidence the selected capsules may rely on (by evidence kind). Keeping the decision in one
place prevents a profile from selecting a capsule under one policy and checking it under another.
-/
structure AssurancePolicy where
  /-- Whether capsules and evidence that delegate to an external implementation are admitted. -/
  allowTrustedExternal : Bool := false
  deriving DecidableEq, Repr

namespace AssurancePolicy

/--
Maintained TorchLean runtime policy.

Checked implementations backed by runtime guards and regression evidence are accepted; trusted
external implementations are not.
-/
def checked : AssurancePolicy := {}

/--
Explicit external-provider policy.

This is the policy used when a caller deliberately delegates a numerical kernel to LibTorch or
another external implementation. The selected boundary remains visible in the execution audit.
-/
def external : AssurancePolicy :=
  { allowTrustedExternal := true }

/-- Whether the policy admits a capsule with the given implementation trust level. -/
def acceptsTrust (policy : AssurancePolicy) : TrustLevel → Bool
  | .checked => true
  | .trustedExternal => policy.allowTrustedExternal

end AssurancePolicy

/-- How a backend capsule treats gradients. -/
inductive VJPMode where
  | none
  | torchLeanTape
  | backendVJP
  deriving DecidableEq, Repr

/-- Provider preference used when selecting an implementation for an operation. -/
inductive ProviderPreference where
  | auto
  | prefer (provider : Provider)
  | only (provider : Provider)
  deriving DecidableEq, Repr

/-- Policy used to select kernel capsules for a device and assurance boundary. -/
structure KernelPolicy where
  device : Device := .cpu
  provider : ProviderPreference := .auto
  assurance : AssurancePolicy := .checked
  vjpMode : VJPMode := .torchLeanTape
  deriving Repr

end Backend
end NN
