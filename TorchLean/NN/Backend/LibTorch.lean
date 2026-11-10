/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Capsule

/-!
# LibTorch Backend Contracts

Contracts of the maintained CUDA primitives that LibTorch ATen tensor operations implement. The
entries of the CUDA implementation table `TorchLean.LibTorch.Kernel.ofOp` build their contracts
with the builders of this file. The multi-head attention entry carries `multiheadAttention`, and
`detach`'s entry, which no planner selects, has none. `KernelContract.capsule` turns a contract
into its capsule for an operator.

TorchLean owns tape traversal and selects each local VJP, and the native forward and backward
routines run without recording a LibTorch autograd graph. The wrappers are a foreign
implementation boundary. Their shape and layout guards and named regression suites are
engineering evidence, and no Lean proof covers ATen, its dependencies or the compiled code.
Numerical reductions keep an implementation-defined order. The CPU and reference capsules, with
their mathematical contracts, live in `Backend.Reference`.

Each contract's VJP mode follows what its entry in `TorchLean.LibTorch.Kernel.ofOp` does: under
`.torchLeanTape` Lean composes the derivative from numerical primitives, and under `.backendVJP`
the entry calls a native backward primitive. In both modes the provider is LibTorch and the global
tape is TorchLean's.
-/

@[expose] public section

namespace TorchLean
namespace Backend
namespace LibTorch

/-- The retained CUDA comparisons of a contract: the suite comparing the forward pass and the one
comparing the VJP. A missing suite leaves a named trust boundary, recorded by
`Backend.LibTorch.suiteEvidence`. -/
structure Coverage where
  /-- Suite comparing the forward pass, if any. -/
  forward : Option String := none
  /-- Suite comparing the VJP, if any. -/
  vjp : Option String := none

/-- Forward pass and VJP both compared by `suite`. -/
def Coverage.full (suite : String) : Coverage := { forward := some suite, vjp := some suite }

/-- Forward pass compared by `suite`, with no retained comparison for the VJP. -/
def Coverage.forwardOnly (suite : String) : Coverage := { forward := some suite }

/-- No retained CUDA comparison. -/
def Coverage.empty : Coverage := {}

/-- The retained elementwise comparisons. -/
def Coverage.elementwise : Coverage := full "TorchLeanTest.CUDA.Elementwise"

/-- The `testSuite` evidence of a retained comparison, or without one a `trustedBoundary` naming
the untested native agreement that is assumed. -/
def suiteEvidence (label : String) (suite? : Option String) (vjp : Bool := false) :
    Contract.Evidence :=
  match suite? with
  | some suite => .testSuite suite
  | none => .trustedBoundary <|
      s!"LibTorch `{label}` implementation; no retained dedicated CUDA " ++
        (if vjp then "VJP parity check" else "forward parity check")

/-- `checked` when the forward pass, and the VJP if there is one, have a retained CUDA comparison,
and `trustedExternal` otherwise. -/
def Coverage.trustLevel (coverage : Coverage) (vjpMode : VJPMode) : TrustLevel :=
  if coverage.forward.isSome && (vjpMode == .none || coverage.vjp.isSome) then
    .checked
  else
    .trustedExternal

/-- The contract `libtorch.{label}` of a maintained LibTorch primitive, with the evidence of
`coverage` and the reduction order `reduction`, which is `.implementationDefined` where ATen selects
the accumulation order, even under deterministic execution settings. `label` also names the
primitive in trust-boundary reasons. -/
def primitiveContract
    (label : String) (coverage : Coverage) (valueSummary vjpSummary : String) (vjpMode : VJPMode)
    (reduction : ReductionPolicy) : KernelContract :=
  { name := s!"libtorch.{label}"
    trustLevel := Coverage.trustLevel coverage vjpMode
    supportsForward := true
    vjpMode
    layout := .libTorchCUDAView
    shape :=
      { summary := "Inputs and outputs are checked against explicit UInt32 dimensions."
        evidence := .runtimeGuard "LibTorch bridge size/rank checks at the Lean/native boundary" }
    layoutNote :=
      { summary := "CUDA buffers retain their dtype in contiguous row-major LibTorch tensors."
        evidence := .runtimeGuard
          "LibTorch bridge dtype, device, contiguity, and element-count checks" }
    value := { summary := valueSummary, evidence := suiteEvidence label coverage.forward }
    vjp :=
      match vjpMode with
      | .none => { summary := vjpSummary, evidence := .notApplicable }
      | _ => { summary := vjpSummary, evidence := suiteEvidence label coverage.vjp (vjp := true) }
    numericalPolicy := { reduction } }

/-- The standard LibTorch CUDA contract of a pointwise operation. -/
def pointwiseContract (label : String) (coverage : Coverage) (vjpMode : VJPMode) :
    KernelContract :=
  primitiveContract
    label
    coverage
    s!"LibTorch CUDA `{label}` follows the pointwise tensor contract."
    s!"Local reverse rule for LibTorch CUDA `{label}`."
    vjpMode .notApplicable

/-- The qualification of a CUDA value or VJP that ATen computes with the CUDA vendor math
library. It is report metadata, with no refinement proof behind it. -/
def vendorMathQualification : String :=
  "computed with the CUDA vendor math library; not bit-exact to the Lean reference, which stays " ++
    "the specification"

/-- The LibTorch CUDA contract of a pointwise operation that ATen evaluates with the CUDA vendor
math library. Its results need not equal the Lean reference, which is the specification, bit for
bit, and the value and VJP notes carry `vendorMathQualification` into the reports. -/
def vendorMathContract (label : String) (coverage : Coverage) (vjpMode : VJPMode) :
    KernelContract :=
  let c := primitiveContract
    label
    coverage
    (s!"LibTorch CUDA `{label}` is computed with the CUDA vendor math library; bit equality " ++
      "with the Lean reference, which is the specification, is not guaranteed.")
    s!"Local reverse rule for LibTorch CUDA `{label}`, computed with the vendor math library."
    vjpMode .notApplicable
  { c with
    value.qualification := vendorMathQualification
    vjp.qualification := vendorMathQualification }

/-- A LibTorch CUDA reduction contract with implementation-defined reduction order. -/
def reductionContract (label : String) (coverage : Coverage) (vjpMode : VJPMode) :
    KernelContract :=
  primitiveContract
    label
    coverage
    s!"LibTorch CUDA `{label}` follows the explicit reduction shape contract."
    s!"Local reduction adjoint for LibTorch CUDA `{label}`."
    vjpMode .implementationDefined

/-- The standard LibTorch CUDA contract of a shape or layout transformation. -/
def viewContract (label : String) (coverage : Coverage) (vjpMode : VJPMode) : KernelContract :=
  primitiveContract
    label
    coverage
    s!"LibTorch CUDA `{label}` follows the explicit shape/layout contract."
    s!"Local shape/layout adjoint for LibTorch CUDA `{label}`."
    vjpMode .notApplicable

/-- A LibTorch CUDA forward-only contract with no registered reverse derivative. -/
def forwardOnlyContract (label : String) (coverage : Coverage) (valueSummary : String) :
    KernelContract :=
  primitiveContract
    label
    coverage
    valueSummary
    s!"LibTorch CUDA `{label}` is a forward-only capsule with no registered VJP."
    .none .notApplicable

open Coverage

/--
Lean-composed multi-head attention over LibTorch numerical primitives.

TorchLean composes the forward pass and the local VJP and owns the saved Q/K/V and probability
buffers. Over leading batch axes the entry is one node over the folded (batch, head) axis, with the
projection gradients summed over every row by the native products, and a leading axis of extent
zero is rejected. `TorchLeanTest.CUDA.Attention` checks it on two sequences against one CPU
tape per sequence (to an absolute 2e-2) and against the entry of each sequence on the device: the
values and the input's gradient bit for bit, the projection gradients (summed per sequence there)
to 2e-2.

No native attention context or fused-provider selection is involved, and full score matrices take
memory quadratic in the sequence length. The numerical primitives run without a LibTorch autograd
graph or a fixed reduction schedule.
-/
def multiheadAttention : KernelContract :=
  { name := "libtorch.multihead_attention"
    trustLevel := trustLevel (full "TorchLeanTest.CUDA.Attention") .torchLeanTape
    supportsForward := true
    vjpMode := .torchLeanTape
    layout := .libTorchCUDAView
    shape :=
      { summary :=
          "Q/K/V use a folded (batch, head, n, headDim) layout, every leading axis folded into " ++
            "the batch; the optional mask broadcasts over the folded batch-head axis; a leading " ++
            "axis of extent zero is rejected."
        evidence := .runtimeGuard
          "Buffer.attentionForward/attentionBackward dimension, scale, and buffer-size checks" }
    layoutNote :=
      { summary :=
          "Row-major LibTorch tensors; the folded batch-head axis is the kernel batch axis."
        evidence := .runtimeGuard
          "LibTorch bridge dtype, device, contiguity, and element-count checks" }
    value :=
      { summary :=
          "Lean-composed attention with hard-mask zero numerators and zero weights in fully " ++
            "blocked rows. Over leading batch axes it is one node, whose values match the " ++
            "entry of each sequence bit for bit on the parity test."
        evidence := .testSuite "TorchLeanTest.CUDA.Attention" }
    vjp :=
      { summary :=
          "Lean composes matrix products and the softmax VJP over saved probabilities for " ++
            "dQ/dK/dV; over leading axes the projection gradients sum over all rows at once."
        evidence := .testSuite "TorchLeanTest.CUDA.Attention" }
    numericalPolicy := { reduction := .implementationDefined } }

end LibTorch
end Backend
end TorchLean
