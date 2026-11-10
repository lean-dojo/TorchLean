/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Capsule
public import TorchLean.Tensor.Op.Need

/-!
# Reference CPU Contracts

The contracts of the reference implementation, which runs a registered operator on the CPU tape
through the operator's rule in `Op.rule?`. Every contract has trust level `checked` and the
reduction order `Reference.reduction` of its rule. The scalar capability a rule reads (`Op.need`)
is checked when the tape runs it.

`Reference.contract` gives the contract of an operator, and `Reference.contract?` picks the
operators whose capsules the reference CPU table `Reference.implementation` offers.
-/

@[expose] public section

namespace TorchLean
namespace Backend
namespace Reference

/-- The checked CPU contract `reference.{tag}` with the reduction order `reduction`: its value is
the forward map of the operator's rule in `Op.rule?`, and its VJP, offered when `vjp` is true, is
the rule's pullback. -/
def checkedContract (tag : String) (vjp : Bool) (reduction : ReductionPolicy) : KernelContract :=
  { name := s!"reference.{tag}"
    trustLevel := .checked
    vjpMode := if vjp then .torchLeanTape else .none
    layout := .canonicalTensor
    shape :=
      { summary := "Shapes are checked in Lean before entering the portable runtime path."
        evidence := .runtimeGuard "portable runtime shape checks" }
    layoutNote :=
      { summary := "Portable paths use TorchLean's canonical tensor representation."
        evidence := .runtimeGuard "typed tensor layout" }
    value :=
      { summary := "The CPU tape runs the forward map of the operator's rule."
        evidence := .sharedDefinition [``Op.rule?] }
    vjp :=
      if vjp then
        { summary := "The CPU tape runs the pullback of the operator's rule."
          evidence := .sharedDefinition [``Op.rule?] }
      else
        { summary := "The operator reads no value input, so it has no VJP."
          evidence := .notApplicable }
    numericalPolicy := { reduction } }

/-- The reduction order of `op`'s rule in `Op.rule?`, whose sums are left folds from zero. It is
implementation-defined for the attention operators, and fixed-left or exact for `scaledMatmul`,
`softmax`, `maskedSoftmax` and `distill`, whose rules replay a sum in exact arithmetic after range
loss or an absorbed nonzero term. It is not applicable when neither the forward map nor the
pullback adds terms, and fixed-left for every other operator. -/
def reduction {ins outs : List Port} : Op ins outs → ReductionPolicy
  | .attentionFromScores .. | .multiheadAttention .. | .attention .. => .implementationDefined
  | .scaledMatmul .. | .softmax .. | .maskedSoftmax .. | .distill .. => .fixedLeftOrExact
  -- Pointwise maps, comparisons and conversions add no terms.
  | .relu | .sigmoid | .tanh | .exp | .abs | .sqrt .. | .sin | .cos | .neg .. | .add .. | .mul ..
  | .max .. | .log | .detach | .softplus | .silu | .square | .reciprocal | .sub .. | .div ..
  | .min .. | .safeLog | .clamp | .scale | .elu | .leakyReLU | .sign | .floor | .ceil | .round
  | .trunc | .frac | .rsqrt | .exp2 | .expm1 | .log2 | .log10 | .log1p | .tan | .asinh | .acosh
  | .atanh | .atan | .asin | .acos | .erf | .atan2 | .fmod | .remainder .. | .hypot | .copysign
  | .fmax | .fmin | .nanToNum | .select .. | .lerp | .addcmul | .addcdiv | .gelu .. | .sinh | .cosh
  | .safeInv | .smoothAbs | .safeDiv | .pow .. | .fullLike | .dropout | .eq .. | .ne .. | .lt ..
  | .le .. | .gt .. | .ge .. | .isNaN | .isInf | .isFinite | .logicalNot | .logicalAnd | .logicalOr
  | .logicalXor | .bitwiseAnd | .bitwiseOr | .bitwiseXor | .bitwiseNot | .bitwiseLeftShift
  | .shiftRightArithmetic | .shiftRightLogical | .populationCount | .countLeadingZeros | .mulhi
  | .bitcast .. | .cbrt | .nextafter | .argmax .. | .argmin .. | .cast .. | .encodeIndex ..
  | .decodeIndex .. => .notApplicable
  -- Moves and selections of entries, random sources and masks add no terms.
  | .sort .. | .topk .. | .randUniform .. | .randn .. | .reshape .. | .permute .. | .flip ..
  | .indices .. | .slice .. | .indexAt .. | .cat .. | .topKMask .. | .bernoulliMask ..
    => .notApplicable
  | _ => .fixedLeft

/-- The reference contract of multi-head attention, which names the multi-head attention
definitions its rule runs. -/
def multiheadAttention : KernelContract :=
  { name := "reference.multiheadAttention"
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    layout := .canonicalTensor
    shape :=
      { summary := "Q/K/V and mask shapes are checked by the typed tensor layer."
        evidence := .runtimeGuard "typed attention shapes" }
    layoutNote :=
      { summary :=
          "Reference multi-head attention uses TorchLean tensor semantics, not a foreign layout."
        evidence := .runtimeGuard "typed tensor layout" }
    value :=
      { summary := "Composed reference attention uses hard-mask zero numerators and replays " ++
          "complete finite scaled contractions after range loss or an absorbed nonzero term."
        evidence := .sharedDefinition [``MultiHeadAttention.forward] }
    vjp :=
      { summary := "The CPU tape runs the VJP of each sequence and adds the projection " ++
          "gradients of the sequences in row-major order of the leading axes."
        evidence := .sharedDefinition
          [``MultiHeadAttention.backwardLeading,
            ``MultiHeadAttention.sum] }
    numericalPolicy := { reduction := .implementationDefined } }

/-- The reference contract of `op`: `multiheadAttention` for multi-head attention, and otherwise
`reference.<tag>` for the wire tag of `op` (`Op.encode`), with a VJP exactly when `op` reads a
value input (`requiresVJP`). -/
def contract : {ins outs : List Port} → Op ins outs → KernelContract
  | _, _, .multiheadAttention .. => multiheadAttention
  | ins, outs, op =>
    checkedContract (Op.encode ⟨ins, outs, op⟩).tag (requiresVJP ins) (reduction op)

/-- The contract of `op` on the CPU tape, or `none` for `detach`, which the session records without
a capsule, and for an operator without a rule (`Op.hasRule`). -/
def contract? {ins outs : List Port} : Op ins outs → Option KernelContract
  | .detach => none
  | op => if op.hasRule then some (contract op) else none

/-- `contract?` offers only `contract op`, and only for an operator with a rule. -/
theorem contract?_eq_some {ins outs : List Port} {op : Op ins outs} {c : KernelContract}
    (h : contract? op = some c) : op.hasRule = true ∧ contract op = c := by
  unfold contract? at h
  split at h
  · simp at h
  · split at h
    · exact ⟨‹_›, Option.some.inj h⟩
    · simp at h

/-- The reference CPU implementation table: the capsule of `contract? op` for each operator `op`. -/
def implementation : Implementation :=
  Implementation.ofContracts "reference" .reference .cpu contract?

end Reference
end Backend
end TorchLean
