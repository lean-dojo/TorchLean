/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Elementwise and Activation IR Lowering

Checked lowering for pointwise arithmetic, unary functions, and activation operations.

Each operation has a named lowerer, called directly by the exhaustive `lowerNode` dispatch.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/-- Validate one parent at the output shape, then apply its tensor operation at execution time. -/
@[simp, inline] def lowerUnary {α : Type} [Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (label : String)
    (operation : Tensor α ctx.node.outShape → Tensor α ctx.node.outShape) :
    NodeLoweringResult ctx := do
  match unaryParent? ctx.node.parents with
  | some pId =>
      let ip ← ctx.parentIdx pId ctx.node.outShape
      pure <| mkForwardNode (fun values => operation (values.read ip))
  | none =>
      throw s!"IRExec: node {ctx.index}: {label} expects 1 parent ({ctx.node.summary})"

/-- Validate the left parent before the right; the right shape may differ, as for `safeLog`. -/
@[simp, inline] def lowerBinary {α : Type} [Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (label : String) (rightShape : Shape)
    (operation : Tensor α ctx.node.outShape → Tensor α rightShape → Tensor α ctx.node.outShape) :
    NodeLoweringResult ctx := do
  match binaryParents? ctx.node.parents with
  | some (aId, bId) =>
      let ia ← ctx.parentIdx aId ctx.node.outShape
      let ib ← ctx.parentIdx bId rightShape
      pure <| mkForwardNode (fun values =>
        operation (values.read ia) (values.read ib))
  | none =>
      throw s!"IRExec: node {ctx.index}: {label} expects 2 parents ({ctx.node.summary})"

/-- Checked lowering for `.add`. -/
def lowerAdd {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "add" ctx.node.outShape (Tensor.addSpec)

/-- Checked lowering for `.sub`. -/
def lowerSub {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "sub" ctx.node.outShape (Tensor.subSpec)

/-- Checked lowering for `.mulElem`. -/
def lowerMulElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "mul_elem" ctx.node.outShape (Tensor.mulSpec)

/-- Checked lowering for `.abs`. -/
def lowerAbs {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "abs" (Tensor.absSpec)

/-- Checked lowering for `.sqrt`. -/
def lowerSqrt {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "sqrt" (Tensor.sqrtSpec)

/-- Checked lowering for `.inv`. -/
def lowerInv {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "inv" (Tensor.invSpec)

/-- Checked lowering for `.maxElem`. -/
def lowerMaxElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "max_elem" ctx.node.outShape (Tensor.maxSpec)

/-- Checked lowering for `.minElem`. -/
def lowerMinElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "min_elem" ctx.node.outShape (Tensor.minSpec)

/-- Checked lowering for `.relu`. -/
def lowerRelu {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "relu" (Activation.reluSpec)

/-- Checked lowering for `.tanh`. -/
def lowerTanh {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "tanh" (Activation.tanhSpec)

/-- Checked lowering for `.sigmoid`. -/
def lowerSigmoid {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "sigmoid" (Activation.sigmoidSpec)

/-- Lower stable softplus through the scalar specification.

Retaining its sign branch preserves both the finite positive tail and the selected computation
at zero when the scalar carries first or higher derivatives. -/
def lowerSoftplus {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "softplus" (Activation.softplusSpec)

/-- Checked lowering for `.safeLog`. -/
def lowerSafeLog {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerBinary ctx "safe_log" .scalar
    (fun value epsilon => Activation.safeLogSpec value epsilon.item)

/-- Checked lowering for `.exp`. -/
def lowerExp {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "exp" (Tensor.expSpec)

/--
Checked lowering for `.log`.

The closure is total: it applies `Tensor.logSpec` to the parent value. The IR evaluator rejects
nonpositive inputs at runtime, which is why the end-to-end equivalence theorem excludes raw `.log`
through `NoRawLog`. A positive-input construction can avoid the domain failure, but its `.log`
node remains outside that syntactic theorem and requires a separate domain-aware argument.
-/
def lowerLog {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "log" (Tensor.logSpec)

/-- Checked lowering for `.sin`. -/
def lowerSin {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "sin" (Tensor.map (fun x => MathFunctions.sin x))

/-- Checked lowering for `.cos`. -/
def lowerCos {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  lowerUnary ctx "cos" (Tensor.map (fun x => MathFunctions.cos x))

/-- Checked lowering for `.softmax axis`. -/
def lowerSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  match unaryParent? n.parents with
  | some pId => do
      match Spec.Shape.axisInBounds? axis τ with
      | none =>
          throw s!"softmax: invalid axis {axis} for rank {Spec.Shape.rank τ}"
      | some h =>
          parentIdx pId τ >>= fun ip =>
            let forward := fun ctx : TensorLookup α Γ =>
              @Activation.softmaxSpec α _ _ τ axis h.down
                (ctx.read ip)
            pure <| mkForwardNode (τ := τ) forward
  | _ => throw s!"IRExec: node {i}: softmax expects 1 parent ({n.summary})"

/-- Checked lowering for `.hardMaskedSoftmax mask`. -/
def lowerHardMaskedSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (mask : NN.IR.HardMask) :
    NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  match unaryParent? n.parents with
  | some pId => do
      let ip ← parentIdx pId τ
      let allowed ←
        match NN.IR.HardMask.toTensorAs? mask τ with
        | .ok value => pure value
        | .error msg =>
            throw s!"IRExec: node {i}: hard_masked_softmax: {msg} ({n.summary})"
      let forward := fun ctx : TensorLookup α Γ =>
        Spec.hardMaskedSoftmaxSpec
          (ctx.read ip) allowed
      pure <| mkForwardNode (τ := τ) forward
  | _ =>
      throw s!"IRExec: node {i}: hard_masked_softmax expects 1 parent ({n.summary})"

end Internal
end IRExec
end Autograd
end Runtime
