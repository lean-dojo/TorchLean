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

Each operation has its own small `lower*` definition. `lowerElementwise` only dispatches on the
operation kind, and the `lowerElementwise_*` equation lemmas let correctness proofs reduce a
dispatch to the branch they care about without unfolding the whole dispatcher.

The `.log` closure applies `Tensor.logSpec` to every input. The IR evaluator additionally rejects
nonpositive inputs, so the end-to-end semantic equivalence theorem carries the `NoRawLog` side
condition; the lowered closure itself is total and never panics.
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

/-- Checked lowering for `.add`. -/
def lowerAdd {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.addSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: add expects 2 parents ({n.summary})"

/-- Checked lowering for `.sub`. -/
def lowerSub {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.subSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: sub expects 2 parents ({n.summary})"

/-- Checked lowering for `.mulElem`. -/
def lowerMulElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.mulSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: mul_elem expects 2 parents ({n.summary})"

/-- Checked lowering for `.abs`. -/
def lowerAbs {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.absSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: abs expects 1 parent ({n.summary})"

/-- Checked lowering for `.sqrt`. -/
def lowerSqrt {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.sqrtSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: sqrt expects 1 parent ({n.summary})"

/-- Checked lowering for `.inv`. -/
def lowerInv {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.invSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: inv expects 1 parent ({n.summary})"

/-- Checked lowering for `.maxElem`. -/
def lowerMaxElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.maxSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: max_elem expects 2 parents ({n.summary})"

/-- Checked lowering for `.minElem`. -/
def lowerMinElem {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.minSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: min_elem expects 2 parents ({n.summary})"

/-- Checked lowering for `.relu`. -/
def lowerRelu {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: relu expects 1 parent ({n.summary})"

/-- Checked lowering for `.tanh`. -/
def lowerTanh {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: tanh expects 1 parent ({n.summary})"

/-- Checked lowering for `.sigmoid`. -/
def lowerSigmoid {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: sigmoid expects 1 parent ({n.summary})"

/-- Lower stable softplus through the scalar specification.

Retaining its sign branch preserves both the finite positive tail and the selected computation
at zero when the scalar carries first or higher derivatives. -/
def lowerSoftplus {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Activation.softplusSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: softplus expects 1 parent ({n.summary})"

/-- Checked lowering for `.safeLog`. -/
def lowerSafeLog {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let ia ← parentIdx aId τ
      let ib ← parentIdx bId .scalar
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Activation.safeLogSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
          ctx) ib).item
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: safe_log expects 2 parents ({n.summary})"

/-- Checked lowering for `.exp`. -/
def lowerExp {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: exp expects 1 parent ({n.summary})"

/--
Checked lowering for `.log`.

The closure is total: it applies `Tensor.logSpec` to the parent value. The IR evaluator rejects
nonpositive inputs at runtime, which is why the end-to-end equivalence theorem excludes raw `.log`
through `NoRawLog`. A positive-input construction can avoid the domain failure, but its `.log`
node remains outside that syntactic theorem and requires a separate domain-aware argument.
-/
def lowerLog {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.logSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: log expects 1 parent ({n.summary})"

/-- Checked lowering for `.sin`. -/
def lowerSin {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.sin x) (getIdx (α := α)
          (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: sin expects 1 parent ({n.summary})"

/-- Checked lowering for `.cos`. -/
def lowerCos {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId τ
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.cos x) (getIdx (α := α)
          (xs := ctx) ip)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: cos expects 1 parent ({n.summary})"

/-- Checked lowering for `.softmax axis`. -/
def lowerSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId => do
      match Spec.Shape.axisInBounds? axis τ with
      | none =>
          throw s!"softmax: invalid axis {axis} for rank {Spec.Shape.rank τ}"
      | some h =>
          parentIdx pId τ >>= fun ip =>
            let forward := fun ctx : TorchLean.TensorPack α Γ =>
              @Activation.softmaxSpec α _ _ τ axis h.down
                (getIdx (α := α) (xs := ctx) ip)
            pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: softmax expects 1 parent ({n.summary})"

/-- Checked lowering for `.hardMaskedSoftmax mask`. -/
def lowerHardMaskedSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (mask : NN.IR.HardMask) :
    NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId => do
      let ip ← parentIdx pId τ
      let allowed ←
        match NN.IR.HardMask.toTensorAs? mask τ with
        | .ok value => pure value
        | .error msg =>
            throw s!"IRExec: node {i}: hard_masked_softmax: {msg} ({n.summary})"
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        Spec.hardMaskedSoftmaxSpec
          (getIdx (α := α) (xs := ctx) ip) allowed
      pure <| fwd forward
  | _ =>
      throw s!"IRExec: node {i}: hard_masked_softmax expects 1 parent ({n.summary})"

/-- Checked lowering for pointwise arithmetic, unary functions, and activation operations. -/
def lowerElementwise {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .add => lowerAdd ctx
  | .sub => lowerSub ctx
  | .mulElem => lowerMulElem ctx
  | .abs => lowerAbs ctx
  | .sqrt => lowerSqrt ctx
  | .inv => lowerInv ctx
  | .maxElem => lowerMaxElem ctx
  | .minElem => lowerMinElem ctx
  | .relu => lowerRelu ctx
  | .tanh => lowerTanh ctx
  | .sigmoid => lowerSigmoid ctx
  | .softplus => lowerSoftplus ctx
  | .safeLog => lowerSafeLog ctx
  | .exp => lowerExp ctx
  | .log => lowerLog ctx
  | .sin => lowerSin ctx
  | .cos => lowerCos ctx
  | .softmax axis => lowerSoftmax ctx axis
  | .hardMaskedSoftmax mask => lowerHardMaskedSoftmax ctx mask
  | _ => throw s!"IRExec: internal error: operation routed to lowerElementwise"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.add`. -/
@[simp] theorem lowerElementwise_add (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .add = lowerAdd ctx := rfl

/-- Dispatch equation for `.sub`. -/
@[simp] theorem lowerElementwise_sub (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .sub = lowerSub ctx := rfl

/-- Dispatch equation for `.mulElem`. -/
@[simp] theorem lowerElementwise_mulElem (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .mulElem = lowerMulElem ctx := rfl

/-- Dispatch equation for `.abs`. -/
@[simp] theorem lowerElementwise_abs (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .abs = lowerAbs ctx := rfl

/-- Dispatch equation for `.sqrt`. -/
@[simp] theorem lowerElementwise_sqrt (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .sqrt = lowerSqrt ctx := rfl

/-- Dispatch equation for `.inv`. -/
@[simp] theorem lowerElementwise_inv (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .inv = lowerInv ctx := rfl

/-- Dispatch equation for `.maxElem`. -/
@[simp] theorem lowerElementwise_maxElem (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .maxElem = lowerMaxElem ctx := rfl

/-- Dispatch equation for `.minElem`. -/
@[simp] theorem lowerElementwise_minElem (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .minElem = lowerMinElem ctx := rfl

/-- Dispatch equation for `.relu`. -/
@[simp] theorem lowerElementwise_relu (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .relu = lowerRelu ctx := rfl

/-- Dispatch equation for `.tanh`. -/
@[simp] theorem lowerElementwise_tanh (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .tanh = lowerTanh ctx := rfl

/-- Dispatch equation for `.sigmoid`. -/
@[simp] theorem lowerElementwise_sigmoid (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .sigmoid = lowerSigmoid ctx := rfl

/-- Dispatch equation for `.softplus`. -/
@[simp] theorem lowerElementwise_softplus (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .softplus = lowerSoftplus ctx := rfl

/-- Dispatch equation for `.safeLog`. -/
@[simp] theorem lowerElementwise_safeLog (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .safeLog = lowerSafeLog ctx := rfl

/-- Dispatch equation for `.exp`. -/
@[simp] theorem lowerElementwise_exp (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .exp = lowerExp ctx := rfl

/-- Dispatch equation for `.log`. -/
@[simp] theorem lowerElementwise_log (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .log = lowerLog ctx := rfl

/-- Dispatch equation for `.sin`. -/
@[simp] theorem lowerElementwise_sin (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .sin = lowerSin ctx := rfl

/-- Dispatch equation for `.cos`. -/
@[simp] theorem lowerElementwise_cos (ctx : NodeLoweringContext α Γ) :
    lowerElementwise ctx .cos = lowerCos ctx := rfl

/-- Dispatch equation for `.softmax axis`. -/
@[simp] theorem lowerElementwise_softmax (ctx : NodeLoweringContext α Γ) (axis : Nat) :
    lowerElementwise ctx (.softmax axis) = lowerSoftmax ctx axis := rfl

/-- Dispatch equation for `.hardMaskedSoftmax mask`. -/
@[simp] theorem lowerElementwise_hardMaskedSoftmax (ctx : NodeLoweringContext α Γ)
    (mask : NN.IR.HardMask) :
    lowerElementwise ctx (.hardMaskedSoftmax mask) = lowerHardMaskedSoftmax ctx mask := rfl

end Internal
end IRExec
end Autograd
end Runtime
