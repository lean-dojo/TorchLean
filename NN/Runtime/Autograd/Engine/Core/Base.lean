/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Tape
public import NN.Spec.Core.Tensor.SomeTensor

/-!
# CPU tape values

The CPU tape is `Tape (Spec.SomeTensor α)`: the one tape of `NN.Runtime.Autograd.Engine.Tape`,
storing shape-erased CPU tensors. This module gives those values their `GradValue` instance and
the typed constructors that every CPU operation is built from.

Design goals:
- Works for arbitrary `Tensor α s` shapes using `Spec.SomeTensor` values.
- Supports DAGs (shared subexpressions): gradients are accumulated by summation.
- Preserves typed tensors at operator boundaries and reports dynamic shape mismatches as errors.

This module defines and executes the tape; it does not by itself prove every registered VJP.
Analytic correctness theorems for supported operations live under `NN.Proofs.Autograd`, while
runtime checks cover the integration between those operations and the tape. For the imperative
frontend over this engine, see `NN.Runtime.Autograd.Torch.Core`.

References (PyTorch / background reading):
- PyTorch docs: `torch.autograd` and "Autograd mechanics":
  https://pytorch.org/docs/stable/autograd.html
  https://pytorch.org/docs/stable/notes/autograd.html
- "micrograd" (small autograd engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace SomeTensor

/--
Accumulate two packed tensors by elementwise addition, with a dynamic shape check.

This is the heart of DAG support: if two different paths contribute gradients to the same parent,
we sum the contributions.
-/
def add {α : Type} [TorchLean.Storage α] [Add α]
  (a b : Spec.SomeTensor α) : Result (Spec.SomeTensor α) := by
  if h : a.shape = b.shape then
    let b' : Tensor α a.shape := b.cast h.symm
    exact .ok (Spec.SomeTensor.ofTensor (addSpec a.tensor b'))
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end SomeTensor

/-- CPU gradients are shape-erased tensors; ownership needs no bookkeeping. -/
instance {α : Type} [TorchLean.Storage α] [Add α] : GradValue (Spec.SomeTensor α) where
  shape v := v.shape
  retag g _ h := Spec.SomeTensor.ofTensor (g.cast h)
  add := SomeTensor.add

/-- The CPU zero gradient of a value. -/
instance {α : Type} [TorchLean.Storage α] [Zero α] : GradZero (Spec.SomeTensor α) where
  zerosLike v := Spec.SomeTensor.ofTensor (Tensor.full v.shape (0 : α))

@[simp] theorem GradValue.shape_someTensor {α : Type} [TorchLean.Storage α] [Add α]
    (v : Spec.SomeTensor α) : GradValue.shape v = v.shape := rfl

@[simp] theorem GradValue.retag_someTensor {α : Type} [TorchLean.Storage α] [Add α]
    (g v : Spec.SomeTensor α) (h : GradValue.shape g = GradValue.shape v) :
    GradValue.retag g v h = Spec.SomeTensor.ofTensor (g.cast h) := rfl

@[simp] theorem GradValue.check_someTensor {α : Type} [TorchLean.Storage α] [Add α]
    (g v : Spec.SomeTensor α) : GradValue.check g v = .ok () := rfl

@[simp] theorem GradValue.add_someTensor {α : Type} [TorchLean.Storage α] [Add α]
    (a b : Spec.SomeTensor α) : GradValue.add a b = SomeTensor.add a b := rfl

@[simp] theorem GradValue.drop_someTensor {α β : Type} [TorchLean.Storage α] [Add α]
    (g : Spec.SomeTensor α) (rest : β) : GradValue.drop g rest = rest := rfl

@[simp] theorem GradZero.zerosLike_someTensor {α : Type} [TorchLean.Storage α] [Zero α]
    (v : Spec.SomeTensor α) :
    GradZero.zerosLike v = Spec.SomeTensor.ofTensor (Tensor.full v.shape (0 : α)) := rfl

/-!
## Tape Construction

The `Tape` namespace provides *pure* constructors for building a recorded computation graph.
Each op appends exactly one node and returns the updated tape plus the new node id.
-/

namespace Tape

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).
-/
def leaf {α : Type} [TorchLean.Storage α] {s : Shape}
  (t : Tape (Spec.SomeTensor α)) (value : Tensor α s) (name : Option String := none) (requiresGrad
      : Bool := true) :
  Tape (Spec.SomeTensor α) × Nat :=
  t.addNode {
    name := name,
    value := Spec.SomeTensor.ofTensor value,
    requiresGrad := requiresGrad,
    parents := #[],
    backward := fun _ => .ok #[]
  }

/--
Read a typed tensor value from a tape node id.

This is the main "dynamic check" boundary in the eager runtime:
- fails if the id is invalid, or
- fails if the stored runtime shape doesn't match the expected dependent shape `s`.
-/
def requireValue {α : Type} [TorchLean.Storage α] {s : Shape}
  (t : Tape (Spec.SomeTensor α)) (id : Nat) : Result (Tensor α s) := by
  match t.getValue? id with
  | none => exact .error "autograd: invalid node id"
  | some any =>
    if h : any.shape = s then
      exact .ok (any.cast h)
    else
      exact .error "autograd: shape mismatch"

/--
Read a typed upstream gradient tensor from a packed tensor.

This is the backward analogue of `Tape.requireValue`: it checks that the upstream gradient has the
expected shape `τ` and then performs the dependent cast.
-/
def requireGrad {α : Type} [TorchLean.Storage α] {τ : Shape}
    (dLdyAny : Spec.SomeTensor α) : Result (Tensor α τ) := by
  if h : dLdyAny.shape = τ then
    exact .ok (dLdyAny.cast h)
  else
    exact .error "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

You provide:
- `forward : Tensor α σ → Tensor α τ`
- `backward : Tensor α σ → Tensor α τ → Tensor α σ` (a VJP rule; note it may depend on the input)

The returned node inherits the parent's gradient flag and stores the forward value and a backward
closure that checks the upstream gradient's shape and returns the parent contribution.
-/
@[inline] def unary {α : Type} [TorchLean.Storage α] {σ τ : Shape}
  (t : Tape (Spec.SomeTensor α)) (opName : String) (xId : Nat)
  (forward : Tensor α σ → Tensor α τ)
  (backward : Tensor α σ → Tensor α τ → Tensor α σ) :
  Result (Tape (Spec.SomeTensor α) × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=σ) xId
  let y := forward x
  let node : Node (Spec.SomeTensor α) :=
    { name := some opName
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := τ) dLdyAny
        let dLdx : Tensor α σ := backward x dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dLdx)]
    }
  pure (t.addNode node)

/--
Record a binary operation with typed inputs and cotangents.

Input lookup, parent metadata, and backward contributions all follow the order `a`, then `b`.
The backward function runs after the upstream shape check and receives both saved inputs.
-/
@[inline] def binary {α : Type} [TorchLean.Storage α] {σ₁ σ₂ τ : Shape}
    (t : Tape (Spec.SomeTensor α)) (opName : String) (aId bId : Nat)
    (forward : Tensor α σ₁ → Tensor α σ₂ → Tensor α τ)
    (backward : Tensor α σ₁ → Tensor α σ₂ → Tensor α τ →
      Tensor α σ₁ × Tensor α σ₂) : Result (Tape (Spec.SomeTensor α) × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := σ₁) aId
  let b ← requireValue (α := α) (t := t) (s := σ₂) bId
  let y := forward a b
  let node : Node (Spec.SomeTensor α) :=
    { name := some opName
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) || (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := τ) dLdyAny
        let (da, db) := backward a b dLdy
        pure #[(aId, Spec.SomeTensor.ofTensor da), (bId, Spec.SomeTensor.ofTensor db)] }
  pure (t.addNode node)
