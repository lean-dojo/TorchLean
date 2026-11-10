/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape
public import TorchLean.Tensor.SomeTensor

/-!
# CPU tape values

The CPU tape is `Tape (TorchLean.SomeTensor α)`: the one tape of
`TorchLean.NN.Tape`, storing shape-erased CPU tensors. Those values get
their `GradValue` and `GradZero` instances here, together with the typed reads of node values and
upstream gradients.

Design goals:
- Works for arbitrary `Tensor α s` shapes using `TorchLean.SomeTensor` values.
- Supports DAGs (shared subexpressions): gradients are accumulated by summation.
- Preserves typed tensors at operator boundaries and reports dynamic shape mismatches as errors.

Analytic correctness theorems for supported operations live in modules such as
`TorchLean.NN.AdjointOp`, `TorchLean.NN.FDeriv` and those under `TorchLean/NN/Tape/Lowering/`, and
runtime checks cover the integration between those operations and the tape. The eager session of
`TorchLean.NN.Session` and the modules under `TorchLean/NN/Session/` hold the imperative frontend
over this engine.

References (PyTorch / background reading):
- PyTorch docs: `torch.autograd` and "Autograd mechanics":
  https://pytorch.org/docs/stable/autograd.html
  https://pytorch.org/docs/stable/notes/autograd.html
- "micrograd" (small autograd engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

namespace SomeTensor

/--
Accumulate two packed tensors by elementwise addition, with a dynamic shape check.

When two paths send gradients to the same parent, the CPU sweeps sum them with this function,
which is how the tape supports DAGs.
-/
def add {α : Type} [Storage α] [Add α]
  (a b : SomeTensor α) : Result (SomeTensor α) := by
  if h : a.shape = b.shape then
    let b' : Tensor α a.shape := b.cast h.symm
    exact .ok (SomeTensor.ofTensor (a.tensor + b'))
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end SomeTensor

/-- CPU gradients are shape-erased tensors whose ownership needs no bookkeeping, so `check` and
`drop` keep their defaults. -/
instance {α : Type} [Storage α] [Add α] : GradValue (SomeTensor α) where
  shape v := v.shape
  retag g _ h := SomeTensor.ofTensor (g.cast h)
  add := SomeTensor.add

/-- The CPU zero gradient of a value. -/
instance {α : Type} [Storage α] [Zero α] : GradZero (SomeTensor α) where
  zerosLike v := SomeTensor.ofTensor (Tensor.full v.shape (0 : α))

/-- The gradient shape of a CPU gradient is its tensor shape. -/
@[simp] theorem GradValue.shape_someTensor {α : Type} [Storage α] [Add α]
    (v : SomeTensor α) : GradValue.shape v = v.shape := rfl

/-- Retagging a CPU gradient casts its tensor to the shape of `v`. -/
@[simp] theorem GradValue.retag_someTensor {α : Type} [Storage α] [Add α]
    (g v : SomeTensor α) (h : GradValue.shape g = GradValue.shape v) :
    GradValue.retag g v h = SomeTensor.ofTensor (g.cast h) := rfl

/-- The device check of a CPU gradient succeeds. -/
@[simp] theorem GradValue.check_someTensor {α : Type} [Storage α] [Add α]
    (g v : SomeTensor α) : GradValue.check g v = .ok () := rfl

/-- CPU gradients are summed with `SomeTensor.add`. -/
@[simp] theorem GradValue.add_someTensor {α : Type} [Storage α] [Add α]
    (a b : SomeTensor α) : GradValue.add a b = SomeTensor.add a b := rfl

/-- Retiring a CPU gradient continues with `rest` unchanged. -/
@[simp] theorem GradValue.drop_someTensor {α β : Type} [Storage α] [Add α]
    (g : SomeTensor α) (rest : β) : GradValue.drop g rest = rest := rfl

/-- The CPU zero gradient of `v` is the zero tensor of its shape. -/
@[simp] theorem GradZero.zerosLike_someTensor {α : Type} [Storage α] [Zero α]
    (v : SomeTensor α) :
    GradZero.zerosLike v = SomeTensor.ofTensor (Tensor.full v.shape (0 : α)) := rfl

/-!
## Typed reads

`Tape.requireValue` and `Tape.requireGrad` read a node value and an upstream gradient at the shape
an operation expects, and fail on a mismatch.
-/

namespace Tape

/--
Read a typed tensor value from a tape node id.

It is the main dynamic check of the eager runtime, and fails if the id is invalid or the stored
runtime shape differs from the expected dependent shape `s`.
-/
def requireValue {α : Type} [Storage α] {s : Shape}
  (t : Tape (SomeTensor α)) (id : Nat) : Result (Tensor α s) := by
  match t.getValue? id with
  | none => exact .error "autograd: invalid node id"
  | some any =>
    if h : any.shape = s then
      exact .ok (any.cast h)
    else
      exact .error "autograd: shape mismatch"

/--
Read a typed upstream gradient tensor from a packed tensor.

The backward analogue of `Tape.requireValue`: it checks that the upstream gradient has the
expected shape `τ` and then performs the dependent cast.
-/
def requireGrad {α : Type} [Storage α] {τ : Shape}
    (dLdyAny : SomeTensor α) : Result (Tensor α τ) := by
  if h : dLdyAny.shape = τ then
    exact .ok (dLdyAny.cast h)
  else
    exact .error "autograd: upstream gradient shape mismatch"

