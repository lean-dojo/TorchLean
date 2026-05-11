/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.Runtime.Context
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Core.Utils

/-!
# Runtime variables

This module defines a small `Variable` record that bundles:
- a name/id,
- a value tensor,
- an optional gradient tensor, and
- a `requires_grad` flag.

This matches the mental model of a PyTorch `torch.Tensor` with `requires_grad`, together with a
place to store an accumulated `.grad`. It is used by the runtime layers
that want an explicit "parameter object", while `NN/Runtime/Context.lean` provides a more generic
name-based registry.
-/

@[expose] public section


namespace Runtime

open Spec
open Tensor

/--
A runtime variable with a fixed shape.

`gradient` is optional so we can represent "no gradient accumulated yet" (or variables that are
used only for inference).
-/
structure Variable (α : Type) (s : Shape) where
  /-- Unique numeric id (allocated by the runtime context). -/
  id : Nat
  /-- Human-readable name (often a parameter name like `"layer1.weight"`). -/
  name : String
  /-- The current tensor value. -/
  value : Tensor α s
  /--
  Optional accumulated gradient.

  This is `none` when no gradient has been accumulated yet (or when `requires_grad = false`).
  -/
  gradient : Option (Tensor α s)
  /-- Whether gradient-based optimizers may update this variable. -/
  requires_grad : Bool

namespace Variable

/--
Create a new variable from an id/name/value.

By default we mark variables as trainable (`requires_grad := true`), matching common ML defaults.
-/
def createVariable {α : Type} {s : Shape} (id : Nat) (name : String) (value : Tensor α s)
    (requires_grad : Bool := true) : Variable α s :=
  { id := id, name := name, value := value, gradient := none, requires_grad := requires_grad }

/-- Read the tensor value stored in a variable. -/
def getValue {α : Type} {s : Shape} (v : Variable α s) : Tensor α s :=
  v.value

/-- Replace the tensor value stored in a variable. -/
def setValue {α : Type} {s : Shape} (v : Variable α s) (new_value : Tensor α s) : Variable α s :=
  { v with value := new_value }

/-- Read the optional gradient stored in a variable. -/
def getGradient {α : Type} {s : Shape} (v : Variable α s) : Option (Tensor α s) :=
  v.gradient

/-- Set (overwrite) the gradient stored in a variable. -/
def setGradient {α : Type} {s : Shape} (v : Variable α s) (grad : Tensor α s) : Variable α s :=
  { v with gradient := some grad }

/-- Clear the gradient stored in a variable. -/
def clearGradient {α : Type} {s : Shape} (v : Variable α s) : Variable α s :=
  { v with gradient := none }

/-- Check whether gradient-based optimizers may update this variable. -/
def checkRequiresGrad {α : Type} {s : Shape} (v : Variable α s) : Bool :=
  v.requires_grad

/--
Accumulate a gradient contribution into the variable.

If no gradient was set yet, this behaves like assignment. Otherwise it adds into the existing
gradient (analogue of PyTorch `param.grad += ...`).
-/
def accumulateGradient {α : Type} [Context α] {s : Shape} [Add α]
    (v : Variable α s) (grad : Tensor α s) : Variable α s :=
  match v.gradient with
  | none => setGradient v grad
  | some existing_grad => setGradient v (Tensor.addSpec existing_grad grad)

/-- Render a variable for debugging (includes value and optional gradient). -/
def variableToString {α : Type} [ToString α] {s : Shape} (v : Variable α s) : String :=
  let grad_str := match v.gradient with
    | none => "none"
    | some g => s!"{pretty g}"
  (s!"Variable({v.name}, id={v.id}, value={pretty v.value}, " ++
    s!"grad={grad_str}, requires_grad={v.requires_grad})")

/-- A simple equality check based on `id` and `name`. -/
def variableEq {α : Type} {s : Shape} (v₁ v₂ : Variable α s) : Bool :=
  v₁.id == v₂.id && v₁.name == v₂.name

/-- Check that the variable's shape equals the given shape. -/
def hasShape {α : Type} {s : Shape} (_v : Variable α s) (expected_shape : Shape) : Bool :=
  Shape.areEqual s expected_shape

/--
Compute the L2 norm of the variable value (Euclidean norm).

This is a small convenience helper for monitoring/debugging.
-/
def variableNorm {α : Type} [Context α] {s : Shape} (v : Variable α s) : α :=
  -- Calculate the L2 norm: sqrt(sum of squares)
  let squared := Tensor.mulSpec v.value v.value
  let sum_squared := sumSpec squared
  MathFunctions.sqrt sum_squared

/-- Clone a variable under a new id, clearing the gradient. -/
def cloneVariable {α : Type} {s : Shape} (v : Variable α s) (new_id : Nat) : Variable α s :=
  { v with id := new_id, gradient := none }

/-- Mark a variable as non-trainable and clear its gradient (inference-only). -/
def detachVariable {α : Type} {s : Shape} (v : Variable α s) : Variable α s :=
  { v with requires_grad := false, gradient := none }

/-- Register this variable's value in a `RuntimeContext` under its `name`. -/
def registerInContext {α : Type} {s : Shape} (ctx : RuntimeContext α) (v : Variable α s) :
  RuntimeContext α :=
  -- Store tensor with its shape information directly
  Runtime.registerVariable ctx v.name v.value

/-- Retrieve this variable's value from a `RuntimeContext` (with shape checking). -/
def getFromContext {α : Type} {s : Shape} [DecidableEq Shape]
    (ctx : RuntimeContext α) (v : Variable α s) : Option (Tensor α s) :=
  -- Get tensor from context with shape checking
  Runtime.getVariable ctx v.name

/-- If `v.gradient` is present, register it in the `RuntimeContext` under `v.name`. -/
def registerGradientInContext {α : Type} {s : Shape}
    (ctx : RuntimeContext α) (v : Variable α s) : RuntimeContext α :=
  match v.gradient with
  | none => ctx
  | some grad =>
    -- Store gradient tensor with its shape information directly
    Runtime.registerGradient ctx v.name grad

/--
Create a sentinel scalar variable used to thread error messages through some pipelines.

This is mainly for demos / ergonomic error handling, not for proofs.
-/
def error {α : Type} [Zero α] (ctx : RuntimeContext α) (msg : String) : Variable α .scalar :=
  { id := ctx.next_id,
    name := msg,
    value := Tensor.scalar 0,
    gradient := none,
    requires_grad := false }

/--
Cast a variable to an equal shape.

This is the shape-indexed analogue of `Eq.mp`: given a proof `s₁ = s₂`, we can transport the
value (and optional gradient) across the equality.
-/
def castShape {α : Type} {s₁ s₂ : Shape} (v : Variable α s₁) (h : s₁ = s₂) : Variable α s₂ :=
  { v with value := Tensor.castShape v.value h, gradient := v.gradient.map (Tensor.castShape · h)
    }

end Variable
end Runtime
