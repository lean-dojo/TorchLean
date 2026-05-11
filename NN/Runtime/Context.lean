/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Utils

/-!
# Runtime context

This module defines the runtime environment for executing TorchLean computations.

The main challenge is that tensors are dependently typed by their `Shape`, but at runtime we want
to store a heterogeneous map from names to values. We solve this by storing an existential wrapper
`AnyTensor` that carries the `Shape` alongside the tensor. Lookup functions then *check* the shape
and use the resulting equality proof to cast the stored tensor to the requested shape.

## Reading map

- `AnyTensor` is the shape-erased wrapper used by the registries.
- `RuntimeContext` is the mutable-looking record that stores values, gradients, and a fresh-id
  counter.
- `register_variable` / `get_variable` are the core value registry operations.
- `register_gradient` / `get_gradient` do the same on the gradient side.
-/

@[expose] public section


namespace Runtime

open Spec
open Tensor

/--
Existential wrapper for tensors of arbitrary shape.

This is the core trick that lets a runtime registry store tensors of different shapes in one
container.
-/
structure AnyTensor (α : Type) where
  /-- The shape carried alongside the tensor value. -/
  s : Shape
  /-- The tensor value, indexed by `s`. -/
  t : Tensor α s

/--
Runtime context for tracking named variables and their gradients.

We keep two registries:
- `var_registry` for values, and
- `gradients` for accumulated gradients (typically produced by backprop).
-/
structure RuntimeContext (α : Type) where
  /--
  Registry of named variable values.

  Implementation note: this is a simple list, so lookup uses the first match (most-recent wins).
  -/
  var_registry : List (String × AnyTensor α)
  /--
  Registry of named gradients.

  As with `var_registry`, this is a simple list; callers are responsible for deciding whether to
  prepend new gradients or update/accumulate existing ones.
  -/
  gradients : List (String × AnyTensor α)
  /--
  Fresh id counter for allocating new variables/parameters.

  This is incremented by `register_variable` and used by higher-level runtime layers.
  -/
  next_id : Nat

/-- An empty context with no variables and no gradients. -/
def emptyContext {α : Type} : RuntimeContext α :=
  { var_registry := [], gradients := [], next_id := 0 }

/--
Register a new variable in the context.

The value is wrapped as an `AnyTensor` so we can store it in a heterogeneous registry.
-/
def registerVariable {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (value : Tensor
  α s) : RuntimeContext α :=
  -- Store tensor with its shape information
  let any_tensor := { s := s, t := value }
  { ctx with
    var_registry := (name, any_tensor) :: ctx.var_registry,
    next_id := ctx.next_id + 1 }

/--
Lookup a variable by name and requested shape.

If the name exists and the stored `Shape` matches `s`, we cast the stored tensor from
`Tensor α any_tensor.s` to `Tensor α s` using the equality proof `any_tensor.s = s`.
-/
def getVariable {α : Type} {s : Shape} [DecidableEq Shape]
  (ctx : RuntimeContext α) (name : String) : Option (Tensor α s) :=
  match ctx.var_registry.find? (fun (n, _) => n == name) with
  | none => none
  | some (_, any_tensor) =>
    if h : any_tensor.s = s then
      -- cast Tensor α any_tensor.s to Tensor α s using the equality proof
      some (Eq.mp (congrArg (Tensor α) h) any_tensor.t)
    else
      none

/--
Update the value of an existing variable (or do nothing if the name is absent).

This performs a simple list-map update of the registry.
-/
def setVariable {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (value : Tensor α s)
  : RuntimeContext α :=
  -- Store tensor with its shape information
  let any_tensor := { s := s, t := value }
  let updated_vars := ctx.var_registry.map (fun (n, v) => if n == name then (n, any_tensor) else (n,
    v))
  { ctx with var_registry := updated_vars }

/--
Register (prepend) a gradient entry in the context.

If you want to *accumulate* gradients under the same name, do that at a higher layer before
calling this helper.
-/
def registerGradient {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (grad : Tensor
  α s) : RuntimeContext α :=
  -- Store gradient tensor with its shape information
  let any_tensor := { s := s, t := grad }
  { ctx with gradients := (name, any_tensor) :: ctx.gradients }

/-- Lookup a gradient by name and requested shape (with the same casting trick as `get_variable`).
  -/
def getGradient {α : Type} {s : Shape} [DecidableEq Shape] (ctx : RuntimeContext α) (name : String)
  : Option (Tensor α s) :=
  -- Get gradient tensor from context and check shape compatibility
  match ctx.gradients.find? (fun (n, _) => n == name) |>.map (fun (_, any_tensor) => any_tensor)
    with
  | none => none
  | some any_tensor =>
    -- Check if the stored shape matches the requested shape
    if h : any_tensor.s = s then
      some (Eq.mp (congrArg (Tensor α) h) any_tensor.t)
    else
      none  -- Shape mismatch

/-- Update a gradient entry in the context (simple list-map update). -/
def setGradient {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (grad : Tensor α s)
  : RuntimeContext α :=
  -- Store gradient tensor with its shape information
  let any_tensor := { s := s, t := grad }
  let updated_grads := ctx.gradients.map (fun (n, g) => if n == name then (n, any_tensor) else (n,
    g))
  { ctx with gradients := updated_grads }

/-- Remove all gradient entries (analogue of PyTorch `optimizer.zero_grad()`). -/
def clearGradients {α : Type} (ctx : RuntimeContext α) : RuntimeContext α :=
  { ctx with gradients := [] }

/-- Return `true` iff the context contains a variable named `name`. -/
def hasVariable {α : Type} (ctx : RuntimeContext α) (name : String) : Bool :=
  ctx.var_registry.any (fun (n, _) => n == name)

/-- List all variable names stored in `ctx`. -/
def variableNames {α : Type} (ctx : RuntimeContext α) : List String :=
  ctx.var_registry.map (fun (n, _) => n)

/-- List all gradient names stored in `ctx`. -/
def gradientNames {α : Type} (ctx : RuntimeContext α) : List String :=
  ctx.gradients.map (fun (n, _) => n)

/-- Number of registered variables. -/
def contextSize {α : Type} (ctx : RuntimeContext α) : Nat :=
  ctx.var_registry.length

/-- Number of registered gradient entries. -/
def gradientCount {α : Type} (ctx : RuntimeContext α) : Nat :=
  ctx.gradients.length

/--
Check a simple invariant: every gradient entry refers to an existing variable name.

This does *not* check shapes; it only checks name presence.
-/
def isValidContext {α : Type} (ctx : RuntimeContext α) : Bool :=
  -- Check that all gradients correspond to existing variables
  ctx.gradients.all (fun (name, _) => hasVariable ctx name)

/-- Render the context contents as a string (for debugging). -/
def contextToString {α : Type} [ToString α] (ctx : RuntimeContext α) : String :=
  let var_str := String.intercalate ", " (ctx.var_registry.map (fun (n, any_tensor) =>
    s!"{n}: {pretty any_tensor.t}"))
  let grad_str := String.intercalate ", " (ctx.gradients.map (fun (n, any_tensor) =>
    s!"{n}: {pretty any_tensor.t}"))
  s!"Context(vars: [{var_str}], grads: [{grad_str}])"
end Runtime
