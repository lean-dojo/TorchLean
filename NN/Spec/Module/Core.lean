/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Mathematical modules

`Spec.Module α σ τ` packages a pure tensor map from shape `σ` to shape `τ`. The input and output
shapes occur in the type, so ill-shaped compositions are rejected by Lean.

`Spec.Module.Chain` composes these maps. The `kind` and `pythonExpr` fields are descriptive
metadata; `forward` alone gives the module its mathematical meaning.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

/-- A pure, shape-indexed tensor map with descriptive code-generation metadata. -/
structure Module (α : Type) [TorchLean.Storage α]
    (inShape outShape : Shape) where
  /-- The mathematical meaning of the module. -/
  forward : Tensor α inShape → Tensor α outShape
  /-- A stable operation name used in reports and exported graphs. -/
  kind : String
  /-- A Python expression used by the source exporter. This field is not part of the semantics. -/
  pythonExpr : String

namespace Module

/--
Dependent composition of modules whose adjacent shapes agree definitionally.
-/
inductive Chain (α : Type) [TorchLean.Storage α] :
    Shape → Shape → Type where
| single {s t} (m : Module α s t) : Chain α s t
| comp {s t u} (a : Chain α s t) (b : Chain α t u) : Chain α s u

namespace Chain

/-- Evaluate a chain from left to right. -/
def forward {α : Type} [TorchLean.Storage α]
    {s t : Shape} (c : Chain α s t) : Tensor α s → Tensor α t :=
  match c with
  | .single m => m.forward
  | .comp a b => fun x =>
      let y := forward a x
      forward b y

/-- Append one module to the output of a chain. -/
def append {α : Type} [TorchLean.Storage α] {s t u : Shape}
    (a : Chain α s t) (b : Module α t u) : Chain α s u :=
  .comp a (.single b)

/-- Return operation names in evaluation order. -/
def kinds {α : Type} [TorchLean.Storage α]
    {s t : Shape} : Chain α s t → Array String
  | .single m => #[m.kind]
  | .comp a b => kinds a ++ kinds b

/-- Return operation names and Python expressions in evaluation order. -/
def layerInfo {α : Type} [TorchLean.Storage α]
    {s t : Shape} : Chain α s t → Array (String × String)
  | .single m => #[(m.kind, m.pythonExpr)]
  | .comp a b => layerInfo a ++ layerInfo b

end Chain

/-- Lift a module to operate independently over one new leading dimension. -/
def liftLeading
  {α : Type} [TorchLean.Storage α] [Context α]
    {n : Nat} {elemIn elemOut : Shape} (m : Module α elemIn elemOut) :
    Module α (.dim n elemIn) (.dim n elemOut) where
  forward tensor :=
    Tensor.dim fun i => m.forward (Tensor.unstack tensor i)
  kind := m.kind
  pythonExpr := m.pythonExpr

/-- Select one coordinate along any valid axis. -/
def select {α : Type} [TorchLean.Storage α]
    {shape : Shape} (axis : Nat)
    [Shape.AxisInBounds axis shape] (i : Fin (Shape.axisSize shape axis)) :
    Module α shape (shape.eraseAxis axis) where
  forward x := Tensor.selectSpec axis x i
  kind := "Select"
  pythonExpr := s!"Select(axis={axis}, index={i.val})"

end Module
end Spec
