/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core
public import NN.Spec.Core.Tensor.Numerics
public import NN.GraphSpec.DAG.Syntax
public import NN.Runtime.Autograd.Model.Norm

/-!
# DAG Normalization Primitives

RMS normalization and regularized L2 normalization with a nonempty final axis.
The L2 epsilon is an unchecked scalar input; callers choose a positive value.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace PrimOp

namespace Internal

/-- RMS normalization of a single vector.

The spec is written for a batch, so a lone vector is normalized by wrapping it in a one-element
batch and reading the result back out. `hWidth` is what rules out dividing by an empty mean. -/
def rmsNormVectorSemantics {α : Type} [TorchLean.Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : TorchLean.Tensor α [width]) :
    TorchLean.Tensor α [width] :=
  Spec.get
    (Spec.rmsNorm (α := α) (.dim fun _ => input) gamma
      (Nat.zero_lt_succ 0) hWidth)
    ⟨0, Nat.zero_lt_succ 0⟩

end Internal

/-- Apply RMS normalization with a shared final-axis scale over an arbitrary leading shape. -/
def rmsNormSemantics {α : Type} [TorchLean.Storage α] [Context α] (leading : Shape) {width : Nat}
    (hWidth : 0 < width) (gamma : TorchLean.Tensor α [width]) :
      TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) :=
  match leading with
  | .scalar => fun input => Internal.rmsNormVectorSemantics hWidth input gamma
  | .dim _ rest => fun input =>
      Tensor.dim fun i => rmsNormSemantics rest hWidth gamma (input.unstack i)

/-- Apply RMS normalization with a pointwise scale over an arbitrary leading shape. -/
def rmsNormElementwiseSemantics {α : Type} [TorchLean.Storage α] [Context α] (leading : Shape)
    {width : Nat} (hWidth : 0 < width) :
    TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) :=
  match leading with
  | .scalar => fun input gamma => Internal.rmsNormVectorSemantics hWidth input gamma
  | .dim _ rest => fun input gamma =>
      Tensor.dim fun i =>
        rmsNormElementwiseSemantics rest hWidth (input.unstack i) (gamma.unstack i)

/-- Batch axes distribute over RMS normalization: each slice is normalized on its own. -/
@[simp] theorem rmsNormElementwiseSemantics_dim {α : Type} [TorchLean.Storage α] [Context α]
    {count width : Nat} {rest : Shape} (hWidth : 0 < width)
    (inputs gammas : Fin count → TorchLean.Tensor α (rest.appendDim width)) :
    rmsNormElementwiseSemantics (.dim count rest) hWidth (.dim inputs) (.dim gammas) =
      .dim (fun index => rmsNormElementwiseSemantics rest hWidth
        (inputs index) (gammas index)) := by
  simp [rmsNormElementwiseSemantics]

/-- Root-mean-square normalization along the final axis of an arbitrary tensor. -/
def rmsNorm (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, [width]] (leading.appendDim width) :=
  { name := s!"rmsNorm({width})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input (.cons gamma .nil) => rmsNormSemantics leading hWidth gamma input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (Runtime.Autograd.Model.Norm.rmsNorm (m := m) (α := α)
          (leading := leading) (width := width) hWidth input gamma :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis RMS normalization with a shared scale. -/
@[simp] theorem rmsNorm_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [TorchLean.Storage α] [Context α]
    (input : TorchLean.Tensor α (leading.appendDim width))
    (gamma : TorchLean.Tensor α [width]) :
    (rmsNorm leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormSemantics leading hWidth gamma input := by
  rfl

/-- RMS normalization with a separate final-axis scale at every leading coordinate. -/
def rmsNormElementwise (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, leading.appendDim width] (leading.appendDim width) :=
  { name := s!"rmsNormElementwise({width})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          rmsNormElementwiseSemantics leading hWidth input gamma
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (do
          let ones ← Runtime.Autograd.Model.const (m := m) (α := α)
            (Tensor.full ([width] : Shape) 1)
          let normalized ← Runtime.Autograd.Model.Norm.rmsNorm (m := m) (α := α)
            (leading := leading) (width := width) hWidth input ones
          Runtime.Autograd.Model.mul (m := m) (α := α)
            (s := leading.appendDim width) normalized gamma :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis RMS normalization with pointwise scale. -/
@[simp] theorem rmsNormElementwise_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [TorchLean.Storage α] [Context α]
    (input gamma : TorchLean.Tensor α (leading.appendDim width)) :
    (rmsNormElementwise leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormElementwiseSemantics leading hWidth input gamma := by
  rfl

/-- Apply regularized L2 normalization over the final axis of an arbitrary tensor. -/
def l2NormalizeSemantics {α : Type} [TorchLean.Storage α] [Context α] (leading : Shape)
    {width : Nat} (epsilon : α) : TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) :=
  match leading with
  | .scalar => fun input => Spec.normalizeL2RegularizedSpec input epsilon
  | .dim _ rest => fun input =>
      Tensor.dim fun i => l2NormalizeSemantics rest epsilon (input.unstack i)

/-- With no batch axes left, the graph node is the regularized L2 normalization spec. -/
@[simp] theorem l2NormalizeSemantics_scalar {α : Type} [TorchLean.Storage α] [Context α]
    {width : Nat} (epsilon : α) (values : TorchLean.Tensor α [width]) :
    l2NormalizeSemantics .scalar epsilon values =
      Spec.normalizeL2RegularizedSpec values epsilon := by
  rfl

/-- Batch axes distribute over L2 normalization. -/
@[simp] theorem l2NormalizeSemantics_dim {α : Type} [TorchLean.Storage α] [Context α]
    {count width : Nat} {rest : Shape} (epsilon : α)
    (values : Fin count → TorchLean.Tensor α (rest.appendDim width)) :
    l2NormalizeSemantics (.dim count rest) epsilon (.dim values) =
      .dim (fun index => l2NormalizeSemantics rest epsilon (values index)) := by
  simp [l2NormalizeSemantics]

/-- Normalize the final axis by `sqrt(sum (x * x) + epsilon)`.

The stabilizer is a scalar graph input rather than a declaration parameter. This keeps the exact
numerical convention visible in captured graphs and permits architectures to select their own
positive epsilon without adding another primitive.
-/
def l2Normalize (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, .scalar] (leading.appendDim width) :=
  { name := s!"l2Normalize({width})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input (.cons epsilon .nil) =>
          l2NormalizeSemantics leading epsilon.item input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input epsilon =>
        (Runtime.Autograd.Model.Norm.l2Normalize (m := m) (α := α)
          (leading := leading) (width := width) hWidth input epsilon :
          m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis regularized L2 normalization. -/
@[simp] theorem l2Normalize_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [TorchLean.Storage α] [Context α]
    (input : TorchLean.Tensor α (leading.appendDim width)) (epsilon : α) :
    (l2Normalize leading width hWidth).specFwd
        (.cons input (.cons (.scalar epsilon) .nil)) =
      l2NormalizeSemantics leading epsilon input := by
  change l2NormalizeSemantics leading (Tensor.scalar epsilon).item input =
    l2NormalizeSemantics leading epsilon input
  rw [Tensor.item_scalar]


end PrimOp

end DAG
end GraphSpec
end NN
