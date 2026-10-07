/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core
public import NN.Spec.Core.Tensor.Numerics
public import NN.Spec.Core.Sequence
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

/-- Adapt the shared leading-axis traversal to the final-axis shape convention. -/
def mapFinalAxis {α : Type} [TorchLean.Storage α]
    (leading : Shape) {width : Nat}
    (f : Tensor α [width] → Tensor α [width])
    (input : Tensor α (leading.appendDim width)) :
    Tensor α (leading.appendDim width) :=
  Tensor.castShape
    (Tensor.mapLeading leading f
      (Tensor.castShape input (Shape.appendDim_eq_concat leading width)))
    (Shape.appendDim_eq_concat leading width).symm

private theorem cast_shape_dim {α : Type} [TorchLean.Storage α]
    {count : Nat} {source target : Shape} (h : source = target)
    (values : Fin count → Tensor α source) :
    Tensor.castShape (.dim values) (congrArg (Shape.dim count) h) =
      .dim (fun index => Tensor.castShape (values index) h) := by
  cases h
  rfl

private theorem mapFinalAxis_dim {α : Type} [TorchLean.Storage α]
    {count width : Nat} {rest : Shape} (f : Tensor α [width] → Tensor α [width])
    (values : Fin count → Tensor α (rest.appendDim width)) :
    mapFinalAxis (.dim count rest) f (.dim values) =
      .dim (fun index => mapFinalAxis rest f (values index)) := by
  simp only [mapFinalAxis, Shape.appendDim, Shape.concat,
    cast_shape_dim (Shape.appendDim_eq_concat rest width),
    cast_shape_dim (Shape.appendDim_eq_concat rest width).symm,
    Tensor.mapLeading, Tensor.unstack_dim]

/-- RMS normalization of a single vector.

The spec is written for a batch, so a lone vector is normalized by wrapping it in a one-element
batch and reading the result back out. `hWidth` is what rules out dividing by an empty mean. -/
def rmsNormVectorSpec {α : Type} [TorchLean.Storage α] [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : TorchLean.Tensor α [width]) :
    TorchLean.Tensor α [width] :=
  Spec.get
    (Spec.rmsNorm (α := α) (.dim fun _ => input) gamma
      (Nat.zero_lt_succ 0) hWidth)
    ⟨0, Nat.zero_lt_succ 0⟩

end Internal

/-- Apply RMS normalization with a shared final-axis scale over an arbitrary leading shape. -/
def rmsNormSpec {α : Type} [TorchLean.Storage α] [Context α] (leading : Shape) {width : Nat}
    (hWidth : 0 < width) (gamma : TorchLean.Tensor α [width]) :
      TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) :=
  Internal.mapFinalAxis leading (fun input => Internal.rmsNormVectorSpec hWidth input gamma)

/-- With no leading axes, RMS normalization uses the single-vector spec. -/
@[simp] theorem rmsNormSpec_scalar {α : Type} [TorchLean.Storage α] [Context α]
    {width : Nat} (hWidth : 0 < width) (gamma input : Tensor α [width]) :
    rmsNormSpec .scalar hWidth gamma input = Internal.rmsNormVectorSpec hWidth input gamma := by
  rfl

/-- Leading axes distribute over RMS normalization without changing the row computation. -/
@[simp] theorem rmsNormSpec_dim {α : Type} [TorchLean.Storage α] [Context α]
    {count width : Nat} {rest : Shape} (hWidth : 0 < width) (gamma : Tensor α [width])
    (values : Fin count → Tensor α (rest.appendDim width)) :
    rmsNormSpec (.dim count rest) hWidth gamma (.dim values) =
      .dim (fun index => rmsNormSpec rest hWidth gamma (values index)) := by
  exact Internal.mapFinalAxis_dim _ values

/-- Root-mean-square normalization along the final axis of an arbitrary tensor. -/
def rmsNorm (leading : Shape) (width : Nat) (hWidth : 0 < width) :
    PrimOp [leading.appendDim width, [width]] (leading.appendDim width) :=
  { name := s!"rmsNorm({width})"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .cons input (.cons gamma .nil) => rmsNormSpec leading hWidth gamma input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (Runtime.Autograd.Model.Norm.rmsNorm (m := m) (α := α)
          (leading := leading) (width := width) hWidth input gamma :
          m (Runtime.Autograd.Model.Ref (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis RMS normalization with a shared scale. -/
@[simp] theorem rmsNorm_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [TorchLean.Storage α] [Context α]
    (input : TorchLean.Tensor α (leading.appendDim width))
    (gamma : TorchLean.Tensor α [width]) :
    (rmsNorm leading width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormSpec leading hWidth gamma input := by
  rfl

/-- Apply regularized L2 normalization over the final axis of an arbitrary tensor. -/
def l2NormalizeSpec {α : Type} [TorchLean.Storage α] [Context α] (leading : Shape)
    {width : Nat} (epsilon : α) : TorchLean.Tensor α (leading.appendDim width) →
      TorchLean.Tensor α (leading.appendDim width) :=
  Internal.mapFinalAxis leading (fun input => Spec.normalizeL2RegularizedSpec input epsilon)

/-- With no batch axes left, the graph node is the regularized L2 normalization spec. -/
@[simp] theorem l2NormalizeSpec_scalar {α : Type} [TorchLean.Storage α] [Context α]
    {width : Nat} (epsilon : α) (values : TorchLean.Tensor α [width]) :
    l2NormalizeSpec .scalar epsilon values =
      Spec.normalizeL2RegularizedSpec values epsilon := by
  rfl

/-- Batch axes distribute over L2 normalization. -/
@[simp] theorem l2NormalizeSpec_dim {α : Type} [TorchLean.Storage α] [Context α]
    {count width : Nat} {rest : Shape} (epsilon : α)
    (values : Fin count → TorchLean.Tensor α (rest.appendDim width)) :
    l2NormalizeSpec (.dim count rest) epsilon (.dim values) =
      .dim (fun index => l2NormalizeSpec rest epsilon (values index)) := by
  exact Internal.mapFinalAxis_dim _ values

/-- Normalize the final axis by `sqrt(sum (x * x) + epsilon)`.

This is not `torch.nn.functional.normalize`, which divides by `max(‖x‖, epsilon)`. The two agree
for nonzero real vectors when `epsilon = 0`; positive epsilon changes the denominator in general.

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
          l2NormalizeSpec leading epsilon.item input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input epsilon =>
        (Runtime.Autograd.Model.Norm.l2Normalize (m := m) (α := α)
          (leading := leading) (width := width) hWidth input epsilon :
          m (Runtime.Autograd.Model.Ref (m := m) (α := α)
            (leading.appendDim width))) }

/-- Pure evaluation of final-axis regularized L2 normalization. -/
@[simp] theorem l2Normalize_specFwd {leading : Shape} {width : Nat}
    (hWidth : 0 < width) {α : Type} [TorchLean.Storage α] [Context α]
    (input : TorchLean.Tensor α (leading.appendDim width)) (epsilon : α) :
    (l2Normalize leading width hWidth).specFwd
        (.cons input (.cons (.scalar epsilon) .nil)) =
      l2NormalizeSpec leading epsilon input := by
  change l2NormalizeSpec leading (Tensor.scalar epsilon).item input =
    l2NormalizeSpec leading epsilon input
  rw [Tensor.item_scalar]


end PrimOp

end DAG
end GraphSpec
end NN
