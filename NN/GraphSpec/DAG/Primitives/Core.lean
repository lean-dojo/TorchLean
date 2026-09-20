/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Primitives
public import NN.GraphSpec.Chain.ToDAG.Core
public import NN.GraphSpec.Primitives.Spatial

/-!
# Core DAG Primitives

Constants, elementwise arithmetic, and standard sequential operations exposed as typed DAG nodes.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.Spec _root_.TorchLean
open TorchLean.Tensor
open _root_.TorchLean.Tensor

namespace PrimOp

/-! ## Basic DAG primitives -/

/-- Produce the all-zero tensor of a statically known shape. -/
def zero (s : Shape) : PrimOp [] s :=
  { name := "zero"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .nil => _root_.TorchLean.Tensor.full s 0
    program := fun {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Model.const (m := m) (α := α) (_root_.TorchLean.Tensor.full s 0) }

/-- The zero DAG node denotes the all-zero tensor of its declared shape. -/
@[simp] theorem zero_specFwd (s : Shape) {α : Type} [TorchLean.Storage α] [Context α] :
    (zero s).specFwd (α := α) .nil = _root_.TorchLean.Tensor.full s 0 := by
  rfl

/-- Produce the all-one tensor of a statically known shape. -/
def one (s : Shape) : PrimOp [] s :=
  { name := "one"
    specFwd := fun {_α} _storage _ctx xs =>
      match xs with
      | .nil => _root_.TorchLean.Tensor.full s 1
    program := fun {α} _ _ =>
      fun {m} _ _ =>
        Runtime.Autograd.Model.const (m := m) (α := α) (_root_.TorchLean.Tensor.full s 1) }

/--
Dense linear layer in DAG form.

Inputs are ordered as `[W, b, x]`:

- `W : Tensor α [outDim, inDim]`,
- `b : Tensor α [outDim]`,
- `x` has shape `[inDim]`.

The output has shape `[outDim]`. This is the DAG embedding of `Primitive.linear`, so the DAG and
sequential authoring surfaces share the same Spec semantics and TorchLean lowering path.
-/
def linear (inDim outDim : Nat) :
    PrimOp [[outDim, inDim], [outDim], [inDim]] [outDim] :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.linear inDim outDim) : PrimOp _ _)

/--
Flatten a tensor to a rank-one tensor in DAG form.

Input: `[x : TorchLean.Tensor s]`.
Output: `Tensor α [Spec.Shape.size s]`.

This is the DAG embedding of `Primitive.flatten`, so it has exactly the same row-major view
semantics as the sequential primitive.
-/
def flatten (s : Shape) : PrimOp [s] [Spec.Shape.size s] :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.flatten s) : PrimOp _ _)

/-! ## Spatial and residual DAG primitives -/

/--
ReLU activation in DAG form.

Input: `[x : s]`, output: `s`.

Semantics: elementwise $\max(x,0)$. This is parameter-free and derived from `Primitive.relu`.

Reference: Nair and Hinton (2010), "Rectified Linear Units Improve Restricted Boltzmann Machines".
-/
def relu (s : Shape) : PrimOp [s] s :=
  (LowerToDAG.Primitive.toDAGPrimOp (Primitive.relu s) : PrimOp _ _)

/--
Add two tensors of the same shape.

Input shapes: `[s, s]`, output shape: `s`.

This is the primitive used for residual/skip connections:
$\mathrm{out}=\operatorname{main}(x)+x$. It is defined
directly because the sequential surface is unary, while residual addition is genuinely multi-input.
-/
def add (s : Shape) : PrimOp [s, s] s :=
  { name := "add"
    specFwd := fun {α} _storage _ctx xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.TorchLean.Tensor.addSpec (α := α) a b
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun a b => Runtime.Autograd.Model.add (m := m) (α := α) (s := s) a b
  }

/-- The pure meaning of the DAG addition node is pointwise tensor addition. -/
@[simp] theorem add_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (left right : _root_.TorchLean.Tensor α s) :
    (add s).specFwd (.cons left (.cons right .nil)) =
      _root_.TorchLean.Tensor.addSpec left right := by
  rfl

/-- Subtract two tensors of the same shape. -/
def sub (s : Shape) : PrimOp [s, s] s :=
  { name := "sub"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.TorchLean.Tensor.subSpec (α := α) a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        Runtime.Autograd.Model.sub (m := m) (α := α) a b }

/-- The pure meaning of the DAG subtraction node is pointwise tensor subtraction. -/
@[simp] theorem sub_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (left right : _root_.TorchLean.Tensor α s) :
    (sub s).specFwd (.cons left (.cons right .nil)) =
      _root_.TorchLean.Tensor.subSpec left right := by
  rfl

/-- Multiply two tensors coordinatewise. -/
def mul (s : Shape) : PrimOp [s, s] s :=
  { name := "mul"
    specFwd := fun {α} _ _ xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.TorchLean.Tensor.mulSpec (α := α) a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        Runtime.Autograd.Model.mul (m := m) (α := α) a b }

/-- The pure meaning of the DAG multiplication node is pointwise tensor multiplication. -/
@[simp] theorem mul_specFwd {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (left right : _root_.TorchLean.Tensor α s) :
    (mul s).specFwd (.cons left (.cons right .nil)) =
      _root_.TorchLean.Tensor.mulSpec left right := by
  rfl


/-- Arbitrary-rank convolution in DAG form, with inputs ordered as `[kernel, bias, x]`. -/
def conv
    {d : Nat} (inC outC : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d]) :
    PrimOp
      [Shape.ofList (outC :: inC :: (Tensor.to kernel (List Nat))), [outC],
        Shape.ofList (inC :: (Tensor.to spatial (List Nat)))]
      (Shape.ofList (outC :: Tensor.to (Spec.convOutSpatial spatial kernel stride padding)
        (List Nat))) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.conv (inC := inC) (outC := outC) kernel stride padding spatial) : PrimOp _ _)

/-- Arbitrary-rank max pooling in DAG form. -/
def maxPool
    {d : Nat} (channels : Nat)
    (kernel stride padding spatial : TorchLean.Tensor Nat [d])
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0} :
    PrimOp [Shape.ofList (channels :: (Tensor.to spatial (List Nat)))]
      (Shape.ofList (channels ::
        Tensor.to (Spec.poolOutSpatialPad spatial kernel stride padding) (List Nat))) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.maxPool (channels := channels) kernel stride padding spatial
        (hKernel := hKernel) (hStride := hStride)) : PrimOp _ _)

/-- Batch normalization over an arbitrary spatial shape in DAG form. -/
def batchNorm (channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim channels spatial).wellFormed) :
    PrimOp [[channels], [channels], .dim channels spatial]
      (.dim channels spatial) :=
  (LowerToDAG.Primitive.toDAGPrimOp
      (Primitive.batchNorm channels spatial hWellFormed) : PrimOp _ _)

end PrimOp

end DAG
end GraphSpec
end NN
