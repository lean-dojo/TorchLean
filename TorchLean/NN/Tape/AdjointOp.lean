/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Algebra
public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Dropout
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Loss
public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.Contract

/-!
# Adjointness interface and semiring certificates

The adjointness interface of the autograd proofs, with the certificates that need only commutative
semiring algebra. A certificate `AdjointOp α σ τ` bundles a `TorchLean.VJPOp`, a selected tangent
map, and a proof of

$$
\left\langle \operatorname{JVP}(x,dx),\delta\right\rangle
=\left\langle dx,\operatorname{VJP}(x,\delta)\right\rangle
$$

for all `x`, `dx` and `δ`, where $\langle\cdot,\cdot\rangle$ is the pairing
`TorchLean.Tensor.Rep.dot`, the sum of entrywise products. Every statement is generic in a
commutative semiring `α`, so one certificate serves `ℝ` and exact backends such as `ℚ`. The
certificates for sigmoid, tanh and the other rules stated over the reals are values of
`AdjointOp ℝ` in `TorchLean.NN.AdjointOp`.

Adjointness does not identify the tangent map with the derivative of the forward map. The
`HasFDerivAt` theorems under `TorchLean.NN.FDeriv` add that fact for the operations they
cover.

Main declarations: `IsAdjointPair`, `AdjointOp`, the chain rule `AdjointOp.compose`, and the
certificates `AdjointOp.relu` and `AdjointOp.linear`. ReLU also assumes `Max`, `BEq`, `LT` and a
decidable `>` on `α`.

PyTorch analogue: the backward rule each primitive supplies to PyTorch Autograd,
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace TorchLean
namespace Tape

open TorchLean.Tensor
open Rep (dot)

noncomputable section

/-- VJP/JVP adjointness for a unary op `σ → τ`. -/
def IsAdjointPair {α : Type} [Storage α] [CommSemiring α] {σ τ : Shape}
  (_forward : Tensor α σ → Tensor α τ)
  (jvp     : Tensor α σ → Tensor α σ → Tensor α τ)
  (vjp     : Tensor α σ → Tensor α τ → Tensor α σ) : Prop :=
  ∀ x dx (δ : Tensor α τ), Rep.dot (α := α) (jvp x dx) δ = Rep.dot (α := α) dx (vjp x δ)

/--
An `VJPOp` over a commutative semiring, a selected tangent map, and a proof that the
tangent map and the backward pass are adjoint for the pairing `dot`.
-/
structure AdjointOp (α : Type) [Storage α] [CommSemiring α] (σ τ : Shape) where
  /-- The operation being certified, forward and backward together. -/
  op : VJPOp α σ τ
  /-- The selected tangent map at a basepoint, applied to a tangent. Identifying it with the
  derivative of `op.forward` takes a separate analytic theorem. -/
  jvp : Tensor α σ → Tensor α σ → Tensor α τ
  /-- The adjointness proof. Bundling it with the operation makes a value of this type a
  certificate: no value exists without a proof that the backward pass is the transpose. -/
  isAdjointPair : IsAdjointPair (α := α) op.forward jvp op.backward

namespace AdjointOp

/--
Composition preserves VJP/JVP correctness (reverse-mode chain rule).

Informally, if $f$ and $g$ satisfy
$\langle\operatorname{JVP},\cdot\rangle
=\langle\cdot,\operatorname{VJP}\rangle$,
then so does $g\circ f$, with the obvious composed JVP and VJP.
-/
def compose {α : Type} [Storage α] [CommSemiring α] {σ τ υ : Shape}
  (f : AdjointOp (α := α) σ τ) (g : AdjointOp (α := α) τ υ) :
  AdjointOp (α := α) σ υ :=
{
  op := VJPOp.compose (α := α) f.op g.op
  jvp := fun x dx => g.jvp (f.op.forward x) (f.jvp x dx)
  isAdjointPair := by
    intro x dx δ
    have hg := g.isAdjointPair (f.op.forward x) (f.jvp x dx) δ
    have hf := f.isAdjointPair x dx (g.op.backward (f.op.forward x) δ)
    simpa [VJPOp.compose, IsAdjointPair] using hg.trans hf
}

end AdjointOp

/--
Correctness of ReLU’s backward rule, stated generically over `α`.

ReLU and its derivative need extra structure on `α`: maximum, scalar equality, order and decidable
comparison.

PyTorch analogue: `torch.relu` / `torch.nn.functional.relu`.
-/
def AdjointOp.relu {α : Type} [Storage α] [CommSemiring α]
  [Max α] [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} :
  AdjointOp (α := α) s s :=
{
  op :=
    { forward := Tensor.map Activation.relu
      backward := fun x g => Tensor.map Activation.reluDeriv x * g }
  jvp := fun x dx => dx * Tensor.map Activation.reluDeriv x
  isAdjointPair := by
    intro x dx δ
    simpa using
        (Tensor.dot_elemwise_adjoint (α := α) (s := s) (dx := dx)
          (df := Tensor.map Activation.reluDeriv x) (δ := δ))
}

/--
Correctness of a linear layer’s backward rule (matrix–vector multiply), stated generically over `α`.

The proof uses only semiring laws and the matrix adjointness lemma
`Tensor.dot_contract_matVec`.
PyTorch analogue: the affine map implemented by `torch.nn.Linear`.
-/
def AdjointOp.linear {α : Type} [Storage α] [CommSemiring α]
  {inDim outDim : Nat} (m : Linear α inDim outDim) :
  AdjointOp (α := α) (.dim inDim .scalar) (.dim outDim .scalar) :=
{
  op :=
    { forward := fun x => Linear.forward (α := α) m x
      backward := fun _ dLdy => Linear.inputDeriv (α := α) m.weights dLdy }
  jvp := fun _x dx => Tensor.contract m.weights dx _ _ [outDim] .matVec
  isAdjointPair := by
    intro x dx δ
    classical
    have hadj :=
      Tensor.dot_contract_matVec (α := α) (W := m.weights) (dLdy := δ) (dx := dx)
    calc
      Rep.dot (α := α) (Tensor.contract m.weights dx _ _ [outDim] .matVec) δ
          = Rep.dot (α := α) δ (Tensor.contract m.weights dx _ _ [outDim] .matVec) := by
              exact Tensor.dot_comm _ _
      _ = Rep.dot (α := α) (Tensor.contract δ m.weights _ _ [inDim] .vecMat) dx := hadj
      _ = Rep.dot (α := α) dx (Tensor.contract δ m.weights _ _ [inDim] .vecMat) := by
              exact Tensor.dot_comm _ _
}

end
end Tape
end TorchLean
