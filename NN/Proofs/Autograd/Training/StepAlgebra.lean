/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness

/-!
# StepAlgebra

Pure training-step algebra for proved autograd graphs.

This file deliberately stays on the *mathematical* side of training:

- `Graph.scalarLoss_grad_correct` specializes the tape/DAG backprop theorem to the scalar-loss
  convention used by training loops.
- `SGD.step` lifts the single-tensor SGD equation to a heterogeneous `TList` of parameters.

It is not another runtime optimizer implementation. Runtime files such as
`Runtime.Autograd.Torch.ParamList.sgdStep{,Fast}` and CUDA eager `sgdStepAllCuda` mutate `IO.Ref`s or
device buffers; their intended mathematical update is the pure context equation stated here.

## PyTorch correspondence / citations
- “Scalar loss” convention: backprop starts with upstream gradient 1.
  https://pytorch.org/docs/stable/autograd.html
- SGD update rule corresponds to `torch.optim.SGD` at the level of pure parameter math.
  https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor
open TensorAlgebra

noncomputable section

namespace Graph

variable {α : Type} [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/--
Seed cotangent for a scalar-loss graph.

The full context has shape list `Γ ++ ss ++ [scalar]`: original inputs/parameters, intermediate
nodes, and the final scalar loss. Reverse-mode training seeds all non-output cotangents with zero
and the scalar-loss output cotangent with `1`, matching `loss.backward()` in PyTorch.
-/
def seedScalarLoss {ss : List Shape} (_g : Graph (α := α) Δ Γ (ss ++ [Shape.scalar]))
    (_x : TList α Γ) : TList α (Γ ++ (ss ++ [Shape.scalar])) :=
  let zPrev : TList α (Γ ++ ss) := TList.zero (α := α) (ss := Γ ++ ss)
  let one : Tensor α Shape.scalar := Tensor.scalar (1 : α)
  let seed' : TList α ((Γ ++ ss) ++ [Shape.scalar]) := TList.snoc (α := α) (ss := Γ ++ ss) (τ :=
    Shape.scalar) zPrev one
  TList.cast (α := α) (h := List.append_assoc Γ ss [Shape.scalar]) seed'

/--
Scalar-loss specialization of `Graph.backprop_correct`.

The left side differentiates the graph by a forward-mode perturbation `dx` and pairs the result
with the scalar-loss seed. The right side pairs the same perturbation with the context cotangent
computed by reverse mode. This is the exact algebraic statement behind “the gradient returned by
backprop is the cotangent for the scalar loss”.
-/
theorem scalarLoss_grad_correct {ss : List Shape} (g : Graph (α := α) Δ Γ (ss ++ [Shape.scalar])) :
    ∀ x dx d,
      let seed := seedScalarLoss (α := α) (Γ := Γ) (ss := ss) g x
      TList.dotList (α := α) (jvpCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss ++ [Shape.scalar]) g x dx
        d) seed
        =
      TList.dotList (α := α) dx (backpropCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss ++ [Shape.scalar])
        g x d seed) := by
  intro x dx d seed
  simpa using (Graph.backprop_correct (α := α) (Δ := Δ) (Γ := Γ) (ss := ss ++ [Shape.scalar]) g x dx
    d seed)

end Graph

namespace SGD

variable {α : Type} [CommRing α]

/--
One pure SGD step over a typed parameter context:

`params := params - lr * grads`

This is the context-level version of the ordinary tensor update
`p := p - lr * g`. Runtime training code may implement the same equation by mutating parameter
references, materializing tensors eagerly, or launching CUDA kernels; this definition is the
side-effect-free algebraic target those implementations are meant to realize.
-/
def step {Γ : List Shape} (params grads : TList α Γ) (lr : α) : TList α Γ :=
  TList.sub (α := α) (ss := Γ) params (TList.scale (α := α) (ss := Γ) lr grads)

/-- The empty parameter context is unchanged by an SGD step. -/
@[simp] theorem step_nil (lr : α) :
    step (α := α) (Γ := []) TList.nil TList.nil lr = TList.nil := by
  rfl

/--
Cons-form unfolding of a context-level SGD step.

This lemma is intentionally small but useful as documentation: the head tensor update is exactly
`subSpec p (scaleSpec g lr)`, and the tail recursively receives the same learning rate.
-/
@[simp] theorem step_cons {s : Shape} {Γ : List Shape} (p g : Tensor α s)
    (ps gs : TList α Γ) (lr : α) :
    step (α := α) (Γ := s :: Γ) (TList.cons p ps) (TList.cons g gs) lr =
      TList.cons (subSpec (α := α) p (scaleSpec (α := α) g lr))
        (step (α := α) (Γ := Γ) ps gs lr) := by
  rfl

end SGD

end

end Algebra
end Autograd
end Proofs
