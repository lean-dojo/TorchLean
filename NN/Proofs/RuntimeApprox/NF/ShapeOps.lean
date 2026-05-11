/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Core.TensorReductionShape

/-!
# NF Shape Operators

NF (rounded) backend: approximation lemmas for shape-only tensor operators.

These operators do not perform arithmetic on scalars (they only permute/replicate entries), so
they preserve existing `approxT` error bounds.

That distinction matters: shape-only ops should not introduce extra rounding error. Their proofs
are mostly transport/indexing arguments rather than numerical analysis.

## PyTorch correspondence / citations
These are the proof analogues of “view-like”/index-rearrangement ops in PyTorch which do not change
floating-point values, only their arrangement:
https://pytorch.org/docs/stable/generated/torch.reshape.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
https://pytorch.org/docs/stable/generated/torch.permute.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
theorem approxT_replicate {s : Shape}
    {xS : SpecTensor .scalar} {xR : Tensor R .scalar} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.replicate (α := SpecScalar) (s := s) xS)
      (Spec.Tensor.replicate (α := R) (s := s) xR)
      eps := by
  classical
  cases xS with
  | scalar x =>
      cases xR with
      | scalar xR =>
          have hscalar :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (x := x) (xR := xR) (eps := eps)).1 hx
          have hε : 0 ≤ eps := le_trans (abs_nonneg _) hscalar
          induction s with
          | scalar =>
              simpa [Spec.Tensor.replicate] using hx
          | dim n inner ih =>
              refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
                (n := n) (s := inner) (xS := Spec.Tensor.replicate (α := SpecScalar) (s := .dim n
                  inner) (.scalar x))
                (xR := Spec.Tensor.replicate (α := R) (s := .dim n inner) (.scalar xR))
                (eps := eps) hε ?_
              intro i
              simpa [Spec.Tensor.replicate] using ih

omit [NeuralValidRndToNearest rnd] in
theorem approxT_broadcastTo
    {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
    {xS : SpecTensor s₁} {xR : Tensor R s₁} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.broadcastTo (α := SpecScalar) (s₁ := s₁) (s₂ := s₂) cb xS)
      (Spec.Tensor.broadcastTo (α := R) (s₁ := s₁) (s₂ := s₂) cb xR)
      eps := by
  classical
  have hε : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) (s := s₁) hx
  induction cb with
  | scalar_to_any s =>
      cases xS with
      | scalar _ =>
          cases xR with
          | scalar _ =>
              simpa [Spec.Tensor.broadcastTo] using
                approxT_replicate (β := β) (fexp := fexp) (rnd := rnd) (s := s) (hx := hx)
  | dim_eq tail ih =>
      cases xS with
      | dim fS =>
          cases xR with
          | dim fR =>
              refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
                (n := _) (s := _)
                (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.dim_eq tail)
                  (Tensor.dim fS))
                (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.dim_eq tail)
                  (Tensor.dim fR))
                (eps := eps) hε ?_
              intro i
              have hx_i :=
                approxT_dim_get (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
              simpa [Spec.Tensor.broadcastTo] using ih (xS := fS i) (xR := fR i) hx_i
  | dim_1_to_n tail ih =>
      cases xS with
      | dim fS =>
          cases xR with
          | dim fR =>
              refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
                (n := _) (s := _)
                (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.dim_1_to_n
                  tail) (Tensor.dim fS))
                (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.dim_1_to_n tail)
                  (Tensor.dim fR))
                (eps := eps) hε ?_
              intro i
              have hx0 :=
                approxT_dim_get (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx (0 : Fin
                  1)
              simpa [Spec.Tensor.broadcastTo] using ih (xS := fS 0) (xR := fR 0) hx0
  | expand_dims tail ih =>
      refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
        (n := _) (s := _)
        (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.expand_dims tail) xS)
        (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.expand_dims tail) xR)
        (eps := eps) hε ?_
      intro i
      simpa [Spec.Tensor.broadcastTo] using ih (xS := xS) (xR := xR) hx

end NFBackend

end
end RuntimeApprox
end Proofs
