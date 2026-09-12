/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common

/-!
# β-phase ReLU relaxations: pointwise soundness

Analytic core of the β-extended ReLU transfer rule, stated without reference to graphs or step
functions.

Given a pre-activation `z` enclosed both by an affine box (propagated from the parent's
certificate) and by an IBP box, the relaxations returned by `phaseRelaxVec?` enclose `relu z`.
The IBP box is what justifies each phase's affine rule (`phaseConsistentScalar?` checks the phase
against it); the affine box is what the resulting slopes are applied to.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/--
Index-wise bounds delivered by a β-consistent phase relaxation.

At index `i`, the lower relaxation applied to the affine lower bound sits below `relu (z i)`, and
the upper relaxation applied to the affine upper bound sits above it. Both slopes are nonnegative,
which is what lets the affine bounds on `z` pass through the relaxations.
-/
theorem phaseRelax_getScalar_bounds {n : Nat}
    (lAff uAff lo hi z αt : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [n])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := n) lo hi αt phases = some (relaxLo, relaxHi))
    (hzAff : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lAff, hi := uAff } z)
    (hzIbp : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } z)
    (i : Fin n) :
    (getScalar relaxLo i).slope * getScalar lAff i + (getScalar relaxLo i).bias ≤
        Activation.Math.reluSpec (α := ℝ) (getScalar z i) ∧
      Activation.Math.reluSpec (α := ℝ) (getScalar z i) ≤
        (getScalar relaxHi i).slope * getScalar uAff i + (getScalar relaxHi i).bias := by
  obtain ⟨hzLo, hzHi⟩ := (encloses_iff_getScalar (n := n) lAff uAff z).1 hzAff i
  obtain ⟨hzIlo, hzIhi⟩ := (encloses_iff_getScalar (n := n) lo hi z).1 hzIbp i
  obtain ⟨ph, hcons, hrHi, hrLo⟩ :=
    (phaseRelaxVec?_some_getScalar lo hi αt phases relaxLo relaxHi hrelax).2 i
  obtain ⟨hai0, hai1⟩ := hαrange i
  rw [hrLo, hrHi]
  have hsLo := phaseRelaxLowerScalar_slope_nonneg (getScalar lo i) (getScalar hi i)
    (getScalar αt i) ph hai0
  have hsHi := phaseRelaxUpperScalar_slope_nonneg (getScalar lo i) (getScalar hi i) ph
  have hlo := NN.MLTheory.CROWN.Proofs.phaseRelaxLowerScalar_sound (getScalar lo i)
    (getScalar hi i) (getScalar αt i) (getScalar z i) hzIlo hzIhi hai0 hai1 ph hcons
  have hhi := NN.MLTheory.CROWN.Proofs.phaseRelaxUpperScalar_sound (getScalar lo i)
    (getScalar hi i) (getScalar z i) hzIlo hzIhi ph hcons
  constructor
  · exact le_trans (add_le_add (mul_le_mul_of_nonneg_left hzLo hsLo) le_rfl) hlo
  · exact le_trans hhi (add_le_add (mul_le_mul_of_nonneg_left hzHi hsHi) le_rfl)

/--
Enclosure of `relu z` by phase relaxations propagated through affine bounds on `z`.

`xLo`, `xHi` are the parent's affine lower and upper bounds (already cast to the IBP dimension
`n`), evaluated at the input `x'`; `lo`, `hi` are the IBP box that `phaseRelaxVec?` checked the
phases against.
-/
theorem encloses_relu_propagateAffine {inDim n : Nat}
    (xLo xHi : AffineVec ℝ inDim n) (x' : Tensor ℝ [inDim])
    (lo hi z αt : Tensor ℝ [n]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [n])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := n) lo hi αt phases = some (relaxLo, relaxHi))
    (hzAff : Theorems.Semantics.encloses (α := ℝ)
      { dim := n
        lo := affineEvalAt (α := ℝ) (inDim := inDim) (outDim := n) xLo x'
        hi := affineEvalAt (α := ℝ) (inDim := inDim) (outDim := n) xHi x' } z)
    (hzIbp : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } z) :
    Theorems.Semantics.encloses (α := ℝ)
      { dim := n
        lo := affineEvalAt (α := ℝ) (inDim := inDim) (outDim := n)
          (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := n) relaxLo xLo) x'
        hi := affineEvalAt (α := ℝ) (inDim := inDim) (outDim := n)
          (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := n) relaxHi xHi) x' }
      (Activation.reluSpec (α := ℝ) z) := by
  refine (encloses_iff_getScalar (n := n) _ _ _).2 ?_
  intro i
  rw [getScalar_affineEvalAt_relu_propagate_affine, getScalar_affineEvalAt_relu_propagate_affine,
    getScalar_relu_spec]
  exact phaseRelax_getScalar_bounds _ _ lo hi z αt phases relaxLo relaxHi hαrange hrelax hzAff
    hzIbp i

/--
Soundness of the β-phase ReLU rule at one node.

The parent value `vp` is enclosed by its certificate entry `xin` (evaluated at `x`) and by its
IBP box `preB`. The node's bounds are the phase relaxations for `preB` propagated through `xin`
cast to `preB.dim`, and they enclose `relu vp`.
-/
theorem enclosesAtInput_relu_beta
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (xin : FlatAffineBounds ℝ) (vp : Val) (preB : FlatBox ℝ)
    (hout : xin.outDim = preB.dim)
    (αt : Tensor ℝ [preB.dim]) (phases : Array Int)
    (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [preB.dim])
    (hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ))
    (hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt phases =
      some (relaxLo, relaxHi))
    (hpar : EnclosesAtInput (α := ℝ) ctx x xin vp)
    (hibp : EnclosesBox preB vp) :
    EnclosesAtInput (α := ℝ) ctx x
      { inDim := xin.inDim
        outDim := preB.dim
        loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxLo
          (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.loAff)
        hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxHi
          (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.hiAff) }
      { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } := by
  obtain ⟨hdim, hzIbp⟩ := hibp
  obtain ⟨hinDim, _, hzAff⟩ := enclosesAtInput_castOut ctx x xin vp hout hdim.symm hpar
  refine ⟨hinDim, hdim, ?_⟩
  rw [relu_spec_castDimScalar]
  exact encloses_relu_propagateAffine
    (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.loAff)
    (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.hiAff)
    (castDimScalar (α := ℝ) hinDim.symm x) preB.lo preB.hi
    (castDimScalar (α := ℝ) hdim.symm vp.v) αt phases relaxLo relaxHi hαrange hrelax hzAff hzIbp

end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
