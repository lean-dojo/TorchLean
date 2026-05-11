/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness
public import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
public import NN.Proofs.Tensor.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Transfer-rule soundness for `alphaCrownStepNode?` (graph dialect, over `ℝ`)

This file proves that the concrete α-CROWN step function
`NN.MLTheory.CROWN.Cert.alphaCrownStepNode?` satisfies the abstract `CrownTransferSound`
assumption used by the generic checker theorem in
`NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`.

Scope (current):
- Proves soundness for the ops handled explicitly by `alphaCrownStepNode?`:
  `.input`, `.const`, `.detach`, `.linear`, `.matmul`, `.sum`, `.relu`, `.reshape`, `.flatten`.
- All other ops are handled via the IBP-fallback in the step function, assuming the provided IBP
  boxes enclose the semantic values (`IBPEnclosesVals`).

The theorem here is pointwise at a fixed input point `x`. To obtain a statement over an input
set/box `B`, quantify this result over all `x ∈ B`.

## Relation to α-CROWN / α,β-CROWN verifiers

The “step function” proved sound here plays the same conceptual role as the per-layer/per-node
bound propagation rules in LiRPA implementations (e.g. `auto_LiRPA`) and in the full verifier
`alpha-beta-CROWN`: given parent enclosures, compute a sound enclosure for the current node.

This file focuses on the **local transfer-rule** theorem that is needed to plug the concrete
step function into the generic checker theorem in
`NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`.

## References (code)

- `auto_LiRPA`: https://github.com/Verified-Intelligence/auto_LiRPA
- `alpha-beta-CROWN`: https://github.com/Verified-Intelligence/alpha-beta-CROWN

## Import policy

This file contains the large op-by-op transfer proofs for the concrete α-CROWN / α/β-CROWN checker
steps. It is imported directly by callers that need those proofs; the lighter overview module keeps
the high-level proof map available without pulling this whole proof development into every build.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness

/-- Alias for the semantic value record used by the generic graph soundness development. -/
abbrev Val := CertSoundness.Val
/-- Alias for the partial node evaluator used in `CertSoundness`. -/
abbrev evalNode? := CertSoundness.evalNode?
/-- Alias for the local semantic side-condition used in checker soundness statements. -/
abbrev SemLocalOK := CertSoundness.SemLocalOK
/-- Alias for the topological-sorting predicate used by the generic soundness theorems. -/
abbrev TopoSorted := CertSoundness.TopoSorted

-- The graph dialect’s `FlatBox` is a dependent record (the tensor shapes depend on `dim`),
-- so Lean does not automatically register a usable extensionality lemma for the `ext` tactic.
-- We add a small `[ext]` lemma locally.
@[ext] private theorem FlatBox.ext' {α : Type} [Context α] {B1 B2 : FlatBox α}
    (hDim : B1.dim = B2.dim)
    (hLo : HEq B1.lo B2.lo)
    (hHi : HEq B1.hi B2.hi) : B1 = B2 := by
  cases B1
  cases B2
  cases hDim
  cases hLo
  cases hHi
  rfl

/-! ## Helper assumptions -/

/-- The designated input entry in `inputs` matches the concrete point `x` (up to a `castDimScalar`).
  -/
def InputsMatch (inputs : Std.HashMap Nat Val) (ctx : AffineCtx)
    (x : Tensor ℝ (.dim ctx.inputDim .scalar)) : Prop :=
  ∃ v : Val,
    inputs[ctx.inputId]? = some v ∧
    ∃ h : v.n = ctx.inputDim,
      castDimScalar (α := ℝ) (n := v.n) (n' := ctx.inputDim) h v.v = x

/-- Pointwise: whenever both arrays contain entries at `id`, the IBP box encloses the semantic
  value. -/
def IBPEnclosesVals (ibp : Array (Option (FlatBox ℝ))) (vals : Array (Option Val)) : Prop :=
  ∀ id : Nat, id < vals.size →
    match ibp[id]!, vals[id]! with
    | some B, some v => CertSoundness.EnclosesBox B v
    | _, _ => True

/-- Well-formedness condition for α vectors: each component lies in `[0,1]`. -/
def AlphaOK (alpha : Array (Option (FlatVec ℝ))) : Prop :=
  ∀ id : Nat, id < alpha.size →
    match alpha[id]! with
    | none => True
    | some a => ∀ i : Fin a.n, (0 : ℝ) ≤ toVec a.v i ∧ toVec a.v i ≤ (1 : ℝ)

/-! ## `Theorems.Semantics.encloses` ↔ componentwise inequalities (via `toVec`) -/

private lemma encloses_iff_toVec {n : Nat}
    (lo hi x : Tensor ℝ (.dim n .scalar)) :
    Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } x ↔
      ∀ i : Fin n, toVec lo i ≤ toVec x i ∧ toVec x i ≤ toVec hi i := by
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      cases x with
      | dim fx =>
        constructor
        · intro h i
          have hi := h i
          cases hlo : flo i with
          | scalar l =>
            cases hhi : fhi i with
            | scalar u =>
              cases hx : fx i with
              | scalar v =>
                simpa [Theorems.Semantics.encloses, getDimScalarFn, toVec, hlo, hhi, hx] using hi
        · intro h i
          have hi := h i
          cases hlo : flo i with
          | scalar l =>
            cases hhi : fhi i with
            | scalar u =>
              cases hx : fx i with
              | scalar v =>
                simpa [Theorems.Semantics.encloses, getDimScalarFn, toVec, hlo, hhi, hx] using hi

/-! ## Small tensor algebra helpers -/

private lemma add_spec_fill_zero_right {n : Nat}
    (t : Tensor ℝ (.dim n .scalar)) :
    Tensor.addSpec (α := ℝ) t (Spec.fill (α := ℝ) (0 : ℝ) (.dim n .scalar)) = t := by
  cases t with
  | dim ft =>
      -- Reduce to pointwise scalar addition by unfolding `add_spec`/`map2_spec`.
      apply congrArg Tensor.dim
      funext i
      cases hti : ft i with
      | scalar x =>
          simp [Tensor.map2Spec, Spec.fill]

private lemma linear_spec_bias_zero_eq_matvec {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar)))
    (x : Tensor ℝ (.dim n .scalar)) :
    Spec.linearSpec (α := ℝ)
        { weights := W
          bias := Spec.fill (α := ℝ) (0 : ℝ) (.dim m .scalar) } x
      =
      Spec.matVecMulSpec (α := ℝ) W x := by
  -- `linear_spec` is `mat_vec_mul + bias`, so the zero bias disappears.
  simp [Spec.linearSpec]

/-! ## Small cast lemmas (avoid `cases` on equalities mentioning record fields) -/

/-- `castDimScalar` composes as expected under transitive equalities. -/
lemma castDimScalar_trans {n n' n'' : Nat}
    (h₁ : n = n') (h₂ : n' = n'') (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) (Eq.trans h₁ h₂) t
      = castDimScalar (α := ℝ) h₂ (castDimScalar (α := ℝ) h₁ t) := by
  cases h₁
  cases h₂
  rfl

/-- `castDimScalar` is proof-irrelevant in its equality argument. -/
lemma castDimScalar_proof_irrel {n n' : Nat}
    (h₁ h₂ : n = n') (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h₁ t = castDimScalar (α := ℝ) h₂ t := by
  have : h₁ = h₂ := Subsingleton.elim _ _
  cases this
  rfl

/-- `toVec` commutes with `castDimScalar` (up to `Fin.cast`). -/
lemma toVec_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ (.dim n .scalar)) (i : Fin n') :
    toVec (castDimScalar (α := ℝ) (n := n) (n' := n') h t) i = toVec t (Fin.cast h.symm i) := by
  cases h
  simp [castDimScalar]

/-- `Activation.relu_spec` commutes with `castDimScalar`. -/
lemma relu_spec_castDimScalar {n n' : Nat} (h : n = n') (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) (n := n) (n' := n') h (Activation.reluSpec (α := ℝ) t)
      =
    Activation.reluSpec (α := ℝ) (castDimScalar (α := ℝ) (n := n) (n' := n') h t) := by
  cases h
  rfl

/-- A small `mat_vec_mul` cast lemma used for single-row “sum” encodings. -/
lemma mat_vec_mul_fill1_castDimScalar {n n' : Nat} (h : n = n') (v : Tensor ℝ (.dim n .scalar)) :
    Spec.matVecMulSpec (α := ℝ)
        (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim n .scalar))) v
      =
    Spec.matVecMulSpec (α := ℝ)
        (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim n' .scalar)))
        (castDimScalar (α := ℝ) (n := n) (n' := n') h v) := by
  cases h
  simp [castDimScalar]

/-- `affineEvalAt` commutes with casting the output dimension of an affine form. -/
lemma affineEvalAt_castAffineOut {inDim outDim outDim' : Nat}
    (h : outDim = outDim') (aff : AffineVec ℝ inDim outDim) (x : Tensor ℝ (.dim inDim .scalar)) :
    CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := outDim')
        (NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := inDim) (m := outDim) (m' := outDim') h
          aff) x
      =
      castDimScalar (α := ℝ) h
        (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := outDim) aff x) := by
  cases h
  rfl

/-- `boundsEvalAt` commutes with casting the output dimension of affine bounds. -/
lemma boundsEvalAt_castAffineOut (xin : FlatAffineBounds ℝ) {outDim' : Nat}
    (h : xin.outDim = outDim') (x : Tensor ℝ (.dim xin.inDim .scalar)) :
    CrownCertSoundness.boundsEvalAt (α := ℝ)
        { inDim := xin.inDim
          outDim := outDim'
          loAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') h xin.loAff
          hiAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') h xin.hiAff } x
      =
      { dim := outDim'
        lo := castDimScalar (α := ℝ) h (CrownCertSoundness.boundsEvalAt (α := ℝ) xin x).lo
        hi := castDimScalar (α := ℝ) h (CrownCertSoundness.boundsEvalAt (α := ℝ) xin x).hi } := by
  cases h
  rfl

/-- `Semantics.encloses` is preserved under casting a box and point to an equal dimension. -/
lemma sem_encloses_castDim {B : FlatBox ℝ} {n' : Nat}
    (h : B.dim = n') (x : Tensor ℝ (.dim B.dim .scalar)) :
    Theorems.Semantics.encloses (α := ℝ) B x →
      Theorems.Semantics.encloses (α := ℝ)
        { dim := n'
          lo := castDimScalar (α := ℝ) h B.lo
          hi := castDimScalar (α := ℝ) h B.hi }
        (castDimScalar (α := ℝ) h x) := by
  intro hx
  cases B with
  | mk n lo hi =>
      cases h
      simpa [Theorems.Semantics.encloses, castDimScalar, getDimScalarFn] using hx

/-- `Semantics.encloses` respects definitional equality of boxes. -/
lemma sem_encloses_of_eq {B1 B2 : FlatBox ℝ}
    (h : B1 = B2) (x : Tensor ℝ (.dim B1.dim .scalar)) :
    Theorems.Semantics.encloses (α := ℝ) B1 x →
      Theorems.Semantics.encloses (α := ℝ) B2
        (castDimScalar (α := ℝ) (congrArg FlatBox.dim h) x) := by
  intro hx
  cases h
  simpa [castDimScalar] using hx

/-- `Semantics.encloses` respects definitional equality of values. -/
lemma sem_encloses_value_eq {B : FlatBox ℝ}
    {x y : Tensor ℝ (.dim B.dim .scalar)} (hxy : x = y) :
    Theorems.Semantics.encloses (α := ℝ) B x →
      Theorems.Semantics.encloses (α := ℝ) B y := by
  intro hx
  cases hxy
  simpa using hx

/-- `EnclosesAtInput` is preserved under casting the output dimension of bounds and value payloads.
  -/
lemma enclosesAtInput_castOut (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (xin : FlatAffineBounds ℝ) (vp : FlatVec ℝ) {outDim' : Nat}
    (hout : xin.outDim = outDim') (hvout : vp.n = outDim') :
    CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x xin vp →
      CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x
        { inDim := xin.inDim
          outDim := outDim'
          loAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.loAff
          hiAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
            (m' := outDim') hout xin.hiAff }
        { n := outDim', v := castDimScalar (α := ℝ) hvout vp.v } := by
  intro hpar
  rcases hpar with ⟨hinDim, hvec⟩
  refine ⟨hinDim, ?_⟩
  -- The `x'` used to evaluate `xin` and the casted bound is the same, since `inDim` is unchanged.
  dsimp
  rcases hvec with ⟨hdim, henc⟩
  -- Cast the enclosure result from `xin.outDim` to `outDim'` via `hout`.
  have henc' :
      Theorems.Semantics.encloses (α := ℝ)
        { dim := outDim'
          lo := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).lo
          hi := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).hi }
        (castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdim.symm vp.v)) := by
    -- `henc` is an enclosure for `boundsEvalAt xin x'` at `castDimScalar hdim.symm vp.v`.
    -- Transport it across `hout`.
    simpa using
      (sem_encloses_castDim (B := CrownCertSoundness.boundsEvalAt (α := ℝ) xin (castDimScalar (α :=
        ℝ) hinDim.symm x))
        (h := hout) (x := castDimScalar (α := ℝ) hdim.symm vp.v) henc)
  -- Simplify the RHS cast: `hout ∘ hdim.symm` is a proof of `vp.n = outDim'`, so it matches
  -- `hvout`.
  have hvout' : Eq.trans hdim.symm hout = hvout := by
    exact Subsingleton.elim _ _
  have hxCast :
      castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdim.symm vp.v)
        = castDimScalar (α := ℝ) hvout vp.v := by
    simpa [hvout', castDimScalar_trans] using
      (castDimScalar_trans (h₁ := hdim.symm) (h₂ := hout) (t := vp.v)).symm
  -- Rewrite the enclosure `henc'` to target `castDimScalar hvout vp.v`.
  have henc1 :
      Theorems.Semantics.encloses (α := ℝ)
        { dim := outDim'
          lo := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).lo
          hi := castDimScalar (α := ℝ) hout (CrownCertSoundness.boundsEvalAt (α := ℝ) xin
            (castDimScalar (α := ℝ) hinDim.symm x)).hi }
        (castDimScalar (α := ℝ) hvout vp.v) := by
    simpa [hxCast] using henc'

  -- Avoid rewriting dependent `boundsEvalAt` equalities: transfer componentwise between the
  -- explicit cast box
  -- and `boundsEvalAt` of the casted affine bound.
  let x0 : Tensor ℝ (.dim xin.inDim .scalar) :=
    castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim) hinDim.symm x
  let B1 : FlatBox ℝ :=
    boundsEvalAt (α := ℝ)
      { inDim := xin.inDim
        outDim := outDim'
        loAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
          (m' := outDim') hout xin.loAff
        hiAff := NN.MLTheory.CROWN.Cert.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
          (m' := outDim') hout xin.hiAff } x0
  let B2 : FlatBox ℝ :=
    { dim := outDim'
      lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x0).lo
      hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x0).hi }

  have hB2 : Theorems.Semantics.encloses (α := ℝ) B2 (castDimScalar (α := ℝ) hvout vp.v) := by
    simpa [B2, x0] using henc1

  have hlo : B1.lo = B2.lo := by
    -- `B1.lo` is `affineEvalAt` of the casted affine map; `B2.lo` is the cast of the original
    -- `boundsEvalAt` lower.
    simpa [B1, B2, x0, CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt] using
      (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x := x0))
  have hhi : B1.hi = B2.hi := by
    simpa [B1, B2, x0, CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt] using
      (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x := x0))

  have hB1 : Theorems.Semantics.encloses (α := ℝ) B1 (castDimScalar (α := ℝ) hvout vp.v) := by
    have hcomp :=
      (encloses_iff_toVec (n := outDim') (lo := B2.lo) (hi := B2.hi)
        (x := castDimScalar (α := ℝ) hvout vp.v)).1 hB2
    refine (encloses_iff_toVec (n := outDim') (lo := B1.lo) (hi := B1.hi)
      (x := castDimScalar (α := ℝ) hvout vp.v)).2 ?_
    intro i
    have hi := hcomp i
    constructor
    · simpa [hlo] using hi.1
    · simpa [hhi] using hi.2

  -- Finish by packaging as `EnclosesVec` (the outer cast is definitional).
  refine ⟨rfl, ?_⟩
  simpa [B1, x0, castDimScalar] using hB1

/-! ## Matrix sign-splitting bound (pointwise, over `ℝ`) -/

private lemma get2_mat_pos {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then Spec.get2 W i j else 0) := by
  cases W with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols j with
      | scalar w =>
        simp [NN.MLTheory.CROWN.IBP.matPos, Spec.get2_eq, Spec.get_eq, hrow, hcol]

private lemma get2_mat_neg {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i j =
      (if Spec.get2 W i j > 0 then 0 else Spec.get2 W i j) := by
  cases W with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols j with
      | scalar w =>
        simp [NN.MLTheory.CROWN.IBP.matNeg, Spec.get2_eq, Spec.get_eq, hrow, hcol]

private lemma signSplit_term_upper (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    w * x ≤ (if 0 < w then w else 0) * u + (if 0 < w then 0 else w) * l := by
  by_cases hw : 0 < w
  · have hw0 : 0 ≤ w := le_of_lt hw
    have : w * x ≤ w * u := mul_le_mul_of_nonneg_left hxu hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this
  · have hw0 : w ≤ 0 := le_of_not_gt hw
    have : w * x ≤ w * l := mul_le_mul_of_nonpos_left hlx hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this

private lemma signSplit_term_lower (w l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    (if 0 < w then w else 0) * l + (if 0 < w then 0 else w) * u ≤ w * x := by
  by_cases hw : 0 < w
  · have hw0 : 0 ≤ w := le_of_lt hw
    have : w * l ≤ w * x := mul_le_mul_of_nonneg_left hlx hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this
  · have hw0 : w ≤ 0 := le_of_not_gt hw
    have : w * u ≤ w * x := mul_le_mul_of_nonpos_left hxu hw0
    simpa [hw, add_assoc, add_left_comm, add_comm] using this

private theorem encloses_linear_signSplit {m n : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar)))
    (b : Tensor ℝ (.dim m .scalar))
    (lo hi x : Tensor ℝ (.dim n .scalar))
    (hx : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } x) :
    Theorems.Semantics.encloses (α := ℝ)
      { dim := m
        lo :=
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi))
            b
        hi :=
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo))
            b }
      (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) := by
  classical
  have hx' := (encloses_iff_toVec (lo := lo) (hi := hi) (x := x)).1 hx
  refine (encloses_iff_toVec (n := m) (lo := _) (hi := _) (x := _)).2 ?_
  intro i
  -- Expand all mat-vec products into finite sums.
  have hW :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ) W x) i =
        ∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k) := by
    simpa using (Spec.toVec_mat_vec_mul_spec (A := W) (v := x) (i := i))
  have hPos_lo :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) lo) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            lo k) := by
    simpa using
      (Spec.toVec_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))
  have hNeg_hi :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) hi) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k) := by
    simpa using
      (Spec.toVec_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hPos_hi :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) hi) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k) := by
    simpa using
      (Spec.toVec_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) (v := hi) (i := i))
  have hNeg_lo :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ)
          (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) lo) i =
        ∑ k : Fin n,
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            lo k) := by
    simpa using
      (Spec.toVec_mat_vec_mul_spec
        (A := NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) (v := lo) (i := i))

  have hUpperTerm :
      ∀ k : Fin n,
        (Spec.get2 W i k) * (Spec.toVec x k) ≤
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k) +
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            lo k) := by
    intro k
    have hk := hx' k
    have hpos := get2_mat_pos (W := W) i k
    have hneg := get2_mat_neg (W := W) i k
    have := signSplit_term_upper (w := Spec.get2 W i k) (l := Spec.toVec lo k) (u := Spec.toVec hi
      k)
      (x := Spec.toVec x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hLowerTerm :
      ∀ k : Fin n,
        (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
          lo k) +
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k)
            ≤ (Spec.get2 W i k) * (Spec.toVec x k) := by
    intro k
    have hk := hx' k
    have hpos := get2_mat_pos (W := W) i k
    have hneg := get2_mat_neg (W := W) i k
    have := signSplit_term_lower (w := Spec.get2 W i k) (l := Spec.toVec lo k) (u := Spec.toVec hi
      k)
      (x := Spec.toVec x k) hk.1 hk.2
    simpa [hpos, hneg, mul_add, add_mul, add_assoc, add_left_comm, add_comm] using this

  have hUpperSum :
      (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) ≤
        (∑ k : Fin n,
          ((Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
            (Spec.toVec hi k) +
           (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
             (Spec.toVec lo k))) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hUpperTerm k))

  have hLowerSum :
      (∑ k : Fin n,
        ((Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
          lo k) +
         (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
           hi k)))
          ≤ (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) := by
    classical
    simpa using (Finset.sum_le_sum (s := Finset.univ) (fun k _ => hLowerTerm k))

  have hlo :
      Spec.toVec
          (Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) lo)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) hi))
            b) i
        ≤
  Spec.toVec (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) i := by
    -- Rewrite `sum (a+b)` into `sum a + sum b` to match `toVec` expansions.
    have hLowerSum' :
        (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (Spec.toVec lo k)) +
          (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (Spec.toVec hi k))
            ≤ (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) := by
      -- Start from `hLowerSum` and distribute the sum.
      let f : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            lo k)
      let g : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k)
      have hLowerSum_fg :
          (∑ k : Fin n, (f k + g k)) ≤ (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) := by
        simpa [f, g] using hLowerSum
      have hdist : (∑ k : Fin n, (f k + g k)) = (∑ k : Fin n, f k) + (∑ k : Fin n, g k) := by
        simp [Finset.sum_add_distrib, f, g]
      simpa [hdist, f, g] using hLowerSum_fg
    have hLowerSum_swapped :
        (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
              (Spec.toVec hi k)) +
          (∑ k : Fin n,
            (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
              (Spec.toVec lo k))
          ≤ (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hLowerSum'
    simp [toVec_add_spec, hW, hPos_lo, hNeg_hi, hLowerSum_swapped, add_comm]

  have hhi :
      Spec.toVec (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) W x) b) i
        ≤
      Spec.toVec
          (Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) hi)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) lo))
            b) i := by
    have hUpperSum' :
        (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) ≤
          (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (Spec.toVec hi k)) +
            (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (Spec.toVec lo k)) := by
      let f : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            hi k)
      let g : Fin n → ℝ :=
        fun k =>
          (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) * (Spec.toVec
            lo k)
      have hUpperSum_fg :
          (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) ≤ (∑ k : Fin n, (f k + g k)) := by
        simpa [f, g] using hUpperSum
      have hdist : (∑ k : Fin n, (f k + g k)) = (∑ k : Fin n, f k) + (∑ k : Fin n, g k) := by
        simp [Finset.sum_add_distrib, f, g]
      simpa [hdist, f, g] using hUpperSum_fg
    have hUpperSum_swapped :
        (∑ k : Fin n, (Spec.get2 W i k) * (Spec.toVec x k)) ≤
          (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n := n) W) i k) *
                (Spec.toVec lo k)) +
            (∑ k : Fin n,
              (Spec.get2 (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n := n) W) i k) *
                (Spec.toVec hi k)) := by
      simpa [add_comm, add_left_comm, add_assoc] using hUpperSum'
    simp [toVec_add_spec, hW, hPos_hi, hNeg_lo, hUpperSum_swapped, add_comm]

  exact ⟨hlo, hhi⟩

/-! ## ReLU relaxations used by α-CROWN -/

private lemma relu_ge_alpha_mul (a z : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    a * z ≤ Activation.Math.reluSpec (α := ℝ) z := by
  by_cases hz : z ≤ 0
  · have : a * z ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ha0 hz
    simpa [Activation.Math.reluSpec, max_eq_right hz] using this
  · have hz' : 0 ≤ z := le_of_not_ge hz
    have : a * z ≤ (1 : ℝ) * z := mul_le_mul_of_nonneg_right ha1 hz'
    simpa [Activation.Math.reluSpec, max_eq_left hz', one_mul] using this

private lemma alphaRelaxLowerScalar_sound
    (l u a x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    let rp := alphaRelaxLowerScalar (α := ℝ) l u a
    rp.slope * x + rp.bias ≤ Activation.Math.reluSpec (α := ℝ) x := by
  unfold alphaRelaxLowerScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec, max_eq_left hxnonneg]
    · simp [hu, hlpos, relu_ge_alpha_mul (a := a) (z := x) ha0 ha1]
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec, max_eq_right hxle]

private lemma relu_relax_scalar_upper_real_runtime
  (l u x : ℝ)
  (hlx : l ≤ x) (hxu : x ≤ u) :
  let rp := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α:=ℝ) l u
  Activation.Math.reluSpec (α:=ℝ) x ≤ rp.slope * x + rp.bias := by
  -- Same structure as `Models/mlp.lean`, but for `Runtime/Ops`.
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec, max_eq_left hxnonneg]
    · have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      simp only [hu, hlpos, if_true, if_false]
      by_cases hxpos : 0 < x
      · have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.Math.reluSpec, max_eq_left hxnonneg]
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          have hrewrite : (u - l) * x - u * (x - l) = l * (u - x) := by ring
          have hxux : 0 ≤ u - x := sub_nonneg.mpr hxu
          have hxmul_le : l * (u - x) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hle0 hxux
          have hmul_goal : (u - l) * x ≤ u * (x - l) := by
            have : (u - l) * x - u * (x - l) ≤ 0 := by simpa [hrewrite] using hxmul_le
            exact sub_nonpos.mp this
          have hx_to_goal' : x ≤ (u * (x - l)) / (u - l) := by
            have : x * (u - l) ≤ u * (x - l) := by simpa [mul_comm] using hmul_goal
            exact (le_div_iff₀ (G₀ := ℝ) hden).mpr this
          simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hx_to_goal'
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by ring
        simpa [h2] using hx_to_goal
      · have hxle : x ≤ 0 := le_of_not_gt hxpos
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · have : 0 ≤ u := le_of_lt hu
            exact div_nonneg this (le_of_lt hden)
          · linarith
        simpa [Activation.Math.reluSpec, max_eq_right hxle, h1] using this
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec, max_eq_right hxle]

private lemma relax_scalar_slope_nonneg (l u : ℝ) :
    0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u).slope := by
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · simp [hu, hlpos]
    · have hden : 0 < (u - l) := by
        have hl0 : l ≤ 0 := le_of_not_gt hlpos
        linarith
      have : 0 ≤ u / (u - l) := by
        have : 0 ≤ u := le_of_lt hu
        exact div_nonneg this (le_of_lt hden)
      simp [hu, hlpos, this]
  · simp [hu]

private lemma alphaRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ha0 : 0 ≤ a) :
    0 ≤ (alphaRelaxLowerScalar (α := ℝ) l u a).slope := by
  unfold alphaRelaxLowerScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · simp [hu, hlpos]
    · simp [hu, hlpos, ha0]
  · simp [hu]

/-! ## ReLU relaxations used by α/β-CROWN (β phase constraints) -/

private lemma phaseConsistentScalar?_inactive {l u : ℝ} :
    phaseConsistentScalar? (α := ℝ) l u ReLUPhase.inactive = some () → u ≤ 0 := by
  intro h
  unfold phaseConsistentScalar? at h
  by_cases hu : u > 0
  · simp [hu] at h
  ·
    have : ¬ (0 : ℝ) < u := by simpa using hu
    exact (not_lt).1 this

private lemma phaseConsistentScalar?_active {l u : ℝ} :
    phaseConsistentScalar? (α := ℝ) l u ReLUPhase.active = some () → 0 ≤ l := by
  intro h
  unfold phaseConsistentScalar? at h
  by_cases hl : l < 0
  · simp [hl] at h
  ·
    have : ¬ l < (0 : ℝ) := by simpa using hl
    exact (not_lt).1 this

private lemma phaseRelaxUpperScalar_slope_nonneg (l u : ℝ) (ph : ReLUPhase) :
    0 ≤ (phaseRelaxUpperScalar (α := ℝ) l u ph).slope := by
  cases ph <;> simp [phaseRelaxUpperScalar, relax_scalar_slope_nonneg]

private lemma phaseRelaxLowerScalar_slope_nonneg (l u a : ℝ) (ph : ReLUPhase) (ha0 : 0 ≤ a) :
    0 ≤ (phaseRelaxLowerScalar (α := ℝ) l u a ph).slope := by
  cases ph <;> simp [phaseRelaxLowerScalar, alphaRelaxLowerScalar_slope_nonneg, ha0]

private lemma phaseRelaxLowerScalar_sound
    (l u a x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (ph : ReLUPhase) (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxLowerScalar (α := ℝ) l u a ph
    rp.slope * x + rp.bias ≤ Activation.Math.reluSpec (α := ℝ) x := by
  cases ph with
  | inactive =>
      -- rp = 0, so this is `0 ≤ relu(x)`.
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec]
  | active =>
      have hl0 : (0 : ℝ) ≤ l := phaseConsistentScalar?_active (l := l) (u := u) hcons
      have hx0 : (0 : ℝ) ≤ x := le_trans hl0 hlx
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec, max_eq_left hx0]
  | unstable =>
      -- Reduce to α-CROWN's lower relaxation.
      simpa [phaseRelaxLowerScalar] using
        (alphaRelaxLowerScalar_sound (l := l) (u := u) (a := a) (x := x) hlx hxu ha0 ha1)

private lemma phaseRelaxUpperScalar_sound
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u)
    (ph : ReLUPhase) (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxUpperScalar (α := ℝ) l u ph
    Activation.Math.reluSpec (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := phaseConsistentScalar?_inactive (l := l) (u := u) hcons
      have hx0 : x ≤ 0 := le_trans hxu hu0
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec, max_eq_right hx0]
  | active =>
      have hl0 : (0 : ℝ) ≤ l := phaseConsistentScalar?_active (l := l) (u := u) hcons
      have hx0 : (0 : ℝ) ≤ x := le_trans hl0 hlx
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec, max_eq_left hx0]
  | unstable =>
      -- Reduce to the runtime upper relaxation.
      simpa [phaseRelaxUpperScalar] using
        (relu_relax_scalar_upper_real_runtime (l := l) (u := u) (x := x) hlx hxu)

/-! ## ReLU transfer helpers (toVec-level) -/

private lemma defaultAlphaVec_range {n : Nat}
    (lo hi : Tensor ℝ (.dim n .scalar)) :
    ∀ i : Fin n, (0 : ℝ) ≤ toVec (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ∧
      toVec (defaultAlphaVec (α := ℝ) (n := n) lo hi) i ≤ (1 : ℝ) := by
  classical
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      intro i
      cases hlo : flo i with
      | scalar l =>
        cases hhi : fhi i with
        | scalar u =>
          -- Default α is either 0 or 1.
          have hone : (Numbers.one : ℝ) = (1 : ℝ) := by rfl
          have hzero : (Numbers.zero : ℝ) = (0 : ℝ) := by rfl
          by_cases h : u > (-l)
          ·
            simp [defaultAlphaVec, toVec, hlo, hhi, h, hone, hzero]
          ·
            simp [defaultAlphaVec, toVec, hlo, hhi, h, hone, hzero]

private lemma toVec_relu_spec {n : Nat} (t : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
    toVec (Activation.reluSpec (α := ℝ) t) i =
      Activation.Math.reluSpec (α := ℝ) (toVec t i) := by
  cases t with
  | dim ft =>
      cases hti : ft i with
      | scalar x =>
          simp [Activation.reluSpec, Tensor.mapSpec, toVec, hti]

private lemma toVec_runtime_relu_relax_vector {n : Nat}
    (lo hi : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
    toVec (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n := n) lo hi) i =
      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) (toVec lo i) (toVec hi i) := by
  classical
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      cases hlo : flo i with
      | scalar l =>
        cases hhi : fhi i with
        | scalar u =>
          simp [NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector, toVec, hlo, hhi]

private lemma toVec_alphaRelaxLowerVec {n : Nat}
    (lo hi αv : Tensor ℝ (.dim n .scalar)) (i : Fin n) :
    toVec (alphaRelaxLowerVec (α := ℝ) (n := n) lo hi αv) i =
      alphaRelaxLowerScalar (α := ℝ) (toVec lo i) (toVec hi i) (toVec αv i) := by
  classical
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      cases αv with
      | dim fa =>
        cases hlo : flo i with
        | scalar l =>
          cases hhi : fhi i with
          | scalar u =>
            cases ha : fa i with
            | scalar a =>
              simp [alphaRelaxLowerVec, toVec, hlo, hhi, ha]

private lemma toVec_affineEvalAt_relu_propagate_affine
    {inDim hidDim : Nat}
    (relax : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) (.dim hidDim .scalar))
    (aff : AffineVec ℝ inDim hidDim)
    (x : Tensor ℝ (.dim inDim .scalar)) (i : Fin hidDim) :
    toVec
        (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := hidDim)
          (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
            (inDim := inDim) (hidDim := hidDim) relax aff) x) i
      =
      let rp := toVec relax i
      rp.slope *
          toVec (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := hidDim) aff x)
            i +
        rp.bias := by
  classical
  -- Reduce everything to scalar coordinates via `toVec_*` lemmas.
  cases relax with
  | dim r =>
    cases aff with
    | mk A c =>
      cases A with
      | dim rows =>
        cases c with
        | dim bias =>
          -- Pick out the relaxation parameter at index `i`.
          cases hri : r i with
          | scalar rp =>
            -- Expand both sides to sums over input coordinates.
            have hget2 :
                ∀ j : Fin inDim,
                  Spec.get2
                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                        (inDim := inDim) (hidDim := hidDim) (Tensor.dim r)
                          { A := Tensor.dim rows, c := Tensor.dim bias }).A
                      i j
                    =
                    Spec.get2 (Tensor.dim rows) i j * rp.slope := by
              intro j
              -- `propagate_affine` scales each matrix entry by `rp.slope`.
              cases hrow : rows i with
              | dim cols =>
                cases hcol : cols j with
                | scalar aij =>
                  simp [NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine, Spec.get2, Spec.get,
                    Spec.getAtSpec,
                    hri, hrow, hcol]
            have hc' :
                toVec
                    (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                      (inDim := inDim) (hidDim := hidDim) (Tensor.dim r)
                        { A := Tensor.dim rows, c := Tensor.dim bias }).c
                    i
                  =
                  rp.slope * toVec (Tensor.dim bias) i + rp.bias := by
              cases hbi : bias i with
              | scalar ci =>
                simp [NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine, toVec, hri, hbi]
            -- Compute both sides pointwise using `toVec_add_spec` and `toVec_mat_vec_mul_spec`.
            let A' :=
              (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                (inDim := inDim) (hidDim := hidDim) (Tensor.dim r)
                  { A := Tensor.dim rows, c := Tensor.dim bias }).A
            let c' :=
              (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                (inDim := inDim) (hidDim := hidDim) (Tensor.dim r)
                  { A := Tensor.dim rows, c := Tensor.dim bias }).c

            have hL_add :
                toVec (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) A' x) c') i
                  =
                  toVec (Spec.matVecMulSpec (α := ℝ) A' x) i + toVec c' i := by
              simp [Spec.toVec_add_spec]

            have hR_add :
                toVec (Tensor.addSpec (α := ℝ) (Spec.matVecMulSpec (α := ℝ) (Tensor.dim rows) x)
                  (Tensor.dim bias)) i
                  =
                  toVec (Spec.matVecMulSpec (α := ℝ) (Tensor.dim rows) x) i + toVec (Tensor.dim
                    bias) i := by
              simp [Spec.toVec_add_spec]

            -- Expand the mat-vec products.
            have hL_mat :
                toVec (Spec.matVecMulSpec (α := ℝ) A' x) i
                  =
                  ∑ k : Fin inDim, (Spec.get2 A' i k) * (toVec x k) := by
              exact (Spec.toVec_mat_vec_mul_spec (A := A') (v := x) (i := i))
            have hR_mat :
                toVec (Spec.matVecMulSpec (α := ℝ) (Tensor.dim rows) x) i
                  =
                  ∑ k : Fin inDim, (Spec.get2 (Tensor.dim rows) i k) * (toVec x k) := by
              exact (Spec.toVec_mat_vec_mul_spec (A := (Tensor.dim rows)) (v := x) (i := i))

            -- Rewrite the scaled-matrix sum to factor out `rp.slope`.
            have hscale :
                (∑ k : Fin inDim, (Spec.get2 (Tensor.dim rows) i k * rp.slope) * toVec x k)
                  =
                  rp.slope * (∑ k : Fin inDim, (Spec.get2 (Tensor.dim rows) i k) * toVec x k) := by
              calc
                (∑ k : Fin inDim, (Spec.get2 (Tensor.dim rows) i k * rp.slope) * toVec x k)
                    =
                    ∑ k : Fin inDim, rp.slope * ((Spec.get2 (Tensor.dim rows) i k) * toVec x k) :=
                      by
                      refine Finset.sum_congr rfl ?_
                      intro k hk
                      simp [mul_left_comm, mul_comm]
                _ = rp.slope * (∑ k : Fin inDim, (Spec.get2 (Tensor.dim rows) i k) * toVec x k) :=
                  by
                      simpa using
                        (Finset.mul_sum (a := rp.slope) (s := Finset.univ)
                          (f := fun k : Fin inDim => (Spec.get2 (Tensor.dim rows) i k) * toVec x
                            k)).symm

            have hscaleAlt :
                (∑ k : Fin inDim, rp.slope * (toVec x k * Spec.get2 (Tensor.dim rows) i k))
                  =
                  rp.slope * (∑ k : Fin inDim, toVec x k * Spec.get2 (Tensor.dim rows) i k) := by
              simpa [mul_assoc, mul_left_comm, mul_comm] using
                (Finset.mul_sum (a := rp.slope) (s := Finset.univ)
                  (f := fun k : Fin inDim => toVec x k * Spec.get2 (Tensor.dim rows) i k)).symm

            -- Put everything together.
            have hget2' :
                ∀ k : Fin inDim, Spec.get2 A' i k = (Spec.get2 (Tensor.dim rows) i k) * rp.slope :=
                  by
              intro k
              simpa [A'] using hget2 k

            -- Unfold `affineEvalAt` on both sides and finish by ring.
            have :
                toVec
                    (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim := hidDim)
                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                        (inDim := inDim) (hidDim := hidDim) (Tensor.dim r)
                          { A := Tensor.dim rows, c := Tensor.dim bias }) x) i
                  =
                  rp.slope *
                      toVec (CrownCertSoundness.affineEvalAt (α := ℝ) (inDim := inDim) (outDim :=
                        hidDim)
                        { A := Tensor.dim rows, c := Tensor.dim bias } x) i +
                    rp.bias := by
              -- Expand `affineEvalAt` and use the equalities above.
              simp [CrownCertSoundness.affineEvalAt, A', c', hL_add, hR_add, hL_mat, hR_mat, hc'] at
                *
              -- Replace `get2 A'` by `get2 rows * rp.slope`.
              simp [hget2'] at *
              -- Normalize the sum and finish.
              simp [mul_add, add_assoc, add_comm, mul_left_comm, mul_comm, hscaleAlt] at *
            simpa [A', c', toVec, hri] using this

/-!
β phase vectors (`AlphaBetaCROWN.phaseRelaxVec?`) are executable, so to reason about them we
extract their per-index consequences from the fact they returned `some ...`.
-/

private lemma List.all_eq_true_of_mem {α : Type} (p : α → Bool) (xs : List α) :
    xs.all p = true → ∀ x : α, x ∈ xs → p x = true := by
  intro hall
  induction xs with
  | nil =>
      intro x hx
      cases hx
  | cons a xs ih =>
      -- Unfold `List.all` without rewriting it into a `∀`-statement.
      have ha' : p a = true ∧ xs.all p = true := by
        simpa [List.all, Bool.and_eq_true] using hall
      rcases ha' with ⟨ha, hxs⟩
      intro x hx
      have hx' : x = a ∨ x ∈ xs := by
        simpa [List.mem_cons] using hx
      cases hx' with
      | inl hxa =>
          cases hxa
          simpa using ha
      | inr hxmem =>
          exact ih hxs x hxmem

private lemma phaseRelaxVec?_some_toVec {n : Nat}
    (lo hi αv : Tensor ℝ (.dim n .scalar)) (phases : Array Int)
    (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) (.dim n .scalar))
    (h : phaseRelaxVec? (α := ℝ) (n := n) lo hi αv phases = some (relaxLo, relaxHi)) :
    phases.size = n ∧
      ∀ i : Fin n,
        ∃ ph : ReLUPhase,
          phaseConsistentScalar? (α := ℝ) (toVec lo i) (toVec hi i) ph = some () ∧
          toVec relaxHi i = phaseRelaxUpperScalar (α := ℝ) (toVec lo i) (toVec hi i) ph ∧
          toVec relaxLo i = phaseRelaxLowerScalar (α := ℝ) (toVec lo i) (toVec hi i) (toVec αv i) ph
            := by
  classical
  by_cases hlen : phases.size = n
  · refine ⟨hlen, ?_⟩
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
          cases αv with
          | dim fa =>
            -- Expand the executable definition in `h`. A successful `some ...` return forces:
            -- (1) all per-index phase checks succeeded, and
            -- (2) the returned `relaxLo`/`relaxHi` are exactly the tensors constructed in the
            -- `then`-branch.
            have hS := h
            simp [phaseRelaxVec?, hlen] at hS
            rcases hS with ⟨hOkAll, hLoEq, hHiEq⟩

            intro i
            have hpTrue := hOkAll i

            cases hlo : flo i with
            | scalar l =>
            cases hhi : fhi i with
            | scalar u =>
                cases ha : fa i with
                | scalar a =>
                    cases hph : ReLUPhase.ofInt? (betaAt phases (↑i)) with
                    | none =>
                        have hph' : ReLUPhase.ofInt? (betaAt phases (↑i)) = none := by
                          simpa using hph
                        have : False := by
                          simp [hlo, hhi, ha, hph'] at hpTrue
                        exact False.elim this
                    | some ph =>
                        have hph' : ReLUPhase.ofInt? (betaAt phases (↑i)) = some ph := by
                          simpa using hph
                        cases hcons : phaseConsistentScalar? (α := ℝ) l u ph with
                        | none =>
                            have : False := by
                              simp [hlo, hhi, ha, hph', hcons] at hpTrue
                            exact False.elim this
                        | some u0 =>
                            cases u0
                            have hcons' : phaseConsistentScalar? (α := ℝ) l u ph = some () := by
                              simpa using hcons
                            refine ⟨ph, ?_, ?_, ?_⟩
                            · simpa [toVec, hlo, hhi] using hcons'
                            ·
                              -- Rewrite to the definitional `dim`-tensor produced by
                              -- `phaseRelaxVec?`.
                              rw [← hHiEq]
                              simp [toVec, hlo, hhi, hph']
                            ·
                              -- Rewrite to the definitional `dim`-tensor produced by
                              -- `phaseRelaxVec?`.
                              rw [← hLoEq]
                              simp [toVec, hlo, hhi, ha, hph']
  ·
    have hnone : phaseRelaxVec? (α := ℝ) (n := n) lo hi αv phases = none := by
      simp [phaseRelaxVec?, hlen]
    have : False := by
      simp [hnone] at h
    exact False.elim this

/-! ## Evaluating `linear_bounds_from_affine` at a point -/

private lemma get2_add_spec {m n : Nat}
    (A B : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (Tensor.addSpec (α := ℝ) A B) i j = Spec.get2 A i j + Spec.get2 B i j := by
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      cases hAi : rowsA i with
      | dim colsA =>
        cases hBi : rowsB i with
        | dim colsB =>
          cases hAj : colsA j with
          | scalar a =>
            cases hBj : colsB j with
            | scalar b =>
              simp [Tensor.addSpec, Tensor.map2Spec, Spec.get2_eq, Spec.get_eq, hAi, hBi, hAj,
                hBj]

private theorem mat_vec_add_matrix {m n : Nat}
    (A B : Tensor ℝ (.dim m (.dim n .scalar)))
    (x : Tensor ℝ (.dim n .scalar)) :
    Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x =
      Tensor.addSpec (α := ℝ)
        (Spec.matVecMulSpec (α := ℝ) A x)
        (Spec.matVecMulSpec (α := ℝ) B x) := by
  classical
  have htoVec :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B) x) =
        Spec.toVec
          (Tensor.addSpec (α := ℝ)
            (Spec.matVecMulSpec (α := ℝ) A x)
            (Spec.matVecMulSpec (α := ℝ) B x)) := by
    funext i
    rw [Spec.toVec_mat_vec_mul_spec (A := Tensor.addSpec (α := ℝ) A B) (v := x) (i := i)]
    simp [Spec.toVec_add_spec]
    rw [Spec.toVec_mat_vec_mul_spec (A := A) (v := x) (i := i)]
    rw [Spec.toVec_mat_vec_mul_spec (A := B) (v := x) (i := i)]
    -- Distribute `get2 (A+B)` and split the sum.
    have :
        (∑ k : Fin n, (Spec.get2 (Tensor.addSpec (α := ℝ) A B) i k) * (Spec.toVec x k)) =
          (∑ k : Fin n, (Spec.get2 A i k) * (Spec.toVec x k)) +
          (∑ k : Fin n, (Spec.get2 B i k) * (Spec.toVec x k)) := by
      classical
      calc
        (∑ k : Fin n, (Spec.get2 (Tensor.addSpec (α := ℝ) A B) i k) * (Spec.toVec x k))
            = ∑ k : Fin n, ((Spec.get2 A i k + Spec.get2 B i k) * (Spec.toVec x k)) := by
                refine Finset.sum_congr rfl ?_
                intro k _
                simp [get2_add_spec]
        _ = ∑ k : Fin n, ((Spec.get2 A i k) * (Spec.toVec x k) + (Spec.get2 B i k) * (Spec.toVec x
          k)) := by
              simp [add_mul]
        _ = (∑ k : Fin n, (Spec.get2 A i k) * (Spec.toVec x k)) +
            (∑ k : Fin n, (Spec.get2 B i k) * (Spec.toVec x k)) := by
              simp [Finset.sum_add_distrib]
    simp [this]
  have hTensor := congrArg Spec.ofVec htoVec
  simpa using
    (Eq.trans (Spec.ofVec_toVec (t := Spec.matVecMulSpec (α := ℝ) (Tensor.addSpec (α := ℝ) A B)
      x)).symm
      (Eq.trans hTensor (Spec.ofVec_toVec (t := Tensor.addSpec (α := ℝ)
        (Spec.matVecMulSpec (α := ℝ) A x)
        (Spec.matVecMulSpec (α := ℝ) B x)))))

private lemma mat_vec_mul_spec_fill_zero {m n : Nat}
    (x : Tensor ℝ (.dim n .scalar)) :
    Spec.matVecMulSpec (α := ℝ) (Spec.fill (α := ℝ) (0 : ℝ) (.dim m (.dim n .scalar))) x =
      Spec.fill (α := ℝ) (0 : ℝ) (.dim m .scalar) := by
  classical
  have htoVec :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ) (Spec.fill (α := ℝ) (0 : ℝ) (.dim m (.dim n
        .scalar))) x) =
        Spec.toVec (Spec.fill (α := ℝ) (0 : ℝ) (.dim m .scalar)) := by
    funext i
    -- Expand the mat-vec coordinate as a finite sum; all terms are zero.
    rw [Spec.toVec_mat_vec_mul_spec (A := Spec.fill (α := ℝ) (0 : ℝ) (.dim m (.dim n .scalar))) (v
      := x) (i := i)]
    simp [Spec.fill, Spec.get2_eq, Spec.get_eq, Spec.toVec]
  have hTensor := congrArg Spec.ofVec htoVec
  simpa using
    (Eq.trans (Spec.ofVec_toVec (t := Spec.matVecMulSpec (α := ℝ)
        (Spec.fill (α := ℝ) (0 : ℝ) (.dim m (.dim n .scalar))) x)).symm
      (Eq.trans hTensor (Spec.ofVec_toVec (t := Spec.fill (α := ℝ) (0 : ℝ) (.dim m .scalar)))))

private lemma mat_vec_mul_spec_aff_identity {n : Nat}
    (x : Tensor ℝ (.dim n .scalar)) :
    Spec.matVecMulSpec (α := ℝ) (affIdentity (α := ℝ) n).A x = x := by
  classical
  have htoVec :
      Spec.toVec (Spec.matVecMulSpec (α := ℝ) (affIdentity (α := ℝ) n).A x) = Spec.toVec x := by
    funext i
    -- Expand mat-vec coordinate; only the diagonal term survives.
    rw [Spec.toVec_mat_vec_mul_spec (A := (affIdentity (α := ℝ) n).A) (v := x) (i := i)]
    simp [affIdentity, Spec.get2_eq, Spec.get_eq, Spec.toVec]
  have hTensor := congrArg Spec.ofVec htoVec
  simpa using
    (Eq.trans (Spec.ofVec_toVec (t := Spec.matVecMulSpec (α := ℝ) (affIdentity (α := ℝ) n).A
      x)).symm
      (Eq.trans hTensor (Spec.ofVec_toVec (t := x))))

private lemma boundsEvalAt_bounds_identity {n : Nat} (x : Tensor ℝ (.dim n .scalar)) :
    boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) n) x = { dim := n, lo := x, hi := x } := by
  classical
  have hMat :
      Spec.matVecMulSpec (α := ℝ) (affIdentity (α := ℝ) n).A x = x :=
    mat_vec_mul_spec_aff_identity (n := n) x
  have hC : (affIdentity (α := ℝ) n).c = Spec.fill (α := ℝ) (0 : ℝ) (.dim n .scalar) := by
    simp [affIdentity]
  ext <;> simp [boundsEvalAt, boundsIdentity, affineEvalAt, hMat, hC]

private lemma boundsEvalAt_bounds_const {inDim outDim : Nat}
    (lo hi : Tensor ℝ (.dim outDim .scalar)) (x : Tensor ℝ (.dim inDim .scalar)) :
    boundsEvalAt (α := ℝ) (boundsConst (α := ℝ) inDim outDim lo hi) x =
      { dim := outDim
        lo := lo
        hi := hi } := by
  classical
  ext <;> simp [boundsEvalAt, boundsConst, affineEvalAt, mat_vec_mul_spec_fill_zero]

private lemma add_spec_left_comm {s : Shape}
    (a b c : Tensor ℝ s) :
    Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b c) =
      Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) a c) := by
  -- Derive left-commutativity from `add_spec_assoc` and `add_spec_comm`.
  calc
    Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b c)
        = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) c := by
            simpa using (add_spec_assoc (a := a) (b := b) (c := c)).symm
    _ = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) b a) c := by
            simp [add_spec_comm]
    _ = Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) a c) := by
            simpa using (add_spec_assoc (a := b) (b := a) (c := c))

private lemma add_spec_pair_distrib {s : Shape}
    (a b c d : Tensor ℝ s) :
    Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) (Tensor.addSpec (α := ℝ) c d) =
      Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a c) (Tensor.addSpec (α := ℝ) b d) := by
  calc
    Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a b) (Tensor.addSpec (α := ℝ) c d)
        = Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) c d)) :=
          by
            simpa using (add_spec_assoc (a := a) (b := b) (c := Tensor.addSpec (α := ℝ) c d))
    _ = Tensor.addSpec (α := ℝ) a (Tensor.addSpec (α := ℝ) c (Tensor.addSpec (α := ℝ) b d)) := by
            -- swap `b` and `c` inside `b + (c + d)`
            have hswap :
                Tensor.addSpec (α := ℝ) b (Tensor.addSpec (α := ℝ) c d) =
                  Tensor.addSpec (α := ℝ) c (Tensor.addSpec (α := ℝ) b d) :=
              add_spec_left_comm (a := b) (b := c) (c := d)
            simp [hswap]
    _ = Tensor.addSpec (α := ℝ) (Tensor.addSpec (α := ℝ) a c) (Tensor.addSpec (α := ℝ) b d) := by
            simpa using (add_spec_assoc (a := a) (b := c) (c := Tensor.addSpec (α := ℝ) b d)).symm

private lemma boundsEvalAt_linear_bounds_from_affine
    {n m : Nat}
    (W : Tensor ℝ (.dim m (.dim n .scalar)))
    (b : Tensor ℝ (.dim m .scalar))
    (xB : FlatAffineBounds ℝ)
    (hout : xB.outDim = n)
    (x : Tensor ℝ (.dim xB.inDim .scalar)) :
    boundsEvalAt (α := ℝ)
        (linearBoundsFromAffine (α := ℝ) (inDim := xB.inDim) (n := n) (m := m) W b xB hout) x =
      { dim := m
        lo :=
          let l := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n) (by simpa [hout] using
            xB.loAff) x
          let u := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n) (by simpa [hout] using
            xB.hiAff) x
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) l)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) u))
            b
        hi :=
          let l := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n) (by simpa [hout] using
            xB.loAff) x
          let u := affineEvalAt (α := ℝ) (inDim := xB.inDim) (outDim := n) (by simpa [hout] using
            xB.hiAff) x
          Tensor.addSpec (α := ℝ)
            (Tensor.addSpec (α := ℝ)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := m) (n :=
                n) W) u)
              (Spec.matVecMulSpec (α := ℝ) (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := m) (n :=
                n) W) l))
            b } := by
  classical
  -- This is algebraic: unfold and use linearity/associativity lemmas.
  refine FlatBox.ext' (α := ℝ) (B1 := boundsEvalAt (α := ℝ)
        (linearBoundsFromAffine (α := ℝ) (inDim := xB.inDim) (n := n) (m := m) W b xB hout) x)
      (B2 := _) rfl ?_ ?_
  · -- `lo`
    apply heq_of_eq
    simp [boundsEvalAt, affineEvalAt, linearBoundsFromAffine, castAffineOut,
      mat_vec_add_matrix, Spec.mat_vec_assoc, Spec.mat_vec_add]
    rw [(add_spec_assoc (a := _) (b := _) (c := b)).symm]
    apply congrArg (fun z => Tensor.addSpec (α := ℝ) z b)
    simpa using (add_spec_pair_distrib (a := _) (b := _) (c := _) (d := _))
  · -- `hi`
    apply heq_of_eq
    simp [boundsEvalAt, affineEvalAt, linearBoundsFromAffine, castAffineOut,
      mat_vec_add_matrix, Spec.mat_vec_assoc, Spec.mat_vec_add]
    rw [(add_spec_assoc (a := _) (b := _) (c := b)).symm]
    apply congrArg (fun z => Tensor.addSpec (α := ℝ) z b)
    simpa using (add_spec_pair_distrib (a := _) (b := _) (c := _) (d := _))

/-! ## Step wrapper -/

/-- Wrapper around `alphaCrownStepNode?` in the `CrownTransferSound` “step function” shape. -/
def stepAlpha (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ))) (alpha : Array (Option (FlatVec ℝ))) (ctx : AffineCtx) :
    Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ) :=
  fun cert id => alphaCrownStepNode? (α := ℝ) g.nodes ps ibp alpha cert ctx id

/-- Wrapper around `alphaBetaCrownStepNode?` in the `CrownTransferSound` “step function” shape. -/
def stepAlphaBeta (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatVec ℝ)))
    (beta : Array (Option (Array Int)))
    (ctx : AffineCtx) :
    Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ) :=
  fun cert id => alphaBetaCrownStepNode? (α := ℝ) g.nodes ps ibp alpha beta cert ctx id

/-! ## Main transfer theorem -/

/-!
In words (α-CROWN, pointwise, graph dialect).

Fix a graph `g`, parameters `ps`, an input point `x`, and a locally-consistent value semantics
array `vals` (i.e. `vals[id]` agrees with evaluating node `id` from its parents’ values).

Assume:

- the designated input node in `inputs` matches `x` (`InputsMatch`),
- the IBP boxes `ibp` enclose the semantic values in `vals` (`IBPEnclosesVals`), and
- the α parameters are well-formed (`AlphaOK`).

Then the concrete step function `alphaCrownStepNode?` satisfies the abstract
`CrownTransferSound` requirement: whenever every parent `p` is enclosed by its certificate entry,
the current node `id` is enclosed by the step-produced certificate entry as well.

This is the key lemma that lets `alphaCrownStepNode?` plug into the generic end-to-end checker
theorem in `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`.
-/
theorem alphaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatVec ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlpha g ps ibp alpha ctx) (cert := cert) := by
  classical
  intro id hid hparents
  cases hs : stepAlpha g ps ibp alpha ctx cert id <;> cases hv : vals[id]!
  all_goals simp
  case some.some b v =>
    -- Semantic evaluation at this node.
    have hEvalEq : vals[id]! = evalNode? g.nodes ps inputs vals id := hsem.2 id hid
    have hEvalSome : evalNode? g.nodes ps inputs vals id = some v := by
      have : some v = evalNode? g.nodes ps inputs vals id := by simpa [hv] using hEvalEq
      simpa using this.symm

    -- Helper: get parent enclosure when both cert/val are present.
    have parentEnc :
        ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
          ∀ (bp : FlatAffineBounds ℝ) (vp : Val),
            cert[p]! = some bp → vals[p]! = some vp → EnclosesAtInput (α := ℝ) ctx x bp vp := by
      intro p hp bp vp hbp hvp
      have h := hparents p hp
      simpa [hbp, hvp] using h

    -- Split by node kind, mirroring `alphaCrownStepNode?`.
    refine (match hk : (g.nodes[id]!).kind with
    | .input => by
        -- Step success forces `id = ctx.inputId` and `b = bounds_identity`.
        have hidCtx : id = ctx.inputId := by
          by_contra hne
          have hnone : stepAlpha g ps ibp alpha ctx cert id = none := by
            simp [stepAlpha, alphaCrownStepNode?, hk, hne]
          have : False := by
            have : (some b) = none := by
              simpa [hs.symm] using hnone
            cases this
          exact this
        subst hidCtx
        have hb : b = boundsIdentity (α := ℝ) ctx.inputDim := by
          have : some (boundsIdentity (α := ℝ) ctx.inputDim) = some b := by
            simpa [stepAlpha, alphaCrownStepNode?, hk] using hs
          cases this
          rfl
        subst hb
        -- Identify the semantic input value and relate it to `x`.
        rcases hinputs with ⟨vin, hmap, ⟨hdim, hxEq⟩⟩
        have hin : inputs[ctx.inputId]? = some vin := hmap
        have hvIn : v = vin := by
          have hiv : inputs[ctx.inputId]? = some v := by
            -- Unfold the evaluator for `.input`.
            simpa [CertSoundness.evalNode?, hk] using hEvalSome
          have : some vin = some v := by
            calc
              some vin = inputs[ctx.inputId]? := by simpa using hin.symm
              _ = some v := hiv
          injection this with h
          exact h.symm
        subst hvIn
        -- Prove enclosure of the point box at `x`.
        refine ⟨rfl, ?_⟩
        dsimp [CrownCertSoundness.EnclosesAtInput]
        simp [castDimScalar]
        refine ⟨hdim.symm, ?_⟩
        -- `v.v` is definitionally `x` by `InputsMatch`.
        have hx : castDimScalar (α := ℝ) hdim v.v = x := by
          simpa using hxEq
        -- Show enclosure by unfolding `boundsEvalAt` and simplifying the identity affine map.
        have hC :
            (affIdentity (α := ℝ) ctx.inputDim).c =
              Spec.fill (α := ℝ) (0 : ℝ) (.dim ctx.inputDim .scalar) := by
          simp [affIdentity]
        have hMat :
            Spec.matVecMulSpec (α := ℝ) (affIdentity (α := ℝ) ctx.inputDim).A x = x :=
          mat_vec_mul_spec_aff_identity (n := ctx.inputDim) x
        have hGoalX : Theorems.Semantics.encloses (α := ℝ)
            (boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim) x) x := by
          -- Unfold to a record and prove coordinatewise using `encloses_iff_toVec`.
          dsimp [boundsEvalAt, boundsIdentity]
          refine (encloses_iff_toVec (n := ctx.inputDim)
              (lo := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := ctx.inputDim)
                (affIdentity (α := ℝ) ctx.inputDim) x)
              (hi := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := ctx.inputDim)
                (affIdentity (α := ℝ) ctx.inputDim) x)
              (x := x)).2 ?_
          intro i
          constructor <;> simp [affineEvalAt, hMat, hC]
        have hxInput :
            boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim)
              (castDimScalar (α := ℝ) hdim v.v)
              =
            boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim) x := by
          exact congrArg (fun t => boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim) t)
            hx
        have hGoalV :
            Theorems.Semantics.encloses (α := ℝ)
              (boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim)
                (castDimScalar (α := ℝ) hdim v.v))
              (castDimScalar (α := ℝ) hdim v.v) := by
          have hGoalX' :=
            sem_encloses_value_eq
              (B := boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim) x)
              (hxy := hx.symm) hGoalX
          have hGoalV0 :=
            sem_encloses_of_eq (h := hxInput.symm) (x := castDimScalar (α := ℝ) hdim v.v) hGoalX'
          have hDim : congrArg FlatBox.dim hxInput.symm = rfl := by
            exact Subsingleton.elim _ _
          cases hDim
          exact hGoalV0
        have hAtCast :
            ∀ {hIn : ctx.inputDim = ctx.inputDim},
              Theorems.Semantics.encloses (α := ℝ)
                (boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim)
                  (castDimScalar (α := ℝ) hIn x))
                (castDimScalar (α := ℝ) hdim v.v) := by
          intro hIn
          cases hIn
          exact sem_encloses_value_eq
            (B := boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim) x)
            (hxy := hx.symm) hGoalX
        have hAtCastOut :
            ∀ {hIn : ctx.inputDim = ctx.inputDim} {hOut : v.n = ctx.inputDim},
              Theorems.Semantics.encloses (α := ℝ)
                (boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim)
                  (castDimScalar (α := ℝ) hIn x))
                (castDimScalar (α := ℝ) hOut v.v) := by
          intro hIn hOut
          exact sem_encloses_value_eq
            (B := boundsEvalAt (α := ℝ) (boundsIdentity (α := ℝ) ctx.inputDim)
              (castDimScalar (α := ℝ) hIn x))
            (hxy := (castDimScalar_proof_irrel (h₁ := hdim) (h₂ := hOut) (t := v.v)).symm)
            (hAtCast (hIn := hIn))
        simpa [castDimScalar] using (hAtCastOut (hIn := rfl) (hOut := hdim))

    | .const _ => by
        -- Both semantics and the step read `ps.constVals[id]?`.
        cases hcv : ps.constVals[id]? with
        | none =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hcv] at hs
        | some vc =>
            have hs' : some (boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v) = some b := by
              simpa [stepAlpha, alphaCrownStepNode?, hk, hcv] using hs
            have hv' : v = vc := by
              have hev : ps.constVals[id]? = some v := by
                simpa [CertSoundness.evalNode?, hk] using hEvalSome
              have : some v = some vc := by
                calc
                  some v = ps.constVals[id]? := by simpa using hev.symm
                  _ = some vc := hcv
              cases this
              rfl
            cases hs'
            -- Keep `vc` (substitute `v`, not `vc`).
            subst v
            -- Now `b` is the constant affine enclosure and `v = vc`.
            refine ⟨rfl, ?_⟩
            dsimp [CrownCertSoundness.EnclosesAtInput]
            simp [castDimScalar]
            refine ⟨rfl, ?_⟩
            simp [castDimScalar]
            -- The evaluated affine bounds are the point box `{vc.v}`.
            have hGoalV : Theorems.Semantics.encloses (α := ℝ)
                (boundsEvalAt (α := ℝ) (boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v) x) vc.v
                  := by
              -- Unfold to a record and prove coordinatewise using `encloses_iff_toVec`.
              dsimp [boundsEvalAt, boundsConst]
              refine (encloses_iff_toVec (n := vc.n)
                  (lo := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := vc.n)
                    { A := Spec.fill (α := ℝ) 0 (.dim vc.n (.dim ctx.inputDim .scalar)), c := vc.v }
                      x)
                  (hi := affineEvalAt (α := ℝ) (inDim := ctx.inputDim) (outDim := vc.n)
                    { A := Spec.fill (α := ℝ) 0 (.dim vc.n (.dim ctx.inputDim .scalar)), c := vc.v }
                      x)
                  (x := vc.v)).2 ?_
              intro i
              constructor <;> simp [affineEvalAt, mat_vec_mul_spec_fill_zero]
            simpa using hGoalV

    | .detach => by
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            -- Semantics returns the parent value; the step returns the parent affine bound.
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            -- From step success: `getAff? cert p1 = some b`.
            by_cases hltC : p1 < cert.size
            · have hbp : cert[p1]! = some b := by
                simpa [stepAlpha, alphaCrownStepNode?, hk, hps, NN.MLTheory.CROWN.Cert.getAff?,
                  hltC] using hs
              -- From semantic success: `getVal? vals p1 = some v`, so `vals[p1]! = some v`.
              have hval : CertSoundness.getVal? vals p1 = some v := by
                simpa [CertSoundness.evalNode?, hk, hps] using hEvalSome
              by_cases hltV : p1 < vals.size
              · have hvp : vals[p1]! = some v := by
                  simpa [CertSoundness.getVal?, hltV] using hval
                have hpar : EnclosesAtInput (α := ℝ) ctx x b v := parentEnc p1 hpMem b v hbp hvp
                simpa using hpar
              · have : CertSoundness.getVal? vals p1 = none := by
                  simp [CertSoundness.getVal?, hltV]
                have : False := by
                  simp [this] at hval
                exact False.elim this
            · -- If `p1` is out of bounds, `getAff?` is `none`, contradiction.
              have hnone : stepAlpha g ps ibp alpha ctx cert id = none := by
                simp [stepAlpha, alphaCrownStepNode?, hk, hps, NN.MLTheory.CROWN.Cert.getAff?, hltC]
              have : False := by
                have : (some b) = none := by
                  simpa [hs.symm] using hnone
                cases this
              exact False.elim this

    | .reshape _ _ => by
          -- Same as detach, but with an out-dimension cast (step checks it; semantics checks it).
          cases hps : (g.nodes[id]!).parents with
          | nil =>
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
          | cons p1 _ =>
              have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
              -- Extract step-side parent affine bound and the out-dimension check.
              have hs' := hs
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
              cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
              | none =>
                  -- Step would be `none`, contradicting `hs : some b = ...`.
                  have : False := by
                    simp [hxin] at hs'
                  exact False.elim this
              | some xin =>
                  by_cases hout : xin.outDim = (g.nodes[id]!).outShape.size
                  ·
                    have hbEq :
                        some
                          { inDim := xin.inDim
                            outDim := (g.nodes[id]!).outShape.size
                            loAff := castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                              (m' := (g.nodes[id]!).outShape.size) hout xin.loAff
                            hiAff := castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                              (m' := (g.nodes[id]!).outShape.size) hout xin.hiAff } = some b := by
                      simpa [hxin, hout] using hs'
                    cases hbEq
                    -- Extract semantic-side parent value and the out-dimension check.
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        -- Peel the evaluator's size check.
                        simp [hgv] at hEval'
                        by_cases hvout : vp.n = (g.nodes[id]!).outShape.size
                        · have hvEq :
                            some { n := (g.nodes[id]!).outShape.size
                                   v := castDimScalar (α := ℝ) hvout vp.v } = some v := by
                            simpa [hvout] using hEval'
                          cases hvEq
                          -- Convert `getAff?/getVal?` equalities into array lookup equalities for
                          -- `parentEnc`.
                          have hcertp : cert[p1]! = some xin := by
                            by_cases hltC : p1 < cert.size
                            · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                            · have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                              have : False := by
                                simp [this] at hxin
                              exact False.elim this
                          have hvpp : vals[p1]! = some vp := by
                            by_cases hltV : p1 < vals.size
                            · simpa [CertSoundness.getVal?, hltV] using hgv
                            · have : CertSoundness.getVal? vals p1 = none := by
                                simp [CertSoundness.getVal?, hltV]
                              have : False := by
                                simp [this] at hgv
                              exact False.elim this
                          have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                            parentEnc p1 hpMem xin vp hcertp hvpp
                          -- Reshape/flatten are value-preserving casts: transport the enclosure
                          -- across output-dim casts.
                          simpa using (enclosesAtInput_castOut (ctx := ctx) (x := x) (xin := xin)
                            (vp := vp)
                            (hout := hout) (hvout := hvout) hpar)
                        · -- Semantic reshape would be `none`, contradicting `hEvalSome`.
                          have : False := by
                            simp [hvout] at hEval'
                          exact False.elim this
                  ·
                    -- Step reshape would be `none`, contradicting `hs`.
                    have hcontra : (none : Option (FlatAffineBounds ℝ)) = some b := by
                      have hs'' := hs'
                      simp [hxin, hout] at hs''
                    cases hcontra

    | .flatten _ => by
          cases hps : (g.nodes[id]!).parents with
          | nil =>
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
          | cons p1 _ =>
              have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
              have hs' := hs
              simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
              cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
              | none =>
                  have : False := by
                    simp [hxin] at hs'
                  exact False.elim this
              | some xin =>
                  by_cases hout : xin.outDim = (g.nodes[id]!).outShape.size
                  · have hbEq :
                      some
                        { inDim := xin.inDim
                          outDim := (g.nodes[id]!).outShape.size
                          loAff := castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                            (m' := (g.nodes[id]!).outShape.size) hout xin.loAff
                          hiAff := castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim)
                            (m' := (g.nodes[id]!).outShape.size) hout xin.hiAff } = some b := by
                      simpa [hxin, hout] using hs'
                    cases hbEq
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        simp [hgv] at hEval'
                        by_cases hvout : vp.n = (g.nodes[id]!).outShape.size
                        · have hvEq :
                            some { n := (g.nodes[id]!).outShape.size
                                   v := castDimScalar (α := ℝ) hvout vp.v } = some v := by
                            simpa [hvout] using hEval'
                          cases hvEq
                          have hcertp : cert[p1]! = some xin := by
                            by_cases hltC : p1 < cert.size
                            · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                            · have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                              have : False := by
                                simp [this] at hxin
                              exact False.elim this
                          have hvpp : vals[p1]! = some vp := by
                            by_cases hltV : p1 < vals.size
                            · simpa [CertSoundness.getVal?, hltV] using hgv
                            · have : CertSoundness.getVal? vals p1 = none := by
                                simp [CertSoundness.getVal?, hltV]
                              have : False := by
                                simp [this] at hgv
                              exact False.elim this
                          have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                            parentEnc p1 hpMem xin vp hcertp hvpp
                          -- Reshape/flatten are value-preserving casts: transport the enclosure
                          -- across output-dim casts.
                          simpa using (enclosesAtInput_castOut (ctx := ctx) (x := x) (xin := xin)
                            (vp := vp)
                            (hout := hout) (hvout := hvout) hpar)
                        · have : False := by
                            simp [hvout] at hEval'
                          exact False.elim this
                  ·
                    have : False := by
                      simp [hxin, hout] at hs'
                    exact False.elim this

    | .linear => by
        -- Linear layer: sound by sign-splitting + parent enclosure.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            -- Step-side: extract the parent affine bound, parameters, and the out-dimension check.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hwb : ps.linearWB[id]? with
                | none =>
                    have : False := by
                      simp [hxin, hwb] at hs'
                    exact False.elim this
                | some p =>
                    by_cases hout : xin.outDim = p.n
                    ·
                      have hbEq :
                          some (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := p.n)
                            (m := p.m) p.w p.b xin hout) =
                            some b := by
                        simpa [hxin, hwb, hout] using hs'
                      cases hbEq

                      -- Semantic-side: extract parent value and the in-dimension check used by
                      -- `.linear`.
                      have hEval' := hEvalSome
                      simp [CertSoundness.evalNode?, hk, hps, hwb] at hEval'
                      cases hgv : CertSoundness.getVal? vals p1 with
                      | none =>
                          have : False := by
                            simp [hgv] at hEval'
                          exact False.elim this
                      | some vp =>
                          simp [hgv] at hEval'
                          by_cases hvIn : vp.n = p.n
                          ·
                            have hvEq :
                                some
                                  { n := p.m
                                    v :=
                                      Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                        (castDimScalar (α := ℝ) hvIn vp.v) } = some v := by
                              simpa [hvIn] using hEval'
                            cases hvEq

                            -- Turn `getAff?/getVal?` equalities into array lookup equalities to use
                            -- `parentEnc`.
                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this
                            have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                              parentEnc p1 hpMem xin vp hcertp hvpp

                            -- Unfold the parent enclosure to obtain a semantic enclosure of
                            -- `boundsEvalAt xin x'`.
                            rcases hpar with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                              castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                hinDim.symm x

                            -- Identify `l/u` as in `boundsEvalAt_linear_bounds_from_affine`.
                            let l : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n) (by simpa
                                [hout] using xin.loAff) x'
                            let u : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n) (by simpa
                                [hout] using xin.hiAff) x'
                            -- Cast the parent box to dimension `p.n` (using `hout`), and rewrite it
                            -- into `[l,u]`.
                            have hxCast :
                                Theorems.Semantics.encloses (α := ℝ) { dim := p.n, lo := l, hi := u
                                  }
                                  (castDimScalar (α := ℝ) hvIn vp.v) := by
                              have hx0 :=
                                sem_encloses_castDim
                                  (B := boundsEvalAt (α := ℝ) xin x')
                                  (h := hout)
                                  (x := castDimScalar (α := ℝ) hdimB.symm vp.v)
                                  hencB
                              have hvIn' : Eq.trans hdimB.symm hout = hvIn := by
                                exact Subsingleton.elim _ _
                              have hxCastVec :
                                  castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                    vp.v) =
                                    castDimScalar (α := ℝ) hvIn vp.v := by
                                have hx1 :
                                    castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                      vp.v) =
                                      castDimScalar (α := ℝ) (Eq.trans hdimB.symm hout) vp.v := by
                                  simpa using (castDimScalar_trans (h₁ := hdimB.symm) (h₂ := hout)
                                    (t := vp.v)).symm
                                simpa [hvIn'] using hx1
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo = l
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, l] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi = u
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, u] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              -- Avoid rewriting dependent boxes: reason componentwise via `toVec`.
                              have hx0' :=
                                (encloses_iff_toVec (n := p.n)
                                  (lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').lo)
                                  (hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').hi)
                                  (x := castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ)
                                    hdimB.symm vp.v))).1 hx0
                              refine (encloses_iff_toVec (n := p.n) (lo := l) (hi := u)
                                (x := castDimScalar (α := ℝ) hvIn vp.v)).2 ?_
                              intro i
                              have hi := hx0' i
                              constructor
                              · exact (by simpa [hl, hxCastVec] using hi.1)
                              · exact (by simpa [hu, hxCastVec] using hi.2)

                            -- Apply sign-splitting enclosure at the point `xv`.
                            let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) hvIn vp.v
                            have hy :
                                Theorems.Semantics.encloses (α := ℝ)
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        p.b
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        p.b }
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv) :=
                                    by
                              -- `encloses_linear_signSplit` encloses `W·x + b`, and `linear_spec`
                              -- is definitional.
                              simpa [Spec.linearSpec, xv] using
                                (encloses_linear_signSplit (m := p.m) (n := p.n) (W := p.w) (b :=
                                  p.b)
                                  (lo := l) (hi := u) (x := xv) hxCast)

                            -- Rewrite the computed bounds to `boundsEvalAt
                            -- (linear_bounds_from_affine ...) x'`.
                            have hBE :
                                boundsEvalAt (α := ℝ)
                                  (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                    p.n) (m := p.m) p.w p.b xin hout) x' =
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        p.b
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        p.b } := by
                              -- This is exactly `boundsEvalAt_linear_bounds_from_affine`.
                              simpa [l, u, x'] using
                                (boundsEvalAt_linear_bounds_from_affine (n := p.n) (m := p.m) (W :=
                                  p.w) (b := p.b) (xB := xin)
                                  (hout := hout) (x := x'))

                            -- Package the result as `EnclosesAtInput`.
                            refine ⟨hinDim, ?_⟩
                            dsimp [CrownCertSoundness.EnclosesVec]
                            refine ⟨rfl, ?_⟩
                            -- Transport the enclosure `hy` across the `boundsEvalAt` equality
                            -- without rewriting in a dependent motive.
                            have hyCast :=
                              sem_encloses_of_eq (h := hBE.symm)
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv)
                                  hy
                            have hdim : congrArg FlatBox.dim hBE.symm = rfl := by
                              exact Subsingleton.elim _ _
                            have hyBout :
                                Theorems.Semantics.encloses (α := ℝ)
                                  (boundsEvalAt (α := ℝ)
                                    (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                      p.n) (m := p.m) p.w p.b xin hout)
                                    x')
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv) :=
                                    by
                              simpa [hdim, castDimScalar] using hyCast
                            simpa [castDimScalar, Spec.linearSpec, xv] using hyBout
                          ·
                            have : False := by
                              simp [hvIn] at hEval'
                            exact False.elim this
                    ·
                      have : False := by
                        simp [hxin, hwb, hout] at hs'
                      exact False.elim this

    | .matmul => by
        -- Matmul is linear with zero bias; same proof strategy as `.linear`.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hwb : ps.matmulW[id]? with
                | none =>
                    have : False := by
                      simp [hxin, hwb] at hs'
                    exact False.elim this
                | some p =>
                    by_cases hout : xin.outDim = p.n
                    ·
                      have hbEq :
                          some (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := p.n)
                            (m := p.m) p.w
                            (Spec.fill (α := ℝ) 0 (.dim p.m .scalar)) xin hout) = some b := by
                        simpa [hxin, hwb, hout] using hs'
                      cases hbEq

                      have hEval' := hEvalSome
                      simp [CertSoundness.evalNode?, hk, hps, hwb] at hEval'
                      cases hgv : CertSoundness.getVal? vals p1 with
                      | none =>
                          have : False := by
                            simp [hgv] at hEval'
                          exact False.elim this
                      | some vp =>
                          simp [hgv] at hEval'
                          by_cases hvIn : vp.n = p.n
                          ·
                            have hvEq :
                                some
                                  { n := p.m
                                    v :=
                                      Spec.linearSpec (α := ℝ)
                                        { weights := p.w, bias := Spec.fill (α := ℝ) 0 (.dim p.m
                                          .scalar) }
                                        (castDimScalar (α := ℝ) hvIn vp.v) } = some v := by
                              simpa [hvIn] using hEval'
                            cases hvEq

                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this
                            have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                              parentEnc p1 hpMem xin vp hcertp hvpp

                            rcases hpar with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                              castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                hinDim.symm x
                            let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m
                              .scalar)
                            let l : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n) (by simpa
                                [hout] using xin.loAff) x'
                            let u : Tensor ℝ (.dim p.n .scalar) :=
                              affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := p.n) (by simpa
                                [hout] using xin.hiAff) x'
                            have hxCast :
                                Theorems.Semantics.encloses (α := ℝ) { dim := p.n, lo := l, hi := u
                                  }
                                  (castDimScalar (α := ℝ) hvIn vp.v) := by
                              have hx0 :=
                                sem_encloses_castDim
                                  (B := boundsEvalAt (α := ℝ) xin x')
                                  (h := hout)
                                  (x := castDimScalar (α := ℝ) hdimB.symm vp.v)
                                  hencB
                              have hvIn' : Eq.trans hdimB.symm hout = hvIn := by
                                exact Subsingleton.elim _ _
                              have hxCastVec :
                                  castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                    vp.v) =
                                    castDimScalar (α := ℝ) hvIn vp.v := by
                                have hx1 :
                                    castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ) hdimB.symm
                                      vp.v) =
                                      castDimScalar (α := ℝ) (Eq.trans hdimB.symm hout) vp.v := by
                                  simpa using (castDimScalar_trans (h₁ := hdimB.symm) (h₂ := hout)
                                    (t := vp.v)).symm
                                simpa [hvIn'] using hx1
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo = l
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, l] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi = u
                                    := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, u] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              -- Avoid rewriting dependent boxes: reason componentwise via `toVec`.
                              have hx0' :=
                                (encloses_iff_toVec (n := p.n)
                                  (lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').lo)
                                  (hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                    x').hi)
                                  (x := castDimScalar (α := ℝ) hout (castDimScalar (α := ℝ)
                                    hdimB.symm vp.v))).1 hx0
                              refine (encloses_iff_toVec (n := p.n) (lo := l) (hi := u)
                                (x := castDimScalar (α := ℝ) hvIn vp.v)).2 ?_
                              intro i
                              have hi := hx0' i
                              constructor
                              · exact (by simpa [hl, hxCastVec] using hi.1)
                              · exact (by simpa [hu, hxCastVec] using hi.2)

                            let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) hvIn vp.v
                            have hy :
                                Theorems.Semantics.encloses (α := ℝ)
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        z
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        z }
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv) := by
                              simpa [Spec.linearSpec, z, xv] using
                                (encloses_linear_signSplit (m := p.m) (n := p.n) (W := p.w) (b := z)
                                  (lo := l) (hi := u) (x := xv) hxCast)

                            have hBE :
                                boundsEvalAt (α := ℝ)
                                  (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                    p.n) (m := p.m) p.w z xin hout) x' =
                                  { dim := p.m
                                    lo :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u))
                                        z
                                    hi :=
                                      Tensor.addSpec (α := ℝ)
                                        (Tensor.addSpec (α := ℝ)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) u)
                                          (Spec.matVecMulSpec (α := ℝ)
                                            (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := p.m) (n :=
                                              p.n) p.w) l))
                                        z } := by
                              simpa [l, u, x', z] using
                                (boundsEvalAt_linear_bounds_from_affine (n := p.n) (m := p.m) (W :=
                                  p.w) (b := z) (xB := xin)
                                  (hout := hout) (x := x'))

                            refine ⟨hinDim, ?_⟩
                            dsimp [CrownCertSoundness.EnclosesVec]
                            refine ⟨rfl, ?_⟩
                            -- Transport the enclosure `hy` across the `boundsEvalAt` equality
                            -- without rewriting in a dependent motive.
                            have hyCast :=
                              sem_encloses_of_eq (h := hBE.symm)
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv) hy
                            have hdim : congrArg FlatBox.dim hBE.symm = rfl := by
                              exact Subsingleton.elim _ _
                            have hyBout :
                                Theorems.Semantics.encloses (α := ℝ)
                                  (boundsEvalAt (α := ℝ)
                                    (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                                      p.n) (m := p.m) p.w z xin hout)
                                    x')
                                  (Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv) := by
                              simpa [hdim, castDimScalar] using hyCast
                            -- Keep the `linear_spec` form (it matches the semantic evaluator for
                            -- `.matmul`).
                            simpa [castDimScalar, xv, z] using hyBout
                          ·
                            have : False := by
                              simp [hvIn] at hEval'
                            exact False.elim this
                    ·
                      have : False := by
                        simp [hxin, hwb, hout] at hs'
                      exact False.elim this

    | .sum => by
        -- Sum is a 1×n linear layer with all-ones weights and zero bias.
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]

            -- Step-side: extract the parent affine bound.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                -- Step forces `b` to be `linear_bounds_from_affine onesRow 0 xin`.
                let onesRow : Tensor ℝ (.dim 1 (.dim xin.outDim .scalar)) :=
                  Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim xin.outDim .scalar))
                let zb : Tensor ℝ (.dim 1 .scalar) :=
                  Spec.fill (α := ℝ) (0 : ℝ) (.dim 1 .scalar)
                have hbEq :
                    some
                        (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := xin.outDim)
                          (m := 1)
                          onesRow zb xin (by rfl)) =
                      some b := by
                  simpa [hxin, onesRow, zb] using hs'
                cases hbEq

                -- Semantic-side: extract the parent value.
                have hEval' := hEvalSome
                simp [CertSoundness.evalNode?, hk, hps] at hEval'
                cases hgv : CertSoundness.getVal? vals p1 with
                | none =>
                    have : False := by
                      simp [hgv] at hEval'
                    exact False.elim this
                | some vp =>
                    -- `v` is exactly the mat-vec multiply by an all-ones row.
                    have hvEq :
                        some
                            { n := 1
                              v :=
                                Spec.matVecMulSpec (α := ℝ)
                                  (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v } =
                          some v := by
                      simpa [hgv] using hEval'
                    cases hvEq

                    -- Connect `getAff?/getVal?` equalities to array lookups to use `parentEnc`.
                    have hcertp : cert[p1]! = some xin := by
                      by_cases hltC : p1 < cert.size
                      · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                      ·
                        have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                          simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                        have : False := by
                          simp [this] at hxin
                        exact False.elim this
                    have hvpp : vals[p1]! = some vp := by
                      by_cases hltV : p1 < vals.size
                      · simpa [CertSoundness.getVal?, hltV] using hgv
                      ·
                        have : CertSoundness.getVal? vals p1 = none := by
                          simp [CertSoundness.getVal?, hltV]
                        have : False := by
                          simp [this] at hgv
                        exact False.elim this
                    have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                      parentEnc p1 hpMem xin vp hcertp hvpp

                    -- Unpack parent enclosure.
                    rcases hpar with ⟨hinDim, hvec⟩
                    dsimp at hvec
                    rcases hvec with ⟨hdimB, hencB⟩
                    let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                      castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim) hinDim.symm x
                    let l : Tensor ℝ (.dim xin.outDim .scalar) :=
                      affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := xin.outDim) xin.loAff x'
                    let u : Tensor ℝ (.dim xin.outDim .scalar) :=
                      affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := xin.outDim) xin.hiAff x'
                    let xv : Tensor ℝ (.dim xin.outDim .scalar) :=
                      castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdimB.symm vp.v

                    have hxCast :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := xin.outDim, lo := l, hi := u } xv := by
                      -- `boundsEvalAt` is definitional in terms of `affineEvalAt`.
                      simpa [CrownCertSoundness.boundsEvalAt, CrownCertSoundness.affineEvalAt, l, u,
                        xv, x'] using hencB

                    -- Apply sign-splitting enclosure (bias is zero, so use the mat-vec form).
                    have hy0 :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb }
                          (Spec.linearSpec (α := ℝ) { weights := onesRow, bias := zb } xv) := by
                      simpa [Spec.linearSpec, onesRow, zb, xv] using
                        (encloses_linear_signSplit (m := 1) (n := xin.outDim) (W := onesRow) (b :=
                          zb)
                          (lo := l) (hi := u) (x := xv) hxCast)
                    have hy :
                        Theorems.Semantics.encloses (α := ℝ)
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb }
                          (Spec.matVecMulSpec (α := ℝ) onesRow xv) := by
                      -- Convert `linear_spec` to mat-vec since the bias is zero.
                      simpa [onesRow, zb, linear_spec_bias_zero_eq_matvec] using hy0

                    -- Rewrite the computed bounds to `boundsEvalAt (linear_bounds_from_affine ...)
                    -- x'`.
                    have hBE :
                        boundsEvalAt (α := ℝ)
                          (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := xin.outDim)
                            (m := 1)
                            onesRow zb xin (by rfl)) x' =
                          { dim := 1
                            lo :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u))
                                zb
                            hi :=
                              Tensor.addSpec (α := ℝ)
                                (Tensor.addSpec (α := ℝ)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matPos (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) u)
                                  (Spec.matVecMulSpec (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.matNeg (α := ℝ) (m := 1) (n :=
                                      xin.outDim) onesRow) l))
                                zb } := by
                      simpa [l, u, x'] using
                        (boundsEvalAt_linear_bounds_from_affine (n := xin.outDim) (m := 1) (W :=
                          onesRow) (b := zb)
                          (xB := xin) (hout := (by rfl)) (x := x'))

                    refine ⟨hinDim, ?_⟩
                    dsimp [CrownCertSoundness.EnclosesVec]
                    refine ⟨rfl, ?_⟩
                    -- Transport the enclosure across `hBE` without rewriting in a dependent motive.
                    have hyCast :=
                      sem_encloses_of_eq (h := hBE.symm) (x := Spec.matVecMulSpec (α := ℝ)
                        onesRow xv) hy
                    have hdim : congrArg FlatBox.dim hBE.symm = rfl := by
                      exact Subsingleton.elim _ _
                    -- Finally, match the semantic evaluator's `onesRow` (built from `vp.n`).
                    have hvOut :
                        Spec.matVecMulSpec (α := ℝ) onesRow xv =
                          Spec.matVecMulSpec (α := ℝ)
                            (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v := by
                      have hdn : xin.outDim = vp.n := by
                        simpa [CrownCertSoundness.boundsEvalAt] using hdimB
                      have hxv : xv = castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm
                        vp.v := by
                        have : hdimB.symm = hdn.symm := by exact Subsingleton.elim _ _
                        simp [xv]
                      have hcast :=
                        mat_vec_mul_fill1_castDimScalar (h := hdn.symm) (v := vp.v)
                      simpa [onesRow, hxv] using hcast.symm
                    have hyBout :
                        Theorems.Semantics.encloses (α := ℝ)
                          (boundsEvalAt (α := ℝ)
                            (linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n :=
                              xin.outDim) (m := 1)
                              onesRow zb xin (by rfl)) x')
                          (Spec.matVecMulSpec (α := ℝ)
                            (Spec.fill (α := ℝ) (1 : ℝ) (.dim 1 (.dim vp.n .scalar))) vp.v) := by
                      simpa [x', hvOut] using hyCast
                    exact hyBout

    | .relu => by
        cases hps : (g.nodes[id]!).parents with
        | nil =>
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
        | cons p1 _ =>
            have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]

            -- Step-side: extract parent affine bounds and IBP pre-activation box.
            have hs' := hs
            simp [stepAlpha, alphaCrownStepNode?, hk, hps] at hs'
            cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
            | none =>
                have : False := by
                  simp [hxin] at hs'
                exact False.elim this
            | some xin =>
                cases hpre : ibp[p1]! with
                | none =>
                    have : False := by
                      simp [hxin, hpre] at hs'
                    exact False.elim this
                | some preB =>
                    -- Semantic-side: extract the parent value `vp`.
                    have hEval' := hEvalSome
                    simp [CertSoundness.evalNode?, hk, hps] at hEval'
                    cases hgv : CertSoundness.getVal? vals p1 with
                    | none =>
                        have : False := by
                          simp [hgv] at hEval'
                        exact False.elim this
                    | some vp =>
                        have hvEq :
                            some { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } = some v :=
                              by
                          simpa [hgv] using hEval'
                        cases hvEq

                        -- Connect `getAff?/getVal?` equalities to array lookups to use `parentEnc`.
                        have hcertp : cert[p1]! = some xin := by
                          by_cases hltC : p1 < cert.size
                          · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                          ·
                            have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                              simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                            have : False := by
                              simp [this] at hxin
                            exact False.elim this
                        have hvpp : vals[p1]! = some vp := by
                          by_cases hltV : p1 < vals.size
                          · simpa [CertSoundness.getVal?, hltV] using hgv
                          ·
                            have : CertSoundness.getVal? vals p1 = none := by
                              simp [CertSoundness.getVal?, hltV]
                            have : False := by
                              simp [this] at hgv
                            exact False.elim this
                        have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
                          parentEnc p1 hpMem xin vp hcertp hvpp

                        -- IBP enclosure for the pre-activation box `preB` at parent `p1`.
                        have hp1ltId : p1 < id := htopo id hid p1 hpMem
                        have hp1ltNodes : p1 < g.nodes.size := lt_trans hp1ltId hid
                        have hp1ltVals : p1 < vals.size := by simpa [hsem.1] using hp1ltNodes
                        have hibpP := hibp p1 hp1ltVals
                        have hEncIbp : CertSoundness.EnclosesBox preB vp := by
                          simpa [hpre, hvpp] using hibpP
                        rcases hEncIbp with ⟨hdimIbp, hencIbp⟩

                        -- Unpack parent enclosure.
                        rcases hpar with ⟨hinDim, hvec⟩
                        dsimp at hvec
                        rcases hvec with ⟨hdimB, hencB⟩
                        have hdn : xin.outDim = vp.n := by
                          simpa [CrownCertSoundness.boundsEvalAt] using hdimB

                        -- Both α-cases share the same main proof, only α-vector differs.
                        cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id with
                        | none =>
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                              let bout : FlatAffineBounds ℝ :=
                                { inDim := xin.inDim
                                  outDim := preB.dim
                                  loAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                        αt)
                                      (by simpa [hout] using xin.loAff)
                                  hiAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n :=
                                        preB.dim) preB.lo preB.hi)
                                      (by simpa [hout] using xin.hiAff) }
                              have hbEq : some bout = some b := by
                                simpa [hxin, hpre, hαopt, hout, bout, αt] using hs'
                              have hb : b = bout := (Option.some.inj hbEq).symm

                              let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.hiAff
                              let lAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xLo
                                  x'
                              let uAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xHi
                                  x'
                              let z : Tensor ℝ (.dim preB.dim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                  vp.v
                              have hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤
                                (1 : ℝ) := by
                                simpa [αt] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                  preB.hi)

                              -- Derive `lAff ≤ z ≤ uAff` by casting the parent enclosure.
                              let zXin : Tensor ℝ (.dim xin.outDim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm vp.v
                              have hzXin :
                                  Theorems.Semantics.encloses (α := ℝ) (boundsEvalAt (α := ℝ) xin
                                    x') zXin := by
                                have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                  exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                    hdn.symm) (t := vp.v)
                                simpa [this, zXin] using hencB
                              have hzCast0 :=
                                sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h := hout)
                                  (x := zXin) hzXin
                              have hzCastZ :
                                  castDimScalar (α := ℝ) hout zXin = z := by
                                have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                  exact Subsingleton.elim _ _
                                -- Cast composition aligns with `z` (up to proof irrelevance).
                                have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                  vp.v)).symm
                                simpa [zXin, z, htrans] using this
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                    lAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                    uAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              have hzAff :
                                  Theorems.Semantics.encloses (α := ℝ)
                                    { dim := preB.dim
                                      lo := lAff
                                      hi := uAff } z := by
                                have hzCast1 :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').lo
                                        hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').hi }
                                      z := by
                                  exact sem_encloses_value_eq
                                    (B := { dim := preB.dim
                                            lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                              xin x').lo
                                            hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                              xin x').hi })
                                    (hxy := hzCastZ) hzCast0
                                have hBoxEq :
                                    ({ dim := preB.dim
                                       lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').lo
                                       hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').hi } : FlatBox ℝ)
                                      =
                                    ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) := by
                                  refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                  · exact heq_of_eq hl
                                  · exact heq_of_eq hu
                                exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                              have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                simpa [CertSoundness.encloses, z] using hencIbp

                              -- Now prove the ReLU enclosure for `boundsEvalAt bout x'`.
                              rw [hb]
                              refine ⟨hinDim, ?_⟩
                              dsimp [CrownCertSoundness.EnclosesVec]
                              refine ⟨hdimIbp, ?_⟩
                              have hreluCast :
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                      (Activation.reluSpec (α := ℝ) vp.v)
                                    =
                                    Activation.reluSpec (α := ℝ) z := by
                                simpa [z] using
                                  (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))
                              -- Reduce to showing `Semantics.encloses` for `Activation.relu_spec
                              -- z`.
                              have :
                                  Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x') (Activation.reluSpec (α := ℝ)
                                        z) := by
                                -- Componentwise enclosure.
                                have hzAffI := (encloses_iff_toVec (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_toVec (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                refine (encloses_iff_toVec (n := preB.dim)
                                  (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                  (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                  (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                intro i
                                have hzLo := (hzAffI i).1
                                have hzHi := (hzAffI i).2
                                have hzIlo := (hzIbpI i).1
                                have hzIhi := (hzIbpI i).2
                                let li := toVec preB.lo i
                                let ui := toVec preB.hi i
                                let zi := toVec z i
                                let ai := toVec αt i
                                have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2
                                have hsLo : 0 ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope :=
                                  alphaRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a := ai)
                                    hai0
                                have hsHi : 0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α
                                  := ℝ) li ui).slope :=
                                  relax_scalar_slope_nonneg (l := li) (u := ui)

                                -- Rewrite the output bounds at index `i`.
                                have hlo_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i
                                      =
                                      let rp := toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim)
                                        preB.lo preB.hi αt) i
                                      rp.slope * toVec lAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                        preB.hi αt)
                                      (aff := xLo) (x := x') (i := i))
                                have hhi_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').hi i
                                      =
                                      let rp := toVec
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n
                                        := preB.dim) preB.lo preB.hi) i
                                      rp.slope * toVec uAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α :=
                                        ℝ) (n := preB.dim) preB.lo preB.hi)
                                      (aff := xHi) (x := x') (i := i))

                                have hrpLo :
                                    toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt) i
                                      =
                                      alphaRelaxLowerScalar (α := ℝ) li ui ai := by
                                  simpa [li, ui, ai] using
                                    (toVec_alphaRelaxLowerVec (lo := preB.lo) (hi := preB.hi) (αv :=
                                      αt) (i := i))
                                have hrpHi :
                                    toVec (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ)
                                      (n := preB.dim) preB.lo preB.hi) i
                                      =
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li ui
                                        := by
                                  simpa [li, ui] using (toVec_runtime_relu_relax_vector (lo :=
                                    preB.lo) (hi := preB.hi) (i := i))

                                -- Lower bound inequality.
                                have hlo1 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec lAff i +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias := by
                                  have hm : (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec
                                    lAff i
                                      ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi := by
                                    exact mul_le_mul_of_nonneg_left hzLo hsLo
                                  have h' :=
                                    add_le_add_right hm (alphaRelaxLowerScalar (α := ℝ) li ui
                                      ai).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hlo2 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  -- Use the scalar α-relaxation soundness on the true
                                  -- pre-activation value.
                                  have := alphaRelaxLowerScalar_sound (l := li) (u := ui) (a := ai)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi) (ha0 := hai0) (ha1 := hai1)
                                  simpa [li, ui, zi, ai] using this
                                have hlo :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  -- Rewrite `lo` and chain inequalities.
                                  simp [hlo_def, hrpLo]
                                  exact le_trans hlo1 hlo2

                                -- Upper bound inequality.
                                have hhi1 :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have := relu_relax_scalar_upper_real_runtime (l := li) (u := ui)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi)
                                  simpa [li, ui, zi] using this
                                have hhi2 :
                                    (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias
                                      ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have hm :
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi
                                        ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i := by
                                    exact mul_le_mul_of_nonneg_left hzHi hsHi
                                  have h' :=
                                    add_le_add_right hm
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hhi :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤ toVec (boundsEvalAt (α
                                      := ℝ) bout x').hi i := by
                                  simp [hhi_def, hrpHi]
                                  exact le_trans hhi1 hhi2

                                -- Combine, using `toVec` of ReLU.
                                have hrelu : toVec (Activation.reluSpec (α := ℝ) z) i =
                                    Activation.Math.reluSpec (α := ℝ) zi := by
                                  simpa [zi] using (toVec_relu_spec (t := z) (i := i))
                                constructor
                                · simpa [hrelu] using hlo
                                · simpa [hrelu] using hhi
                              -- Apply `hreluCast` to match the required casted value.
                              have this' :
                                  Theorems.Semantics.encloses (boundsEvalAt (α := ℝ) bout x')
                                    (Activation.reluSpec (α := ℝ) z) := by
                                simpa [x'] using this
                              exact sem_encloses_value_eq
                                (B := boundsEvalAt (α := ℝ) bout x')
                                (hxy := hreluCast.symm) this'
                            ·
                              have : False := by
                                simp [hxin, hpre, hαopt, hout] at hs'
                              exact False.elim this
                        | some αv =>
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                if hα : αv.n = preB.dim then
                                  castDimScalar (α := ℝ) (n := αv.n) (n' := preB.dim) hα αv.v
                                else
                                  defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                              let bout : FlatAffineBounds ℝ :=
                                { inDim := xin.inDim
                                  outDim := preB.dim
                                  loAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                        αt)
                                      (by simpa [hout] using xin.loAff)
                                  hiAff :=
                                    NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                      (inDim := xin.inDim) (hidDim := preB.dim)
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n :=
                                        preB.dim) preB.lo preB.hi)
                                      (by simpa [hout] using xin.hiAff) }
                              have hbEq : some bout = some b := by
                                simpa [hxin, hpre, hαopt, hout, bout, αt] using hs'
                              have hb : b = bout := (Option.some.inj hbEq).symm
                              have hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤
                                (1 : ℝ) := by
                                classical
                                by_cases hα : αv.n = preB.dim
                                ·
                                  -- Use `AlphaOK` for the provided α vector.
                                  have hidA : id < alpha.size := by
                                    by_cases hlt : id < alpha.size
                                    · exact hlt
                                    ·
                                      have : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id =
                                        none := by
                                        simp [NN.MLTheory.CROWN.Cert.getAlpha?, hlt]
                                      have : False := by
                                        simp [this] at hαopt
                                      exact False.elim this
                                  have hentry : alpha[id]! = some αv := by
                                    simpa [NN.MLTheory.CROWN.Cert.getAlpha?, hidA] using hαopt
                                  have hrange0 : ∀ i : Fin αv.n, (0 : ℝ) ≤ toVec αv.v i ∧ toVec αv.v
                                    i ≤ (1 : ℝ) := by
                                    simpa [hentry] using halpha id hidA
                                  intro i
                                  have hri := hrange0 (Fin.cast hα.symm i)
                                  simpa [αt, hα, toVec_castDimScalar] using hri
                                ·
                                  -- Fallback: default α is 0/1.
                                  simpa [αt, hα] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                    preB.hi)

                              -- Reuse the previous proof by reduction to the `none` case.
                              -- (The only difference is the choice of `αt` and its range proof.)
                              -- We rerun the enclosure proof with this `αt`.
                              -- This is structurally identical to the `none` branch above.
                              -- For brevity, we invoke it via a local `have` using the same steps.
                              -- (See the `none` branch for detailed commentary.)
                              -- Keep this duplicated rather than abstracting; dependent elaboration
                              -- gets fragile here.
                              let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.hiAff
                              let lAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xLo
                                  x'
                              let uAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim) xHi
                                  x'
                              let z : Tensor ℝ (.dim preB.dim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                  vp.v

                              let zXin : Tensor ℝ (.dim xin.outDim .scalar) :=
                                castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim) hdn.symm vp.v
                              have hzXin :
                                  Theorems.Semantics.encloses (α := ℝ) (boundsEvalAt (α := ℝ) xin
                                    x') zXin := by
                                have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                  exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                    hdn.symm) (t := vp.v)
                                simpa [this, zXin] using hencB
                              have hzCast0 :=
                                sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h := hout)
                                  (x := zXin) hzXin
                              have hzCastZ :
                                  castDimScalar (α := ℝ) hout zXin = z := by
                                have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                  exact Subsingleton.elim _ _
                                have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                  vp.v)).symm
                                simpa [zXin, z, htrans] using this
                              have hl :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                    lAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                    x')).symm
                              have hu :
                                  castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                    uAff := by
                                simpa [CrownCertSoundness.boundsEvalAt,
                                  CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                  (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                    x')).symm
                              have hzAff :
                                  Theorems.Semantics.encloses (α := ℝ)
                                    { dim := preB.dim
                                      lo := lAff
                                      hi := uAff } z := by
                                have hzCast1 :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').lo
                                        hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                          x').hi }
                                      z := by
                                      exact sem_encloses_value_eq
                                        (B := { dim := preB.dim
                                                lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α
                                                  := ℝ) xin x').lo
                                                hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α
                                                  := ℝ) xin x').hi })
                                        (hxy := hzCastZ) hzCast0
                                have hBoxEq :
                                    ({ dim := preB.dim
                                       lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').lo
                                       hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin
                                         x').hi } : FlatBox ℝ)
                                      =
                                    ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) := by
                                  refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                  · exact heq_of_eq hl
                                  · exact heq_of_eq hu
                                exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                              have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                simpa [CertSoundness.encloses, z] using hencIbp

                              rw [hb]
                              refine ⟨hinDim, ?_⟩
                              dsimp [CrownCertSoundness.EnclosesVec]
                              refine ⟨hdimIbp, ?_⟩
                              have hreluCast :
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                      (Activation.reluSpec (α := ℝ) vp.v)
                                    =
                                    Activation.reluSpec (α := ℝ) z := by
                                simpa [z] using
                                  (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))
                              have :
                                  Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x') (Activation.reluSpec (α := ℝ)
                                        z) := by
                                have hzAffI := (encloses_iff_toVec (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_toVec (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                refine (encloses_iff_toVec (n := preB.dim)
                                  (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                  (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                  (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                intro i
                                have hzLo := (hzAffI i).1
                                have hzHi := (hzAffI i).2
                                have hzIlo := (hzIbpI i).1
                                have hzIhi := (hzIbpI i).2
                                let li := toVec preB.lo i
                                let ui := toVec preB.hi i
                                let zi := toVec z i
                                let ai := toVec αt i
                                have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2
                                have hsLo : 0 ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope :=
                                  alphaRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a := ai)
                                    hai0
                                have hsHi : 0 ≤ (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α
                                  := ℝ) li ui).slope :=
                                  relax_scalar_slope_nonneg (l := li) (u := ui)
                                have hlo_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i
                                      =
                                      let rp := toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim)
                                        preB.lo preB.hi αt) i
                                      rp.slope * toVec lAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                        preB.hi αt)
                                      (aff := xLo) (x := x') (i := i))
                                have hhi_def :
                                    toVec (boundsEvalAt (α := ℝ) bout x').hi i
                                      =
                                      let rp := toVec
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ) (n
                                        := preB.dim) preB.lo preB.hi) i
                                      rp.slope * toVec uAff i + rp.bias := by
                                  simpa [hb, bout, CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                    (toVec_affineEvalAt_relu_propagate_affine
                                      (relax := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α :=
                                        ℝ) (n := preB.dim) preB.lo preB.hi)
                                      (aff := xHi) (x := x') (i := i))
                                have hrpLo :
                                    toVec (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt) i
                                      =
                                      alphaRelaxLowerScalar (α := ℝ) li ui ai := by
                                  simpa [li, ui, ai] using
                                    (toVec_alphaRelaxLowerVec (lo := preB.lo) (hi := preB.hi) (αv :=
                                      αt) (i := i))
                                have hrpHi :
                                    toVec (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := ℝ)
                                      (n := preB.dim) preB.lo preB.hi) i
                                      =
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li ui
                                        := by
                                  simpa [li, ui] using (toVec_runtime_relu_relax_vector (lo :=
                                    preB.lo) (hi := preB.hi) (i := i))
                                have hlo1 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec lAff i +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias := by
                                  have hm : (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * toVec
                                    lAff i
                                      ≤ (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi := by
                                    exact mul_le_mul_of_nonneg_left hzLo hsLo
                                  have h' :=
                                    add_le_add_right hm (alphaRelaxLowerScalar (α := ℝ) li ui
                                      ai).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hlo2 :
                                    (alphaRelaxLowerScalar (α := ℝ) li ui ai).slope * zi +
                                        (alphaRelaxLowerScalar (α := ℝ) li ui ai).bias
                                      ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  have := alphaRelaxLowerScalar_sound (l := li) (u := ui) (a := ai)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi) (ha0 := hai0) (ha1 := hai1)
                                  simpa [li, ui, zi, ai] using this
                                have hlo :
                                    toVec (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                  simp [hlo_def, hrpLo]
                                  exact le_trans hlo1 hlo2
                                have hhi1 :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have := relu_relax_scalar_upper_real_runtime (l := li) (u := ui)
                                    (x := zi)
                                    (hlx := hzIlo) (hxu := hzIhi)
                                  simpa [li, ui, zi] using this
                                have hhi2 :
                                    (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).slope * zi +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias
                                      ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i +
                                        (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                          ui).bias := by
                                  have hm :
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * zi
                                        ≤
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                        ui).slope * toVec uAff i := by
                                    exact mul_le_mul_of_nonneg_left hzHi hsHi
                                  have h' :=
                                    add_le_add_right hm
                                      (NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) li
                                      ui).bias
                                  simpa [add_comm, add_left_comm, add_assoc] using h'
                                have hhi :
                                    Activation.Math.reluSpec (α := ℝ) zi ≤ toVec (boundsEvalAt (α
                                      := ℝ) bout x').hi i := by
                                  simp [hhi_def, hrpHi]
                                  exact le_trans hhi1 hhi2
                                have hrelu : toVec (Activation.reluSpec (α := ℝ) z) i =
                                    Activation.Math.reluSpec (α := ℝ) zi := by
                                  simpa [zi] using (toVec_relu_spec (t := z) (i := i))
                                constructor
                                · simpa [hrelu] using hlo
                                · simpa [hrelu] using hhi
                              have this' :
                                  Theorems.Semantics.encloses (boundsEvalAt (α := ℝ) bout x')
                                    (Activation.reluSpec (α := ℝ) z) := by
                                simpa [x'] using this
                              exact sem_encloses_value_eq
                                (B := boundsEvalAt (α := ℝ) bout x')
                                (hxy := hreluCast.symm) this'
                            ·
                              have : False := by
                                simp [hxin, hpre, hαopt, hout] at hs'
                              exact False.elim this

    | .permute _ | .randUniform _ | .bernoulliMask _ | .add | .sub | .mul_elem | .abs | .sqrt |
      .inv
    | .maxElem | .minElem | .maxPool2d .. | .maxPool2dPad .. | .avgPool2d .. | .avgPool2dPad
      ..
    | .broadcastTo .. | .reduceSum _ | .reduceMean _ | .conv2d .. | .tanh | .sigmoid | .exp | .log
      | .sin | .cos
    | .softmax _ | .layernorm _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .mseLoss =>
      by
        -- Unsupported ops: `alphaCrownStepNode?` can only succeed via the IBP-derived constant
        -- enclosure.
        have hibpHere :
            match ibp[id]!, vals[id]! with
            | some B0, some v0 => CertSoundness.EnclosesBox B0 v0
            | _, _ => True := by
              have : id < vals.size := by simpa [hsem.1] using hid
              simpa using hibp id this
        cases hib : ibp[id]!
        · simp [stepAlpha, alphaCrownStepNode?, hk, hib] at hs
        · rename_i B0
          have hEnc : CertSoundness.EnclosesBox B0 v := by simpa [hib, hv] using hibpHere
          have hbEq : some (boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) = some b := by
            simpa [stepAlpha, alphaCrownStepNode?, hk, hib] using hs
          cases hbEq
          refine ⟨rfl, ?_⟩
          dsimp [CrownCertSoundness.EnclosesAtInput]
          simp [castDimScalar]
          rcases hEnc with ⟨hdim, hbox⟩
          refine ⟨hdim, ?_⟩
          have hBoxEq :
              boundsEvalAt (α := ℝ)
                  (boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) x =
                { dim := B0.dim, lo := B0.lo, hi := B0.hi } := by
            simpa using
              (boundsEvalAt_bounds_const (inDim := ctx.inputDim) (outDim := B0.dim) (lo := B0.lo)
                (hi := B0.hi)
                (x := x))
          have hBoxEval :
              boundsEvalAt (α := ℝ)
                  (boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) x = B0 := by
            cases B0
            simpa using hBoxEq
          have hbox' :=
            sem_encloses_of_eq (h := hBoxEval.symm) (x := castDimScalar (α := ℝ) hdim.symm v.v) hbox
          have hdim' : congrArg FlatBox.dim hBoxEval.symm = rfl := by
            exact Subsingleton.elim _ _
          simpa [hdim', castDimScalar] using hbox'
    )

/-!
In words (α/β-CROWN, pointwise, graph dialect).

This is the β-extended analog of `alphaCrown_transfer_sound`.

Compared to plain α-CROWN, the step function additionally receives a `beta` array encoding
per-ReLU phase constraints (active/inactive/unstable). When a phase is consistent with the IBP
pre-activation interval, the relaxation reduces to an exact affine rule for that unit; otherwise
the step falls back to the corresponding sound α-CROWN relaxation, or to an IBP-derived constant
enclosure for operators outside this affine-transfer subset.

The theorem states that this concrete step function satisfies `CrownTransferSound`, and thus can
be used as the trusted “checker semantics” in `graph_crown_cert_soundness`.
-/
theorem alphaBetaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatVec ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlphaBeta g ps ibp alpha beta ctx) (cert := cert) := by
  classical
  -- Reuse the α-CROWN transfer theorem for all nodes where `α/β` reduces to plain α.
  have hsoundAlpha :
      CrownTransferSound
        (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
        (ctx := ctx) (x := x)
        (step := stepAlpha g ps ibp alpha ctx) (cert := cert) :=
    alphaCrown_transfer_sound (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (cert := cert)
      (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
      (htopo := htopo) (hsem := hsem) (hinputs := hinputs) (hibp := hibp) (halpha := halpha)

  intro id hid hparents
  cases hs : stepAlphaBeta g ps ibp alpha beta ctx cert id <;> cases hv : vals[id]!
  all_goals simp
  case some.some b v =>
      -- Semantic evaluation at this node.
      have hEvalEq : vals[id]! = evalNode? g.nodes ps inputs vals id := hsem.2 id hid
      have hEvalSome : evalNode? g.nodes ps inputs vals id = some v := by
        have : some v = evalNode? g.nodes ps inputs vals id := by
          simpa [hv] using hEvalEq
        simpa using this.symm

      -- Split by node kind, mirroring `alphaBetaCrownStepNode?`.
      cases hk : (g.nodes[id]!).kind
      case relu =>
          -- If there is no β vector, α/β-CROWN is definitionally α-CROWN.
          cases hbeta : getBeta? (beta := beta) id with
          | none =>
              have hsAlpha : stepAlpha g ps ibp alpha ctx cert id = some b := by
                simpa [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk, hbeta] using hs
              have hA := hsoundAlpha id hid hparents
              simpa [hsAlpha, hv] using hA
          | some phases =>
              cases hps : (g.nodes[id]!).parents with
              | nil =>
                  -- ReLU needs a parent; the step cannot succeed.
                  simp [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hbeta, hps] at hs
              | cons p1 _ =>
                have hpMem : p1 ∈ (g.nodes[id]!).parents := by simp [hps]

                -- Step-side: extract parent affine bounds + IBP box.
                have hs' := hs
                simp [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hps, hbeta] at hs'
                cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
                | none =>
                    have : False := by
                      simp [hxin] at hs'
                    exact False.elim this
                | some xin =>
                    cases hpre : ibp[p1]! with
                    | none =>
                        have : False := by
                          simp [hxin, hpre] at hs'
                        exact False.elim this
                    | some preB =>
                        -- Semantic-side: extract the parent value `vp` and identify `v` as
                        -- `relu(vp)`.
                        have hEval' := hEvalSome
                        simp [CertSoundness.evalNode?, hk, hps] at hEval'
                        cases hgv : CertSoundness.getVal? vals p1 with
                        | none =>
                            have : False := by
                              simp [hgv] at hEval'
                            exact False.elim this
                        | some vp =>
                            have hvEq :
                                some { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } = some v
                                  := by
                              simpa [hgv] using hEval'
                            cases hvEq

                            -- Connect `getAff?/getVal?` equalities to array lookups to use
                            -- `hparents`.
                            have hcertp : cert[p1]! = some xin := by
                              by_cases hltC : p1 < cert.size
                              · simpa [NN.MLTheory.CROWN.Cert.getAff?, hltC] using hxin
                              ·
                                have : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = none := by
                                  simp [NN.MLTheory.CROWN.Cert.getAff?, hltC]
                                have : False := by
                                  simp [this] at hxin
                                exact False.elim this
                            have hvpp : vals[p1]! = some vp := by
                              by_cases hltV : p1 < vals.size
                              · simpa [CertSoundness.getVal?, hltV] using hgv
                              ·
                                have : CertSoundness.getVal? vals p1 = none := by
                                  simp [CertSoundness.getVal?, hltV]
                                have : False := by
                                  simp [this] at hgv
                                exact False.elim this

                            have parentEnc : CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x xin
                              vp := by
                              have := hparents p1 hpMem
                              simpa [hcertp, hvpp] using this
                            rcases parentEnc with ⟨hinDim, hvec⟩
                            dsimp at hvec
                            rcases hvec with ⟨hdimB, hencB⟩
                            have hdn : xin.outDim = vp.n := by
                              simpa [CrownCertSoundness.boundsEvalAt] using hdimB

                            -- IBP enclosure for the parent value.
                            have hibpHere :
                                match ibp[p1]!, vals[p1]! with
                                | some B0, some v0 => CertSoundness.EnclosesBox B0 v0
                                | _, _ => True := by
                              have : p1 < vals.size := by
                                have : p1 < g.nodes.size := lt_trans (htopo id hid p1 hpMem) hid
                                simpa [hsem.1] using this
                              simpa using hibp p1 this
                            have hencIbp : CertSoundness.EnclosesBox preB vp := by
                              simpa [hpre, hvpp] using hibpHere

                            rcases hencIbp with ⟨hdimIbp, hboxIbp⟩

                            -- `hout` is needed to align the parent affine out-dimension with the
                            -- IBP dimension.
                            by_cases hout : xin.outDim = preB.dim
                            ·
                              -- Common local proof once we have a concrete `αt` + phase
                              -- relaxations.
                              let x' : Tensor ℝ (.dim xin.inDim .scalar) :=
                                castDimScalar (α := ℝ) (n := ctx.inputDim) (n' := xin.inDim)
                                  hinDim.symm x
                              let xLo : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.loAff
                              let xHi : AffineVec ℝ xin.inDim preB.dim := by
                                simpa [hout] using xin.hiAff

                              have relu_beta_common
                                  (αt : Tensor ℝ (.dim preB.dim .scalar))
                                  (hαrange : ∀ i : Fin preB.dim,
                                    (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤ (1 : ℝ))
                                  (relaxLo relaxHi :
                                    Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) (.dim preB.dim
                                      .scalar))
                                  (hrelax :
                                    phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt
                                      phases =
                                      some (relaxLo, relaxHi))
                                  (hbEq :
                                    some
                                        { inDim := xin.inDim
                                          outDim := preB.dim
                                          loAff :=
                                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                              ℝ)
                                              (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                          hiAff :=
                                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                              ℝ)
                                              (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                                                } =
                                      some b) :
                                  CrownCertSoundness.EnclosesAtInput (α := ℝ) ctx x b
                                    { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } := by
                                have hb :
                                    b =
                                      { inDim := xin.inDim
                                        outDim := preB.dim
                                        loAff :=
                                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                            ℝ)
                                            (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                        hiAff :=
                                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α :=
                                            ℝ)
                                            (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi }
                                              := by
                                  exact (Option.some.inj hbEq).symm
                                -- Cast the semantic parent value into `preB.dim` so the ReLU is
                                -- well-typed.
                                let z : Tensor ℝ (.dim preB.dim .scalar) :=
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                    vp.v
                                have hreluCast :
                                    castDimScalar (α := ℝ) (n := vp.n) (n' := preB.dim) hdimIbp.symm
                                        (Activation.reluSpec (α := ℝ) vp.v)
                                      =
                                      Activation.reluSpec (α := ℝ) z := by
                                  simpa [z] using
                                    (relu_spec_castDimScalar (h := hdimIbp.symm) (t := vp.v))

                                -- Derive the affine enclosure `lAff ≤ z ≤ uAff` from the parent's
                                -- enclosure.
                                let zXin : Tensor ℝ (.dim xin.outDim .scalar) :=
                                  castDimScalar (α := ℝ) (n := vp.n) (n' := xin.outDim)
                                    hdn.symm vp.v
                                have hzXin :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) xin x') zXin := by
                                  have : castDimScalar (α := ℝ) hdimB.symm vp.v = zXin := by
                                    exact castDimScalar_proof_irrel (h₁ := hdimB.symm) (h₂ :=
                                      hdn.symm) (t := vp.v)
                                  simpa [this, zXin] using hencB

                                have hzCast0 :=
                                  sem_encloses_castDim (B := boundsEvalAt (α := ℝ) xin x') (h :=
                                    hout) (x := zXin) hzXin
                                have hzCastZ :
                                    castDimScalar (α := ℝ) hout zXin = z := by
                                  have htrans : Eq.trans hdn.symm hout = hdimIbp.symm := by
                                    exact Subsingleton.elim _ _
                                  have := (castDimScalar_trans (h₁ := hdn.symm) (h₂ := hout) (t :=
                                    vp.v)).symm
                                  simpa [zXin, z, htrans] using this
                                let lAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                  affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim)
                                    xLo x'
                                let uAff : Tensor ℝ (.dim preB.dim .scalar) :=
                                  affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := preB.dim)
                                    xHi x'
                                have hl :
                                    castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').lo =
                                      lAff := by
                                  simpa [CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, lAff, x', xLo] using
                                    (affineEvalAt_castAffineOut (h := hout) (aff := xin.loAff) (x :=
                                      x')).symm
                                have hu :
                                    castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ) xin x').hi =
                                      uAff := by
                                  simpa [CrownCertSoundness.boundsEvalAt,
                                    CrownCertSoundness.affineEvalAt, uAff, x', xHi] using
                                    (affineEvalAt_castAffineOut (h := hout) (aff := xin.hiAff) (x :=
                                      x')).symm
                                have hzAff :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      { dim := preB.dim
                                        lo := lAff
                                        hi := uAff } z := by
                                  -- Convert `hzCast0` into the `lAff/uAff` box via `hl/hu`.
                                  have hzCast1 :
                                      Theorems.Semantics.encloses (α := ℝ)
                                        { dim := preB.dim
                                          lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                            xin x').lo
                                          hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                            xin x').hi }
                                        z := by
                                    exact sem_encloses_value_eq
                                      (B := { dim := preB.dim
                                              lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α :=
                                                ℝ) xin x').lo
                                              hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α :=
                                                ℝ) xin x').hi })
                                      (hxy := hzCastZ) hzCast0
                                  have hBoxEq :
                                      ({ dim := preB.dim
                                         lo := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                           xin x').lo
                                         hi := castDimScalar (α := ℝ) hout (boundsEvalAt (α := ℝ)
                                           xin x').hi } : FlatBox ℝ)
                                        =
                                      ({ dim := preB.dim, lo := lAff, hi := uAff } : FlatBox ℝ) :=
                                        by
                                    refine FlatBox.ext' (hDim := rfl) (hLo := ?_) (hHi := ?_)
                                    · exact heq_of_eq hl
                                    · exact heq_of_eq hu
                                  exact sem_encloses_of_eq (h := hBoxEq) (x := z) hzCast1

                                have hzIbp : Theorems.Semantics.encloses (α := ℝ) preB z := by
                                  simpa [CertSoundness.encloses, z] using hboxIbp

                                let bout : FlatAffineBounds ℝ :=
                                  { inDim := xin.inDim
                                    outDim := preB.dim
                                    loAff :=
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                        (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                                    hiAff :=
                                      NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
                                        (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi }
                                have hzAffI := (encloses_iff_toVec (n := preB.dim) (lo := lAff) (hi
                                  := uAff) (x := z)).1 hzAff
                                have hzIbpI := (encloses_iff_toVec (n := preB.dim) (lo := preB.lo)
                                  (hi := preB.hi) (x := z)).1 hzIbp
                                have hphase := phaseRelaxVec?_some_toVec (n := preB.dim)
                                  (lo := preB.lo) (hi := preB.hi) (αv := αt) (phases := phases)
                                  (relaxLo := relaxLo) (relaxHi := relaxHi) hrelax

                                have hConcrete :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x')
                                      (Activation.reluSpec (α := ℝ) z) := by
                                  -- Now it suffices to show enclosure against the concrete lo/hi
                                  -- tensors.
                                  refine (encloses_iff_toVec (n := preB.dim)
                                    (lo := (boundsEvalAt (α := ℝ) bout x').lo)
                                    (hi := (boundsEvalAt (α := ℝ) bout x').hi)
                                    (x := Activation.reluSpec (α := ℝ) z)).2 ?_
                                  intro i
                                  have hzLo := (hzAffI i).1
                                  have hzHi := (hzAffI i).2
                                  have hzIlo := (hzIbpI i).1
                                  have hzIhi := (hzIbpI i).2
                                  rcases hphase.2 i with ⟨ph, hcons, hrHi, hrLo⟩

                                  let li := toVec preB.lo i
                                  let ui := toVec preB.hi i
                                  let zi := toVec z i
                                  let ai := toVec αt i
                                  have hai0 : (0 : ℝ) ≤ ai := (hαrange i).1
                                  have hai1 : ai ≤ (1 : ℝ) := (hαrange i).2

                                  let rpLo := toVec relaxLo i
                                  let rpHi := toVec relaxHi i
                                  have hsLo : 0 ≤ rpLo.slope := by
                                    have hs :
                                        0 ≤ (phaseRelaxLowerScalar (α := ℝ) li ui ai ph).slope :=
                                      phaseRelaxLowerScalar_slope_nonneg (l := li) (u := ui) (a :=
                                        ai) (ph := ph) hai0
                                    simpa [rpLo, li, ui, ai, hrLo] using hs
                                  have hsHi : 0 ≤ rpHi.slope := by
                                    have hs :
                                        0 ≤ (phaseRelaxUpperScalar (α := ℝ) li ui ph).slope :=
                                      phaseRelaxUpperScalar_slope_nonneg (l := li) (u := ui) (ph :=
                                        ph)
                                    simpa [rpHi, li, ui, hrHi] using hs

                                  have hlo_def :
                                      toVec (boundsEvalAt (α := ℝ) bout x').lo i
                                        =
                                        let rp := toVec relaxLo i
                                        rp.slope * toVec lAff i + rp.bias := by
                                    simpa [CrownCertSoundness.boundsEvalAt,
                                      CrownCertSoundness.affineEvalAt, bout, lAff, x', xLo] using
                                      (toVec_affineEvalAt_relu_propagate_affine
                                        (relax := relaxLo) (aff := xLo) (x := x') (i := i))
                                  have hhi_def :
                                      toVec (boundsEvalAt (α := ℝ) bout x').hi i
                                        =
                                        let rp := toVec relaxHi i
                                        rp.slope * toVec uAff i + rp.bias := by
                                    simpa [CrownCertSoundness.boundsEvalAt,
                                      CrownCertSoundness.affineEvalAt, bout, uAff, x', xHi] using
                                      (toVec_affineEvalAt_relu_propagate_affine
                                        (relax := relaxHi) (aff := xHi) (x := x') (i := i))

                                  have hlo1 :
                                      rpLo.slope * toVec lAff i + rpLo.bias
                                        ≤
                                      rpLo.slope * zi + rpLo.bias := by
                                    have hm : rpLo.slope * toVec lAff i ≤ rpLo.slope * zi := by
                                      exact mul_le_mul_of_nonneg_left hzLo hsLo
                                    have h' := add_le_add_right hm rpLo.bias
                                    simpa [add_comm, add_left_comm, add_assoc] using h'
                                  have hlo2 :
                                      rpLo.slope * zi + rpLo.bias
                                        ≤
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                    have :=
                                      phaseRelaxLowerScalar_sound (l := li) (u := ui) (a := ai) (x
                                        := zi)
                                        (hlx := hzIlo) (hxu := hzIhi) (ha0 := hai0) (ha1 := hai1)
                                        (ph := ph) (hcons := hcons)
                                    simpa [rpLo, li, ui, ai, zi, hrLo] using this
                                  have hlo :
                                      toVec (boundsEvalAt (α := ℝ) bout x').lo i ≤
                                        Activation.Math.reluSpec (α := ℝ) zi := by
                                    simp [hlo_def]
                                    exact le_trans hlo1 hlo2

                                  have hhi1 :
                                      Activation.Math.reluSpec (α := ℝ) zi ≤
                                        rpHi.slope * zi + rpHi.bias := by
                                    have :=
                                      phaseRelaxUpperScalar_sound (l := li) (u := ui) (x := zi)
                                        (hlx := hzIlo) (hxu := hzIhi) (ph := ph) (hcons := hcons)
                                    simpa [rpHi, li, ui, zi, hrHi] using this
                                  have hhi2 :
                                      rpHi.slope * zi + rpHi.bias
                                        ≤
                                      rpHi.slope * toVec uAff i + rpHi.bias := by
                                    have hm : rpHi.slope * zi ≤ rpHi.slope * toVec uAff i := by
                                      exact mul_le_mul_of_nonneg_left hzHi hsHi
                                    have h' := add_le_add_right hm rpHi.bias
                                    simpa [add_comm, add_left_comm, add_assoc] using h'
                                  have hhi :
                                      Activation.Math.reluSpec (α := ℝ) zi ≤
                                        toVec (boundsEvalAt (α := ℝ) bout x').hi i := by
                                    simp [hhi_def]
                                    exact le_trans hhi1 hhi2

                                  have hrelu : toVec (Activation.reluSpec (α := ℝ) z) i =
                                      Activation.Math.reluSpec (α := ℝ) zi := by
                                    simpa [zi] using (toVec_relu_spec (t := z) (i := i))
                                  constructor
                                  · simpa [hrelu] using hlo
                                  · simpa [hrelu] using hhi
                                rw [hb]
                                refine ⟨hinDim, ?_⟩
                                dsimp [CrownCertSoundness.EnclosesVec]
                                refine ⟨hdimIbp, ?_⟩
                                have hConcrete' :
                                    Theorems.Semantics.encloses (α := ℝ)
                                      (boundsEvalAt (α := ℝ) bout x')
                                      (Activation.reluSpec (α := ℝ) z) := by
                                  simpa [x'] using hConcrete
                                exact sem_encloses_value_eq
                                  (B := boundsEvalAt (α := ℝ) bout x')
                                  (hxy := hreluCast.symm) hConcrete'

                              -- Now instantiate `relu_beta_common` according to whether α is
                              -- present.
                              cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id with
                              | some αv =>
                                  have hs'' := hs'
                                  simp [hxin, hpre, hαopt, hout] at hs''
                                  by_cases hα : αv.n = preB.dim
                                  ·
                                    let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                      castDimScalar (α := ℝ) (n := αv.n) (n' := preB.dim) hα αv.v
                                    have hαrange : ∀ i : Fin preB.dim,
                                        (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤ (1 : ℝ) := by
                                      have hidA : id < alpha.size := by
                                        by_cases hltA : id < alpha.size
                                        · exact hltA
                                        ·
                                          have : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id
                                            = none := by
                                            simp [NN.MLTheory.CROWN.Cert.getAlpha?, hltA]
                                          have : False := by
                                            simp [this] at hαopt
                                          exact False.elim this
                                      have hentry : alpha[id]! = some αv := by
                                        simpa [NN.MLTheory.CROWN.Cert.getAlpha?, hidA] using hαopt
                                      have hrange0 : ∀ i : Fin αv.n, (0 : ℝ) ≤ toVec αv.v i ∧ toVec
                                        αv.v i ≤ (1 : ℝ) := by
                                        simpa [hentry] using halpha id hidA
                                      intro i
                                      have hri := hrange0 (Fin.cast hα.symm i)
                                      simpa [αt, hα, toVec_castDimScalar] using hri
                                    simp [hα] at hs''
                                    cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo
                                      preB.hi αt phases with
                                    | none =>
                                        have : False := by
                                          simp [αt, hrelax] at hs''
                                        exact False.elim this
                                    | some rpair =>
                                        cases rpair with
                                        | mk relaxLo relaxHi =>
                                            have hbEq :
                                                some
                                                    ({ inDim := xin.inDim
                                                       outDim := preB.dim
                                                       loAff :=
                                                         Runtime.Ops.ReLU.propagateAffine
                                                           (α := ℝ)
                                                           (inDim := xin.inDim) (hidDim := preB.dim)
                                                           relaxLo xLo
                                                       hiAff :=
                                                         Runtime.Ops.ReLU.propagateAffine
                                                           (α := ℝ)
                                                           (inDim := xin.inDim) (hidDim := preB.dim)
                                                           relaxHi xHi } : FlatAffineBounds ℝ) =
                                                  some b := by
                                              simpa [αt, hrelax] using hs''
                                            exact relu_beta_common αt hαrange relaxLo relaxHi hrelax
                                              hbEq
                                  ·
                                    -- Dimension mismatch: step cannot succeed.
                                    simp [hα] at hs''
                              | none =>
                                  have hs'' := hs'
                                  simp [hxin, hpre, hαopt, hout] at hs''
                                  let αt : Tensor ℝ (.dim preB.dim .scalar) :=
                                    defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi
                                  have hαrange : ∀ i : Fin preB.dim,
                                      (0 : ℝ) ≤ toVec αt i ∧ toVec αt i ≤ (1 : ℝ) := by
                                    simpa [αt] using defaultAlphaVec_range (lo := preB.lo) (hi :=
                                      preB.hi)
                                  cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo
                                    preB.hi αt phases with
                                  | none =>
                                      have : False := by
                                        simp [αt, hrelax] at hs''
                                      exact False.elim this
                                  | some rpair =>
                                      cases rpair with
                                      | mk relaxLo relaxHi =>
                                          have hbEq :
                                              some
                                                  ({ inDim := xin.inDim
                                                     outDim := preB.dim
                                                     loAff :=
                                                       Runtime.Ops.ReLU.propagateAffine
                                                         (α := ℝ)
                                                         (inDim := xin.inDim) (hidDim := preB.dim)
                                                         relaxLo xLo
                                                     hiAff :=
                                                       Runtime.Ops.ReLU.propagateAffine
                                                         (α := ℝ)
                                                         (inDim := xin.inDim) (hidDim := preB.dim)
                                                         relaxHi xHi } : FlatAffineBounds ℝ) =
                                                some b := by
                                            simpa [αt, hrelax] using hs''
                                          exact relu_beta_common αt hαrange relaxLo relaxHi hrelax
                                            hbEq
                            ·
                              -- If `xin.outDim ≠ preB.dim` then the step returns `none`.
                              have hs'' := hs'
                              -- First reduce the `match` on the known `some xin` / `some preB`.
                              simp [hxin, hpre] at hs''
                              -- Now the outer `if hout : xin.outDim = preB.dim` is forced to take
                              -- the `else` branch.
                              cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id <;> (
                                have : False := by
                                  simp [hαopt, hout] at hs''
                                exact False.elim this
                              )

      all_goals
        -- All other node kinds delegate to α-CROWN.
        have hsAlpha : stepAlpha g ps ibp alpha ctx cert id = some b := by
          simpa [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk] using hs
        have hA := hsoundAlpha id hid hparents
        simpa [hsAlpha, hv] using hA

  end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
