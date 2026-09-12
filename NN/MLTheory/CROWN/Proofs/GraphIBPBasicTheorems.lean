/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp
public import NN.MLTheory.CROWN.BoundOps.Lawful
public import NN.MLTheory.CROWN.Graph.Theorems

/-!
# GraphIBPBasicTheorems

Basic theorems about the graph-level IBP engine:

* A `Valid` predicate for interval boxes (`lo ≤ hi` componentwise).
* Preservation of validity for core interval ops (add/sub/relu) and runtime monotone-activation IBP.
* Validity of the linear/matmul IBP rules (derived from the existing `ibp_linear_sound_real` proof).
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open Spec TorchLean
open TorchLean.Tensor

namespace Box

/-- Componentwise validity of a 1D interval box: `lo ≤ hi` for every coordinate. -/
def Valid {n : Nat} (B : Box ℝ (.dim n .scalar)) : Prop :=
  ∀ i : Fin n, TorchLean.Tensor.getScalar B.lo i ≤ TorchLean.Tensor.getScalar B.hi i

/-- If a box contains any point, then it is componentwise valid (`lo ≤ hi`). -/
theorem valid_of_contains {n : Nat} (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n])
  (hx : Box.contains (α := ℝ) B x) : Valid B := by
  intro i
  exact le_trans (hx i).1 (hx i).2

/-- A valid box contains its lower endpoint `lo`. -/
theorem contains_lo_of_valid {n : Nat} (B : Box ℝ (.dim n .scalar)) (hB : Valid B) :
    Box.contains (α := ℝ) B B.lo := by
  intro i
  exact ⟨le_rfl, hB i⟩

/-- Validity is preserved by (definitional) casts of the vector dimension. -/
theorem valid_castBoxDim {n n' : Nat} (h : n = n')
    (B : Box ℝ (.dim n .scalar)) (hB : Valid B) :
    Valid (NN.MLTheory.CROWN.Graph.castBoxDim (α := ℝ) h B) := by
  cases h
  simpa [NN.MLTheory.CROWN.Graph.castBoxDim] using hB

end Box

end NN.MLTheory.CROWN

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace FlatBoxTheorems

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Converting a valid `Box` into a `FlatBox` preserves validity. -/
theorem valid_toFlatBox_real {n : Nat} (B : Box ℝ (.dim n .scalar)) (hB :
  NN.MLTheory.CROWN.Box.Valid B) :
    (toFlatBox (α := ℝ) n B).Valid := by
  exact hB

/-- Converting a valid `FlatBox` into a `Box` preserves validity. -/
theorem valid_ofFlatBox_real (B : FlatBox ℝ) (hB : B.Valid) :
    NN.MLTheory.CROWN.Box.Valid (ofFlatBox (α := ℝ) B) := by
  exact hB

/-- Validity is preserved by interval addition on `FlatBox` (over `ℝ`). -/
theorem valid_box_add_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxAdd (α := ℝ) B1 B2).Valid := by
  let : BoundOps ℝ := instBoundOpsReal
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · subst h
        -- use the canonical form lemma for the equal-dimension case
        have hEq :
            boxAdd (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n1, lo := lo2, hi := hi2 } =
              { dim := n1
                lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
                hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 } := by
          simpa using (NN.MLTheory.CROWN.Graph.Theorems.box_add_on_eq (α := ℝ) n1 lo1 hi1 lo2 hi2)
        -- reduce the goal to scalar arithmetic
        rw [hEq]
        intro i
        have hdown :
            (@BoundOps.addDown ℝ inferInstance inferInstance instBoundOpsReal) =
              (fun x y : ℝ => x + y) := rfl
        have hup :
            (@BoundOps.addUp ℝ inferInstance inferInstance instBoundOpsReal) =
              (fun x y : ℝ => x + y) := rfl
        rw [Tensor.getScalar_map2Spec, Tensor.getScalar_map2Spec, hdown, hup]
        exact add_le_add (h1 i) (h2 i)
      · -- mismatch branch: returns B1 unchanged
        have hEq :
            boxAdd (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n2, lo := lo2, hi := hi2 } =
              { dim := n1, lo := lo1, hi := hi1 } := by
          simp [boxAdd, h]
        simpa [hEq] using h1

/-- Validity is preserved by interval subtraction on `FlatBox` (over `ℝ`). -/
theorem valid_box_sub_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxSub (α := ℝ) B1 B2).Valid := by
  let : BoundOps ℝ := instBoundOpsReal
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · subst h
        have hEq :
            boxSub (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n1, lo := lo2, hi := hi2 } =
              { dim := n1
                lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
                hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 } := by
          simpa using (NN.MLTheory.CROWN.Graph.Theorems.box_sub_on_eq (α := ℝ) n1 lo1 hi1 lo2 hi2)
        rw [hEq]
        intro i
        have hdown :
            (@BoundOps.subDown ℝ inferInstance inferInstance instBoundOpsReal) =
              (fun x y : ℝ => x - y) := rfl
        have hup :
            (@BoundOps.subUp ℝ inferInstance inferInstance instBoundOpsReal) =
              (fun x y : ℝ => x - y) := rfl
        rw [Tensor.getScalar_map2Spec, Tensor.getScalar_map2Spec, hdown, hup]
        exact sub_le_sub (h1 i) (h2 i)
      ·
        have hEq :
            boxSub (α := ℝ)
                { dim := n1, lo := lo1, hi := hi1 }
                { dim := n2, lo := lo2, hi := hi2 } =
              { dim := n1, lo := lo1, hi := hi1 } := by
          simp [boxSub, h]
        simpa [hEq] using h1

/-- Validity is preserved by ReLU-IBP on `FlatBox` (over `ℝ`). -/
theorem valid_box_relu_real (B : FlatBox ℝ) (hB : B.Valid) :
    (boxRelu (α := ℝ) B).Valid := by
  change ∀ i : Fin B.dim,
    (Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := ℝ) x) B.lo).getScalar i ≤
      (Tensor.mapSpec (fun x => Activation.Math.reluSpec (α := ℝ) x) B.hi).getScalar i
  intro i
  rw [Tensor.getScalar_mapSpec, Tensor.getScalar_mapSpec]
  simpa only [Activation.Math.reluSpec_eq_max] using max_le_max (hB i) (le_refl 0)

end FlatBoxTheorems

namespace Theorems

open NN.MLTheory.CROWN.Box

/-- Validity of the `IBP.linear` transfer rule (over `ℝ`). -/
theorem ibp_linear_valid_real {m n : Nat}
  (W : Tensor ℝ [m, n])
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar))
  (hxB : Valid xB) (hbB : Valid bB) :
  Valid (IBP.linear (α := ℝ) W xB bB) := by
  -- pick witnesses `x = xB.lo` and `b = bB.lo`
  have hx : Box.contains (α := ℝ) xB xB.lo := Box.contains_lo_of_valid (B := xB) hxB
  have hb : Box.contains (α := ℝ) bB bB.lo := Box.contains_lo_of_valid (B := bB) hbB
  have hout :
      Box.contains (α := ℝ) (IBP.linear (α := ℝ) W xB bB)
        (Spec.linearSpec (α := ℝ) { weights := W, bias := bB.lo } xB.lo) :=
    NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real W xB bB xB.lo bB.lo hx hb
  exact Box.valid_of_contains (B := IBP.linear (α := ℝ) W xB bB) _ hout

/-- Validity of the graph-level `linear` IBP rule (over `ℝ`), if the parameters are present. -/
theorem graph_ibp_linear_valid_real (id : Nat) (ps : ParamStore ℝ) (Xin : FlatBox ℝ)
  (hXin : FlatBox.Valid (α := ℝ) Xin) :
  match ibpLinear (α := ℝ) id ps Xin with
  | none => True
  | some Bout => FlatBox.Valid (α := ℝ) Bout := by
  classical
  unfold ibpLinear
  cases hlin : ps.linearWB[id]? with
  | none => simp
  | some p =>
      by_cases hdim : Xin.dim = p.n
      · -- dimension match: reduce to `IBP.linear_valid_real`
        simp [ibpLinearParams, hdim]
        -- show the produced box is valid
        -- Convert input to a dimension-safe `Box` and prove it is valid.
        have hxBoxValid : Valid (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) := by
          -- validity is preserved by definitional cast
          have : Valid (ofFlatBox (α := ℝ) Xin) := FlatBoxTheorems.valid_ofFlatBox_real (B := Xin)
            hXin
          exact NN.MLTheory.CROWN.Box.valid_castBoxDim (h := hdim) (B := ofFlatBox (α := ℝ) Xin)
            this
        have hbBoxValid : Valid (Box.point (α := ℝ) p.b) := by
          intro i
          exact le_rfl
        have hyValid : Valid (IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
            (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (Box.point (α := ℝ) p.b)) :=
          ibp_linear_valid_real (W := p.w) (xB := castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (bB := Box.point (α := ℝ) p.b) hxBoxValid hbBoxValid
        exact FlatBoxTheorems.valid_toFlatBox_real (B := IBP.linear (α := ℝ) (m := p.m) (n := p.n)
          p.w
          (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) (Box.point (α := ℝ) p.b)) hyValid
      · simp [ibpLinearParams, hdim]

/-- Validity of the graph-level `matmul` IBP rule (over `ℝ`), if the parameters are present. -/
theorem graph_ibp_matmul_valid_real (id : Nat) (ps : ParamStore ℝ) (Xin : FlatBox ℝ)
  (hXin : FlatBox.Valid (α := ℝ) Xin) :
  match ibpMatmul (α := ℝ) id ps Xin with
  | none => True
  | some Bout => FlatBox.Valid (α := ℝ) Bout := by
  classical
  unfold ibpMatmul
  cases hmat : ps.matmulW[id]? with
  | none => simp
  | some p =>
      by_cases hdim : Xin.dim = p.n
      · simp [hdim]
        have hxBoxValid : Valid (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) := by
          have : Valid (ofFlatBox (α := ℝ) Xin) := FlatBoxTheorems.valid_ofFlatBox_real (B := Xin)
            hXin
          exact NN.MLTheory.CROWN.Box.valid_castBoxDim (h := hdim) (B := ofFlatBox (α := ℝ) Xin)
            this
        -- zero bias point box is valid
        let z : Tensor ℝ [p.m] := Tensor.full (α := ℝ) (.dim p.m .scalar) 0
        have hbBoxValid : Valid (Box.point (α := ℝ) z) := by
          intro i
          exact le_rfl
        have hyValid : Valid (IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
            (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (Box.point (α := ℝ) z)) :=
          ibp_linear_valid_real (W := p.w) (xB := castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin))
            (bB := Box.point (α := ℝ) z) hxBoxValid hbBoxValid
        exact FlatBoxTheorems.valid_toFlatBox_real (B := IBP.linear (α := ℝ) (m := p.m) (n := p.n)
          p.w
          (castBoxDim (α := ℝ) hdim (ofFlatBox (α := ℝ) Xin)) (Box.point (α := ℝ) z)) hyValid
      · simp [hdim]

end Theorems

end NN.MLTheory.CROWN.Graph
