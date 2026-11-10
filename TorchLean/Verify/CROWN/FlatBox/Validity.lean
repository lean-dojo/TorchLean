/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Real

/-!
# Valid boxes over `ℝ`

A flat box is valid (`FlatBox.Valid`) when `lo ≤ hi` at every coordinate. We call a vector box
`B : Box ℝ (.dim n .scalar)` valid when its flat view `Graph.toFlatBox n B` is, so one predicate
serves both box types, and `Graph.valid_toFlatBox_real` converts the componentwise order of the
endpoints to it.

Over `ℝ`, a box that contains a point is valid (`Box.valid_of_contains`) and a valid box contains
its lower endpoint (`Box.contains_lo_of_valid`). Point boxes are valid (`Box.valid_point`), and so
is a valid flat box viewed at a checked dimension (`FlatBox.valid_getScalarBox`). Interval
addition, subtraction and ReLU keep boxes valid (`Graph.valid_boxAdd_real`,
`Graph.valid_boxSub_real`, `Graph.valid_boxReLU_real`). On valid inputs the linear rules give
valid boxes: `IBP.linear` (`IBP.linear_valid_real`, from `IBP.linear_sound_real`), and
`Graph.ibpLinearParams` whenever it gives a box (`Graph.ibpLinearParams_valid_real`), also with
zero bias (`Graph.ibp_matmul_valid_real`). The op-table matmul uses `Graph.binaryMatmulBox`.
-/

@[expose] public section

namespace TorchLean.CROWN

open Tensor

namespace Box

/-- If the vector box `B` contains a point `x`, then its flat view `Graph.toFlatBox n B` is
valid. -/
theorem valid_of_contains {n : Nat} (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n])
    (hx : Box.Contains (α := ℝ) B x) : (Graph.toFlatBox (α := ℝ) n B).Valid := by
  intro i
  exact le_trans (hx i).1 (hx i).2

/-- If the flat view `Graph.toFlatBox n B` is valid, then the vector box `B` contains its lower
endpoint `B.lo`. -/
theorem contains_lo_of_valid {n : Nat} (B : Box ℝ (.dim n .scalar))
    (hB : (Graph.toFlatBox (α := ℝ) n B).Valid) : Box.Contains (α := ℝ) B B.lo := by
  intro i
  exact ⟨le_rfl, hB i⟩

/-- The flat view of the point box `Box.point t` is valid. -/
theorem valid_point {n : Nat} (t : Tensor ℝ [n]) :
    (Graph.toFlatBox (α := ℝ) n (Box.point (α := ℝ) t)).Valid :=
  FlatBox.valid_ofTensor le_refl t

end Box

/-- If the flat box `B` is valid, then so is the flat view of the vector box `B.getScalarBox h`
that views it at the checked dimension `h : B.dim = m`. -/
theorem FlatBox.valid_getScalarBox (B : FlatBox ℝ) {m : Nat} (h : B.dim = m) (hB : B.Valid) :
    (Graph.toFlatBox (α := ℝ) m (B.getScalarBox h)).Valid := by
  subst h
  exact hB

namespace Graph

/-- If `B.lo ≤ B.hi` at every coordinate of the vector box `B`, then `toFlatBox n B` is valid. -/
theorem valid_toFlatBox_real {n : Nat} (B : Box ℝ (.dim n .scalar))
    (hB : ∀ i : Fin n, Tensor.getScalar B.lo i ≤ Tensor.getScalar B.hi i) :
    (toFlatBox (α := ℝ) n B).Valid :=
  hB

/-- The interval sum `boxAdd B1 B2` of two valid flat boxes over `ℝ` is valid. -/
theorem valid_boxAdd_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxAdd (α := ℝ) B1 B2).Valid := by
  rcases B1 with ⟨n1, lo1, hi1⟩
  rcases B2 with ⟨n2, lo2, hi2⟩
  by_cases h : n1 = n2
  · subst n2
    simp only [boxAdd, ↓reduceDIte]
    intro i
    rw [getScalar_zipWith, getScalar_zipWith]
    exact add_le_add (h1 i) (h2 i)
  · simpa only [boxAdd, h, ↓reduceDIte] using h1

/-- The interval difference `boxSub B1 B2` of two valid flat boxes over `ℝ` is valid. -/
theorem valid_boxSub_real (B1 B2 : FlatBox ℝ) (h1 : B1.Valid) (h2 : B2.Valid) :
    (boxSub (α := ℝ) B1 B2).Valid := by
  rcases B1 with ⟨n1, lo1, hi1⟩
  rcases B2 with ⟨n2, lo2, hi2⟩
  by_cases h : n1 = n2
  · subst n2
    simp only [boxSub, ↓reduceDIte]
    intro i
    rw [getScalar_zipWith, getScalar_zipWith]
    exact sub_le_sub (h1 i) (h2 i)
  · simpa only [boxSub, h, ↓reduceDIte] using h1

/-- The ReLU image `boxReLU B` of a valid flat box over `ℝ` is valid. -/
theorem valid_boxReLU_real (B : FlatBox ℝ) (hB : B.Valid) :
    (boxReLU (α := ℝ) B).Valid := by
  change ∀ i : Fin B.dim,
    (Tensor.map (fun x => Activation.relu (α := ℝ) x) B.lo).getScalar i ≤
      (Tensor.map (fun x => Activation.relu (α := ℝ) x) B.hi).getScalar i
  intro i
  rw [getScalar_map, getScalar_map]
  simpa only [Activation.relu_eq_max] using max_le_max (hB i) (le_refl 0)

end Graph

/-- If the flat views of the vector boxes `xB` and `bB` are valid, then so is the flat view of
`IBP.linear W xB bB` (over `ℝ`). -/
theorem IBP.linear_valid_real {m n : Nat}
    (W : Tensor ℝ [m, n])
    (xB : Box ℝ (.dim n .scalar))
    (bB : Box ℝ (.dim m .scalar))
    (hxB : (Graph.toFlatBox (α := ℝ) n xB).Valid) (hbB : (Graph.toFlatBox (α := ℝ) m bB).Valid) :
    (Graph.toFlatBox (α := ℝ) m (IBP.linear (α := ℝ) W xB bB)).Valid :=
  Box.valid_of_contains (IBP.linear (α := ℝ) W xB bB) _
    (IBP.linear_sound_real W xB bB xB.lo bB.lo
      (Box.contains_lo_of_valid xB hxB) (Box.contains_lo_of_valid bB hbB))

namespace Graph

/-- On a valid input box `Xin`, `ibpLinearParams p Xin` is none or a valid box (over `ℝ`). -/
theorem ibpLinearParams_valid_real (p : SomeLinear ℝ) (Xin : FlatBox ℝ) (hXin : Xin.Valid) :
    match ibpLinearParams p Xin with
    | none => True
    | some Bout => Bout.Valid := by
  by_cases hdim : Xin.dim = p.inDim
  · simp only [ibpLinearParams, hdim, ↓reduceDIte]
    exact IBP.linear_valid_real p.spec.weights _ _ (FlatBox.valid_getScalarBox Xin hdim hXin)
      (Box.valid_point p.spec.bias)
  · simp [ibpLinearParams, hdim]

/-- On a valid input box `Xin`, `ibpLinearParams` with weights `W` and zero bias is none or a
valid box (over `ℝ`). The op-table matmul uses `Graph.binaryMatmulBox`. -/
theorem ibp_matmul_valid_real {m n : Nat} (W : Tensor ℝ [m, n]) (Xin : FlatBox ℝ)
    (hXin : Xin.Valid) :
    match ibpLinearParams
        { inDim := n, outDim := m, spec := { weights := W, bias := Tensor.full [m] 0 } } Xin with
    | none => True
    | some Bout => Bout.Valid :=
  ibpLinearParams_valid_real
    { inDim := n, outDim := m, spec := { weights := W, bias := Tensor.full [m] 0 } } Xin hXin

end Graph

end TorchLean.CROWN
