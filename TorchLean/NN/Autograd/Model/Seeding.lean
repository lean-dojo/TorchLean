/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.TensorVectorization
public import TorchLean.NN.Tape.Lowering.HigherOrder

/-!
# Fixed-state derivative seeds

Input derivatives use zero directions for parameters, but a subsequent pullback differentiates
with respect to the whole context. The runtime seed is therefore read as a jet on the full
parameter-and-input space, which keeps the parameter dependence and needs no normed-space instance
on the heterogeneous tensor pack or on the sealed model-state type.
-/

public section

namespace TorchLean.TensorPack.JetRelated

private theorem seed_get {shapes : List Shape} {shape : Shape} {n : Nat}
    (state : TensorPack ℝ shapes) (input : Tensor ℝ shape)
    (directions : Fin n → Tensor ℝ shape) {s : Shape} (index : Idx (shapes ++ [shape]) s) :
    getIdx ((state.map (Tensor.map (Dual.Nested.ofPrimal n))).append
      (Pack.singleton (Dual.Nested.seedTensor directions input))) index =
      Dual.Nested.seedTensor (fun k => getIdx
        ((TensorPack.zero (ss := shapes)).append (Pack.singleton (directions k))) index)
        (getIdx (state.append (Pack.singleton input)) index) := by
  induction state with
  | nil =>
    obtain ⟨i, hi⟩ := index
    cases i using Fin.cases with
    | zero => cases hi; rfl
    | succ i => exact Fin.elim0 i
  | cons x xs ih =>
    obtain ⟨i, hi⟩ := index
    cases i using Fin.cases with
    | zero =>
      cases hi
      exact (Dual.Nested.seedTensor_zero n x).symm
    | succ i => exact ih ⟨i, hi⟩

/-- Constant state followed by a seeded input is the full-context jet in input-only directions.

The base point contains all state entries. Zero parameter directions make this jet take input
derivatives, and the differentiated map keeps its dependence on the parameters.
-/
theorem const_append_seed {shapes : List Shape} {shape : Shape} {n : Nat}
    (state : TensorPack ℝ shapes) (input : Tensor ℝ shape)
    (directions : Fin n → Tensor ℝ shape) :
    JetRelated
      (fun k => toVec ((TensorPack.zero (ss := shapes)).append
        (Pack.singleton (directions k))))
      (toVec (state.append (Pack.singleton input))) ofVec
      ((state.map (Tensor.map (Dual.Nested.ofPrimal n))).append
        (Pack.singleton (Dual.Nested.seedTensor directions input))) := by
  apply of_get
  intro s index
  have hread := funext (CtxVec.getTensorCLM_apply index)
  constructor
  · rw [← hread]
    exact (CtxVec.getTensorCLM index).contDiff
  · rw [seed_get, ← hread]
    simpa only [CtxVec.getTensorCLM_apply, ofVec_toVec] using
      (DualTensor.jet_linear
        (fun k => toVec ((TensorPack.zero (ss := shapes)).append
          (Pack.singleton (directions k)))) (CtxVec.getTensorCLM index)
        (toVec (state.append (Pack.singleton input)))).symm

end TorchLean.TensorPack.JetRelated
