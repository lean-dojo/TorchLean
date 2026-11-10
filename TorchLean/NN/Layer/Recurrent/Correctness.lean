/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer.Recurrent
public import TorchLean.NN.Ops.Pure
public import TorchLean.Tensor.Layers.LSTM
public import TorchLean.NN.Op.Arms

/-!
# The recurrent cells under the pure interpreter

At the pure interpreter `Ops Id α`, for every scalar with `Storage` and `Context`, the cells
`rnnCell`, `lstmCell` and `gruCell` return the values of the Spec cells `TorchLean.RNN.cell`,
`LSTM.cell`, `GRU.cell` and `GRU.ResetAfter.cell`. `Recurrent.scan` returns
the stacked outputs and the final state of `Sequence.mapAccum`, and `mapBatch f` returns
`f` on every row, stacked, with empty sequences included in both. The sequence specs are therefore
the cells run by `Recurrent.scan`. These are equalities of returned values and say nothing about
the operations an interpreter records.
-/

public section

open TorchLean.Tensor TorchLean.Ops
open TorchLean.Shape.Coord (linearize unlinearize)

namespace TorchLean

open Shape (Coord)

variable {α : Type} [Storage α] [Context α]

omit [Context α] in
/-- In `Id`, `x >>= f` is `f x`. -/
theorem nn.Layer.id_bind {β γ : Type} (x : β) (f : β → Id γ) : (x >>= f : Id γ) = f x := rfl

omit [Context α] in
/-- In `Id`, `pure x` is `x`. -/
theorem nn.Layer.id_pure {β : Type} (x : β) : (pure x : Id β) = x := rfl

omit [Context α] in
private theorem nn.Layer.reshape_eq_cast {s t : Shape} (h : s = t) (x : Tensor α s)
    (hs : s.size = t.size) :
    Tensor.reshape x t hs = h ▸ x := by
  subst h; rfl

private theorem nn.Layer.reshape_slice_vector {n : Nat} (v : Tensor α [n])
    (hd : 0 < (Shape.dim n .scalar).rank) (start len : Nat)
    (p : Shape.SliceValid (.dim n .scalar) [⟨0, hd⟩] [start] [start + len] [1])
    (q : ((Shape.dim n .scalar).slice [⟨0, hd⟩] [start] [start + len] [1]).size =
      (Shape.dim len []).size) :
    (v.slice [⟨0, hd⟩] [start] [start + len] [1] p).reshape (.dim len []) q =
      Ref.cast (m := Id) (α := α) (Shape.slice_singleton (.dim n .scalar) ⟨0, hd⟩ start len)
        (v.slice [⟨0, hd⟩] [start] [start + len] [1] p) :=
  reshape_eq_cast (Shape.slice_singleton (.dim n .scalar) ⟨0, hd⟩ start len) _ _

omit [Context α] in
private theorem nn.Layer.eq_dim_of_unstack {n : Nat} {tail : Shape} (x : Tensor α (.dim n tail))
    (f : Fin n → Tensor α tail) (h : ∀ i, Tensor.unstack x i = f i) : x = Tensor.dim f := by
  refine Rep.ext fun c => ?_
  obtain ⟨i, c⟩ := c
  simp only [Tensor.dim_apply]
  rw [← h i]
  simp only [Tensor.unstack_apply]

omit [Context α] in
private theorem nn.Layer.unstack_cast_dim {n k : Nat} {tail : Shape} (e : n = k)
    (x : Tensor α (.dim n tail)) (i : Fin k) :
    Tensor.unstack (e ▸ x : Tensor α (.dim k tail)) i = Tensor.unstack x ⟨i.val, e ▸ i.isLt⟩ := by
  subst e; rfl

omit [Context α] in
private theorem nn.Layer.unstack_cat_two {n₁ n₂ : Nat} {tail : Shape} (a : Tensor α (.dim n₁ tail))
    (b : Tensor α (.dim n₂ tail)) (i : Fin (n₁ + n₂)) :
    Tensor.unstack (Tensor.cat (s := .dim 0 tail) ⟨0, Shape.AxisInBounds.proof⟩ ![n₁, n₂]
        (Fin.cons a (Fin.cons b finZeroElim)) (.dim (n₁ + n₂) tail)) i =
      if h : i.val < n₁ then Tensor.unstack a ⟨i.val, h⟩
      else Tensor.unstack b ⟨i.val - n₁, by have := i.isLt; omega⟩ := by
  refine Rep.ext fun c => ?_
  simp only [Tensor.unstack_apply]
  rw [Tensor.cat_cons_apply]
  split_ifs <;> simp only [Tensor.unstack_apply]

/-- Under the pure interpreter `cat` of two tensors along the leading axis is `Tensor.cat` of
the pair. -/
theorem Ops.apply_cat_two {n₁ n₂ : Nat} {tail : Shape} (a : Tensor α (.dim n₁ tail))
    (b : Tensor α (.dim n₂ tail)) :
    Ops.apply (self := instOpsId) (.cat (s := .dim 0 tail) ⟨0, by simp⟩ [n₁, n₂]) (a, b, ()) =
      Tensor.cat (s := .dim 0 tail) ⟨0, Shape.AxisInBounds.proof⟩ ![n₁, n₂]
        (Fin.cons a (Fin.cons b finZeroElim)) (.dim (n₁ + n₂) tail) :=
  Op.cat_catElementParts_two a b

omit [Context α] in
private theorem nn.Layer.unstack_reshape_one {tail : Shape} (v : Tensor α tail)
    (h : tail.size = (Shape.dim 1 tail).size) :
    Tensor.unstack (Tensor.reshape v (Shape.dim 1 tail) h) ⟨0, Nat.one_pos⟩ = v := by
  simp only [Tensor.reshape]
  rw [Rep.reshape_one_cons, Tensor.unstack_dim]

/-- Read back in a shape equal to `1 :: T`, a coordinate of `T` has index zero in front and its own
indices after it. -/
private theorem nn.Layer.get_ofFin_toFin {S T : Shape} (h : S = .dim 1 T)
    (hs : T.size = S.size) (c : T.Coord) (d : Fin S.rank) :
    (Shape.Coord.get (Coord.unlinearize (Fin.cast hs (Coord.linearize c)) : S.Coord) d).val =
      if hd : d.val = 0 then 0
      else (c.get ⟨d.val - 1, by subst h; have := d.isLt; simp at this; omega⟩).val := by
  subst h
  have key : (Coord.unlinearize (Fin.cast hs (Coord.linearize c)) :
      (Shape.dim 1 T).Coord) = (0, c) := by
    rw [← Coord.unlinearize_linearize (s := 1 :: T) ((0 : Fin 1), c)]
    unfold Coord.unlinearize
    congr 1
  rw [key]
  obtain ⟨d, hd⟩ := d
  cases d with
  | zero => rfl
  | succ k => simp; rfl

/-- Selecting index `t` of the leading axis under the pure interpreter is `Tensor.unstack`. -/
theorem Ops.apply_indexAt_zero {n : Nat} {σ : Shape} (xs : Tensor α (.dim n σ)) (t : Fin n) :
    Ops.apply (self := instOpsId) (.indexAt (s := .dim n σ) ⟨0, Shape.AxisInBounds.proof⟩ t)
      (xs, ()) = Tensor.unstack xs t := by
  refine Rep.ext fun c => ?_
  erw [Tensor.unstack_apply, Tensor.reshape_apply, Tensor.slice_apply]
  congr 1
  apply Shape.Coord.ext
  intro d
  rw [Shape.Coord.get_ofFun]
  apply Fin.ext
  show _ + _ * _ = _
  rw [nn.Layer.get_ofFin_toFin (Shape.slice_singleton _ _ _ _)]
  obtain ⟨d, hd⟩ := d
  cases d with
  | zero => simp only [Fin.cast_mk, ↓reduceDIte]; rfl
  | succ k => simp [Fin.ext_iff]; rfl
  · rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ _

/-- Row `i` of the leading-axis slice of `len` rows of `x` from `start`, cast to `len` rows, is row
`start + i` of `x`. -/
private theorem nn.Layer.unstack_slice_zero {n : Nat} {σ : Shape} (x : Tensor α (.dim n σ))
    (start len : Nat) (h : start + len ≤ n)
    (e : (Shape.dim n σ).slice [⟨0, Shape.AxisInBounds.proof⟩] [start] [start + len] [1] =
      .dim len σ) (i : Fin len) :
    Tensor.unstack (Ref.cast (m := Id) (α := α) e
      (x.slice [⟨0, Shape.AxisInBounds.proof⟩] [start] [start + len] [1]
        (Shape.sliceValid_singleton _ ⟨0, Shape.AxisInBounds.proof⟩ start len h))) i =
      Tensor.unstack x ⟨start + i.val, by omega⟩ := by
  refine Rep.ext fun c => ?_
  simp only [Tensor.unstack_apply]
  erw [Ref.cast, eqRec_eq_castShape, castShape_apply, Tensor.slice_apply]
  congr 1
  refine Shape.Coord.ext_index fun k hk => ?_
  rw [Shape.Coord.index_ofFun, dite_eq_left hk]
  rcases k with _ | k
  · simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast]
    rw [Shape.Coord.index_cast e.symm]
    simp only [Nat.one_mul, ↓reduceIte]
    rfl
  · simp only [Shape.valueAt_singleton, Shape.Coord.val_get, Fin.val_cast]
    rw [Shape.Coord.index_cast e.symm]
    simp [Fin.ext_iff]

/-- Under the pure interpreter `mapBatch f` is `f` on every row. -/
theorem Ops.mapBatch_eq {n : Nat} {σ τ : Shape} (f : Tensor α σ → Tensor α τ)
    (inputs : Tensor α (Shape.dim n σ)) :
    mapBatch (m := Id) (α := α) f inputs =
      Tensor.dim fun t => f (Tensor.unstack inputs t) := by
  unfold mapBatch
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n, inputs, ih with
    | 0, _, _ => exact nn.Layer.eq_dim_of_unstack _ _ fun i => i.elim0
    | 1, inputs, _ =>
      rw [TorchLean.mapBatch.eq_2]
      simp only [nn.Layer.id_bind]
      refine nn.Layer.eq_dim_of_unstack _ _ fun i => ?_
      obtain rfl : i = 0 := Fin.fin_one_eq_zero i
      erw [nn.Layer.unstack_reshape_one _ (Nat.one_mul _).symm]
      exact congrArg f (nn.Layer.unstack_reshape_one _ (Nat.one_mul _).symm).symm
    | k + 2, inputs, ih =>
      rw [TorchLean.mapBatch.eq_3]
      simp only [nn.Layer.id_bind, nn.Layer.id_pure]
      rw [ih ((k + 2) / 2) (by omega), ih (k + 2 - (k + 2) / 2) (by omega)]
      refine nn.Layer.eq_dim_of_unstack _ _ fun i => ?_
      erw [nn.Layer.unstack_cast_dim, apply_cat_two, nn.Layer.unstack_cat_two]
      split_ifs with hi
      · erw [Tensor.unstack_dim,
          nn.Layer.unstack_slice_zero _ _ _ (by omega) (Shape.slice_singleton _ _ _ _)]
        congr 2
        exact Fin.ext (Nat.zero_add _)
      · erw [Tensor.unstack_dim,
          nn.Layer.unstack_slice_zero _ _ _ (by omega) (Shape.slice_singleton _ _ _ _)]
        congr 2
        apply Fin.ext
        simp only at hi ⊢
        omega

/-! ## Cells -/

/-- Under the pure interpreter the RNN cell is `RNN.cell`. -/
theorem nn.Layer.rnnCell_eq {inputWidth hiddenWidth : Nat}
    (cell : Linear α (inputWidth + hiddenWidth) hiddenWidth)
    (input : Tensor α [inputWidth])
    (previous : Tensor α [hiddenWidth]) :
    rnnCell (m := Id) (.cons cell.weights (.cons cell.bias .nil) : TensorPack α _) input previous =
      RNN.cell cell input previous := by
  simp only [RNN.cell]
  rw [← Op.cat_catElementParts_two]
  rfl

/-- Under the pure interpreter the LSTM cell is `LSTM.cell`, in both state components. -/
theorem nn.Layer.lstmCell_eq {inputWidth hiddenWidth : Nat}
    (cell : LSTM α inputWidth hiddenWidth) (input : Tensor α [inputWidth])
    (previous : LSTM.State α hiddenWidth) :
    lstmCell (m := Id)
        (.cons cell.forget.weights (.cons cell.forget.bias
          (.cons cell.input.weights (.cons cell.input.bias (.cons cell.candidate.weights
            (.cons cell.candidate.bias (.cons cell.output.weights (.cons cell.output.bias
              .nil))))))) : TensorPack α _)
        input (previous.hidden, previous.cell) =
      ((LSTM.cell cell input previous).hidden,
        (LSTM.cell cell input previous).cell) := by
  simp only [LSTM.cell]
  rw [← Op.cat_catElementParts_two]
  rfl

/-- Under the pure interpreter the reset-before GRU cell is `GRU.cell`, whose complement
gate is `sigmoid (-updatePre)`. -/
theorem nn.Layer.gruCell_resetBefore_eq {inputWidth hiddenWidth : Nat}
    (cell : GRU α inputWidth hiddenWidth) (input : Tensor α [inputWidth])
    (previous : Tensor α [hiddenWidth]) :
    gruCell (m := Id) .resetBefore
        (.cons cell.reset.weights (.cons cell.reset.bias
          (.cons cell.update.weights (.cons cell.update.bias (.cons cell.candidate.weights
            (.cons cell.candidate.bias .nil))))) : TensorPack α _)
        input previous =
      GRU.cell cell input previous := by
  simp only [GRU.cell]
  rw [← Op.cat_catElementParts_two, ← Op.cat_catElementParts_two]
  rfl

/-- Under the pure interpreter the reset-after GRU cell is `GRU.ResetAfter.cell`, with the
recurrent candidate bias inside the reset product. -/
theorem nn.Layer.gruCell_resetAfter_eq {inputWidth hiddenWidth : Nat}
    (cell : GRU.ResetAfter α inputWidth hiddenWidth) (input : Tensor α [inputWidth])
    (previous : Tensor α [hiddenWidth]) :
    gruCell (m := Id) .resetAfter
        (.cons cell.inputWeight (.cons cell.hiddenWeight (.cons cell.inputBias
          (.cons cell.hiddenBias .nil))) : TensorPack α _)
        input previous =
      GRU.ResetAfter.cell cell input previous := by
  simp only [GRU.ResetAfter.cell, reshape_slice_vector]
  rfl

/-! ## The scan -/

namespace nn.Layer

/-- Under the pure interpreter the balanced stack of `count` rows from `start` is the tensor whose
row `i` is row `start + i`. -/
theorem stackRows_eq_dim {rows : Nat} {tail : Shape}
    (row : Fin rows → Tensor α (Shape.dim 1 tail)) :
    (count start : Nat) → (h : start + count ≤ rows) → (hc : 0 < count) →
      stackRows (m := Id) (α := α) row start count h hc =
        Tensor.dim fun i : Fin count =>
          Tensor.unstack (row ⟨start + i.val, by omega⟩) ⟨0, Nat.one_pos⟩
  | 1, start, h, hc => by
      rw [stackRows]
      refine Rep.ext fun c => ?_
      obtain ⟨i, c⟩ := c
      obtain rfl : i = 0 := Fin.fin_one_eq_zero i
      simp only [Tensor.dim_apply,
        Tensor.unstack_apply]
      rfl
  | count + 2, start, h, hc => by
      rw [stackRows]
      have hl := stackRows_eq_dim row ((count + 2) / 2) start (by omega) (by omega)
      have hr := stackRows_eq_dim row (count + 2 - (count + 2) / 2) (start + (count + 2) / 2)
        (by omega) (by omega)
      rw [hl, hr]
      refine eq_dim_of_unstack _ _ fun i => ?_
      erw [unstack_cast_dim, apply_cat_two, unstack_cat_two]
      split_ifs with hi
      · simp only [Tensor.unstack_dim]
      · simp only [Tensor.unstack_dim]
        congr 2
        apply Fin.ext
        simp only at hi ⊢
        omega

/-- Under the pure interpreter `collectRows` is the loop of `Sequence.mapAccum`: the same
results, each reshaped to one row, and the same final state. -/
theorem collectRows_eq_mapAccumLoop {State : Type} {rows : Nat} {tail : Shape}
    (step : Fin rows → State → State × Tensor α tail)
    (h : tail.size = (Shape.dim 1 tail).size) :
    (index : Nat) → (hIndex : index ≤ rows) → (state : State) →
      (acc : Array (Tensor α tail)) → (hAcc : acc.size = index) →
      (acc' : Array (Tensor α (Shape.dim 1 tail))) → (hAcc' : acc'.size = index) →
      acc' = acc.map (Tensor.reshape · (Shape.dim 1 tail) h) →
      (collectRows (m := Id) (α := α) step index hIndex state acc' hAcc').1.val =
          (Sequence.mapAccumLoop rows step index hIndex state acc hAcc).2.val.map
            (Tensor.reshape · (Shape.dim 1 tail) h) ∧
        (collectRows (m := Id) (α := α) step index hIndex state acc' hAcc').2 =
          (Sequence.mapAccumLoop rows step index hIndex state acc hAcc).1
  | index, hIndex, state, acc, hAcc, acc', hAcc', hrel => by
    rw [collectRows.eq_1, Sequence.mapAccumLoop.eq_1]
    by_cases hLt : index < rows
    · simp only [hLt, ↓reduceDIte]
      exact collectRows_eq_mapAccumLoop step h (index + 1) hLt (step ⟨index, hLt⟩ state).1
        (acc.push (step ⟨index, hLt⟩ state).2) (by rw [Array.size_push, hAcc]) _ _
        (by rw [hrel, Array.map_push]; rfl)
    · simp only [hLt, ↓reduceDIte]
      exact ⟨hrel, rfl⟩
  termination_by index => rows - index

/-- Under the pure interpreter `scanRows` stacks the results of `Sequence.mapAccum` and
returns its final state. With no rows it returns the zero tensor and `initial`. -/
theorem scanRows_eq_mapAccum {State : Type} {rows : Nat} {tail : Shape} (initial : State)
    (step : Fin rows → State → State × Tensor α tail) :
    scanRows (m := Id) (α := α) initial step =
      (Tensor.dim (Sequence.mapAccum rows initial step).2.getScalar,
        (Sequence.mapAccum rows initial step).1) := by
  have h : tail.size = (Shape.dim 1 tail).size := by simp
  have hc := collectRows_eq_mapAccumLoop (α := α) step h 0 (Nat.zero_le rows) initial #[] rfl
    #[] rfl (by rw [Array.map_empty])
  unfold scanRows Sequence.mapAccum
  rcases hcr : collectRows (m := Id) (α := α) step 0 (Nat.zero_le rows) initial #[] rfl with
    ⟨⟨arr, harr⟩, final⟩
  rw [hcr] at hc
  obtain ⟨hv, hs⟩ := hc
  rcases hr : Sequence.mapAccumLoop rows step 0 (Nat.zero_le rows) initial #[] rfl
    with ⟨final', results, hsize⟩
  rw [hr] at hv hs
  simp only at hv hs
  subst hv hs
  simp only [id_bind, id_pure]
  by_cases hRows : 0 < rows
  · simp only [hRows, ↓reduceDIte]
    rw [stackRows_eq_dim]
    congr 1
    congr 1
    funext i
    simp only [Array.getElem_map, Tensor.getScalar_ofFn, unstack_reshape_one, Nat.zero_add]
  · simp only [hRows, ↓reduceDIte]
    exact Prod.ext (eq_dim_of_unstack _ _ fun i => absurd i.isLt (by omega)) rfl

end nn.Layer

namespace Recurrent

open nn.Layer (gruCell lstmCell rnnCell)

/-- Under the pure interpreter `Recurrent.scan` is `Sequence.mapAccum` over the rows of the
input with the outputs stacked, and an empty sequence gives the zero tensor and `initial`. -/
theorem scan_eq_mapAccum {State : Type} {length : Nat} {σ τ : Shape}
    (step : Tensor α σ → State → State × Tensor α τ) (initial : State)
    (inputs : Tensor α (Shape.dim length σ)) :
    scan (m := Id) (α := α) step initial inputs =
      let result := Sequence.mapAccum length initial fun t state =>
        step (Tensor.unstack inputs t) state
      (Tensor.dim result.2.getScalar, result.1) := by
  unfold scan
  convert nn.Layer.scanRows_eq_mapAccum (α := α) initial
    (fun t state => step (Tensor.unstack inputs t) state) using 2
  funext t state
  simp only [nn.Layer.id_bind]
  erw [apply_indexAt_zero]

omit [Context α] in
private theorem mapAccumLoop_map {S S' R : Type} (f : S → S') {n : Nat}
    (step : Fin n → S → S × R) (step' : Fin n → S' → S' × R)
    (h : ∀ i s, step' i (f s) = (f (step i s).1, (step i s).2)) :
    (i : Nat) → (hi : i ≤ n) → (s : S) → (acc : Array R) → (hacc : acc.size = i) →
      Sequence.mapAccumLoop n step' i hi (f s) acc hacc =
        (f (Sequence.mapAccumLoop n step i hi s acc hacc).1,
          (Sequence.mapAccumLoop n step i hi s acc hacc).2)
  | i, hi, s, acc, hacc => by
    rw [Sequence.mapAccumLoop.eq_1, Sequence.mapAccumLoop.eq_1]
    by_cases hlt : i < n
    · have e := h ⟨i, hlt⟩ s
      rcases hs : step ⟨i, hlt⟩ s with ⟨next, result⟩
      rw [hs] at e
      simp only [hlt, ↓reduceDIte, e, hs]
      exact mapAccumLoop_map f step step' h (i + 1) hlt next (acc.push result) _
    · simp only [hlt, ↓reduceDIte]
  termination_by i => n - i

/-- Under the pure interpreter, `Recurrent.scan` with the RNN cell is `RNN.sequence`: the
stacked hidden states and the final hidden state. -/
theorem scan_rnnCell {length inputWidth hiddenWidth : Nat}
    (cell : Linear α (inputWidth + hiddenWidth) hiddenWidth)
    (inputs : Tensor α [length, inputWidth]) (initial : Tensor α [hiddenWidth]) :
    scan (m := Id) (α := α) (fun input previous =>
        let next := rnnCell (m := Id)
          (.cons cell.weights (.cons cell.bias .nil) : TensorPack α _) input previous
        (next, next)) initial inputs =
      RNN.sequence cell inputs initial := by
  rw [scan_eq_mapAccum]
  simp only [nn.Layer.rnnCell_eq]
  rfl

/-- Under the pure interpreter, `Recurrent.scan` with the reset-before GRU cell is
`GRU.sequence`: the stacked hidden states and the final hidden state. -/
theorem scan_gruCell_resetBefore {length inputWidth hiddenWidth : Nat}
    (cell : GRU α inputWidth hiddenWidth) (inputs : Tensor α [length, inputWidth])
    (initial : Tensor α [hiddenWidth]) :
    scan (m := Id) (α := α) (fun input previous =>
        let next := gruCell (m := Id) .resetBefore
          (.cons cell.reset.weights (.cons cell.reset.bias
            (.cons cell.update.weights (.cons cell.update.bias (.cons cell.candidate.weights
              (.cons cell.candidate.bias .nil))))) : TensorPack α _)
          input previous
        (next, next)) initial inputs =
      GRU.sequence cell inputs initial := by
  rw [scan_eq_mapAccum]
  simp only [nn.Layer.gruCell_resetBefore_eq]
  rfl

/-- Under the pure interpreter, `Recurrent.scan` with the reset-after GRU cell is
`GRU.ResetAfter.sequence`: the stacked hidden states and the final hidden state. -/
theorem scan_gruCell_resetAfter {length inputWidth hiddenWidth : Nat}
    (cell : GRU.ResetAfter α inputWidth hiddenWidth)
    (inputs : Tensor α [length, inputWidth]) (initial : Tensor α [hiddenWidth]) :
    scan (m := Id) (α := α) (fun input previous =>
        let next := gruCell (m := Id) .resetAfter
          (.cons cell.inputWeight (.cons cell.hiddenWeight (.cons cell.inputBias
            (.cons cell.hiddenBias .nil))) : TensorPack α _)
          input previous
        (next, next)) initial inputs =
      GRU.ResetAfter.sequence cell inputs initial := by
  rw [scan_eq_mapAccum]
  simp only [nn.Layer.gruCell_resetAfter_eq]
  rfl

/-- Running the LSTM cell with `Recurrent.scan` gives `LSTM.sequence`: the stacked hidden
states, and the final hidden and cell states. -/
theorem scan_lstmCell {length inputWidth hiddenWidth : Nat}
    (cell : LSTM α inputWidth hiddenWidth) (inputs : Tensor α [length, inputWidth])
    (initial : LSTM.State α hiddenWidth) :
    scan (m := Id) (α := α) (fun input previous =>
        let next := lstmCell (m := Id)
          (.cons cell.forget.weights (.cons cell.forget.bias
            (.cons cell.input.weights (.cons cell.input.bias (.cons cell.candidate.weights
              (.cons cell.candidate.bias (.cons cell.output.weights (.cons cell.output.bias
                .nil))))))) : TensorPack α _)
          input previous
        (next, next.1)) (initial.hidden, initial.cell) inputs =
      let result := LSTM.sequence cell inputs initial
      (result.1, (result.2.hidden, result.2.cell)) := by
  rw [scan_eq_mapAccum]
  have h := mapAccumLoop_map (fun state : LSTM.State α hiddenWidth =>
      ((state.hidden, state.cell) :
        Ref (m := Id) (α := α) [hiddenWidth] × Ref (m := Id) (α := α) [hiddenWidth]))
    (fun t previous =>
      let state := LSTM.cell cell (Tensor.unstack inputs t) previous
      (state, state.hidden))
    (fun t previous =>
      let next := lstmCell (m := Id)
        (.cons cell.forget.weights (.cons cell.forget.bias
          (.cons cell.input.weights (.cons cell.input.bias (.cons cell.candidate.weights
            (.cons cell.candidate.bias (.cons cell.output.weights (.cons cell.output.bias
              .nil))))))) : TensorPack α _)
        (Tensor.unstack inputs t) previous
      (next, next.1))
    (fun t previous => by simp only [nn.Layer.lstmCell_eq]) 0 (Nat.zero_le length) initial
    #[] rfl
  simp only [Sequence.mapAccum, LSTM.sequence, h]

end Recurrent

end TorchLean
