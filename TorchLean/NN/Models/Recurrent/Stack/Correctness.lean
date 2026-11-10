/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Recurrent.Stack
public import TorchLean.NN.Layer.Recurrent.Correctness

/-!
# The recurrent models under the pure interpreter

At the pure interpreter `Ops Id α`, for every scalar with `Storage` and `Context`, `run` is the
state recursion `TorchLean.Sequence.mapAccum` with its hidden vectors stacked, and `zipRows` is
`Tensor.zipEach` over the leading axis. A cell whose state is its hidden vector ends in its last
hidden row, or in the initial state for an empty sequence. Each model is its Spec formula, built
from the recurrent sequence specs of the cells, heads and embeddings on every row
(`Tensor.mapLeading`), the time reversal `Tensor.flip`, and the row concatenation `Tensor.cat`. A
final section states the structural contracts of the models: empty sequences and stacks,
per-layer states, the state count of `languageModel`, both LSTM state components, the row
alignment of `bidirectional`, and the residual feature `residual` returns.
-/

public section

open TorchLean.Tensor

namespace TorchLean.Recurrent

open nn.Layer (id_bind id_pure)
open Ops (mapBatch_eq apply_indexAt_zero)

variable {α : Type} [Storage α] [Context α]

omit [Context α] in
private theorem mapAccum_fst_eq {S R : Type} [Storage R] (n : Nat) (s : S)
    (step : Fin n → S → S × R) :
    (Sequence.mapAccum n s step).1 =
      (Sequence.mapAccumLoop n step 0 (Nat.zero_le n) s #[] rfl).1 := rfl

omit [Context α] in
private theorem mapAccum_getScalar {S R : Type} [Storage R] (n : Nat) (s : S)
    (step : Fin n → S → S × R) (i : Fin n) :
    (Sequence.mapAccum n s step).2.getScalar i =
      (Sequence.mapAccumLoop n step 0 (Nat.zero_le n) s #[] rfl).2.val[i.val]'(by
        rw [(Sequence.mapAccumLoop n step 0 (Nat.zero_le n) s #[] rfl).2.property]
        exact i.isLt) :=
  Tensor.getScalar_ofFn _ i

omit [Context α] in
/-- The results of a traversal whose step results do not depend on the state. -/
private theorem mapAccumLoop_results {S R : Type} {n : Nat} (step : Fin n → S → S × R)
    (g : Fin n → R) (h : ∀ i s, (step i s).2 = g i) :
    (i : Nat) → (hi : i ≤ n) → (s : S) → (acc : Array R) → (hacc : acc.size = i) →
      (∀ (j : Nat) (hj : j < i), acc[j]'(by omega) = g ⟨j, by omega⟩) →
      ∀ (j : Nat) (hj : j < n),
        (Sequence.mapAccumLoop n step i hi s acc hacc).2.val[j]'(by
          rw [(Sequence.mapAccumLoop n step i hi s acc hacc).2.property]; exact hj) =
          g ⟨j, hj⟩
  | i, hi, s, acc, hacc, hacc', j, hj => by
    rw [Sequence.mapAccumLoop.eq_1]
    by_cases hlt : i < n
    · simp only [hlt, ↓reduceDIte]
      refine mapAccumLoop_results step g h (i + 1) hlt _ _ _ (fun k hk => ?_) j hj
      by_cases hki : k < i
      · rw [Array.getElem_push_lt (by omega : k < acc.size)]; exact hacc' k hki
      · have hk' : k = i := by omega
        subst hk'
        simp only [Array.getElem_push, hacc, lt_irrefl, ↓reduceDIte]
        exact h _ _
    · simp only [hlt, ↓reduceDIte]
      exact hacc' j (by omega)
  termination_by i => n - i

omit [Context α] in
/-- A traversal whose step results do not depend on the state collects `g`. -/
private theorem mapAccum_snd {S R : Type} [Storage R] {n : Nat} (s : S) (step : Fin n → S → S × R)
    (g : Fin n → R) (h : ∀ i s, (step i s).2 = g i) :
    (Sequence.mapAccum n s step).2 = Tensor.ofFn g := by
  have hres := mapAccumLoop_results step g h 0 (Nat.zero_le n) s #[] rfl
    (fun j hj => absurd hj (Nat.not_lt_zero j))
  unfold Sequence.mapAccum
  rcases hr : Sequence.mapAccumLoop n step 0 (Nat.zero_le n) s #[] rfl with
    ⟨final, results⟩
  rw [hr] at hres
  dsimp only
  congr 1
  funext i
  exact hres i.val i.isLt

omit [Context α] in
/-- A traversal whose step reports `k` of its next state ends in a state whose `k` is the last
result, or `k s` if `n = 0`. -/
private theorem mapAccumLoop_last {S R : Type} {n : Nat} (g : Fin n → S → S) (k : S → R) :
    (i : Nat) → (hi : i ≤ n) → (s : S) → (acc : Array R) → (hacc : acc.size = i) →
      (∀ hpos : 0 < i, acc[i - 1]'(by omega) = k s) →
      k (Sequence.mapAccumLoop n (fun t s => (g t s, k (g t s))) i hi s acc hacc).1 =
        if hn : 0 < n then
          (Sequence.mapAccumLoop n (fun t s => (g t s, k (g t s))) i hi s acc
            hacc).2.val[n - 1]'(by
              rw [(Sequence.mapAccumLoop n (fun t s => (g t s, k (g t s))) i hi s
                acc hacc).2.property]; omega)
        else k s
  | i, hi, s, acc, hacc, hlast => by
    rw [Sequence.mapAccumLoop.eq_1]
    by_cases hlt : i < n
    · have hn : 0 < n := by omega
      simp only [hlt, ↓reduceDIte]
      have ih := mapAccumLoop_last g k (i + 1) hlt (g ⟨i, hlt⟩ s) (acc.push (k (g ⟨i, hlt⟩ s)))
        (by rw [Array.size_push, hacc]) (fun _ => by
          simp only [Nat.add_sub_cancel, Array.getElem_push, hacc, lt_irrefl, ↓reduceDIte])
      simp only [hn, ↓reduceDIte] at ih ⊢
      exact ih
    · simp only [hlt, ↓reduceDIte]
      have hin : i = n := by omega
      subst hin
      by_cases hn : 0 < i
      · simp only [hn, ↓reduceDIte]; exact (hlast hn).symm
      · simp only [hn, ↓reduceDIte]
  termination_by i => n - i

/-- Under the pure interpreter `run` is `Sequence.mapAccum` over the rows of the input: the
hidden vectors of the successive states stacked, and the final state. -/
theorem run_eq {State : Type} {length inputWidth hiddenWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    run (m := Id) cell initial inputs =
      let result := Sequence.mapAccum length initial fun t state =>
        let next := cell.step (Tensor.unstack inputs t) state
        (next, (cell.hidden next : Tensor α [hiddenWidth]))
      (Tensor.dim result.2.getScalar, result.1) :=
  scan_eq_mapAccum (α := α) (σ := [inputWidth]) (τ := [hiddenWidth]) _ initial inputs

/-- Under the pure interpreter `zipRows f` is `f` on every pair of rows. -/
theorem zipRows_eq {length : Nat} {σ σ' τ : Shape} (f : Tensor α σ → Tensor α σ' → Tensor α τ)
    (left : Tensor α (Shape.dim length σ)) (right : Tensor α (Shape.dim length σ')) :
    zipRows (m := Id) (α := α) f left right =
      Tensor.dim fun t => f (Tensor.unstack left t) (Tensor.unstack right t) := by
  unfold zipRows
  calc _ = Prod.fst <$> nn.Layer.scanRows (m := Id) (α := α) ()
        (fun t (u : Unit) => (u, f (Tensor.unstack left t) (Tensor.unstack right t))) := by
        congr 2
        funext t u
        erw [apply_indexAt_zero, apply_indexAt_zero]
        rfl
    _ = _ := ?_
  erw [nn.Layer.scanRows_eq_mapAccum (α := α) (tail := τ) ()
    (fun t u => (u, f (Tensor.unstack left t) (Tensor.unstack right t)))]
  rw [mapAccum_snd () _ (fun t => f (Tensor.unstack left t) (Tensor.unstack right t))
    (fun _ _ => rfl)]
  change Tensor.dim (Tensor.ofFn _).getScalar = _
  congr 1
  funext t
  exact Tensor.getScalar_ofFn _ t

/-- Under the pure interpreter the hidden vector of the final state of `run` is its last hidden row,
and for an empty sequence the hidden vector of `initial`. -/
theorem hidden_run_snd_eq_unstack {State : Type} {length inputWidth hiddenWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    cell.hidden (run (m := Id) cell initial inputs).2 =
      if hLength : length = 0 then cell.hidden initial
      else Tensor.unstack (run (m := Id) cell initial inputs).1
        ⟨length - 1, Nat.sub_one_lt hLength⟩ := by
  rw [run_eq]
  dsimp only
  have key := mapAccumLoop_last (fun t state => cell.step (Tensor.unstack inputs t) state)
    cell.hidden 0 (Nat.zero_le length) initial #[] rfl (fun h => absurd h (Nat.lt_irrefl 0))
  beta_reduce at key
  rw [mapAccum_fst_eq]
  by_cases hLength : length = 0
  · subst hLength
    simp only [Nat.lt_irrefl, ↓reduceDIte] at key ⊢
    exact key
  · have hpos : 0 < length := Nat.pos_of_ne_zero hLength
    simp only [hpos, ↓reduceDIte] at key
    simp only [hLength, ↓reduceDIte, Tensor.unstack_dim]
    rw [mapAccum_getScalar]
    exact key

/-- Under the pure interpreter a cell whose state is its hidden vector ends in its last hidden row,
and an empty sequence keeps the initial state. -/
theorem run_snd_eq_unstack {length inputWidth hiddenWidth : Nat}
    (cell : Cell Id α (Tensor α [hiddenWidth]) inputWidth hiddenWidth)
    (hidden_eq : ∀ state, cell.hidden state = state) (initial : Tensor α [hiddenWidth])
    (inputs : Tensor α [length, inputWidth]) :
    (run (m := Id) cell initial inputs).2 =
      if hLength : length = 0 then initial
      else Tensor.unstack (run (m := Id) cell initial inputs).1
        ⟨length - 1, Nat.sub_one_lt hLength⟩ := by
  have h := hidden_run_snd_eq_unstack cell initial inputs
  simp only [hidden_eq] at h
  exact h

/-- Running the RNN cell is `RNN.sequence`: the hidden output stream and the final hidden
state. -/
theorem run_rnn {length inputWidth hiddenWidth : Nat}
    (cell : Linear α (inputWidth + hiddenWidth) hiddenWidth)
    (initial : Tensor α [hiddenWidth]) (inputs : Tensor α [length, inputWidth]) :
    run (m := Id) (Cell.rnn (.cons cell.weights (.cons cell.bias .nil) : TensorPack α _))
        initial inputs = RNN.sequence cell inputs initial :=
  scan_rnnCell cell inputs initial

/-- Running the `.resetBefore` GRU cell is `GRU.sequence`: the hidden output stream and the
final hidden state. -/
theorem run_gru_resetBefore {length inputWidth hiddenWidth : Nat}
    (cell : GRU α inputWidth hiddenWidth) (initial : Tensor α [hiddenWidth])
    (inputs : Tensor α [length, inputWidth]) :
    run (m := Id) (Cell.gru .resetBefore
        (.cons cell.reset.weights (.cons cell.reset.bias
          (.cons cell.update.weights (.cons cell.update.bias (.cons cell.candidate.weights
            (.cons cell.candidate.bias .nil))))) : TensorPack α _))
        initial inputs = GRU.sequence cell inputs initial :=
  scan_gruCell_resetBefore cell inputs initial

/-- Running the `.resetAfter` GRU cell is `GRU.ResetAfter.sequence`: the hidden output
stream and the final hidden state. -/
theorem run_gru_resetAfter {length inputWidth hiddenWidth : Nat}
    (cell : GRU.ResetAfter α inputWidth hiddenWidth) (initial : Tensor α [hiddenWidth])
    (inputs : Tensor α [length, inputWidth]) :
    run (m := Id) (Cell.gru .resetAfter
        (.cons cell.inputWeight (.cons cell.hiddenWeight (.cons cell.inputBias
          (.cons cell.hiddenBias .nil))) : TensorPack α _)) initial inputs =
      GRU.ResetAfter.sequence cell inputs initial :=
  scan_gruCell_resetAfter cell inputs initial

/-- Running the LSTM cell is `LSTM.sequence`: the hidden output stream, and both final
state components. -/
theorem run_lstm {length inputWidth hiddenWidth : Nat}
    (cell : LSTM α inputWidth hiddenWidth) (initial : LSTM.State α hiddenWidth)
    (inputs : Tensor α [length, inputWidth]) :
    run (m := Id) (Cell.lstm (.cons cell.forget.weights (.cons cell.forget.bias
        (.cons cell.input.weights (.cons cell.input.bias (.cons cell.candidate.weights
          (.cons cell.candidate.bias (.cons cell.output.weights (.cons cell.output.bias
            .nil))))))) : TensorPack α _))
        (initial.hidden, initial.cell) inputs =
      let result := LSTM.sequence cell inputs initial
      (result.1, (result.2.hidden, result.2.cell)) :=
  scan_lstmCell cell inputs initial

/-! ## Models -/

variable {State : Type}

/-- Under the pure interpreter `sequence` is `head` on every row of `run`, with its final state. -/
theorem sequence_eq {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    sequence (m := Id) cell head initial inputs =
      (Tensor.mapLeading [length] head (run (m := Id) cell initial inputs).1,
        (run (m := Id) cell initial inputs).2) := by
  unfold sequence
  simp only [id_bind, id_pure]
  erw [mapBatch_eq (α := α) (σ := [hiddenWidth]) head]
  rfl

/-- Under the pure interpreter `stacked` is `head` on every row of `Stack.run`, with its final
states. -/
theorem stacked_eq {StateOf : Nat → Type} {length inputWidth hiddenWidth outputWidth : Nat}
    (layers : Stack Id α StateOf inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : layers.States)
    (inputs : Tensor α [length, inputWidth]) :
    stacked (m := Id) layers head initial inputs =
      (Tensor.mapLeading [length] head (layers.run (m := Id) initial inputs).1,
        (layers.run (m := Id) initial inputs).2) := by
  unfold stacked
  simp only [id_bind, id_pure]
  erw [mapBatch_eq (α := α) (σ := [hiddenWidth]) head]
  rfl

/-- Under the pure interpreter `classifier` is `head` on the hidden vector of the final state. -/
theorem classifier_eq {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    classifier (m := Id) cell head initial inputs =
      head (cell.hidden (run (m := Id) cell initial inputs).2) := rfl

/-- Under the pure interpreter `generator` is `sequence` on the embedded tokens. -/
theorem generator_eq {length tokenWidth inputWidth hiddenWidth outputWidth : Nat}
    (embedding : Tensor α [tokenWidth] → Tensor α [inputWidth])
    (cell : Cell Id α State inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (tokens : Tensor α [length, tokenWidth]) :
    generator (m := Id) embedding cell head initial tokens =
      sequence (m := Id) cell head initial (Tensor.mapLeading [length] embedding tokens) := by
  unfold generator
  simp only [id_bind]
  erw [mapBatch_eq (α := α) (σ := [tokenWidth]) embedding]
  rfl

/-- Under the pure interpreter `bidirectional` is `head` on `[forward; backward]` at every step,
where the backward stream runs on the time-reversed inputs and is reversed back. -/
theorem bidirectional_eq {State' : Type}
    {length inputWidth hiddenWidth hiddenWidth' outputWidth : Nat}
    (forwardCell : Cell Id α State inputWidth hiddenWidth)
    (backwardCell : Cell Id α State' inputWidth hiddenWidth')
    (head : Tensor α [hiddenWidth + hiddenWidth'] → Tensor α [outputWidth])
    (forwardInitial : State) (backwardInitial : State')
    (inputs : Tensor α [length, inputWidth]) :
    bidirectional (m := Id) forwardCell backwardCell head forwardInitial backwardInitial inputs =
      Tensor.mapLeading [length] head
        (Tensor.zipEach [length] [hiddenWidth + hiddenWidth']
          (fun a b => Tensor.cat (s := [0]) ⟨0, by simp⟩ ![hiddenWidth, hiddenWidth']
            (Fin.cons a (Fin.cons b finZeroElim)) [hiddenWidth + hiddenWidth'])
          (run (m := Id) forwardCell forwardInitial inputs).1
          (Tensor.flip (run (m := Id) backwardCell backwardInitial
            (Tensor.flip inputs [⟨0, by simp⟩])).1 [⟨0, by simp⟩])) := by
  unfold bidirectional
  simp only [id_bind]
  erw [zipRows_eq (α := α) (σ := [hiddenWidth]) (σ' := [hiddenWidth'])]
  show Tensor.dim _ = Tensor.dim _
  congr 1
  funext t
  erw [Ops.apply_cat_two]
  congr 1
  simp only [Tensor.zipEach]
  rw [Tensor.unstack_dim]
  rfl

/-- Under the pure interpreter `languageModel` is `stacked` over the cells of `layers` on the
embedded tokens, with the final states as a list, when `initial` has one state per layer, and `none`
otherwise. -/
theorem languageModel_eq {length tokenWidth width outputWidth : Nat}
    (embedding : Tensor α [tokenWidth] → Tensor α [width])
    (layers : List (Cell Id α State width width))
    (head : Tensor α [width] → Tensor α [outputWidth]) (initial : List State)
    (tokens : Tensor α [length, tokenWidth]) :
    languageModel (m := Id) embedding layers head initial tokens =
      ((Stack.ofList layers).states? initial).map fun states =>
        let result := stacked (m := Id) (Stack.ofList layers) head states
          (Tensor.mapLeading [length] embedding tokens)
        (result.1, (Stack.ofList layers).statesList result.2) := by
  unfold languageModel
  cases (Stack.ofList layers).states? initial with
  | none => rfl
  | some states =>
      simp only [id_bind, id_pure, Option.map_some]
      erw [mapBatch_eq (α := α) (σ := [tokenWidth]) embedding]
      rfl

/-- Under the pure interpreter `encoderDecoder` runs `generator` from the encoder's final state. -/
theorem encoderDecoder_eq {sourceLength targetLength sourceWidth targetWidth sourceInput
      targetInput hiddenWidth outputWidth : Nat}
    (sourceEmbedding : Tensor α [sourceWidth] → Tensor α [sourceInput])
    (encoder : Cell Id α State sourceInput hiddenWidth)
    (targetEmbedding : Tensor α [targetWidth] → Tensor α [targetInput])
    (decoder : Cell Id α State targetInput hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (source : Tensor α [sourceLength, sourceWidth])
    (target : Tensor α [targetLength, targetWidth]) :
    encoderDecoder (m := Id) sourceEmbedding encoder targetEmbedding decoder head initial source
        target =
      let encoded := (run (m := Id) encoder initial
        (Tensor.mapLeading [sourceLength] sourceEmbedding source)).2
      let decoded := generator (m := Id) targetEmbedding decoder head encoded target
      (decoded.1, encoded, decoded.2) := by
  unfold encoderDecoder
  simp only [id_bind, id_pure]
  erw [mapBatch_eq (α := α) (σ := [sourceWidth]) sourceEmbedding]
  rfl

/-- Under the pure interpreter `residual` is `head` on the features `h_t + projection x_t` at
every step, with the last feature (the hidden vector of `initial` for an empty sequence). -/
theorem residual_eq {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (projection : Tensor α [inputWidth] → Tensor α [hiddenWidth])
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    residual (m := Id) cell projection head initial inputs =
      let feature := fun t : Fin length =>
        Tensor.unstack (run (m := Id) cell initial inputs).1 t +
          projection (Tensor.unstack inputs t)
      (Tensor.dim fun t => head (feature t),
        if hLength : 0 < length then feature ⟨length - 1, Nat.sub_one_lt (Nat.ne_of_gt hLength)⟩
        else cell.hidden initial) := by
  unfold residual
  simp only [id_bind, id_pure]
  erw [zipRows_eq (α := α) (σ := [hiddenWidth]) (σ' := [inputWidth])]
  erw [mapBatch_eq (α := α) (σ := [hiddenWidth]) head]
  simp only [Tensor.unstack_dim]
  by_cases hLength : 0 < length
  · simp only [hLength, ↓reduceDIte]
    erw [apply_indexAt_zero]
    exact Prod.ext rfl (Tensor.unstack_dim (α := α) _ _)
  · simp only [hLength, ↓reduceDIte]
    rfl

/-! ## Contracts

Each contract below is stated against a Spec function where one exists and in its own terms
otherwise. They cover empty sequences and stacks, the state of every stack layer, the state count
of `languageModel`, both LSTM state components, the row alignment of `bidirectional`, and the
value `residual` returns.
-/
/-- Under the pure interpreter an empty sequence leaves the state of any cell unchanged. -/
theorem run_snd_of_length_zero {State : Type} {inputWidth hiddenWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth) (initial : State)
    (inputs : Tensor α [0, inputWidth]) : (run (m := Id) cell initial inputs).2 = initial := by
  rw [run_eq]
  unfold Sequence.mapAccum
  rw [Sequence.mapAccumLoop.eq_1]
  simp

/-- Under the pure interpreter `classifier` on an empty sequence is `head` on the hidden vector of
the initial state. -/
theorem classifier_of_length_zero {State : Type} {inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [0, inputWidth]) :
    classifier (m := Id) cell head initial inputs = head (cell.hidden initial) := by
  rw [classifier_eq, run_snd_of_length_zero]

namespace Stack

variable {StateOf : Nat → Type}

/-- Under the pure interpreter a stack runs its first cell on the inputs from the first state and
the rest of the stack on that hidden stream, and each layer's final state is the final state of
its own run. -/
theorem run_cons {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α (StateOf hiddenWidth) inputWidth hiddenWidth)
    (rest : Stack Id α StateOf hiddenWidth outputWidth) (states : (Stack.cons cell rest).States)
    (inputs : Tensor α [length, inputWidth]) :
    (Stack.cons cell rest).run (m := Id) states inputs =
      let first := Recurrent.run (m := Id) cell states.1 inputs
      let others := rest.run (m := Id) states.2 first.1
      (others.1, ((first.2, others.2) : StateOf hiddenWidth × rest.States)) := rfl

/-- Under the pure interpreter an empty sequence leaves every layer state of a stack unchanged. -/
theorem run_snd_of_length_zero : {inputWidth outputWidth : Nat} →
    (layers : Stack Id α StateOf inputWidth outputWidth) → (states : layers.States) →
    (inputs : Tensor α [0, inputWidth]) → (layers.run (m := Id) states inputs).2 = states
  | _, _, .nil, _, _ => rfl
  | _, _, .cons cell rest, states, inputs => by
      rw [run_cons]
      dsimp only
      rw [Recurrent.run_snd_of_length_zero, run_snd_of_length_zero rest]
      rfl

/-- The states of the stack of a list of cells are defined exactly when there is one state per
cell. -/
theorem isSome_states?_ofList {m : Type → Type} [Monad m] [Ops m α] {State : Type} {width : Nat}
    (layers : List (Cell m α State width width)) (initial : List State) :
    ((Stack.ofList layers).states? initial).isSome ↔ initial.length = layers.length := by
  induction layers generalizing initial with
  | nil =>
      cases initial with
      | nil => exact iff_of_true rfl rfl
      | cons _ _ => simp [Stack.ofList, Stack.states?]
  | cons cell cells ih =>
      cases initial with
      | nil => simp [Stack.ofList, Stack.states?]
      | cons state states =>
          simp only [Stack.ofList, Stack.states?, List.length_cons, Nat.add_right_cancel_iff]
          rw [← ih states]
          cases (Stack.ofList cells).states? states <;> rfl

/-- The states accepted from a list give that list back. -/
theorem statesList_of_states?_eq_some {m : Type → Type} [Monad m] [Ops m α] {State : Type}
    {width : Nat} (layers : List (Cell m α State width width)) (initial : List State)
    (states : (Stack.ofList layers).States)
    (h : (Stack.ofList layers).states? initial = some states) :
    (Stack.ofList layers).statesList states = initial := by
  induction layers generalizing initial with
  | nil => cases initial <;> simp_all [Stack.ofList, Stack.states?, Stack.statesList]
  | cons cell cells ih =>
      cases initial with
      | nil => simp [Stack.ofList, Stack.states?] at h
      | cons state rest =>
          simp only [Stack.ofList, Stack.states?] at h
          cases htail : (Stack.ofList cells).states? rest with
          | none => rw [htail] at h; cases h
          | some tail =>
              rw [htail] at h
              cases h
              show state :: (Stack.ofList cells).statesList tail = state :: rest
              rw [ih rest tail htail]

/-- The state list of the stack of a list of cells has one state per cell. -/
theorem length_statesList_ofList {m : Type → Type} [Monad m] [Ops m α] {State : Type}
    {width : Nat} (layers : List (Cell m α State width width))
    (states : (Stack.ofList layers).States) :
    ((Stack.ofList layers).statesList states).length = layers.length := by
  induction layers with
  | nil => rfl
  | cons cell cells ih => exact congrArg Nat.succ (ih states.2)

end Stack

/-- Under the pure interpreter `stacked` with no recurrent layers is `head` on every input row, and
it returns its states unchanged. -/
theorem stacked_nil {StateOf : Nat → Type} {length width outputWidth : Nat}
    (head : Tensor α [width] → Tensor α [outputWidth])
    (states : (Stack.nil : Stack Id α StateOf width width).States)
    (inputs : Tensor α [length, width]) :
    stacked (m := Id) Stack.nil head states inputs =
      (Tensor.mapLeading [length] head inputs, states) := by
  rw [stacked_eq]; rfl

/-- Under the pure interpreter `languageModel` returns `none` exactly when `initial` does not have
one state per layer. -/
theorem languageModel_eq_none_iff {State : Type} {length tokenWidth width outputWidth : Nat}
    (embedding : Tensor α [tokenWidth] → Tensor α [width])
    (layers : List (Cell Id α State width width))
    (head : Tensor α [width] → Tensor α [outputWidth]) (initial : List State)
    (tokens : Tensor α [length, tokenWidth]) :
    languageModel (m := Id) embedding layers head initial tokens = none ↔
      initial.length ≠ layers.length := by
  rw [languageModel_eq, Option.map_eq_none_iff, ← Option.not_isSome_iff_eq_none,
    Stack.isSome_states?_ofList]

/-- Under the pure interpreter `sequence` with the LSTM cell is the head on every hidden vector of
`LSTM.sequence`, with both final state components. -/
theorem sequence_lstm {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : LSTM α inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth])
    (initial : LSTM.State α hiddenWidth) (inputs : Tensor α [length, inputWidth]) :
    sequence (m := Id) (Cell.lstm (.cons cell.forget.weights
        (.cons cell.forget.bias (.cons cell.input.weights (.cons cell.input.bias
          (.cons cell.candidate.weights (.cons cell.candidate.bias (.cons cell.output.weights
            (.cons cell.output.bias .nil))))))) : TensorPack α _))
        head (initial.hidden, initial.cell) inputs =
      let result := LSTM.sequence cell inputs initial
      (Tensor.mapLeading [length] head result.1, (result.2.hidden, result.2.cell)) := by
  rw [sequence_eq, run_lstm]

omit [Context α] in
/-- Reversing the leading axis reads row `t` from row `n - 1 - t`. -/
theorem unstack_flip_zero {n : Nat} {tail : Shape} (x : Tensor α (.dim n tail)) (t : Fin n) :
    Tensor.unstack (Tensor.flip x [⟨0, by simp⟩]) t = Tensor.unstack x t.rev := by
  refine Rep.ext fun c => ?_
  simp only [Tensor.unstack_apply, Tensor.flip_apply]
  congr 1
  refine Shape.Coord.ext fun d => ?_
  rw [Shape.Coord.get_ofFun]
  match d with
  | ⟨0, _⟩ => rfl
  | ⟨d + 1, h⟩ => rfl

/-- Under the pure interpreter row `t` of `bidirectional` with the identity head joins row `t` of
the forward run and row `length - 1 - t` of the backward run over the reversed inputs. -/
theorem unstack_bidirectional {State State' : Type}
    {length inputWidth hiddenWidth hiddenWidth' : Nat}
    (forwardCell : Cell Id α State inputWidth hiddenWidth)
    (backwardCell : Cell Id α State' inputWidth hiddenWidth')
    (forwardInitial : State) (backwardInitial : State')
    (inputs : Tensor α [length, inputWidth]) (t : Fin length) :
    Tensor.unstack (bidirectional (m := Id) forwardCell backwardCell (fun both => both)
        forwardInitial backwardInitial inputs) t =
      Tensor.cat (s := [0]) ⟨0, by simp⟩ ![hiddenWidth, hiddenWidth']
        (Fin.cons (Tensor.unstack (run (m := Id) forwardCell forwardInitial inputs).1 t)
          (Fin.cons (Tensor.unstack (run (m := Id) backwardCell backwardInitial
            (Tensor.flip inputs [⟨0, by simp⟩])).1 t.rev) finZeroElim))
        [hiddenWidth + hiddenWidth'] := by
  rw [bidirectional_eq]
  simp only [Tensor.mapLeading, Tensor.zipEach, Tensor.unstack_dim]
  have h := unstack_flip_zero (α := α) (tail := [hiddenWidth'])
    (run (m := Id) backwardCell backwardInitial (Tensor.flip inputs [⟨0, by simp⟩])).1 t
  exact congrArg (fun b : Tensor α [hiddenWidth'] =>
    Tensor.cat (s := [0]) ⟨0, by simp⟩ ![hiddenWidth, hiddenWidth']
      (Fin.cons (Tensor.unstack (run (m := Id) forwardCell forwardInitial inputs).1 t)
        (Fin.cons b finZeroElim)) [hiddenWidth + hiddenWidth']) h

/-- Under the pure interpreter the second result of `residual` is the last residual feature
`h + projection x`, or the hidden vector of `initial` for an empty sequence. -/
theorem residual_snd {State : Type} {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (projection : Tensor α [inputWidth] → Tensor α [hiddenWidth])
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [length, inputWidth]) :
    (residual (m := Id) cell projection head initial inputs).2 =
      if hLength : 0 < length then
        let last : Fin length := ⟨length - 1, Nat.sub_one_lt (Nat.ne_of_gt hLength)⟩
        Tensor.unstack (run (m := Id) cell initial inputs).1 last +
          projection (Tensor.unstack inputs last)
      else cell.hidden initial := by
  rw [residual_eq]

/-- Under the pure interpreter `classifier` on a nonempty sequence is `head` on the last hidden row
of `run`. -/
theorem classifier_eq_unstack {State : Type} {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell Id α State inputWidth hiddenWidth)
    (head : Tensor α [hiddenWidth] → Tensor α [outputWidth]) (initial : State)
    (inputs : Tensor α [length, inputWidth]) (hLength : length ≠ 0) :
    classifier (m := Id) cell head initial inputs =
      head (Tensor.unstack (run (m := Id) cell initial inputs).1
        ⟨length - 1, Nat.sub_one_lt hLength⟩) := by
  rw [classifier_eq, hidden_run_snd_eq_unstack]
  simp only [hLength, ↓reduceDIte]

end TorchLean.Recurrent
