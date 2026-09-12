/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Concatenation Correctness

Semantic preservation for lowering `.concat axis`.

The lowering and the IR evaluator both read every parent, fold `Tensor.concatAxisSpec` over the
parents along the leading axis, and (for a nonzero axis) permute the concatenated axis to the front
first and back afterwards. The proofs here relate the two `mapM` traversals over the parent ids and
the two folds.

## Main definitions

- `array_mapM_ok_of_pointwise`: successful `Except` traversals are pointwise determined.
- `evalConcatLeadingAxisFold_eq_concatInputsForward`: the evaluator's leading-axis fold agrees with
  the lowering's `concatInputsForward`.
- `permuteSomeTensor_frontInput`: the evaluator permutes a parent exactly as the lowering does,
  using the evidence recorded in `ConcatFrontInput`.
- `buildFrom_denoteAllFrom_concat_zero`, `buildFrom_denoteAllFrom_concat_pos`,
  `buildFrom_denoteAllFrom_concat`: the step lemma for axis `0`, for a nonzero axis, and combined.

## Implementation notes

- `simp` normalizes the context spelling `[inShape] ++ ss` inside implicit arguments and the array
  bounds of `extract`/`foldl`; the proofs use `erw` or exclude those simp lemmas where a syntactic
  match with a lemma statement is needed.
- Dependent `if` guards are handled with `split at` so their proofs stay out of the simp set.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-- A successful `Except` traversal of a list determines any pointwise related traversal. -/
theorem list_mapM_ok_of_pointwise {β γ δ : Type}
    (f : β → Except String γ) (g : β → Except String δ) (h : γ → δ)
    (hpt : ∀ x y, f x = .ok y → g x = .ok (h y)) :
    ∀ (xs : List β) (ys : List γ), xs.mapM f = .ok ys → xs.mapM g = .ok (ys.map h) := by
  intro xs
  induction xs with
  | nil =>
      intro ys hys
      simp [Pure.pure, Except.pure] at hys
      subst hys
      simp [Pure.pure, Except.pure]
  | cons x xs ih =>
      intro ys hys
      rw [List.mapM_cons] at hys
      cases hx : f x with
      | error msg => simp [hx] at hys
      | ok y =>
          rw [hx] at hys
          cases hxs : xs.mapM f with
          | error msg => simp [hxs] at hys
          | ok ys' =>
              rw [hxs] at hys
              simp [Pure.pure, Except.pure] at hys
              subst hys
              rw [List.mapM_cons, hpt x y hx, ih ys' hxs]
              simp [Pure.pure, Except.pure]

/-- Array form of `list_mapM_ok_of_pointwise`. -/
theorem array_mapM_ok_of_pointwise {β γ δ : Type}
    (f : β → Except String γ) (g : β → Except String δ) (h : γ → δ)
    (hpt : ∀ x y, f x = .ok y → g x = .ok (h y))
    (xs : Array β) (ys : Array γ) (hys : xs.mapM f = .ok ys) :
    xs.mapM g = .ok (ys.map h) := by
  rw [Array.mapM_eq_mapM_toList] at hys ⊢
  cases hl : xs.toList.mapM f with
  | error msg => simp [hl, Functor.map, Except.map] at hys
  | ok l =>
      rw [hl] at hys
      simp [Functor.map, Except.map] at hys
      subst hys
      rw [list_mapM_ok_of_pointwise f g h hpt xs.toList l hl]
      simp [Functor.map, Except.map]

/-- A successful `Except` traversal preserves the list length. -/
theorem list_length_of_mapM_ok {β γ : Type} (f : β → Except String γ) :
    ∀ (xs : List β) (ys : List γ), xs.mapM f = .ok ys → ys.length = xs.length := by
  intro xs
  induction xs with
  | nil =>
      intro ys hys
      simp [Pure.pure, Except.pure] at hys
      subst hys
      rfl
  | cons x xs ih =>
      intro ys hys
      rw [List.mapM_cons] at hys
      cases hx : f x with
      | error msg => simp [hx] at hys
      | ok y =>
          rw [hx] at hys
          cases hxs : xs.mapM f with
          | error msg => simp [hxs] at hys
          | ok ys' =>
              rw [hxs] at hys
              simp [Pure.pure, Except.pure] at hys
              subst hys
              simp [ih ys' hxs]

/-- A successful `Except` traversal preserves the array size. -/
theorem array_size_of_mapM_ok {β γ : Type} (f : β → Except String γ)
    (xs : Array β) (ys : Array γ) (hys : xs.mapM f = .ok ys) : ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at hys
  cases hl : xs.toList.mapM f with
  | error msg => simp [hl, Functor.map, Except.map] at hys
  | ok l =>
      rw [hl] at hys
      simp [Functor.map, Except.map] at hys
      subst hys
      simpa using list_length_of_mapM_ok f xs.toList l hl

/-- Decoding a leading-axis parent whose stored tail already matches succeeds without a cast. -/
@[simp] theorem expectLeadingAxisInput_mk {α : Type} [TorchLean.Storage α] [Context α]
    (i : Nat) (rest : Shape) (size : Nat) (t : Tensor α (.dim size rest)) :
    NN.IR.Graph.expectLeadingAxisInput (α := α) i rest
        (Spec.SomeTensor.mk (α := α) (.dim size rest) t) =
      pure ⟨size, t⟩ := by
  simp [NN.IR.Graph.expectLeadingAxisInput]

/-- `Eq.mp` along an equality of tensor types is the shape cast. -/
theorem eq_mp_eq_cast_shape {α : Type} [TorchLean.Storage α] {s t : Shape}
    (x : Tensor α s) (h : Tensor α s = Tensor α t) (h' : s = t) :
    Eq.mp h x = Tensor.castShape x h' := by
  subst h'
  rfl

/-- `concatInputsForward` unfolded to a cast of the leading-axis fold. -/
theorem concatInputsForward_eq {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} {rest : Shape} (inputs : Array (ConcatInput α Γ rest)) (nOut : Nat)
    (hSum : inputs.foldl (fun acc input => acc + input.1) 0 = nOut)
    (context : TorchLean.TensorPack α Γ) :
    concatInputsForward (α := α) inputs nOut hSum context =
      Tensor.castShape (concatLeadingAxisFromInputs (α := α) context inputs).2
        (congrArg (fun k => Shape.dim k rest)
          ((concatLeadingAxisFromInputs_size_eq_sum (α := α) context inputs).trans hSum)) := rfl

/-- Folding from the first element over the remaining ones is `concatLeadingAxisList`. -/
theorem foldl_extract_one_eq_concatLeadingAxisList {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (first : Sigma fun n => Tensor α (.dim n rest))
    (others : List (Sigma fun n => Tensor α (.dim n rest))) :
    ((first :: others).toArray.extract 1).foldl
        (fun acc nxt =>
          match acc, nxt with
          | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
              ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                (suffix := rest) t1 t2⟩)
        first =
      concatLeadingAxisList (α := α) (rest := rest) (first :: others) := by
  rw [← Array.foldl_toList]
  simp [concatLeadingAxisList]

/-- Folding from the first array element over the remaining ones is `concatLeadingAxisList`. -/
theorem foldl_extract_one_eq_concatLeadingAxisList' {α : Type} [TorchLean.Storage α] [Context α]
    {rest : Shape} (sigs : Array (Sigma fun n => Tensor α (.dim n rest))) (h : 0 < sigs.size) :
    (sigs.extract 1).foldl
        (fun acc nxt =>
          match acc, nxt with
          | ⟨n1, t1⟩, ⟨n2, t2⟩ =>
              ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                (suffix := rest) t1 t2⟩)
        sigs[0] =
      concatLeadingAxisList (α := α) (rest := rest) sigs.toList := by
  obtain ⟨first, others, hl⟩ :
      ∃ first others, sigs.toList = first :: others := by
    cases hs : sigs.toList with
    | nil =>
        exfalso
        have hZero : sigs.size = 0 := by simpa using congrArg List.length hs
        simp [hZero] at h
    | cons first others => exact ⟨first, others, rfl⟩
  have hsigs : sigs = (first :: others).toArray := by
    apply Array.ext'
    simpa using hl
  subst hsigs
  exact foldl_extract_one_eq_concatLeadingAxisList first others

/-- Projection-form variant of `foldl_extract_one_eq_concatLeadingAxisList'`, matching the fold
shape `simp` produces from the evaluator's pattern match. -/
theorem foldl_extract_one_eq_concatLeadingAxisList_proj {α : Type} [TorchLean.Storage α]
    [Context α] {rest : Shape} (sigs : Array (Sigma fun n => Tensor α (.dim n rest)))
    (h : 0 < sigs.size) :
    (sigs.extract 1).foldl
        (fun acc nxt =>
          (⟨acc.1 + nxt.1, Tensor.concatAxisSpec .scalar (α := α) (n := acc.1) (m := nxt.1)
            (suffix := rest) acc.2 nxt.2⟩ : Sigma fun n => Tensor α (.dim n rest)))
        sigs[0] =
      concatLeadingAxisList (α := α) (rest := rest) sigs.toList := by
  obtain ⟨first, others, hl⟩ :
      ∃ first others, sigs.toList = first :: others := by
    cases hs : sigs.toList with
    | nil =>
        exfalso
        have hZero : sigs.size = 0 := by simpa using congrArg List.length hs
        simp [hZero] at h
    | cons first others => exact ⟨first, others, rfl⟩
  have hsigs : sigs = (first :: others).toArray := by
    apply Array.ext'
    simpa using hl
  subst hsigs
  rw [← Array.foldl_toList]
  simp [concatLeadingAxisList]

/--
The evaluator's leading-axis concat fold over parents that already carry the tail `rest` is the
lowering's `concatInputsForward`.
-/
theorem evalConcatLeadingAxisFold_eq_concatInputsForward
    {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape} {rest : Shape}
    (i nOut : Nat) (inputs : Array (ConcatInput α Γ rest))
    (context : TorchLean.TensorPack α Γ) (hSize : 0 < inputs.size)
    (hSum : inputs.foldl (fun acc input => acc + input.1) 0 = nOut) :
    NN.IR.Graph.evalConcatLeadingAxisFold (α := α) i nOut rest
        (inputs.map fun input =>
          Spec.SomeTensor.mk (α := α) (.dim input.1 rest) (input.2 context)) =
      .ok (Spec.SomeTensor.mk (α := α) (.dim nOut rest)
        (concatInputsForward (α := α) inputs nOut hSum context)) := by
  unfold NN.IR.Graph.evalConcatLeadingAxisFold
  have hSigs :
      (inputs.map fun input =>
          Spec.SomeTensor.mk (α := α) (.dim input.1 rest) (input.2 context)).mapM
          (NN.IR.Graph.expectLeadingAxisInput (α := α) i rest) =
        pure (inputs.map fun input =>
          (⟨input.1, input.2 context⟩ : Sigma fun n => Tensor α (.dim n rest))) := by
    simp only [Array.mapM_map, Function.comp_def, expectLeadingAxisInput_mk, Array.mapM_pure]
  rw [hSigs]
  simp only [Pure.pure, Except.pure, Except.ok_bind]
  have hSize' : 0 < (inputs.map fun input =>
      (⟨input.1, input.2 context⟩ : Sigma fun n => Tensor α (.dim n rest))).size := by
    simpa using hSize
  rw [Array.getElem?_eq_getElem hSize']
  simp only
  rw [foldl_extract_one_eq_concatLeadingAxisList' _ hSize']
  have hIR :
      concatLeadingAxisList (α := α) (rest := rest)
          (inputs.map fun input =>
            (⟨input.1, input.2 context⟩ : Sigma fun n => Tensor α (.dim n rest))).toList =
        concatLeadingAxisFromInputs (α := α) context inputs := by
    simp [concatLeadingAxisFromInputs]
  rw [hIR, concatInputsForward_eq]
  have hY : (concatLeadingAxisFromInputs (α := α) context inputs).1 = nOut :=
    (concatLeadingAxisFromInputs_size_eq_sum (α := α) context inputs).trans hSum
  rw [dif_pos hY]
  change Except.ok _ = Except.ok _
  congr 2
  exact eq_mp_eq_cast_shape _ _ _

/-- The lowering context used by `buildFrom` for node `n` at position `i`. -/
abbrev loweringContext {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape) (ss : List Shape)
    (i : Nat) (n : NN.IR.Node) : NodeLoweringContext α ([inShape] ++ ss) :=
  { graph := g, payload := payload, index := i, node := n,
    parentIdx := fun pid s => mkIdx (inShape := inShape) (ss := ss) pid s }

/-- Each successful axis-zero concat input reads a parent value of the recorded shape. -/
theorem concatAxisZeroInputs_getParentValue {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (x : Tensor α inShape) (i : Nat) (n : NN.IR.Node)
    (rest : Shape) (inputs : Array (ConcatInput α ([inShape] ++ ss) rest))
    (h : concatAxisZeroInputs (α := α) (loweringContext g payload inShape ss i n) rest =
      .ok inputs) :
    n.parents.mapM
        (NN.IR.Graph.getParentValue (denoteAllState (α := α) inShape ⟨ss, gd⟩ x) i n) =
      .ok (inputs.map fun input =>
        Spec.SomeTensor.mk (α := α) (.dim input.1 rest)
          (input.2 (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)))) := by
  unfold concatAxisZeroInputs at h
  refine array_mapM_ok_of_pointwise _ _ _ ?_ n.parents inputs h
  intro pid input hpid
  cases hP : g.getNode pid with
  | error msg => simp [hP] at hpid
  | ok pNode =>
  cases hS : pNode.outShape with
  | scalar => simp [hP, hS] at hpid
  | dim nP restP =>
  by_cases hRest : restP = rest
  · subst hRest
    simp [hP, hS] at hpid
    cases hIdx : mkIdx (inShape := inShape) (ss := ss) pid (.dim nP restP) with
    | error msg => simp [hIdx] at hpid
    | ok ip =>
        simp [hIdx] at hpid
        subst hpid
        simp [NN.IR.Graph.getParentValue,
          denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss) (gd := gd) (x := x) hIdx,
          Pure.pure, Except.pure]
  · simp [hP, hS, hRest] at hpid

/-- Each successful axis-zero concat input records its parent's declared shape. -/
theorem concatAxisZeroInputs_shapes {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (i : Nat) (n : NN.IR.Node) (rest : Shape)
    (inputs : Array (ConcatInput α ([inShape] ++ ss) rest))
    (h : concatAxisZeroInputs (α := α) (loweringContext g payload inShape ss i n) rest =
      .ok inputs) :
    n.parents.mapM (fun pid => (fun pNode : NN.IR.Node => pNode.outShape) <$> g.getNode pid) =
      .ok (inputs.map fun input => Shape.dim input.1 rest) := by
  unfold concatAxisZeroInputs at h
  refine array_mapM_ok_of_pointwise _ _ _ ?_ n.parents inputs h
  intro pid input hpid
  cases hP : g.getNode pid with
  | error msg => simp [hP] at hpid
  | ok pNode =>
  cases hS : pNode.outShape with
  | scalar => simp [hP, hS] at hpid
  | dim nP restP =>
  by_cases hRest : restP = rest
  · subst hRest
    simp [hP, hS] at hpid
    cases hIdx : mkIdx (inShape := inShape) (ss := ss) pid (.dim nP restP) with
    | error msg => simp [hIdx] at hpid
    | ok ip =>
        simp [hIdx] at hpid
        subst hpid
        simp [hS, Functor.map, Except.map]
  · simp [hP, hS, hRest] at hpid

/-- Retag a shape-erased tensor along a shape equality. -/
theorem someTensor_mk_castShape {α : Type} [TorchLean.Storage α] [Context α] {s s' : Shape}
    (t : Tensor α s) (h : s = s') :
    Spec.SomeTensor.mk (α := α) s t = Spec.SomeTensor.mk (α := α) s' (Tensor.castShape t h) := by
  subst h
  rfl

/-- `cast` along an equality of tensor types is the shape cast. -/
theorem cast_eq_cast_shape {α : Type} [TorchLean.Storage α] {s t : Shape}
    (x : Tensor α s) (h : Tensor α s = Tensor α t) (h' : s = t) :
    cast h x = Tensor.castShape x h' := by
  subst h'
  rfl

/-- Casting the second components of equal leading-axis sigmas gives equal tensors. -/
theorem sigma_cast_shape_congr {α : Type} [TorchLean.Storage α] {rest t : Shape}
    (X Y : Sigma fun n => Tensor α (.dim n rest)) (hXY : X = Y)
    (h₁ : Shape.dim X.1 rest = t) (h₂ : Shape.dim Y.1 rest = t) :
    Tensor.castShape X.2 h₁ = Tensor.castShape Y.2 h₂ := by
  subst hXY
  rfl

/-- `Eq.mpr` along a reflexive tensor-type equality is the identity. -/
@[simp] theorem eq_mpr_tensor_refl {α : Type} [TorchLean.Storage α] {s : Shape}
    (h : Tensor α s = Tensor α s) (t : Tensor α s) : Eq.mpr h t = t := rfl

/-- Permuting a validated front input's parent value moves the concatenated axis to the front. -/
theorem permuteSomeTensor_frontInput {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} {permFront : Array Nat} {restFront : Shape}
    (f : ConcatFrontInput α Γ permFront restFront) (t : Tensor α f.sIn) :
    NN.IR.Graph.permuteSomeTensor (α := α) (Spec.SomeTensor.mk (α := α) f.sIn t) permFront =
      .ok (Spec.SomeTensor.mk (α := α) (.dim f.nP restFront)
        (Tensor.castShape (applySwapsTensor (α := α) (s := f.sIn) (swaps := f.swaps) t)
          f.final_eq)) := by
  rw [permuteSomeTensor_eq_applySwapsTensor (α := α) t permFront (.dim f.nP restFront) f.swaps
    f.perm_eq f.swaps_eq]
  rw [someTensor_mk_castShape _ f.final_eq]

/-- Each successful front concat input reads the parent value at the recorded shape. -/
theorem concatAxisFrontInputs_getParentValue {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (x : Tensor α inShape) (i : Nat) (n : NN.IR.Node)
    (permFront : Array Nat) (restFront : Shape)
    (frontInputs : Array (ConcatFrontInput α ([inShape] ++ ss) permFront restFront))
    (h : concatAxisFrontInputs (α := α) (loweringContext g payload inShape ss i n) permFront
      restFront = .ok frontInputs) :
    n.parents.mapM
        (NN.IR.Graph.getParentValue (denoteAllState (α := α) inShape ⟨ss, gd⟩ x) i n) =
      .ok (frontInputs.map fun f =>
        Spec.SomeTensor.mk (α := α) f.sIn
          (getIdx (α := α)
            (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
            f.ip)) := by
  unfold concatAxisFrontInputs at h
  refine array_mapM_ok_of_pointwise _ _ _ ?_ n.parents frontInputs h
  intro pid f hpid
  cases hP : g.getNode pid with
  | error msg => simp [hP] at hpid
  | ok pNode =>
  cases hIdx : mkIdx (inShape := inShape) (ss := ss) pid pNode.outShape with
  | error msg => simp [hP, hIdx] at hpid
  | ok ip =>
  simp only [hP, hIdx, Except.ok_bind] at hpid
  split at hpid
  · simp at hpid
  · rename_i nP restP hPerm
    split at hpid
    · rename_i hRest
      split at hpid
      · simp at hpid
      · rename_i swaps hSwaps
        split at hpid
        · rename_i hFinal
          simp only [Pure.pure, Except.pure, Except.ok.injEq] at hpid
          subst hpid
          simp [NN.IR.Graph.getParentValue,
            denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss) (gd := gd) (x := x) hIdx,
            Pure.pure, Except.pure]
        · simp at hpid
    · simp at hpid
  · simp at hpid

/-- Each successful front concat input records its parent's declared shape. -/
theorem concatAxisFrontInputs_shapes {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (i : Nat) (n : NN.IR.Node) (permFront : Array Nat) (restFront : Shape)
    (frontInputs : Array (ConcatFrontInput α ([inShape] ++ ss) permFront restFront))
    (h : concatAxisFrontInputs (α := α) (loweringContext g payload inShape ss i n) permFront
      restFront = .ok frontInputs) :
    n.parents.mapM (fun pid => Node.outShape <$> g.getNode pid) =
      .ok (frontInputs.map fun f => f.sIn) := by
  unfold concatAxisFrontInputs at h
  refine array_mapM_ok_of_pointwise _ _ _ ?_ n.parents frontInputs h
  intro pid f hpid
  cases hP : g.getNode pid with
  | error msg => simp [hP] at hpid
  | ok pNode =>
  cases hIdx : mkIdx (inShape := inShape) (ss := ss) pid pNode.outShape with
  | error msg => simp [hP, hIdx] at hpid
  | ok ip =>
  simp only [hP, hIdx, Except.ok_bind] at hpid
  split at hpid
  · simp at hpid
  · rename_i nP restP hPerm
    split at hpid
    · rename_i hRest
      split at hpid
      · simp at hpid
      · rename_i swaps hSwaps
        split at hpid
        · rename_i hFinal
          simp only [Pure.pure, Except.pure, Except.ok.injEq] at hpid
          subst hpid
          simp [Functor.map, Except.map]
        · simp at hpid
    · simp at hpid
  · simp at hpid

/-- Semantic preservation for `.concat 0`. -/
theorem buildFrom_denoteAllFrom_concat_zero
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hk : n.kind = .concat 0) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  rcases n with ⟨nId, nParents, nKind, nOutShape⟩
  have hkKind : nKind = .concat 0 := by simpa using hk
  subst hkKind
  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN] at hBuild
  by_cases hSize : nParents.size < 2
  · exact False.elim <| throw_bind_ne_ok (by simpa [lowerConcat, hSize] using hBuild)
  simp (config := { failIfUnchanged := false }) [lowerConcat, hSize] at hBuild
  cases hShapes : nParents.mapM (fun pid => Node.outShape <$> g.getNode pid) with
  | error msg => simp [hShapes] at hBuild
  | ok parentShapes =>
  simp (config := { failIfUnchanged := false }) [hShapes] at hBuild
  cases hInfer : OpContracts.inferConcatOutShape 0 parentShapes with
  | error msg => exact False.elim <| throw_bind_ne_ok (by simpa [hInfer] using hBuild)
  | ok expected =>
  simp (config := { failIfUnchanged := false }) [hInfer] at hBuild
  by_cases hB : expected = nOutShape
  swap
  · exact False.elim <| throw_bind_ne_ok (by simpa [hB] using hBuild)
  simp (config := { failIfUnchanged := false }) [hB] at hBuild
  cases nOutShape with
  | scalar => exact False.elim <| throw_bind_ne_ok (by simpa using hBuild)
  | dim nOut rest =>
  simp (config := { failIfUnchanged := false }) at hBuild
  cases hInputs :
      concatAxisZeroInputs (α := α)
        (loweringContext g payload inShape ss i
          { id := nId, parents := nParents, kind := .concat 0, outShape := .dim nOut rest })
        rest with
  | error msg => simp [hInputs] at hBuild
  | ok inputs =>
  simp (config := { failIfUnchanged := false }) [hInputs] at hBuild
  -- The size guard is a dependent `if`; split it so its proof is available to the closure.
  split at hBuild
  · rename_i hSum
    simp (config := { failIfUnchanged := false }) at hBuild
    let nodeData : ForwardNode α ([inShape] ++ ss) (.dim nOut rest) :=
      mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := .dim nOut rest) (fun context =>
        Tensor.castShape (concatInputsForward (α := α) inputs nOut hSum context) rfl)
    let st1 : State α inShape := ⟨ss ++ [.dim nOut rest], .snoc (ss := ss) gd nodeData⟩
    have hRec :
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' := by
      simpa [st1, nodeData] using hBuild
    have hInputsSize : 0 < inputs.size := by
      rw [array_size_of_mapM_ok _ nParents inputs hInputs]
      exact Nat.lt_of_lt_of_le (by decide) (Nat.le_of_not_lt hSize)
    have hParents := concatAxisZeroInputs_getParentValue (α := α) g payload gd x i
      { id := nId, parents := nParents, kind := .concat 0, outShape := .dim nOut rest } rest inputs
      hInputs
    have hShapes' := concatAxisZeroInputs_shapes (α := α) g payload (inShape := inShape) (ss := ss)
      i { id := nId, parents := nParents, kind := .concat 0, outShape := .dim nOut rest } rest
      inputs hInputs
    have hParentShapes : parentShapes = inputs.map fun input => Shape.dim input.1 rest := by
      have h := hShapes.symm.trans hShapes'
      exact Except.ok.inj h
    subst hParentShapes
    have hFold := evalConcatLeadingAxisFold_eq_concatInputsForward (α := α) i nOut inputs
      (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)) hInputsSize hSum
    have hEval :
        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
            (input := Spec.SomeTensor.mk (α := α) inShape x)
            (vals := denoteAllState (α := α) inShape ⟨ss, gd⟩ x) (i := i) =
          .ok (Spec.SomeTensor.mk (α := α) (.dim nOut rest)
            (nodeData.eval (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
              (.cons x .nil)))) := by
      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, hN, -NN.IR.Graph.getParentValue]
      rw [hParents]
      -- `erw` because simp normalizes the context spelling `[inShape] ++ ss` inside implicit
      -- arguments, which blocks syntactic matching against the lemma statements.
      simp [NN.IR.Graph.evalConcat, Array.map_map, Function.comp_def]
      erw [hInfer]
      simp [hB]
      erw [hFold]
      simp [nodeData]
    apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
      (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
      (nodeData := nodeData)
    · exact ih st1 hRec
    · exact hEval
  · exact False.elim <| throw_bind_ne_ok hBuild

/-- Semantic preservation for `.concat axis` with `axis ≠ 0`. -/
theorem buildFrom_denoteAllFrom_concat_pos
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (axis : Nat) (hAxis : axis ≠ 0)
    (hN : g.getNode i = .ok n) (hk : n.kind = .concat axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  rcases n with ⟨nId, nParents, nKind, nOutShape⟩
  have hkKind : nKind = .concat axis := by simpa using hk
  subst hkKind
  unfold buildFrom at hBuild
  simp (config := { failIfUnchanged := false }) [hi, hN] at hBuild
  by_cases hSize : nParents.size < 2
  · exact False.elim <| throw_bind_ne_ok (by simpa [lowerConcat, hSize] using hBuild)
  simp (config := { failIfUnchanged := false }) [lowerConcat, hSize, hAxis] at hBuild
  cases hShapes : nParents.mapM (fun pid => Node.outShape <$> g.getNode pid) with
  | error msg => simp [hShapes] at hBuild
  | ok parentShapes =>
  simp (config := { failIfUnchanged := false }) [hShapes] at hBuild
  cases hInfer : OpContracts.inferConcatOutShape axis parentShapes with
  | error msg => exact False.elim <| throw_bind_ne_ok (by simpa [hInfer] using hBuild)
  | ok expected =>
  simp (config := { failIfUnchanged := false }) [hInfer] at hBuild
  by_cases hB : expected = nOutShape
  swap
  · exact False.elim <| throw_bind_ne_ok (by simpa [hB] using hBuild)
  simp (config := { failIfUnchanged := false }) [hB] at hBuild
  cases hPermFront : OpContracts.permMoveAxisToFront axis nOutShape with
  | error msg => exact False.elim <| throw_bind_ne_ok (by simpa [hPermFront] using hBuild)
  | ok permFront =>
  simp (config := { failIfUnchanged := false }) [hPermFront] at hBuild
  cases hPermBack : OpContracts.inversePerm permFront with
  | error msg => exact False.elim <| throw_bind_ne_ok (by simpa [hPermBack] using hBuild)
  | ok permBack =>
  simp (config := { failIfUnchanged := false }) [hPermBack] at hBuild
  cases hOFE : Spec.Shape.permute? nOutShape permFront.toList with
  | none => exact False.elim <| throw_bind_ne_ok (by simpa [hOFE] using hBuild)
  | some outFrontExpected =>
  simp (config := { failIfUnchanged := false }) [hOFE] at hBuild
  cases outFrontExpected with
  | scalar => exact False.elim <| throw_bind_ne_ok (by simpa using hBuild)
  | dim nOutFront restFront =>
  simp (config := { failIfUnchanged := false }) at hBuild
  cases hPB : Spec.Shape.permute? (Shape.dim nOutFront restFront) permBack.toList with
  | none => exact False.elim <| throw_bind_ne_ok (by simpa [hPB] using hBuild)
  | some outBack =>
  simp (config := { failIfUnchanged := false }) [hPB] at hBuild
  -- `simp` spells the rank of the permuted output as `restFront.rank + 1`.
  cases hSwapsBack : NN.IR.Graph.swapDepthsForPerm permBack (Spec.Shape.rank restFront + 1) with
  | error msg => simp [hSwapsBack] at hBuild
  | ok swapsBack =>
  simp (config := { failIfUnchanged := false }) [hSwapsBack] at hBuild
  split at hBuild
  · rename_i hOutBackFinal
    cases hFront :
        concatAxisFrontInputs (α := α)
          (loweringContext g payload inShape ss i
            { id := nId, parents := nParents, kind := .concat axis, outShape := nOutShape })
          permFront restFront with
    | error msg => simp [hFront] at hBuild
    | ok frontInputs =>
    simp (config := { failIfUnchanged := false }) [hFront] at hBuild
    split at hBuild
    · rename_i hSum
      simp (config := { failIfUnchanged := false }) at hBuild
      let inputs : Array (ConcatInput α ([inShape] ++ ss) restFront) :=
        frontInputs.map ConcatFrontInput.toInput
      have hSum' : inputs.foldl (fun acc input => acc + input.1) 0 = nOutFront := by
        simpa [inputs] using hSum
      let nodeData : ForwardNode α ([inShape] ++ ss) nOutShape :=
        mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := nOutShape) (fun context =>
          Tensor.castShape
            (applySwapsTensor (α := α) (s := .dim nOutFront restFront) (swaps := swapsBack)
              (concatInputsForward (α := α) inputs nOutFront hSum' context))
            hOutBackFinal)
      let st1 : State α inShape := ⟨ss ++ [nOutShape], .snoc (ss := ss) gd nodeData⟩
      have hRec :
          buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
            (i := i + 1) st1 = .ok st' := by
        simpa [st1, nodeData, inputs] using hBuild
      have hFrontSize : 0 < frontInputs.size := by
        rw [array_size_of_mapM_ok _ nParents frontInputs hFront]
        exact Nat.lt_of_lt_of_le (by decide) (Nat.le_of_not_lt hSize)
      have hParents := concatAxisFrontInputs_getParentValue (α := α) g payload gd x i
        { id := nId, parents := nParents, kind := .concat axis, outShape := nOutShape }
        permFront restFront frontInputs hFront
      have hShapes' := concatAxisFrontInputs_shapes (α := α) g payload (inShape := inShape)
        (ss := ss) i { id := nId, parents := nParents, kind := .concat axis, outShape := nOutShape }
        permFront restFront frontInputs hFront
      have hParentShapes : parentShapes = frontInputs.map fun f => f.sIn :=
        Except.ok.inj (hShapes.symm.trans hShapes')
      subst hParentShapes
      have hEval :
          NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
              (input := Spec.SomeTensor.mk (α := α) inShape x)
              (vals := denoteAllState (α := α) inShape ⟨ss, gd⟩ x) (i := i) =
            .ok (Spec.SomeTensor.mk (α := α) nOutShape
              (nodeData.eval (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
                (.cons x .nil)))) := by
        simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, hN, -NN.IR.Graph.getParentValue]
        rw [hParents]
        simp [NN.IR.Graph.evalConcat, Array.map_map, Function.comp_def, -Array.getElem?_map,
          -Array.size_map, -Array.size_extract]
        erw [hInfer]
        -- Keep the array sizes and `sigs[0]` unnormalized so the fold lemma matches syntactically.
        simp [hB, hAxis, hPermFront, hPermBack, hOFE, Array.mapM_map, Function.comp_def,
          permuteSomeTensor_frontInput, Array.mapM_pure, -Array.getElem?_map, -Array.size_map,
          -Array.size_extract]
        have hSigsSize :
            0 < (frontInputs.map fun fi =>
              (⟨fi.nP, Tensor.castShape
                (applySwapsTensor (α := α) (s := fi.sIn) (swaps := fi.swaps)
                  (getIdx (α := α)
                    (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
                    fi.ip))
                fi.final_eq⟩ : Sigma fun n => Tensor α (.dim n restFront))).size := by
          simpa using hFrontSize
        have hIR :
            concatLeadingAxisList (α := α) (rest := restFront)
                (frontInputs.map fun fi =>
                  (⟨fi.nP, Tensor.castShape
                    (applySwapsTensor (α := α) (s := fi.sIn) (swaps := fi.swaps)
                      (getIdx (α := α)
                        (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
                          (.cons x .nil))
                        fi.ip))
                    fi.final_eq⟩ : Sigma fun n => Tensor α (.dim n restFront))).toList =
              concatLeadingAxisFromInputs (α := α)
                (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
                inputs := by
          simp [concatLeadingAxisFromInputs, inputs, ConcatFrontInput.toInput, Function.comp_def]
        have hY :
            (concatLeadingAxisFromInputs (α := α)
              (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)) inputs).1 =
              nOutFront :=
          (concatLeadingAxisFromInputs_size_eq_sum (α := α) _ inputs).trans hSum'
        erw [Array.getElem?_eq_getElem hSigsSize]
        simp only
        split
        · rename_i hFoldEq
          rw [permuteSomeTensor_eq_applySwapsTensor (α := α) _ permBack outBack swapsBack hPB
            (by simpa using hSwapsBack)]
          simp only [Graph.expectShape_mk_of_eq hOutBackFinal, Functor.map, Except.map,
            Except.ok_bind, NN.IR.Graph.normalizeNodeOutput_nodeShape, nodeData,
            mkForwardNode_eval]
          have hX :=
            (foldl_extract_one_eq_concatLeadingAxisList_proj (α := α) (rest := restFront) _
              hSigsSize).trans hIR
          rw [cast_eq_cast_shape _ _ (congrArg (fun k => Shape.dim k restFront) hFoldEq),
            concatInputsForward_eq]
          erw [sigma_cast_shape_congr _ _ hX (congrArg (fun k => Shape.dim k restFront) hFoldEq)
            (congrArg (fun k => Shape.dim k restFront) hY)]
        · rename_i hFoldNe
          exfalso
          apply hFoldNe
          erw [foldl_extract_one_eq_concatLeadingAxisList_proj _ hSigsSize, hIR]
          exact hY
      apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
        (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (hi := hi)
        (nodeData := nodeData)
      · exact ih st1 hRec
      · exact hEval
    · exact False.elim <| throw_bind_ne_ok hBuild
  · exact False.elim <| throw_bind_ne_ok hBuild

/-- Semantic preservation for `.concat axis` along any axis. -/
theorem buildFrom_denoteAllFrom_concat
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .concat axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  by_cases hAxis : axis = 0
  · subst hAxis
    exact buildFrom_denoteAllFrom_concat_zero (α := α) g payload gd i st' x n hN hk hi hBuild ih
  · exact buildFrom_denoteAllFrom_concat_pos (α := α) g payload gd i st' x n axis hAxis hN hk hi
      hBuild ih

end IRExec
end Autograd
end Runtime
