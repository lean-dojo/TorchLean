/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.StateSpace

/-!
# Proofs for affine selective scan

The Mamba/S4 scan theorem is an algebra theorem about affine maps. A sequential recurrent update and
a parallel prefix scan agree because composition of affine transitions is associative:

$$
(a_2,b_2)\circ(a_1,b_1)=(a_2a_1,a_2b_1+b_2).
$$

Over a semiring, prefix summaries denote the same state as the left-to-right recurrence.
Floating-point addition is not associative, so a float scan evaluated in another order may round
differently, and native kernels are outside these statements. The stateful scan and
prefix-causality results keep the sequential step order and need no semiring laws, so they hold at
every scalar type. `scanStates_eq_scalarAffineScan` makes each fiber of `Tensor.selectiveScan` a
`scalarAffineScan`, so these results describe the operator itself, and
`DiagonalSSM.scan_eq_selectiveScan` proves the array scan of a diagonal SSM equal to the operator's
rows.
-/

@[expose] public section

namespace TorchLean
namespace StateSpace

namespace ScalarAffineTransition

variable {α : Type}

/-- Composing scalar affine transitions agrees with function composition. -/
@[simp] theorem compose_apply [Semiring α] (t₂ t₁ : ScalarAffineTransition α)
    (h : α) :
    (ScalarAffineTransition.compose t₂ t₁).apply h = t₂.apply (t₁.apply h) := by
  cases t₁
  cases t₂
  simp [ScalarAffineTransition.compose, ScalarAffineTransition.apply,
    mul_add, mul_assoc, add_assoc]

/-- The identity transition is a left identity for composition. -/
@[simp] theorem compose_id_left [Semiring α] (t : ScalarAffineTransition α) :
    ScalarAffineTransition.compose ScalarAffineTransition.id t = t := by
  cases t
  simp [ScalarAffineTransition.compose, ScalarAffineTransition.id]

/-- The identity transition is a right identity for composition. -/
@[simp] theorem compose_id_right [Semiring α] (t : ScalarAffineTransition α) :
    ScalarAffineTransition.compose t ScalarAffineTransition.id = t := by
  cases t
  simp [ScalarAffineTransition.compose, ScalarAffineTransition.id]

/-- Scalar affine transition composition is associative. -/
@[simp] theorem compose_assoc [Semiring α]
    (t₃ t₂ t₁ : ScalarAffineTransition α) :
    ScalarAffineTransition.compose
        (ScalarAffineTransition.compose t₃ t₂) t₁ =
      ScalarAffineTransition.compose t₃
        (ScalarAffineTransition.compose t₂ t₁) := by
  cases t₁
  cases t₂
  cases t₃
  simp [ScalarAffineTransition.compose, mul_add, mul_assoc, add_assoc]

/-- Zero is a fixed point of a homogeneous scalar transition. -/
@[simp] theorem _root_.TorchLean.StateSpace.homogeneous_apply_zero [Semiring α] (a : α) :
    (ScalarAffineTransition.apply { a := a, b := 0 } 0) = 0 := by
  simp [ScalarAffineTransition.apply]

end ScalarAffineTransition

/-- Seeding a scan with existing output preserves its state evolution and prepends that output. -/
theorem scanArrayFrom_eq {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    scanArrayFrom step initial initialOutputs xs =
      let result := scanArray step initial xs
      (result.1, initialOutputs ++ result.2) := by
  unfold scanArray scanArrayFrom
  let prepend : State × Array Output → State × Array Output :=
    fun result => (result.1, initialOutputs ++ result.2)
  let advance : State × Array Output → Input → State × Array Output :=
    fun result input => ((step result.1 input).1, result.2.push (step result.1 input).2)
  change xs.foldl advance (initial, initialOutputs) = prepend (xs.foldl advance (initial, #[]))
  simpa [prepend] using
    (Array.foldl_hom prepend (g₁ := advance) (g₂ := advance)
      (xs := xs) (init := (initial, #[]))
      (fun _ _ => by simp [prepend, advance]))

/-- The output buffer carried by a scan does not affect its final state. -/
theorem scanArrayFrom_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State)
    (initialOutputs : Array Output) (xs : Array Input) :
    (scanArrayFrom step initial initialOutputs xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  unfold scanArrayFrom
  exact (Array.foldl_hom Prod.fst (xs := xs) (init := (initial, initialOutputs))
    (fun stateAndOutputs input => by
      rcases stateAndOutputs with ⟨state, outputs⟩
      rcases hstep : step state input with ⟨nextState, output⟩
      simp [hstep])).symm

/-- The state component of `scanArray` is the ordinary state-only left fold. -/
theorem scanArray_state_eq_foldl {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (scanArray step initial xs).1 =
      xs.foldl (fun state input => (step state input).1) initial := by
  exact scanArrayFrom_state_eq_foldl step initial #[] xs

/-- A scan over appended inputs is the prefix scan followed by the state-dependent suffix scan. -/
theorem scanArray_append {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    scanArray step initial (xs ++ ys) =
      let first := scanArray step initial xs
      let second := scanArray step first.1 ys
      (second.1, first.2 ++ second.2) := by
  unfold scanArray scanArrayFrom
  rw [Array.foldl_append]
  generalize hfirst :
    Array.foldl
      (fun stateAndOutputs input =>
        let (state, outputs) := stateAndOutputs
        let (nextState, output) := step state input
        (nextState, outputs.push output))
      (initial, #[]) xs = first
  rcases first with ⟨firstState, firstOutputs⟩
  exact scanArrayFrom_eq step firstState firstOutputs ys

/-- Appending future inputs cannot change outputs already emitted by a stateful scan. -/
theorem scanArray_append_outputs_take {State Input Output : Type}
    (step : State → Input → State × Output) (initial : State) (xs ys : Array Input) :
    (scanArray step initial (xs ++ ys)).2.take xs.size =
      (scanArray step initial xs).2 := by
  rw [show (scanArray step initial (xs ++ ys)).2 =
      (scanArray step initial xs).2 ++
        (scanArray step
          (scanArray step initial xs).1 ys).2 by
    simpa using congrArg Prod.snd (scanArray_append step initial xs ys)]
  rw [← scanArray_outputs_size step initial xs, Array.take_eq_extract,
    Array.extract_append_left, Array.extract_size]

/-- Running appended scalar transitions factors through the state reached after the prefix. -/
theorem runScalarAffine_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (ScalarAffineTransition α)) :
    runScalarAffine h0 (xs ++ ys) =
      runScalarAffine (runScalarAffine h0 xs) ys := by
  simpa [runScalarAffine] using
    congrArg Prod.fst (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- Running one scalar transition is the same as applying it. -/
@[simp] theorem runScalarAffine_singleton {α : Type} [Mul α] [Add α] (h0 : α)
    (tr : ScalarAffineTransition α) :
    runScalarAffine h0 #[tr] = tr.apply h0 := by
  rfl

/-- Over a semiring, applying the affine summary to `h0` gives the state the sequential recurrence
reaches. -/
theorem summarizeScalarAffine_apply {α : Type} [Semiring α] (h0 : α)
    (transitions : Array (ScalarAffineTransition α)) :
    (summarizeScalarAffine transitions).apply h0 =
      runScalarAffine h0 transitions := by
  unfold runScalarAffine
  rw [scanArray_state_eq_foldl]
  unfold summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldl_toList]
  generalize transitions.toList = items
  induction items generalizing h0 with
  | nil =>
      simp [ScalarAffineTransition.id,
        ScalarAffineTransition.apply]
  | cons transition rest ih =>
      simp only [List.foldr_cons, List.foldl_cons]
      rw [ScalarAffineTransition.compose_apply, ih]

private theorem foldr_compose_summary {α : Type} [Semiring α]
    (initial : ScalarAffineTransition α)
    (transitions : Array (ScalarAffineTransition α)) :
    transitions.foldr
        (fun transition summary =>
          ScalarAffineTransition.compose summary transition) initial =
      ScalarAffineTransition.compose initial
        (summarizeScalarAffine transitions) := by
  unfold summarizeScalarAffine
  rw [← Array.foldr_toList, ← Array.foldr_toList]
  generalize transitions.toList = items
  induction items with
  | nil => simp
  | cons transition rest ih =>
      simp only [List.foldr_cons]
      rw [ih]
      exact ScalarAffineTransition.compose_assoc initial
        (List.foldr
          (fun transition summary =>
            ScalarAffineTransition.compose summary transition)
          ScalarAffineTransition.id rest) transition

/-- Over a semiring, the summary of `xs ++ ys` is the summary of `ys` composed after the summary of
`xs`. -/
theorem summarizeScalarAffine_append {α : Type} [Semiring α]
    (xs ys : Array (ScalarAffineTransition α)) :
    summarizeScalarAffine (xs ++ ys) =
      ScalarAffineTransition.compose
        (summarizeScalarAffine ys)
        (summarizeScalarAffine xs) := by
  unfold summarizeScalarAffine
  rw [Array.foldr_append]
  exact foldr_compose_summary
    (Array.foldr
      (fun transition summary =>
        ScalarAffineTransition.compose summary transition)
      ScalarAffineTransition.id ys) xs

/-- Scanning appended scalar transitions is the prefix scan followed by the suffix scan. -/
theorem scalarAffineScan_append {α : Type} [Mul α] [Add α] (h0 : α)
    (xs ys : Array (ScalarAffineTransition α)) :
    scalarAffineScan h0 (xs ++ ys) =
      scalarAffineScan h0 xs ++
        scalarAffineScan (runScalarAffine h0 xs) ys := by
  simpa [scalarAffineScan, runScalarAffine] using
    congrArg Prod.snd (scanArray_append
      (fun state transition =>
        let nextState := transition.apply state
        (nextState, nextState)) h0 xs ys)

/-- Each fiber of `Tensor.selectiveScan` is a scalar affine scan: the states of the transitions
`h ↦ a t * h + b t * x t` from `initial`, in order, at every scalar. -/
theorem scanStates_eq_scalarAffineScan {α : Type} [Storage α] [Context α] {n : Nat}
    (a b x : Fin n → α) (initial : α) :
    scanStates a b x initial =
      (scalarAffineScan initial
        (Array.ofFn fun t => (⟨a t, b t * x t⟩ : ScalarAffineTransition α))).toList := by
  let tr : Fin n → ScalarAffineTransition α := fun t => ⟨a t, b t * x t⟩
  let advance : α × Array α → ScalarAffineTransition α → α × Array α :=
    fun result transition =>
      (transition.apply result.1, result.2.push (transition.apply result.1))
  let cons : α × List α → Fin n → α × List α := fun acc t =>
    (a t * acc.1 + b t * x t, (a t * acc.1 + b t * x t) :: acc.2)
  have harray : scalarAffineScan initial (Array.ofFn tr) =
      ((List.finRange n).foldl (fun acc t => advance acc (tr t)) (initial, #[])).2 := by
    change ((Array.ofFn tr).foldl advance (initial, #[])).2 = _
    rw [← Array.foldl_toList, Array.toList_ofFn, List.ofFn_eq_map, List.foldl_map]
  -- The list fold keeps the states newest first, so reversing it gives the array scan's order.
  have hlist := List.foldl_hom (l := List.finRange n) (init := ((initial, #[]) : α × Array α))
    (fun acc : α × Array α => (acc.1, acc.2.toList.reverse))
    (g₁ := fun acc t => advance acc (tr t)) (g₂ := cons)
    (fun acc t => by simp [cons, advance, tr, ScalarAffineTransition.apply])
  change ((List.finRange n).foldl cons (initial, [])).2.reverse =
    (scalarAffineScan initial (Array.ofFn tr)).toList
  rw [harray]
  simpa using congrArg (fun acc : α × List α => acc.2.reverse) hlist

/-- The entry of `Tensor.selectiveScan dim` at step `t` and the other indices `rest` is the state
after step `t` of the scalar affine scan of that fiber. -/
theorem _root_.TorchLean.Tensor.selectiveScan_apply {α : Type} [Storage α] [Context α] {s : Shape}
    (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor α (Tensor.scanCoefficients s dim sharedA))
    (b : Tensor α (Tensor.scanCoefficients s dim sharedB))
    (x : Tensor α s) (initial : Tensor α (s.eraseIdx dim.val))
    (t : Fin (s.axisSize dim.val)) (rest : Shape.Coord (s.eraseIdx dim.val)) :
    Tensor.selectiveScan dim sharedA sharedB a b x initial
        ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest)) =
      (scalarAffineScan (initial rest) (Array.ofFn fun t' =>
        (⟨Tensor.scanCoefficient dim sharedA a t' rest,
          Tensor.scanCoefficient dim sharedB b t' rest *
            x ((Shape.Coord.axisEquiv dim.val (shape := s)).symm
              (t', rest))⟩ :
          ScalarAffineTransition α))).toList.getD t.val 0 := by
  rw [← scanStates_eq_scalarAffineScan]
  simp only [Tensor.selectiveScan, Tensor.Rep.get_ofFn,
    Equiv.apply_symm_apply]

/-- Splitting the steps of a fiber: the states of the first `n` steps, then those of the last `k`
steps from the state the first `n` reach. -/
theorem scanStates_append {α : Type} [Storage α] [Context α] {n k : Nat}
    (a b x : Fin (n + k) → α) (initial : α) :
    scanStates a b x initial =
      scanStates (fun t => a (t.castLE (Nat.le_add_right n k)))
          (fun t => b (t.castLE (Nat.le_add_right n k)))
          (fun t => x (t.castLE (Nat.le_add_right n k))) initial ++
        scanStates (fun t => a (t.natAdd n))
          (fun t => b (t.natAdd n)) (fun t => x (t.natAdd n))
          (runScalarAffine initial (Array.ofFn fun t =>
            (⟨a (t.castLE (Nat.le_add_right n k)),
              b (t.castLE (Nat.le_add_right n k)) * x (t.castLE (Nat.le_add_right n k))⟩ :
              ScalarAffineTransition α))) := by
  rw [scanStates_eq_scalarAffineScan, scanStates_eq_scalarAffineScan,
    scanStates_eq_scalarAffineScan, Array.ofFn_add, scalarAffineScan_append, Array.toList_append]

namespace DiagonalSSM

/-- The scan of appended inputs is the scan of the prefix, then the scan of the suffix from the
state the prefix reaches. -/
theorem _root_.TorchLean.Dynamics.DiagonalSSM.scan_append {α : Type} [Storage α] [Add α]
    [Mul α] {dim : Nat} (m : Dynamics.DiagonalSSM α dim) (h0 : Tensor α [dim])
    (xs ys : Array (Tensor α [dim])) :
    m.scan h0 (xs ++ ys) = m.scan h0 xs ++ m.scan (xs.foldl m.step h0) ys := by
  have h := congrArg Prod.snd
    (scanArray_append (fun h x => let next := m.step h x; (next, next)) h0 xs ys)
  dsimp only at h
  rw [scanArray_state_eq_foldl] at h
  exact h

/-- Channel `c` of the scan of a diagonal SSM is the scalar affine scan of the transitions
`h ↦ A c * h + B c * x c` from `h0 c`. -/
theorem _root_.TorchLean.Dynamics.DiagonalSSM.scan_map_apply {α : Type} [Storage α] [Add α]
    [Mul α] {dim : Nat} (m : Dynamics.DiagonalSSM α dim) (h0 : Tensor α [dim])
    (xs : Array (Tensor α [dim])) (c : Shape.Coord [dim]) :
    (m.scan h0 xs).map (fun h => h c) = scalarAffineScan (h0 c)
      (xs.map fun x => (⟨m.A c, m.B c * x c⟩ : ScalarAffineTransition α)) := by
  let channel : Tensor α [dim] → ScalarAffineTransition α :=
    fun x => ⟨m.A c, m.B c * x c⟩
  let advance : Tensor α [dim] × Array (Tensor α [dim]) →
      Tensor α [dim] → Tensor α [dim] × Array (Tensor α [dim]) :=
    fun result x => (m.step result.1 x, result.2.push (m.step result.1 x))
  let advanceChannel : α × Array α → ScalarAffineTransition α → α × Array α :=
    fun result transition =>
      (transition.apply result.1, result.2.push (transition.apply result.1))
  let project : Tensor α [dim] × Array (Tensor α [dim]) → α × Array α :=
    fun result => (result.1 c, result.2.map fun h => h c)
  have hhom := Array.foldl_hom (xs := xs) (init := (h0, #[])) project (g₁ := advance)
    (g₂ := fun result x => advanceChannel result (channel x))
    (fun result x => by
      simp [project, advance, advanceChannel, channel, ScalarAffineTransition.apply])
  change (xs.foldl advance (h0, #[])).2.map (fun h => h c) =
    ((xs.map channel).foldl advanceChannel (h0 c, #[])).2
  rw [Array.foldl_map]
  simpa [project] using (congrArg Prod.snd hhom).symm

/-- The scan of a diagonal SSM is `Tensor.selectiveScan` of the stacked inputs along the step
axis, with `A` and `B` shared over the steps, read row by row. -/
theorem _root_.TorchLean.Dynamics.DiagonalSSM.scan_eq_selectiveScan {α : Type} [Storage α]
    [Context α] {dim : Nat} (m : Dynamics.DiagonalSSM α dim) (h0 : Tensor α [dim])
    (xs : Array (Tensor α [dim])) :
    m.scan h0 xs = Array.ofFn (Tensor.unstack
      (Tensor.selectiveScan (s := [xs.size, dim]) ⟨0, Nat.zero_lt_two⟩ true true
        m.A m.B (Tensor.dim fun t => xs[t]) h0)) := by
  apply Array.ext
  · simp
  · intro t hscan _
    apply Tensor.Rep.ext
    intro c
    have hxs : t < xs.size := by simpa using hscan
    -- Channel `c` of row `t` on the left is entry `t` of the channel's scalar affine scan.
    have hleft : (m.scan h0 xs)[t] c = (scalarAffineScan (h0 c)
        (xs.map fun x => (⟨m.A c, m.B c * x c⟩ : ScalarAffineTransition α))).toList.getD
          t 0 := by
      rw [← Dynamics.DiagonalSSM.scan_map_apply m h0 xs c]
      simp [hxs]
    -- On the right it is the same entry, through the fiber of the operator along axis 0.
    have hright : Tensor.selectiveScan (s := [xs.size, dim]) ⟨0, Nat.zero_lt_two⟩ true
        true m.A m.B (Tensor.dim fun t => xs[t]) h0 (⟨t, hxs⟩, c) =
        (scalarAffineScan (h0 c) (Array.ofFn fun t' : Fin xs.size =>
          (⟨m.A c, m.B c * xs[t'] c⟩ : ScalarAffineTransition α))).toList.getD t 0 := by
      rw [← scanStates_eq_scalarAffineScan]
      simp only [Tensor.selectiveScan, Tensor.Rep.get_ofFn,
        Tensor.dim_apply]
      rfl
    have hmap : (xs.map fun x => (⟨m.A c, m.B c * x c⟩ : ScalarAffineTransition α)) =
        Array.ofFn fun t' : Fin xs.size =>
          (⟨m.A c, m.B c * xs[t'] c⟩ : ScalarAffineTransition α) := by
      apply Array.ext <;> simp
    rw [hleft, hmap, ← hright, Array.getElem_ofFn]
    simp only [Tensor.unstack_apply]

end DiagonalSSM

/--
If $|a|\leq\rho$, the homogeneous transition $h\mapsto ah$ over $\mathbb{R}$ satisfies
$|ah|\leq\rho|h|$, which for this linear map is the Lipschitz bound with factor $\rho$.

This is the one-channel bound behind `abs_homogeneous_apply_le_self` (the case `ρ = 1`).
-/
theorem abs_homogeneous_apply_le (a ρ h : ℝ) (ha : |a| ≤ ρ) :
    |(ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ ρ * |h| := by
  calc
    |(ScalarAffineTransition.apply { a := a, b := 0 } h)|
        = |a * h| := by simp [ScalarAffineTransition.apply]
    _ = |a| * |h| := by rw [abs_mul]
    _ ≤ ρ * |h| := mul_le_mul_of_nonneg_right ha (abs_nonneg h)

/-- A homogeneous scalar transition with $|a|\leq 1$ is non-expansive. -/
theorem abs_homogeneous_apply_le_self (a h : ℝ) (ha : |a| ≤ 1) :
    |(ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ |h| := by
  calc
    |(ScalarAffineTransition.apply { a := a, b := 0 } h)| ≤ 1 * |h| :=
      abs_homogeneous_apply_le a 1 h ha
    _ = |h| := by simp

end StateSpace
end TorchLean
