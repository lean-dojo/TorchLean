/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Reduce

/-!
# Selective scan specs

The recurrence behind state-space sequence models such as S4 and Mamba, in a small form that proofs
can use. Mamba's hardware-aware parallel scan rests on the fact that each per-token recurrent update
is an affine map

`h ↦ A_t h + b_t`.

Over a semiring, affine maps compose associatively, so a recurrent scan can run either as a
left-to-right recurrence or as a parallel prefix scan over affine summaries. With IEEE arithmetic
the two schedules can round differently. The scalar definitions below are kept compact so that
`TorchLean/NN/StateSpace/Scan.lean` can prove the algebra without depending on a particular
runtime backend.

`Tensor.selectiveScan dim`, the diagonal scan of the operator table and of the CUDA kernel
contract, scans along any axis, with each coefficient shared over the steps or given one per step.
Each of its fibers is a `scalarAffineScan` (`scanStates_eq_scalarAffineScan`).
`Tensor.selectiveScanBackward` is its pullback by the adjoint recurrence and
`Tensor.selectiveScanJVP` its tangent by the tangent recurrence.

On a fiber this is Mamba's eq. (2a), `h_t = Ā h_{t-1} + B̄ x_t`, with a diagonal `Ā`, so every
state coordinate runs on its own with the scalar `a t` and the term `b t * x t`. We transcribed it
by hand. One coefficient per step gives the time-varying scan of Mamba's Algorithm 2 (S6), and
coefficients shared over the steps the time-invariant one of its Algorithm 1. S4's Sec. 2.4 starts
eq. (3) from `x_{-1} = 0`, and ours starts from `initial`. The discretization belongs to the
caller: neither Mamba's zero-order hold, eq. (4), nor S4's bilinear rule is in the spec, and the
readout `y_t = C h_t` of eq. (2b) isn't either. With `a = 0.5`, `b = 1`, `initial = 1` and inputs
`2, 4, 0` the states are `2.5, 5.25, 2.625`, the values `TorchLeanTest/CUDA/SelectiveScan.lean`
expects from the CUDA kernel. The scan has no PyTorch test that we know of.

References:
- Gu, Goel, Ré. "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), ICLR 2022,
  arXiv:2111.00396, eq. (3) of Sec. 2.3.
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", COLM 2024,
  arXiv:2312.00752: eqs. (2a), (2b) and (4) of Sec. 2, Algorithms 1 and 2 of Sec. 3.2, and the
  parallel scan of Sec. 3.3.2.
- Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
  State Space Duality" (Mamba-2), ICML 2024, arXiv:2405.21060.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean

/--
Run a stateful step over an array, appending each emitted value to `initialOutputs`.

The initial output buffer makes this suitable for chunked execution without changing the state
transition being specified.
-/
def StateSpace.scanArrayFrom {State : Type*} {Input : Type*} {Output : Type*}
    (step : State → Input → State × Output) (initial : State) (initialOutputs : Array Output)
    (xs : Array Input) : State × Array Output :=
  xs.foldl
    (fun (state, outputs) input =>
      let (nextState, output) := step state input
      (nextState, outputs.push output))
    (initial, initialOutputs)

/-- Run a stateful step over an array and return the final state and emitted values. -/
def StateSpace.scanArray {State : Type*} {Input : Type*} {Output : Type*}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    State × Array Output :=
  scanArrayFrom step initial #[] xs

/-- Scanning an empty array returns the initial state and no outputs. -/
@[simp] theorem StateSpace.scanArray_empty {State : Type*} {Input : Type*} {Output : Type*}
    (step : State → Input → State × Output) (initial : State) :
    scanArray step initial (#[] : Array Input) = (initial, (#[] : Array Output)) := by
  rfl

/-- A stateful scan emits exactly one value for each input. -/
@[simp] theorem StateSpace.scanArray_outputs_size {State : Type*} {Input : Type*} {Output : Type*}
    (step : State → Input → State × Output) (initial : State) (xs : Array Input) :
    (scanArray step initial xs).2.size = xs.size := by
  unfold scanArray scanArrayFrom
  let f : (State × Array Output) → Input → State × Array Output :=
    fun (state, outputs) input =>
      let (nextState, output) := step state input
      (nextState, outputs.push output)
  change (Array.foldl f (initial, #[]) xs).2.size = xs.size
  refine Array.foldl_induction (as := xs) (init := (initial, #[])) (f := f)
    (motive := fun i result => result.2.size = i) ?_ ?_
  · rfl
  · intro i result h
    rcases step result.1 xs[i] with ⟨nextState, output⟩
    simp [f, h]

/-- A scalar affine transition `h ↦ a*h + b`. -/
structure StateSpace.ScalarAffineTransition (α : Type*) where
  /-- Linear multiplier. In diagonal SSMs this is one channel of the discretized state matrix. -/
  a : α
  /-- Additive input contribution for the current token. -/
  b : α
deriving Repr

namespace StateSpace.ScalarAffineTransition

variable {α : Type*}

/-- Apply a scalar affine transition. -/
def apply [Mul α] [Add α] (tr : ScalarAffineTransition α) (h : α) : α :=
  tr.a * h + tr.b

/-- Identity affine transition. -/
def id [One α] [Zero α] : ScalarAffineTransition α :=
  { a := 1, b := 0 }

/--
Compose two affine transitions.

`compose t₂ t₁` means "first apply `t₁`, then apply `t₂`".
-/
def compose [Mul α] [Add α] (t₂ t₁ : ScalarAffineTransition α) :
    ScalarAffineTransition α :=
  { a := t₂.a * t₁.a
    b := t₂.a * t₁.b + t₂.b }

end StateSpace.ScalarAffineTransition

/-- Sequentially run scalar affine transitions from an initial state. -/
def StateSpace.runScalarAffine {α : Type*} [Mul α] [Add α] (h0 : α)
    (transitions : Array (ScalarAffineTransition α)) : α :=
  (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).1

/--
Summarize an array of transitions as one affine transition.

Parallel selective scan rests on these summaries: any associative scan algorithm can produce the
prefix summaries, and over a semiring applying the summary to `h0` gives the state the recurrence
reaches (`summarizeScalarAffine_apply`).
-/
def StateSpace.summarizeScalarAffine {α : Type*} [Semiring α]
    (transitions : Array (ScalarAffineTransition α)) : ScalarAffineTransition α :=
  transitions.foldr
    (fun transition summary => ScalarAffineTransition.compose summary transition)
    ScalarAffineTransition.id

/-- Return every recurrent state after each scalar affine transition. -/
def StateSpace.scalarAffineScan {α : Type*} [Mul α] [Add α] (h0 : α) :
    Array (ScalarAffineTransition α) → Array α :=
  fun transitions => (scanArray (fun state transition =>
    let nextState := transition.apply state
    (nextState, nextState)) h0 transitions).2

/-- The scalar affine scan has one state per transition. -/
@[simp] theorem StateSpace.scalarAffineScan_size {α : Type*} [Mul α] [Add α] (h0 : α)
    (transitions : Array (ScalarAffineTransition α)) :
    (scalarAffineScan h0 transitions).size = transitions.size := by
  exact scanArray_outputs_size _ h0 transitions

end TorchLean

namespace TorchLean

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- The states of a diagonal affine scan of length `n`, `h t = a t * h (t - 1) + b t * x t` from
`h (-1) = initial`, in order. -/
def StateSpace.scanStates {n : Nat} (a b x : Fin n → α) (initial : α) : List α :=
  ((List.finRange n).foldl (fun (acc : α × List α) t =>
    let next := a t * acc.1 + b t * x t
    (next, next :: acc.2)) (initial, [])).2.reverse

/-- The adjoint states of the scan at the upstream gradients `g`,
`λ t = g t + λ (t + 1) * a (t + 1)` from the last step, in order. -/
def StateSpace.scanAdjoints {n : Nat} (a g : Fin n → α) : List α :=
  ((List.finRange n).reverse.foldl (fun (acc : Option (Fin n) × α × List α) t =>
    let adjoint := match acc.1 with
      | none => g t
      | some later => g t + acc.2.1 * a later
    (some t, adjoint, adjoint :: acc.2.2)) (none, 0, [])).2.2

/-- The tangents of the states of `scanStates a b x initial` along `da`, `db`, `dx` and
`dinitial`: `dh t = da t * h (t - 1) + a t * dh (t - 1) + db t * x t + b t * dx t`, added left to
right, from `dh (-1) = dinitial`, in order. -/
def StateSpace.scanTangents {n : Nat} (a b x da db dx : Fin n → α) (initial dinitial : α) :
    List α :=
  ((List.finRange n).foldl (fun (acc : α × α × List α) t =>
    let tangent := da t * acc.1 + a t * acc.2.1 + db t * x t + b t * dx t
    (a t * acc.1 + b t * x t, tangent, tangent :: acc.2.2)) (initial, dinitial, [])).2.2.reverse

/-- The shape of a coefficient of a scan of `s` along `dim`: shared over the steps, or one per
step. -/
abbrev Tensor.scanCoefficients (s : Shape) (dim : Fin s.rank) (shared : Bool) : Shape :=
  bif shared then s.eraseIdx dim.val else s

/-- A coefficient of a scan along `dim` at step `t` and the other indices `rest`. -/
def Tensor.scanCoefficient (dim : Fin (Shape.rank s)) :
    (shared : Bool) → Tensor α (scanCoefficients s dim shared) →
      Fin (Shape.axisSize s dim.val) → Shape.Coord (s.eraseIdx dim.val) → α
  | true, a, _, rest => a rest
  | false, a, t, rest => a ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest))

/-- A coefficient gradient of a scan along `dim`: `f t rest` at each step, or for a shared
coefficient its sum over the steps, added from zero in increasing `t`. -/
def Tensor.scanCoefficientGrad (dim : Fin (Shape.rank s)) :
    (shared : Bool) → (Fin (Shape.axisSize s dim.val) →
      Shape.Coord (s.eraseIdx dim.val) → α) →
      Tensor α (scanCoefficients s dim shared)
  | true, f => Rep.ofFn fun rest =>
      (List.finRange (Shape.axisSize s dim.val)).foldl (fun acc t => acc + f t rest) 0
  | false, f => Rep.ofFn fun c =>
      let split := Shape.Coord.axisEquiv dim.val (shape := s) c
      f split.1 split.2

/-- The diagonal selective scan of `x` along `dim`: the state after each step,
`h t = a t * h (t - 1) + b t * x t` from `h (-1) = initial`, each coefficient shared over the
steps or one per step. -/
def Tensor.selectiveScan (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor α (scanCoefficients s dim sharedA)) (b : Tensor α (scanCoefficients s dim sharedB))
    (x : Tensor α s) (initial : Tensor α (s.eraseIdx dim.val)) : Tensor α s :=
  Rep.ofFn fun c =>
    let split := Shape.Coord.axisEquiv dim.val (shape := s) c
    let along := fun (t : Fin (s.axisSize dim.val)) =>
      (Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, split.2)
    (StateSpace.scanStates (scanCoefficient dim sharedA a · split.2)
      (scanCoefficient dim sharedB b · split.2) (fun t => x (along t))
      (initial split.2)).getD split.1.val 0

/-- The pullback of `selectiveScan dim sharedA sharedB a b x initial` at `g`: with the adjoint
states `λ t = g t + λ (t + 1) * a (t + 1)` of each fiber, the gradients `λ t * h (t - 1)` of `a`
and `λ t * x t` of `b` (each summed over the steps when shared), `λ t * b t` of `x`, and
`λ 0 * a 0` of `initial`, zero for an empty scan. -/
def Tensor.selectiveScanBackward (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor α (scanCoefficients s dim sharedA)) (b : Tensor α (scanCoefficients s dim sharedB))
    (x : Tensor α s) (initial : Tensor α (s.eraseIdx dim.val)) (g : Tensor α s) :
    Tensor α (scanCoefficients s dim sharedA) × Tensor α (scanCoefficients s dim sharedB) ×
      Tensor α s × Tensor α (s.eraseIdx dim.val) :=
  let along := fun (t : Fin (s.axisSize dim.val)) (rest : Shape.Coord (s.eraseIdx dim.val)) =>
    (Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest)
  let aAt := fun t rest => scanCoefficient dim sharedA a t rest
  let bAt := fun t rest => scanCoefficient dim sharedB b t rest
  let states := fun rest =>
    StateSpace.scanStates (aAt · rest) (bAt · rest) (fun t => x (along t rest)) (initial rest)
  let adjoints := fun rest => StateSpace.scanAdjoints (aAt · rest) (fun t => g (along t rest))
  let previous := fun (t : Fin (s.axisSize dim.val)) rest =>
    if t.val = 0 then initial rest else (states rest).getD (t.val - 1) 0
  let adjoint := fun (t : Fin (s.axisSize dim.val)) rest => (adjoints rest).getD t.val 0
  (scanCoefficientGrad dim sharedA fun t rest => adjoint t rest * previous t rest,
    scanCoefficientGrad dim sharedB fun t rest => adjoint t rest * x (along t rest),
    Rep.ofFn (fun c =>
      let split := Shape.Coord.axisEquiv dim.val (shape := s) c
      adjoint split.1 split.2 * bAt split.1 split.2),
    Rep.ofFn fun rest =>
      if h : 0 < s.axisSize dim.val then adjoint ⟨0, h⟩ rest * aAt ⟨0, h⟩ rest else 0)

/-- The tangent of `selectiveScan dim sharedA sharedB a b x initial` along `da`, `db`, `dx` and
`dinitial`: on each fiber the tangent recurrence of `StateSpace.scanTangents`, each coefficient's
tangent shared over the steps or one per step as the coefficient is. -/
def Tensor.selectiveScanJVP (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor α (scanCoefficients s dim sharedA)) (b : Tensor α (scanCoefficients s dim sharedB))
    (x : Tensor α s) (initial : Tensor α (s.eraseIdx dim.val))
    (da : Tensor α (scanCoefficients s dim sharedA))
    (db : Tensor α (scanCoefficients s dim sharedB)) (dx : Tensor α s)
    (dinitial : Tensor α (s.eraseIdx dim.val)) : Tensor α s :=
  Rep.ofFn fun c =>
    let split := Shape.Coord.axisEquiv dim.val (shape := s) c
    let along := fun (t : Fin (s.axisSize dim.val)) =>
      (Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, split.2)
    (StateSpace.scanTangents (scanCoefficient dim sharedA a · split.2)
      (scanCoefficient dim sharedB b · split.2) (fun t => x (along t))
      (scanCoefficient dim sharedA da · split.2)
      (scanCoefficient dim sharedB db · split.2) (fun t => dx (along t))
      (initial split.2) (dinitial split.2)).getD split.1.val 0

end TorchLean
