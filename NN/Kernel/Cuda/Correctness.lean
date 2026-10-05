/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda

/-!
# Correctness of named kernel lowering

CUDA rendering uses numbered temporaries and mutable loop locals. Allocation must keep earlier
source bindings live, including across branch-result copies and repeated loop updates. The proofs
here establish that names advance monotonically and generated destinations cannot overwrite the
older allocation prefix. `eval_lower` then proves result agreement, including checked read errors,
for every scoped block. Scalar arithmetic is supplied without reassociation laws.

These are statements about the named evaluator. They do not establish correctness of CUDA text,
NVRTC compilation or GPU execution.
-/

@[expose] public section

namespace NN.Kernel.Cuda

namespace Internal

/-- Every destination is outside the local names allocated before `first`. -/
def writesAfter {α : Type} (first : Nat) : Statement α → Prop
  | .skip => True
  | .assign value => ¬ value.name.Before first
  | .seq firstBlock secondBlock => writesAfter first firstBlock ∧ writesAfter first secondBlock
  | .branch name _ yes _ no _ => ¬ name.Before first ∧ writesAfter first yes ∧ writesAfter first no
  | .loop acc counter _ _ body _ =>
      ¬ acc.Before first ∧ ¬ counter.Before first ∧ writesAfter first body

/-- Increasing the allocation bound retains every name already in scope. -/
theorem before_mono {t : Ty} {m n : Nat} (name : Name t) (h : m ≤ n)
    (hName : name.Before m) : name.Before n := by
  cases name with
  | index => trivial
  | temporary number => exact Nat.lt_of_lt_of_le hName h

/-- A block that only writes newer locals also leaves an earlier prefix untouched. -/
theorem writesAfter_mono {α : Type} {m n : Nat} (statement : Statement α) (h : m ≤ n)
    (hWrites : writesAfter n statement) : writesAfter m statement := by
  induction statement with
  | skip => trivial
  | assign value => exact fun hName => hWrites (before_mono value.name h hName)
  | seq first second ihFirst ihSecond => exact ⟨ihFirst hWrites.1, ihSecond hWrites.2⟩
  | branch name condition yes y no r ihYes ihNo =>
      exact ⟨fun hName => hWrites.1 (before_mono name h hName),
        ihYes hWrites.2.1, ihNo hWrites.2.2⟩
  | loop acc counter count initial body next ih =>
      exact ⟨fun hName => hWrites.1 (before_mono acc h hName),
        fun hName => hWrites.2.1 (before_mono counter h hName), ih hWrites.2.2⟩

/-- Allocation advances monotonically, returns an in-scope name and only writes fresh locals. -/
theorem lower_bounds {α : Type} {Γ : List Ty} {t : Ty} (block : Target.Block α Γ t)
    (names : Names Γ) (first : Nat)
    (hNames : ∀ {s : Ty} (v : Var Γ s), (names.get v).Before first) :
    let ((statement, result), finish) := (lower names block).run first
    first ≤ finish ∧ result.Before finish ∧ writesAfter first statement := by
  induction block generalizing first with
  | result value =>
      cases value with
      | scalar value =>
          change first ≤ first + 1 ∧ first < first + 1 ∧ ¬ first < first
          omega
      | index value =>
          change first ≤ first + 1 ∧ first < first + 1 ∧ ¬ first < first
          omega
      | predicate value =>
          change first ≤ first + 1 ∧ first < first + 1 ∧ ¬ first < first
          omega
      | var v => exact ⟨Nat.le_refl _, hNames v, trivial⟩
  | compute operation =>
      change first ≤ first + 1 ∧ first < first + 1 ∧ ¬ first < first
      omega
  | branch p yes no ihYes ihNo =>
      change let ((yc, y), ny) := (lower names yes).run (first + 1)
        let ((nc, n), nn) := (lower names no).run ny
        first ≤ nn ∧ (Name.temporary first : Name _).Before nn ∧
          writesAfter first (.branch (.temporary first) (Atom.lower names.get p) yc y nc n)
      have hYes := ihYes names (first + 1) (fun v =>
        before_mono (names.get v) (Nat.le_succ first) (hNames v))
      generalize hY : (lower names yes).run (first + 1) = y at hYes ⊢
      rcases y with ⟨⟨yc, y⟩, ny⟩
      dsimp only at ⊢
      have hNo := ihNo names ny (fun v =>
        before_mono (names.get v) (Nat.le_trans (Nat.le_succ first) hYes.1) (hNames v))
      generalize hN : (lower names no).run ny = n at hNo ⊢
      rcases n with ⟨⟨nc, n⟩, nn⟩
      change first ≤ nn ∧ first < nn ∧
        (¬ first < first ∧ writesAfter first yc ∧ writesAfter first nc)
      refine ⟨?_, ?_, by omega, ?_, ?_⟩
      · omega
      · omega
      · exact writesAfter_mono yc (Nat.le_succ first) hYes.2.2
      · exact writesAfter_mono nc (by omega) hNo.2.2
  | bind value body ihValue ihBody =>
      change let ((vc, v), nv) := (lower names value).run first
        let ((bc, b), nb) := (lower (names.push v) body).run nv
        first ≤ nb ∧ b.Before nb ∧ writesAfter first (.seq vc bc)
      have hValue := ihValue names first hNames
      generalize hV : (lower names value).run first = v at hValue ⊢
      rcases v with ⟨⟨vc, v⟩, nv⟩
      dsimp only at ⊢
      have hScope : ∀ {s : Ty} (w : Var _ s), ((names.push v).get w).Before nv := by
        intro s w
        cases w with
        | zero => exact hValue.2.1
        | succ w => exact before_mono (names.get w) hValue.1 (hNames w)
      have hBody := ihBody (names.push v) nv hScope
      generalize hB : (lower (names.push v) body).run nv = b at hBody ⊢
      rcases b with ⟨⟨bc, b⟩, nb⟩
      change first ≤ nb ∧ b.Before nb ∧ (writesAfter first vc ∧ writesAfter first bc)
      exact ⟨Nat.le_trans hValue.1 hBody.1, hBody.2.1, hValue.2.2,
        writesAfter_mono bc hValue.1 hBody.2.2⟩
  | loop count initial body ih =>
      change let ((bc, b), nb) := (lower
          ((names.push (.temporary (first + 1) : Name .index)).push
            (.temporary first : Name .scalar)) body).run (first + 2)
        first ≤ nb ∧ (Name.temporary first : Name .scalar).Before nb ∧
          writesAfter first (.loop (.temporary first) (.temporary (first + 1))
            (Atom.lower names.get count) (Atom.lower names.get initial) bc b)
      have hActualScope : ∀ {s : Ty} (w : Var _ s),
          ((names.push (.temporary (first + 1) : Name .index)).push
            (.temporary first : Name .scalar)).get w |>.Before (first + 2) := by
        intro s w
        cases w with
        | zero => change first < first + 2; omega
        | succ w =>
            cases w with
            | zero => change first + 1 < first + 2; omega
            | succ w => exact before_mono (names.get w) (by omega) (hNames w)
      have hBody := ih
        ((names.push (.temporary (first + 1) : Name .index)).push
          (.temporary first : Name .scalar)) (first + 2) hActualScope
      generalize hB : (lower
        ((names.push (.temporary (first + 1) : Name .index)).push
          (.temporary first : Name .scalar)) body).run (first + 2) = b at hBody ⊢
      rcases b with ⟨⟨bc, b⟩, nb⟩
      change first ≤ nb ∧ first < nb ∧
        (¬ first < first ∧ ¬ first + 1 < first ∧ writesAfter first bc)
      refine ⟨by omega, by omega, by omega, by omega, ?_⟩
      exact writesAfter_mono bc (by omega) hBody.2.2

/-- An older local and a fresh destination have different CUDA spellings. -/
theorem before_ne {s t : Ty} {first : Nat} (earlier : Name s) (later : Name t)
    (hBefore : earlier.Before first) (hAfter : ¬ later.Before first) :
    earlier.render ≠ later.render := by
  cases later with
  | index => exact False.elim (hAfter trivial)
  | temporary n =>
      cases earlier with
      | index => exact Name.index_ne_temporary n
      | temporary m =>
          intro h
          have hNumber := Name.temporary_render_inj.mp h
          subst n
          exact hAfter hBefore

/-- Writing outside an allocation prefix leaves every name in that prefix unchanged. -/
theorem get_set_fresh {α : Type} {s t : Ty} {first : Nat} (locals : Locals α)
    (name : Name t) (value : t.Value α) (other : Name s)
    (hBefore : other.Before first) (hAfter : ¬ name.Before first) :
    (locals.set name value).get other = locals.get other :=
  Locals.get_set_of_ne locals name value other (before_ne other name hBefore hAfter)

/-- Two local valuations agree on every name allocated before `first`. -/
def frame {α : Type} (first : Nat) (before after : Locals α) : Prop :=
  ∀ (t : Ty) (name : Name t), name.Before first → after.get name = before.get name

/-- Agreement on an allocation prefix composes across successive blocks. -/
theorem frame_trans {α : Type} {first : Nat} {a b c : Locals α}
    (hAB : frame first a b) (hBC : frame first b c) : frame first a c := by
  intro t name h
  rw [hBC _ name h, hAB _ name h]

/-- Counter writes, body execution and accumulator copies retain every older local. -/
theorem steps_frame {α : Type} (first : Nat)
    (body : Locals α → ExceptT Error Option (Locals α))
    (hBody : ∀ before after, (body before).run = some (Except.ok (ε := Error) after) →
      frame first before after)
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar) (count : UInt64)
    (hAcc : ¬ acc.Before first) (hCounter : ¬ counter.Before first)
    (n : Nat) (i : UInt64) (locals after : Locals α)
    (hResult : (Statement.Internal.steps body acc counter next count n i locals).run =
      some (Except.ok (ε := Error) after)) : frame first locals after := by
  induction n generalizing i locals with
  | zero =>
      change some (Except.ok (ε := Error) locals) = some (Except.ok (ε := Error) after) at hResult
      cases hResult
      exact fun _ _ _ => rfl
  | succ n ih =>
      by_cases h : i < count
      · simp only [Statement.Internal.steps, h, ↓reduceIte] at hResult
        cases hEval : (body (locals.set counter i)).run with
        | none =>
            change ((body (locals.set counter i)).run >>= _) =
              some (Except.ok (ε := Error) after) at hResult
            simp [hEval] at hResult
        | some result =>
            cases result with
            | error error =>
                change ((body (locals.set counter i)).run >>= _) =
                  some (Except.ok (ε := Error) after) at hResult
                simp [hEval, ExceptT.bindCont] at hResult
            | ok current =>
                change ((body (locals.set counter i)).run >>= _) =
                  some (Except.ok (ε := Error) after) at hResult
                simp only [hEval] at hResult
                have hTail : frame first (current.set acc (current.get next)) after :=
                  ih (i + 1) (current.set acc (current.get next)) hResult
                intro t name hName
                rw [hTail _ name hName, get_set_fresh _ _ _ _ hName hAcc,
                  hBody _ _ hEval _ name hName, get_set_fresh _ _ _ _ hName hCounter]
      · simp only [Statement.Internal.steps, h, ↓reduceIte] at hResult
        change some (Except.ok (ε := Error) locals) = some (Except.ok (ε := Error) after) at hResult
        cases hResult
        exact fun _ _ _ => rfl

/-- Successfully executing a block with fresh destinations preserves the earlier valuation. -/
theorem eval_frame {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    (statement : Statement α) (first : Nat) (hWrites : writesAfter first statement)
    (locals after : Locals α)
    (hResult : (statement.eval arithmetic read locals).run = some (Except.ok (ε := Error) after)) :
    frame first locals after := by
  induction statement generalizing locals after with
  | skip =>
      change some (Except.ok (ε := Error) locals) = some (Except.ok (ε := Error) after) at hResult
      cases hResult
      exact fun _ _ _ => rfl
  | assign value =>
      change value.eval arithmetic read locals = some (Except.ok (ε := Error) after) at hResult
      cases hEval : value.operation.eval arithmetic read locals.get with
      | none => simp [Assignment.eval, hEval] at hResult
      | some result =>
          cases result with
          | error error => simp [Assignment.eval, hEval, Except.map] at hResult
          | ok result =>
              simp only [Assignment.eval, hEval, Option.map_some, Except.map,
                Option.some.injEq, Except.ok.injEq] at hResult
              subst after
              exact fun _ name hName => get_set_fresh _ _ _ _ hName hWrites
  | seq a b ihA ihB =>
      change ((a.eval arithmetic read locals).run >>= _) =
        some (Except.ok (ε := Error) after) at hResult
      cases hEval : (a.eval arithmetic read locals).run with
      | none => simp [hEval] at hResult
      | some result =>
          cases result with
          | error error => simp [hEval, ExceptT.bindCont] at hResult
          | ok current =>
              simp only [hEval] at hResult
              exact frame_trans (ihA hWrites.1 _ _ hEval) (ihB hWrites.2 _ _ hResult)
  | branch name condition yes y no n ihYes ihNo =>
      cases hCondition : condition.eval locals.get with
      | false =>
          simp only [Statement.eval, hCondition, Bool.false_eq_true, ↓reduceIte] at hResult
          cases hEval : (no.eval arithmetic read locals).run with
          | none =>
              change ((no.eval arithmetic read locals).run >>= _) =
                some (Except.ok (ε := Error) after) at hResult
              simp [hEval] at hResult
          | some result =>
              cases result with
              | error error =>
                  change ((no.eval arithmetic read locals).run >>= _) =
                    some (Except.ok (ε := Error) after) at hResult
                  simp [hEval, ExceptT.bindCont] at hResult
              | ok current =>
                  change ((no.eval arithmetic read locals).run >>= _) =
                    some (Except.ok (ε := Error) after) at hResult
                  simp only [hEval] at hResult
                  change some (Except.ok (ε := Error) (current.set name (current.get n))) =
                    some (Except.ok (ε := Error) after) at hResult
                  cases hResult
                  intro t other hName
                  rw [get_set_fresh _ _ _ _ hName hWrites.1]
                  exact ihNo hWrites.2.2 _ _ hEval _ other hName
      | true =>
          simp only [Statement.eval, hCondition, ↓reduceIte] at hResult
          cases hEval : (yes.eval arithmetic read locals).run with
          | none =>
              change ((yes.eval arithmetic read locals).run >>= _) =
                some (Except.ok (ε := Error) after) at hResult
              simp [hEval] at hResult
          | some result =>
              cases result with
              | error error =>
                  change ((yes.eval arithmetic read locals).run >>= _) =
                    some (Except.ok (ε := Error) after) at hResult
                  simp [hEval, ExceptT.bindCont] at hResult
              | ok current =>
                  change ((yes.eval arithmetic read locals).run >>= _) =
                    some (Except.ok (ε := Error) after) at hResult
                  simp only [hEval] at hResult
                  change some (Except.ok (ε := Error) (current.set name (current.get y))) =
                    some (Except.ok (ε := Error) after) at hResult
                  cases hResult
                  intro t other hName
                  rw [get_set_fresh _ _ _ _ hName hWrites.1]
                  exact ihYes hWrites.2.1 _ _ hEval _ other hName
  | loop acc counter count initial body next ih =>
      have hLoop := steps_frame first (body.eval arithmetic read)
        (fun before after h => ih hWrites.2.2 before after h) acc counter next
        (count.eval locals.get) hWrites.1 hWrites.2.1 _ 0
        (locals.set acc (initial.eval locals.get)) after hResult
      intro t name hName
      rw [hLoop _ name hName, get_set_fresh _ _ _ _ hName hWrites.1]

/-- A successful projected result comes from a successful unprojected valuation. -/
theorem mapped_ok {α β : Type} (value : Option (Except Error α)) (f : α → β) (y : β)
    (h : value.map (·.map f) = some (.ok y)) :
    ∃ x, value = some (.ok x) ∧ f x = y := by
  cases value with
  | none => cases h
  | some result =>
      cases result with
      | error error => cases h
      | ok x => exact ⟨x, rfl, by simpa [Except.map] using h⟩

/-- Projecting successful values cannot create or change a checked error. -/
theorem mapped_error {α β : Type} (value : Option (Except Error α)) (f : α → β)
    (error : Error) (h : value.map (·.map f) = some (.error error)) :
    value = some (.error error) := by
  cases value with
  | none => cases h
  | some result =>
      cases result with
      | error e => simpa [Except.map] using h
      | ok x => cases h

/-- A body simulation and preserved invariant compose over the exact bounded step order. -/
theorem steps_correct {α : Type} (body : Locals α → ExceptT Error Option (Locals α))
    (step : UInt64 → α → Except Error α) (invariant : Locals α → Prop)
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar) (count : UInt64)
    (hBody : ∀ i locals, invariant locals →
      (body (locals.set counter i)).run.map (·.map (·.get next)) =
        some (step i (locals.get acc)))
    (hNext : ∀ i locals after, invariant locals →
      (body (locals.set counter i)).run = some (.ok after) →
        invariant (after.set acc (after.get next)))
    (n : Nat) (i : UInt64) (locals : Locals α) (hInvariant : invariant locals)
    (hBound : n + i.toNat ≤ count.toNat) :
    (Statement.Internal.steps body acc counter next count n i locals).run.map
      (·.map (·.get acc)) = some (iterate step n i (locals.get acc)) := by
  induction n generalizing i locals with
  | zero => rfl
  | succ n ih =>
      have h : i < count := by
        simp only [UInt64.lt_iff_toNat_lt]
        omega
      have hNextBound : n + (i + 1).toNat ≤ count.toNat := by
        rw [Target.Internal.increment_toNat h]
        omega
      have hCurrent := hBody i locals hInvariant
      cases hStep : step i (locals.get acc) with
      | error error =>
          rw [hStep] at hCurrent
          have hEval := mapped_error _ _ _ hCurrent
          simp only [Statement.Internal.steps, h, ↓reduceIte, iterate, hStep]
          change (((body (locals.set counter i)).run >>= _).map _) = _
          rw [hEval]
          rfl
      | ok value =>
          rw [hStep] at hCurrent
          obtain ⟨after, hEval, hValue⟩ := mapped_ok _ _ _ hCurrent
          simp only [Statement.Internal.steps, h, ↓reduceIte, iterate, hStep]
          change (((body (locals.set counter i)).run >>= _).map _) = _
          rw [hEval]
          change (Statement.Internal.steps body acc counter next count n (i + 1)
            (after.set acc (after.get next))).run.map (·.map (·.get acc)) = _
          rw [ih (i + 1) (after.set acc (after.get next))
            (hNext i locals after hInvariant hEval) hNextBound, Locals.get_set, hValue]
          rfl

/-- Copying the selected branch result retains its value and its read failures. -/
theorem branch_result {α : Type} {t : Ty} (arithmetic : Arithmetic α) (read : Reader α)
    (name : Name t) (condition : Atom α Name .predicate) (yes no : Statement α)
    (y n : Name t) (locals : Locals α) :
    ((Statement.branch name condition yes y no n).eval arithmetic read locals).run.map
        (·.map (·.get name)) =
      if condition.eval locals.get then
        (yes.eval arithmetic read locals).run.map (·.map (·.get y))
      else (no.eval arithmetic read locals).run.map (·.map (·.get n)) := by
  cases hCondition : condition.eval locals.get with
  | false =>
      cases hEval : (no.eval arithmetic read locals).run with
      | none => simp [Statement.eval, hCondition, hEval]
      | some result =>
          cases result <;>
            simp [Statement.eval, hCondition, hEval, Except.map, Locals.get_set]
  | true =>
      cases hEval : (yes.eval arithmetic read locals).run with
      | none => simp [Statement.eval, hCondition, hEval]
      | some result =>
          cases result <;>
            simp [Statement.eval, hCondition, hEval, Except.map, Locals.get_set]

end Internal

/-- Running an allocated block leaves every previously allocated local unchanged.

The initial source names must lie before the allocation counter. This theorem applies to a
successful run; failed reads do not produce an output valuation.
-/
theorem lower_preserves_local {α : Type} {Γ : List Ty} {s t : Ty}
    (block : Target.Block α Γ t) (names : Internal.Names Γ) (first : Nat)
    (hNames : ∀ {u : Ty} (v : Var Γ u), (names.get v).Before first)
    (arithmetic : Arithmetic α) (read : Reader α) (locals after : Locals α)
    (other : Name s) (hBefore : other.Before first)
    (hResult : (((Internal.lower names block).run first).1.1.eval
      arithmetic read locals).run = some (.ok after)) :
    after.get other = locals.get other := by
  have hBounds := Internal.lower_bounds block names first hNames
  exact Internal.eval_frame arithmetic read _ first hBounds.2.2 locals after hResult s other hBefore

open Internal

/-- Allocating scoped blocks into named statements preserves their result and checked errors.

The initial named valuation must represent the source scope, whose names lie below the allocation
counter. Other initial local values are arbitrary. Arithmetic is supplied without algebraic laws:
the theorem preserves the specified operation and accumulation order, not a reassociated formula.
It concerns named semantics, not execution of rendered CUDA by NVIDIA's compiler and device.
-/
theorem eval_lower {α : Type} {Γ : List Ty} {t : Ty} (block : Target.Block α Γ t)
    (names : Names Γ) (first : Nat) (locals : Locals α) (env : Env α Γ)
    (hNames : ∀ {s : Ty} (v : Var Γ s), (names.get v).Before first)
    (hScope : ∀ {s : Ty} (v : Var Γ s), locals.get (names.get v) = env.get v)
    (arithmetic : Arithmetic α) (read : Reader α) :
    let ((statement, result), _) := (lower names block).run first
    (statement.eval arithmetic read locals).run.map (·.map (·.get result)) =
      some (block.eval arithmetic read env) := by
  induction block generalizing first locals with
  | result value =>
      cases value with
      | scalar value =>
          change (some (Except.ok (ε := Error)
            (locals.set (.temporary first : Name .scalar) value))).map
            (·.map (·.get (.temporary first : Name .scalar))) = some (.ok value)
          simp [Except.map, Locals.get_set]
      | index value =>
          change (some (Except.ok (ε := Error)
            (locals.set (.temporary first : Name .index) value))).map
            (·.map (·.get (.temporary first : Name .index))) = some (.ok value)
          simp [Except.map, Locals.get_set]
      | predicate value =>
          change (some (Except.ok (ε := Error)
            (locals.set (.temporary first : Name .predicate) value))).map
            (·.map (·.get (.temporary first : Name .predicate))) = some (.ok value)
          simp [Except.map, Locals.get_set]
      | var v =>
          change some (Except.ok (ε := Error) (locals.get (names.get v))) =
            some (.ok (env.get v))
          rw [hScope]
  | compute operation =>
      change (Assignment.eval ⟨.temporary first, Primitive.lower names.get operation⟩
        arithmetic read locals).map (·.map (·.get (.temporary first))) =
          some (operation.eval arithmetic read env)
      rw [Assignment.eval_lower names.get locals env hScope]
      cases operation.eval arithmetic read env <;> simp [Except.map, Locals.get_set]
  | branch p yes no ihYes ihNo =>
      change let ((yc, y), ny) := (lower names yes).run (first + 1)
        let ((nc, n), _) := (lower names no).run ny
        ((Statement.branch (.temporary first) (Atom.lower names.get p) yc y nc n).eval
          arithmetic read locals).run.map (·.map (·.get (.temporary first))) =
            some ((Target.Block.branch p yes no).eval arithmetic read env)
      have hYesNames : ∀ {s : Ty} (v : Var _ s), (names.get v).Before (first + 1) :=
        fun v => before_mono _ (Nat.le_succ first) (hNames v)
      have hYes := ihYes names (first + 1) locals env hYesNames hScope
      have hBounds := lower_bounds yes names (first + 1) hYesNames
      generalize hY : (lower names yes).run (first + 1) = yr at hYes hBounds ⊢
      rcases yr with ⟨⟨yc, y⟩, ny⟩
      dsimp only at ⊢
      have hNoNames : ∀ {s : Ty} (v : Var _ s), (names.get v).Before ny :=
        fun v => before_mono _ (by omega) (hNames v)
      have hNo := ihNo names ny locals env hNoNames hScope
      generalize hN : (lower names no).run ny = nr at hNo ⊢
      rcases nr with ⟨⟨nc, n⟩, nn⟩
      dsimp only at ⊢
      rw [branch_result, Atom.eval_lower names.get locals.get env hScope]
      cases hP : p.eval env with
      | false => simpa only [Target.Block.eval, hP, Bool.false_eq_true, ↓reduceIte] using hNo
      | true => simpa only [Target.Block.eval, hP, ↓reduceIte] using hYes
  | bind value body ihValue ihBody =>
      change let ((vc, v), nv) := (lower names value).run first
        let ((bc, b), _) := (lower (names.push v) body).run nv
        ((Statement.seq vc bc).eval arithmetic read locals).run.map (·.map (·.get b)) =
          some ((Target.Block.bind value body).eval arithmetic read env)
      have hValue := ihValue names first locals env hNames hScope
      have hBounds := lower_bounds value names first hNames
      generalize hV : (lower names value).run first = vr at hValue hBounds ⊢
      rcases vr with ⟨⟨vc, v⟩, nv⟩
      dsimp only at ⊢
      have hBodyNames : ∀ {s : Ty} (w : Var _ s), ((names.push v).get w).Before nv := by
        intro s w
        cases w with
        | zero => exact hBounds.2.1
        | succ w => exact before_mono _ hBounds.1 (hNames w)
      cases hValueEval : value.eval arithmetic read env with
      | error error =>
          rw [hValueEval] at hValue
          have hEval := mapped_error _ _ _ hValue
          simp only [Statement.eval, Target.Block.eval, hValueEval]
          change (((vc.eval arithmetic read locals).run >>= _).map _) = _
          rw [hEval]
          rfl
      | ok x =>
          rw [hValueEval] at hValue
          obtain ⟨after, hEval, hStored⟩ := mapped_ok _ _ _ hValue
          have hBodyScope : ∀ {s : Ty} (w : Var _ s),
              after.get ((names.push v).get w) = (env.push x).get w := by
            intro s w
            cases w with
            | zero => exact hStored
            | succ w =>
                change after.get (names.get w) = env.get w
                rw [show after.get (names.get w) = locals.get (names.get w) from
                  eval_frame arithmetic read vc first hBounds.2.2 locals after hEval
                    _ (names.get w) (hNames w)]
                exact hScope w
          have hBody := ihBody (names.push v) nv after (env.push x) hBodyNames hBodyScope
          generalize hB : (lower (names.push v) body).run nv = br at hBody ⊢
          rcases br with ⟨⟨bc, b⟩, nb⟩
          simp only [Statement.eval, Target.Block.eval, hValueEval]
          change (((vc.eval arithmetic read locals).run >>= _).map _) = _
          rw [hEval]
          exact hBody
  | loop count initial body ih =>
      change let ((bc, next), _) := (lower
          ((names.push (.temporary (first + 1) : Name .index)).push
            (.temporary first : Name .scalar)) body).run (first + 2)
        ((Statement.loop (.temporary first) (.temporary (first + 1))
          (Atom.lower names.get count) (Atom.lower names.get initial) bc next).eval
          arithmetic read locals).run.map (·.map (·.get (.temporary first))) =
            some ((Target.Block.loop count initial body).eval arithmetic read env)
      let bodyNames := (names.push (.temporary (first + 1) : Name .index)).push
        (.temporary first : Name .scalar)
      have hBodyNames : ∀ {s : Ty} (w : Var _ s), (bodyNames.get w).Before (first + 2) := by
        intro s w
        cases w with
        | zero => change first < first + 2; omega
        | succ w =>
            cases w with
            | zero => change first + 1 < first + 2; omega
            | succ w => exact before_mono _ (by omega) (hNames w)
      have hBounds := lower_bounds body bodyNames (first + 2) hBodyNames
      generalize hB : (lower bodyNames body).run (first + 2) = br at hBounds ⊢
      rcases br with ⟨⟨bc, next⟩, nb⟩
      let invariant := fun current : Locals α =>
        ∀ (s : Ty) (v : Var _ s), current.get (names.get v) = env.get v
      let step := fun i acc => body.eval arithmetic read
        (Env.push (t := .scalar) acc (Env.push (t := .index) i env))
      have hCounter : ¬ (Name.temporary (first + 1) : Name .index).Before first := by
        change ¬ first + 1 < first; omega
      have hAcc : ¬ (Name.temporary first : Name .scalar).Before first := by
        change ¬ first < first; omega
      have hBody : ∀ i current, invariant current →
          (bc.eval arithmetic read (current.set (.temporary (first + 1) : Name .index) i)).run.map
            (·.map (·.get next)) = some (step i (current.get (.temporary first))) := by
        intro i current hInvariant
        have hBodyScope : ∀ {s : Ty} (w : Var _ s),
            (current.set (.temporary (first + 1) : Name .index) i).get (bodyNames.get w) =
              ((env.push i).push (current.get (.temporary first : Name .scalar))).get w := by
          intro s w
          cases w with
          | zero =>
              apply Locals.get_set_of_ne
              intro h
              have hn := Name.temporary_render_inj.mp h
              omega
          | succ w =>
              cases w with
              | zero => exact Locals.get_set _ _ _
              | succ w =>
                  change (current.set (.temporary (first + 1) : Name .index) i).get
                    (names.get w) = env.get w
                  rw [get_set_fresh _ _ _ _ (hNames w) hCounter]
                  exact hInvariant _ w
        have h := ih bodyNames (first + 2)
          (current.set (.temporary (first + 1) : Name .index) i)
          ((env.push i).push (current.get (.temporary first : Name .scalar)))
          hBodyNames hBodyScope
        simpa only [hB] using h
      have hNext : ∀ (i : UInt64) (current after : Locals α), invariant current →
          (bc.eval arithmetic read (current.set (.temporary (first + 1) : Name .index) i)).run =
            some (.ok after) →
              invariant (after.set (.temporary first : Name .scalar) (after.get next)) := by
        intro i current after hInvariant hEval s v
        rw [get_set_fresh _ _ _ _ (hNames v) hAcc]
        have hFrame := eval_frame arithmetic read bc (first + 2) hBounds.2.2
          (current.set (.temporary (first + 1) : Name .index) i) after hEval
        rw [hFrame _ (names.get v) (before_mono _ (by omega) (hNames v)),
          get_set_fresh _ _ _ _ (hNames v) hCounter]
        exact hInvariant _ v
      have hInitial : invariant
          (locals.set (.temporary first) ((Atom.lower names.get initial).eval locals.get)) := by
        intro s v
        rw [get_set_fresh _ _ _ _ (hNames v) hAcc]
        exact hScope v
      have hSteps := steps_correct (bc.eval arithmetic read) step invariant
        (.temporary first) (.temporary (first + 1)) next
        ((Atom.lower names.get count).eval locals.get) hBody hNext
        ((Atom.lower names.get count).eval locals.get).toNat 0
        (locals.set (.temporary first) ((Atom.lower names.get initial).eval locals.get))
        hInitial (by simp)
      rw [Target.Block.eval, Target.loop_eq_iterate]
      simpa only [Statement.eval, Locals.get_set, UInt64.toNat_zero, Nat.sub_zero,
        Atom.eval_lower names.get locals.get env hScope] using hSteps

end NN.Kernel.Cuda
