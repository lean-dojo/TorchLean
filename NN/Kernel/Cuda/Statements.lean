/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda.Parsing
import NN.Kernel.Cuda.Correctness

/-!
# Semantics of emitted kernel statements

The text grammar composes checked assignments in source order. A branch decodes its condition,
destination and result names, runs only its selected arm, then copies that arm's result. Errors
and undefined expressions short-circuit subsequent writes. Unsigned loops re-read their guard and
stop before copying a failed body result. Allocation preserves the bound and counter, connecting
these guard-based semantics to bounded evaluation. This grammar covers the emitter's structured
statements, not arbitrary CUDA syntax. `Cuda.Source` supplies the global entrypoint semantics.
-/

@[expose] public section

namespace NN.Kernel.Cuda

/-- Finite execution of an unsigned `for` loop, with its guard re-evaluated at every step.

The counter is scoped to the loop. Its active value is supplied to the guard and body; after the
loop, its slot is outside the live scope. Successful steps copy the result into the accumulator
and increment the counter left by the body. Errors and undefined arithmetic stop before that copy.
There is no fuel parameter: only an exhausted guard or an early failure ends a finite derivation.
-/
inductive Loop.Denotes {α : Type} (body : Locals α → ExceptT Error Option (Locals α))
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar)
    (count : Atom α Name .index) :
    UInt64 → Locals α → Option (Except Error (Locals α)) → Prop where
  | halt {i : UInt64} {locals : Locals α}
      (h : ¬i < count.eval (locals.set counter i).get) :
      Loop.Denotes body acc counter next count i locals (some (.ok locals))
  | undefined {i : UInt64} {locals : Locals α}
      (h : i < count.eval (locals.set counter i).get)
      (hBody : (body (locals.set counter i)).run = none) :
      Loop.Denotes body acc counter next count i locals none
  | error {i : UInt64} {locals : Locals α} {error : Error}
      (h : i < count.eval (locals.set counter i).get)
      (hBody : (body (locals.set counter i)).run = some (.error error)) :
      Loop.Denotes body acc counter next count i locals (some (.error error))
  | step {i : UInt64} {locals after : Locals α} {result : Option (Except Error (Locals α))}
      (h : i < count.eval (locals.set counter i).get)
      (hBody : (body (locals.set counter i)).run = some (.ok after))
      (hNext : Loop.Denotes body acc counter next count (after.get counter + 1)
        (after.set acc (after.get next)) result) :
      Loop.Denotes body acc counter next count i locals result

/-- Bounded evaluation has finite guard-based semantics when each successful step preserves the
bound and active counter. The equality on remaining steps prevents fuel from truncating a loop.
Unsigned increment does not wrap while the guard is true, including at the maximum bound.
-/
theorem Loop.denotes_steps {α : Type} (body : Locals α → ExceptT Error Option (Locals α))
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar)
    (count : Atom α Name .index)
    (hCount : ∀ (locals : Locals α) i,
      count.eval (locals.set counter i).get = count.eval locals.get)
    (hStep : ∀ (locals : Locals α) i after,
      (body (locals.set counter i)).run = some (.ok after) →
      after.get counter = i ∧
        count.eval (after.set acc (after.get next)).get = count.eval locals.get)
    (bound : UInt64) (n : Nat) (i : UInt64) (locals : Locals α)
    (hValue : count.eval locals.get = bound) (hRemaining : n + i.toNat = bound.toNat) :
    Loop.Denotes body acc counter next count i locals
      (Statement.Internal.steps body acc counter next bound n i locals).run := by
  induction n generalizing i locals with
  | zero =>
      apply Loop.Denotes.halt
      rw [hCount, hValue, UInt64.lt_iff_toNat_lt]
      omega
  | succ n ih =>
      have hGuard : i < bound := by
        rw [UInt64.lt_iff_toNat_lt]
        omega
      have hTextGuard : i < count.eval (locals.set counter i).get := by
        rw [hCount, hValue]
        exact hGuard
      have hNext : n + (i + 1).toNat = bound.toNat := by
        rw [Target.Internal.increment_toNat hGuard]
        omega
      simp only [Statement.Internal.steps, hGuard, ↓reduceIte]
      cases hBody : (body (locals.set counter i)).run with
      | none =>
          rw [show body (locals.set counter i) = ExceptT.mk none from hBody]
          exact Loop.Denotes.undefined hTextGuard hBody
      | some result =>
          rw [show body (locals.set counter i) = ExceptT.mk (some result) from hBody]
          cases result with
          | error error => exact Loop.Denotes.error hTextGuard hBody
          | ok after =>
              have hStable := hStep locals i after hBody
              apply Loop.Denotes.step hTextGuard hBody
              rw [hStable.1]
              exact ih (i + 1) (after.set acc (after.get next)) (hStable.2.trans hValue) hNext

/-- Source-order semantics for empty text, assignments, sequences, conditionals and unsigned loops.

The buffer and arithmetic contracts are inherited from `Assignment.Denotes`. The relation uses
decoded tokens rather than the renderer's type and name selection. Both branch arms have source
semantics, but execution evaluates only the selected arm. Loops use finite guard-based derivations,
not a fuel-truncated interpreter. `Cuda.Source` covers entrypoint text; native atomics, memory,
compilation and device execution remain external boundaries.
-/
inductive Statement.Denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (inputs : Nat) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value) :
    String → (Locals format.Value → ExceptT Error Option (Locals format.Value)) → Prop where
  | skip : Statement.Denotes format arithmetic inputs size load "" (fun locals => pure locals)
  | assign {text : String}
      {run : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (h : ∀ locals, Assignment.Denotes format arithmetic inputs size load locals text
        (run locals).run) :
      Statement.Denotes format arithmetic inputs size load text run
  | seq {firstText secondText : String}
      {first second : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (hFirst : Statement.Denotes format arithmetic inputs size load firstText first)
      (hSecond : Statement.Denotes format arithmetic inputs size load secondText second) :
      Statement.Denotes format arithmetic inputs size load (firstText ++ secondText)
        (fun locals => do
          let after ← first locals
          second after)
  | branch {t : Ty} {typeText nameText conditionText yesText yesName noText noName : String}
      {yes no : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (name : Name t) (condition : Atom format.Value Name .predicate) (y n : Name t)
      (ht : Internal.type? format typeText = some t)
      (hn : Name.parse t nameText = some name)
      (hp : Atom.parse format .predicate conditionText = some condition)
      (hy : Name.parse t yesName = some y) (hno : Name.parse t noName = some n)
      (hYes : Statement.Denotes format arithmetic inputs size load yesText yes)
      (hNo : Statement.Denotes format arithmetic inputs size load noText no) :
      Statement.Denotes format arithmetic inputs size load
        (s!"{typeText} {nameText};\nif ({conditionText}) \{\n" ++ yesText ++
          s!"{nameText} = {yesName};\n} else \{\n" ++ noText ++
          s!"{nameText} = {noName};\n}\n")
        (fun locals => do
          if condition.eval locals.get then
            let after ← yes locals
            pure (after.set name (after.get y))
          else
            let after ← no locals
            pure (after.set name (after.get n)))
  | loop {typeText accText counterText countText initialText bodyText nextText : String}
      {body run : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (acc : Name .scalar) (counter : Name .index) (next : Name .scalar)
      (count : Atom format.Value Name .index) (initial : Atom format.Value Name .scalar)
      (ht : Internal.type? format typeText = some .scalar)
      (ha : Name.parse .scalar accText = some acc)
      (hi : Name.parse .index counterText = some counter)
      (hc : Atom.parse format .index countText = some count)
      (hv : Atom.parse format .scalar initialText = some initial)
      (hn : Name.parse .scalar nextText = some next)
      (hBody : Statement.Denotes format arithmetic inputs size load bodyText body)
      (hLoop : ∀ locals, Loop.Denotes body acc counter next count 0
        (locals.set acc (initial.eval locals.get)) (run locals).run) :
      Statement.Denotes format arithmetic inputs size load
        (s!"{typeText} {accText} = {initialText};\n" ++
          s!"for (unsigned long long {counterText} = 0ULL; {counterText} < {countText}; " ++
          s!"++{counterText}) \{\n" ++ bodyText ++ s!"{accText} = {nextText};\n}\n") run

/-- A rendered assignment retains its result and error behavior inside a statement sequence. -/
theorem Statement.denotes_render_assign (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value)
    (hRead : ∀ operand, operand < inputs → ∀ index,
      read operand index = if index ≥ size operand then
        .error (.bounds operand index (size operand).toNat) else .ok (load operand index))
    {t : Ty} (assignment : Assignment format.Value t) {text : String}
    (h : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs
      (.assign assignment) = .ok text) :
    Statement.Denotes format arithmetic inputs size load text
      ((Statement.assign assignment).eval arithmetic read) := by
  apply Statement.Denotes.assign
  intro locals
  exact Assignment.denotes_render format arithmetic read inputs size load hRead locals assignment h

/-- Rendering a sequence preserves source-order composition, including early errors. -/
theorem Statement.denotes_render_seq (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value) (first second : Statement format.Value)
    (hFirst : ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits) inputs first = .ok text →
      Statement.Denotes format arithmetic inputs size load text (first.eval arithmetic read))
    (hSecond : ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits) inputs second = .ok text →
      Statement.Denotes format arithmetic inputs size load text (second.eval arithmetic read))
    {text : String}
    (h : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs
      (.seq first second) = .ok text) :
    Statement.Denotes format arithmetic inputs size load text
      ((Statement.seq first second).eval arithmetic read) := by
  cases hf : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs first with
  | error error =>
      simp [Internal.render, Internal.renderWith, hf, bind, Except.bind] at h
  | ok firstText =>
      cases hs : Internal.render format (fun x => (Literal.ofValue format x).bits)
          inputs second with
      | error error =>
          simp [Internal.render, Internal.renderWith, hf, hs, bind, Except.bind] at h
      | ok secondText =>
          simp only [Internal.render, Internal.renderWith, hf, hs] at h
          dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
            Except.pure] at h
          injection h with htext
          subst text
          exact Statement.Denotes.seq (hFirst hf) (hSecond hs)

/-- Rendering a conditional preserves its typed result copy and lazy arm selection.

The premises supply text semantics for each arm, so the result composes with further statements
without requiring either arm to execute during the proof or run.
-/
theorem Statement.denotes_render_branch (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value) {t : Ty} (name : Name t)
    (condition : Atom format.Value Name .predicate) (yes no : Statement format.Value)
    (y n : Name t)
    (hYes : ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits) inputs yes = .ok text →
      Statement.Denotes format arithmetic inputs size load text (yes.eval arithmetic read))
    (hNo : ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits) inputs no = .ok text →
      Statement.Denotes format arithmetic inputs size load text (no.eval arithmetic read))
    {text : String}
    (h : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs
      (.branch name condition yes y no n) = .ok text) :
    Statement.Denotes format arithmetic inputs size load text
      ((Statement.branch name condition yes y no n).eval arithmetic read) := by
  cases hp : Internal.atom format (fun x => (Literal.ofValue format x).bits) condition with
  | error error =>
      simp [Internal.render, Internal.renderWith, hp, bind, Except.bind] at h
  | ok conditionText =>
      cases hy : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs yes with
      | error error =>
          simp [Internal.render, Internal.renderWith, hp, hy, bind, Except.bind] at h
      | ok yesText =>
          cases hn : Internal.render format (fun x => (Literal.ofValue format x).bits)
              inputs no with
          | error error =>
              simp [Internal.render, Internal.renderWith, hp, hy, hn, bind, Except.bind] at h
          | ok noText =>
              simp only [Internal.render, Internal.renderWith, hp, hy, hn] at h
              dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
                Except.pure] at h
              injection h with htext
              subst text
              simp only [Internal.typeName_native]
              apply Statement.Denotes.branch name condition y n
                (Internal.type_typeName format t) (Name.parse_render name) _
                (Name.parse_render y) (Name.parse_render n) (hYes hy) (hNo hn)
              have hparse := Atom.parse_render format condition
              simp only [hp, Except.toOption, Option.bind_some] at hparse
              exact hparse

/-- Rendering an unsigned loop agrees with bounded evaluation when the guard's bound and active
counter remain unchanged by initialization and successful body steps.

The bound is re-read in the text semantics, not assumed to be captured once. The premises exclude
mutating it or the counter; they do not exclude body failures or undefined arithmetic. Fresh name
allocation is needed to discharge these premises for generated blocks.
-/
theorem Statement.denotes_render_loop (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value) (acc : Name .scalar) (counter : Name .index)
    (count : Atom format.Value Name .index) (initial : Atom format.Value Name .scalar)
    (body : Statement format.Value) (next : Name .scalar)
    (hCount : ∀ (locals : Locals format.Value) i,
      count.eval (locals.set counter i).get = count.eval locals.get)
    (hInitial : ∀ (locals : Locals format.Value),
      count.eval (locals.set acc (initial.eval locals.get)).get = count.eval locals.get)
    (hStep : ∀ (locals : Locals format.Value) i after,
      (body.eval arithmetic read (locals.set counter i)).run = some (.ok after) →
      after.get counter = i ∧
        count.eval (after.set acc (after.get next)).get = count.eval locals.get)
    (hBody : ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits) inputs body = .ok text →
      Statement.Denotes format arithmetic inputs size load text (body.eval arithmetic read))
    {text : String}
    (h : Internal.render format (fun x => (Literal.ofValue format x).bits) inputs
      (.loop acc counter count initial body next) = .ok text) :
    Statement.Denotes format arithmetic inputs size load text
      ((Statement.loop acc counter count initial body next).eval arithmetic read) := by
  cases hc : Internal.atom format (fun x => (Literal.ofValue format x).bits) count with
  | error error => simp [Internal.render, Internal.renderWith, hc, bind, Except.bind] at h
  | ok countText =>
      cases hv : Internal.atom format (fun x => (Literal.ofValue format x).bits) initial with
      | error error => simp [Internal.render, Internal.renderWith, hc, hv, bind, Except.bind] at h
      | ok initialText =>
          cases hb : Internal.render format (fun x => (Literal.ofValue format x).bits)
              inputs body with
          | error error =>
              simp [Internal.render, Internal.renderWith, hc, hv, hb, bind, Except.bind] at h
          | ok bodyText =>
              simp only [Internal.render, Internal.renderWith, hc, hv, hb] at h
              dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
                Except.pure] at h
              injection h with htext
              subst text
              have ht : Internal.type? format (Internal.Format.name format) = some .scalar := by
                cases format <;> rfl
              have hCountParse := Atom.parse_render format count
              simp only [hc, Except.toOption, Option.bind_some] at hCountParse
              have hInitialParse := Atom.parse_render format initial
              simp only [hv, Except.toOption, Option.bind_some] at hInitialParse
              refine Statement.Denotes.loop acc counter next count initial ht
                (Name.parse_render acc) (Name.parse_render counter) hCountParse hInitialParse
                (Name.parse_render next) (hBody hb) ?_
              intro locals
              exact Loop.denotes_steps (body.eval arithmetic read) acc counter next count
                hCount hStep (count.eval locals.get) (count.eval locals.get).toNat 0
                (locals.set acc (initial.eval locals.get)) (hInitial locals) (by simp)

/-- Every successfully rendered allocated block has the named evaluator's text semantics.

The source names lie before the allocation counter. Freshness supplies the read-only bound and
counter invariants for every generated loop, including nested loops and branches. Arithmetic and
buffer contents remain explicit contracts; this theorem does not assume native execution agrees.
-/
theorem Statement.denotes_lower (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value)
    (hRead : ∀ operand, operand < inputs → ∀ index,
      read operand index = if index ≥ size operand then
        .error (.bounds operand index (size operand).toNat) else .ok (load operand index))
    {Γ : List Ty} {t : Ty} (block : Target.Block format.Value Γ t)
    (names : Internal.Names Γ) (first : Nat)
    (hNames : ∀ {s : Ty} (v : Var Γ s), (names.get v).Before first) :
    let ((statement, _), _) := (Internal.lower names block).run first
    ∀ {text : String},
      Internal.render format (fun x => (Literal.ofValue format x).bits)
        inputs statement = .ok text →
      Statement.Denotes format arithmetic inputs size load text
        (statement.eval arithmetic read) := by
  induction block generalizing first with
  | result value =>
      cases value with
      | scalar value =>
          intro text h
          exact Statement.denotes_render_assign format arithmetic read inputs size load hRead _ h
      | index value =>
          intro text h
          exact Statement.denotes_render_assign format arithmetic read inputs size load hRead _ h
      | predicate value =>
          intro text h
          exact Statement.denotes_render_assign format arithmetic read inputs size load hRead _ h
      | var v =>
          intro text h
          change Except.ok "" = Except.ok text at h
          cases h
          exact Statement.Denotes.skip
  | compute operation =>
      intro text h
      exact Statement.denotes_render_assign format arithmetic read inputs size load hRead _ h
  | branch p yes no ihYes ihNo =>
      change let ((yc, y), ny) := (Internal.lower names yes).run (first + 1)
        let ((nc, n), _) := (Internal.lower names no).run ny
        ∀ {text : String}, Internal.render format (fun x => (Literal.ofValue format x).bits)
          inputs (.branch (.temporary first) (Atom.lower names.get p) yc y nc n) = .ok text →
          Statement.Denotes format arithmetic inputs size load text
            ((Statement.branch (.temporary first) (Atom.lower names.get p) yc y nc n).eval
              arithmetic read)
      have hYesNames : ∀ {s : Ty} (v : Var _ s), (names.get v).Before (first + 1) :=
        fun v => Internal.before_mono _ (Nat.le_succ first) (hNames v)
      have hYes := @ihYes names (first + 1) hYesNames
      have hBounds := Internal.lower_bounds yes names (first + 1) hYesNames
      generalize hY : (Internal.lower names yes).run (first + 1) = yr at hYes hBounds ⊢
      rcases yr with ⟨⟨yc, y⟩, ny⟩
      dsimp only at ⊢
      have hNoNames : ∀ {s : Ty} (v : Var _ s), (names.get v).Before ny :=
        fun v => Internal.before_mono _ (by omega) (hNames v)
      have hNo := @ihNo names ny hNoNames
      generalize hN : (Internal.lower names no).run ny = nr at hNo ⊢
      rcases nr with ⟨⟨nc, n⟩, nn⟩
      intro text h
      exact Statement.denotes_render_branch format arithmetic read inputs size load
        (.temporary first) (Atom.lower names.get p) yc nc y n hYes hNo h
  | bind value body ihValue ihBody =>
      change let ((vc, v), nv) := (Internal.lower names value).run first
        let ((bc, _), _) := (Internal.lower (names.push v) body).run nv
        ∀ {text : String}, Internal.render format (fun x => (Literal.ofValue format x).bits)
          inputs (.seq vc bc) = .ok text → Statement.Denotes format arithmetic inputs size load
            text ((Statement.seq vc bc).eval arithmetic read)
      have hValue := @ihValue names first hNames
      have hBounds := Internal.lower_bounds value names first hNames
      generalize hV : (Internal.lower names value).run first = vr at hValue hBounds ⊢
      rcases vr with ⟨⟨vc, v⟩, nv⟩
      dsimp only at ⊢
      have hBodyNames : ∀ {s : Ty} (w : Var _ s), ((names.push v).get w).Before nv := by
        intro s w
        cases w with
        | zero => exact hBounds.2.1
        | succ w => exact Internal.before_mono _ hBounds.1 (hNames w)
      have hBody := @ihBody (names.push v) nv hBodyNames
      generalize hB : (Internal.lower (names.push v) body).run nv = br at hBody ⊢
      rcases br with ⟨⟨bc, b⟩, nb⟩
      intro text h
      exact Statement.denotes_render_seq format arithmetic read inputs size load
        vc bc hValue hBody h
  | loop count initial body ih =>
      change let ((bc, next), _) := (Internal.lower
          ((names.push (.temporary (first + 1) : Name .index)).push
            (.temporary first : Name .scalar)) body).run (first + 2)
        ∀ {text : String}, Internal.render format (fun x => (Literal.ofValue format x).bits)
          inputs (.loop (.temporary first) (.temporary (first + 1))
            (Atom.lower names.get count) (Atom.lower names.get initial) bc next) = .ok text →
          Statement.Denotes format arithmetic inputs size load text
            ((Statement.loop (.temporary first) (.temporary (first + 1))
              (Atom.lower names.get count) (Atom.lower names.get initial) bc next).eval
                arithmetic read)
      let bodyNames := (names.push (.temporary (first + 1) : Name .index)).push
        (.temporary first : Name .scalar)
      have hBodyNames : ∀ {s : Ty} (w : Var _ s), (bodyNames.get w).Before (first + 2) := by
        intro s w
        cases w with
        | zero => change first < first + 2; omega
        | succ w =>
            cases w with
            | zero => change first + 1 < first + 2; omega
            | succ w => exact Internal.before_mono _ (by omega) (hNames w)
      have hBody := @ih bodyNames (first + 2) hBodyNames
      have hBounds := Internal.lower_bounds body bodyNames (first + 2) hBodyNames
      generalize hB : (Internal.lower bodyNames body).run (first + 2) = br at hBody hBounds ⊢
      rcases br with ⟨⟨bc, next⟩, nb⟩
      have hCounter : ¬ (Name.temporary (first + 1) : Name .index).Before first := by
        change ¬ first + 1 < first
        omega
      have hAcc : ¬ (Name.temporary first : Name .scalar).Before first := by
        change ¬ first < first
        omega
      intro text h
      apply Statement.denotes_render_loop format arithmetic read inputs size load
        (.temporary first) (.temporary (first + 1)) (Atom.lower names.get count)
        (Atom.lower names.get initial) bc next _ _ _ hBody h
      · intro locals i
        cases count with
        | index value => rfl
        | var v =>
            exact Internal.get_set_fresh locals (.temporary (first + 1)) i
              (names.get v) (hNames v) hCounter
      · intro locals
        cases count with
        | index value => rfl
        | var v =>
            exact Internal.get_set_fresh locals (.temporary first)
              ((Atom.lower names.get initial).eval locals.get) (names.get v) (hNames v) hAcc
      · intro locals i after hResult
        have hFrame := Internal.eval_frame arithmetic read bc (first + 2) hBounds.2.2
          (locals.set (.temporary (first + 1) : Name .index) i) after hResult
        constructor
        · rw [hFrame .index (.temporary (first + 1)) (by change first + 1 < first + 2; omega),
            Locals.get_set]
        · cases count with
          | index value => rfl
          | var v =>
              change (after.set (.temporary first : Name .scalar) (after.get next)).get
                (names.get v) = locals.get (names.get v)
              rw [Internal.get_set_fresh _ _ _ _ (hNames v) hAcc,
                hFrame .index (names.get v) (Internal.before_mono _ (by omega) (hNames v)),
                Internal.get_set_fresh _ _ _ _ (hNames v) hCounter]

end NN.Kernel.Cuda
