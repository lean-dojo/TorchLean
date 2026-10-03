/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Builder
public import NN.Spec.Core.Context.Rational
public import NN.Tensor

/-!
# TapeBuilderTest

Exercises `NN.Proofs.Autograd.Tape.Builder` on a user-style eager program over `ℚ`.

The point of the module under test is that a `do`-block written in `TapeM` can be reasoned about,
not merely run. So this file does both to the same program: it runs it and checks the value, and it
*proves*, from nothing but "the block ran successfully", the pure-`Tape` fact behind every
statement in the block. The proof is the peel: `run_bind_inv` once per statement, then `run_leaf`
or `record_run_inv` at each op.
-/

open scoped Spec.RationalAlgebraic

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tests
namespace Rationals
namespace TapeBuilder

open Runtime.Autograd
open Proofs.Autograd.Builder

abbrev tag : String := "tape_builder_test (Rat)"

/-- First input. -/
def xa : Tensor ℚ [2] := (Tensor.from #[(2 : ℚ), 3]).reshape [2] (by dsimp; decide)

/-- Second input. -/
def xb : Tensor ℚ [2] := (Tensor.from #[(5 : ℚ), 7]).reshape [2] (by dsimp; decide)

/-- The scaling constant. -/
def c : ℚ := 4

/-- A user-style eager program: two leaves, a product, a scaling. -/
def prog : TapeM ℚ Nat := do
  let a ← TapeM.leaf (s := [2]) xa
  let b ← TapeM.leaf (s := [2]) xb
  let m ← TapeM.mul (s := [2]) a b
  TapeM.scale (s := [2]) m c

/--
A successful run of `prog` yields the pure-`Tape` fact behind each of its four statements.

Nothing about the tape's contents is assumed: the hypothesis is only that the block ran, and the
conclusion names the intermediate tapes and node ids the block threaded implicitly.
-/
theorem prog_peel {t t' : Tape ℚ} {ids : Nat} (h : prog.run t = .ok (ids, t')) :
    ∃ (t1 t2 t3 : Tape ℚ) (ida idb idm : Nat),
      Tape.leaf (t := t) xa = (t1, ida)
      ∧ Tape.leaf (t := t1) xb = (t2, idb)
      ∧ Tape.mul (t := t2) (s := [2]) ida idb = .ok (t3, idm)
      ∧ Tape.scale (t := t3) (s := [2]) idm c = .ok (t', ids) := by
  obtain ⟨ida, t1, hA, hrest⟩ := run_bind_inv h
  obtain ⟨idb, t2, hB, hrest⟩ := run_bind_inv hrest
  obtain ⟨idm, t3, hM, hS⟩ := run_bind_inv hrest
  refine ⟨t1, t2, t3, ida, idb, idm, ?_, ?_, record_run_inv hM, record_run_inv hS⟩
  · rw [run_leaf (t := t) xa] at hA
    simp only [Except.ok.injEq, Prod.mk.injEq] at hA
    exact Prod.ext hA.2 hA.1
  · rw [run_leaf (t := t1) xb] at hB
    simp only [Except.ok.injEq, Prod.mk.injEq] at hB
    exact Prod.ext hB.2 hB.1

/-- The program runs, and `4 * (xa * xb)` is `[40, 84]`. -/
def checkProg : Result Bool := do
  let (id, t) ← TapeM.run (Tape.empty) prog
  let out ← Tape.requireValue (t := t) (s := [2]) id
  let expected : Tensor ℚ [2] := (Tensor.from #[(40 : ℚ), 84]).reshape [2] (by dsimp; decide)
  pure (decide (pretty out = pretty expected))

-- `Tape.empty` has no native implementation available to the elaborator, so `checkProg` cannot be
-- forced by a compile-time `#guard`; `run` below is what executes it, under the test suite.
def run : IO Unit := do
  match checkProg with
  | .ok true => IO.println s!"{tag}: OK"
  | .ok false => throw <| IO.userError s!"{tag}: FAILED"
  | .error msg => throw <| IO.userError s!"{tag}: {msg}"

end TapeBuilder
end Rationals
end Tests
