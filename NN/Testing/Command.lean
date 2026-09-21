/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Testing.Compare
public meta import Lean.Elab.BuiltinEvalCommand

/-!
# Differential checks in the editor

`#compare candidate with reference on cases` runs pure functions on an explicit, nonempty array.
Add `using relation` to choose the comparison. This is a test command, not a proof tactic.
Import `NN.Testing.Compare` instead for the effectful runner in a compiled executable.
-/

public meta section

/-- Execute a differential test of pure functions on explicit inputs. This is not a proof. -/
syntax (name := compareCommand) "#compare " term " with " term " on " term
  (" using " term)? : command

macro_rules
  | `(#compare $candidate with $reference on $cases using $agrees) =>
    `(#eval do
      let inputs := $cases
      NN.Testing.compareOn inputs (fun x => pure ($candidate x))
        (fun x => pure ($reference x)) $agrees
      IO.println s!"Compared {inputs.size} cases (test only; not a proof).")
  | `(#compare $candidate with $reference on $cases) =>
    `(#compare $candidate with $reference on $cases using (· == ·))
