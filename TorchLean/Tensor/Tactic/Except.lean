/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import Lean.Elab.Tactic.Simpa

/-!
# Successful validation chains

`except_cases` splits an `Except` computation at an intermediate step and continues with the
successful value. The error branch is closed by a success hypothesis already in context: `simp`
unfolds the bind and turns that hypothesis into a contradiction.
-/

public meta section

namespace TorchLean

/-- Split an intermediate `Except` result, using the enclosing computation's success to rule out
the error branch. Unfold the enclosing computation in `hOk` first if necessary.

`except_cases hf : f using hOk with value => ...` provides `hf : f = .ok value` in the body.
The error and payload types are unrestricted.
-/
syntax (name := exceptCases) "except_cases " ident " : " term " using " term
  " with " ident " => " tacticSeq : tactic

macro_rules
  | `(tactic| except_cases $h:ident : $e:term using $hOk:term with $v:ident => $body:tacticSeq) =>
      `(tactic|
        cases $h:ident : $e with
        | error err =>
            have success := $hOk
            simp [Bind.bind, Except.bind, *] at success
        | ok $v =>
            $body)

end TorchLean
