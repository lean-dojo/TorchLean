/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Lean.Parser.Tactic

public import NN.Spec.RL.Core

/-!
# RL Proof Tactics

This file defines a small set of proof-focused tactics/macros used across `NN/Proofs/RL`.

Design goals:

* **Lightweight**: no heavyweight automation, just convenience wrappers.
* **Stable**: avoid tactics that are fragile under small definitional changes.
* **Local**: the API is intentionally small; add new helpers only when repeated boilerplate
  has shown up in multiple RL proof files.

References:

- Lean 4 documentation (macros, tactics, `simp`): https://lean-lang.org/lean4/doc/
-/

@[expose] public section

namespace Proofs
namespace RL

/-- `simp_rl` is a `simp`-wrapper used across `NN/Proofs/RL`.

It unfolds the core one-step formulas by default:

- `Spec.RL.discountedBackup`
- `Spec.RL.tdTarget`
- `Spec.RL.tdResidual`

`continueMask` is intentionally *not* unfolded by default, since many proofs keep it symbolic and
only unfold it at the point they need case-splits on `done`.

Usage:

```lean
simp_rl
simp_rl [myLemma, myOtherLemma]
```
-/
syntax (name := simp_rl) "simp_rl" ("[" Lean.Parser.Tactic.simpLemma,* "]")? : tactic

macro_rules
  | `(tactic| simp_rl) =>
      `(tactic|
        simp [Spec.RL.discountedBackup, Spec.RL.tdTarget, Spec.RL.tdResidual])
  | `(tactic| simp_rl [$lemmas,*]) =>
      `(tactic|
        simp [Spec.RL.discountedBackup, Spec.RL.tdTarget, Spec.RL.tdResidual,
          $lemmas,*])

/-- `except_cases` peels a successful `Except` do-chain by case-splitting an intermediate step.

This is useful in trust-boundary proofs (where checkers return `Except String _`):

- In the `.ok` branch, it gives you an equality `h : e = .ok v` and continues with payload `v`.
- In the `.error` branch, it closes the goal by contradiction using the hypothesis that the *whole*
  chain succeeded.

Example:

```lean
have hOk : (do _ ← f; g) = .ok x := ...
except_cases hf : f using hOk with v =>
  -- `v` is the `.ok` payload from `f`.
  ...
```
-/
syntax (name := except_cases) "except_cases " ident " : " term " using " term
  " with " ident " => " tacticSeq : tactic

macro_rules
  | `(tactic| except_cases $h:ident : $e:term using $hOk:term with $v:ident => $body:tacticSeq) =>
      `(tactic|
        cases $h:ident : $e with
        | error err =>
            cases (by simpa [*] using $hOk)
        | ok $v =>
            $body)

end RL
end Proofs
