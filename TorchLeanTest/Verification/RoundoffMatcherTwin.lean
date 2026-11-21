/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

/-!
# A private matcher of the name of one of `TorchLeanTest.Verification.RoundoffMatcher`

`pick` is private to this module, as the `pick` of `TorchLeanTest/Verification/RoundoffMatcher.lean`
is to that one, so their matchers have one name less their private prefixes, and their splitters
and equations, realized in one module, have the same names. That module checks that
`restoreDeclarations` realizes them again from the right matcher.
-/

namespace TorchLeanTest.RoundoffMatcher

/-- A private definition of this module, with overlapping patterns, compiled to a matcher private to
this module whose splitter does not use it. -/
private def pick (n : Nat) : Bool :=
  match n with
  | 0 => true
  | _ => false

end TorchLeanTest.RoundoffMatcher
