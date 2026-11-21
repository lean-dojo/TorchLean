/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
import all TorchLean.Tensor.Numeric.Enclosure
import all TorchLeanTest.Verification.RoundoffMatcherTwin

/-!
# Restoring the splitter and equations of a private matcher

When a state restored by `#roundoff_coverage` lacks the splitter or an equation of a matcher,
`restoreDeclarations` realizes them again with `Match.getEquationsFor`, from the matcher
`auxiliaryMatchers` finds. `Match.getEquationsFor` declares them under the private prefix of the
current module, after removing the matcher's own private prefix (`mkPrivateName`), so the name of a
private matcher is not the one their names carry, and private matchers of one name in different
modules have splitters and equations of the same names. The checks realize the equations of the
matchers of `pick`, private to this module, of the `pick` private to
`TorchLeanTest/Verification/RoundoffMatcherTwin.lean`, and of `coefficientBits`, private to
`TorchLean/Tensor/Numeric/Enclosure.lean`, drop them with a restored state, and restore them. The
splitter of the second alone, which does not use its matcher, may split either `pick`, and is not
restored.
-/

namespace TorchLeanTest.RoundoffMatcher

/-- A private definition of this module, compiled to a matcher private to this module. -/
private def pick (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | _ + 1 => 2

end TorchLeanTest.RoundoffMatcher

meta section

namespace TorchLeanTest.RoundoffMatcher

open Lean Meta TorchLean.Roundoff

/-- Realize the equations of `matcher`, check that `auxiliaryMatchers` finds `matcher` alone from
the splitter and equations, restore the state from before, which lacks them, and check that
`restoreDeclarations` realizes them again: the matcher's equations can then still be read. The state
from before is restored at the end. -/
def checkMatcher (matcher : Name) : MetaM Unit := do
  unless isMatcherCore (← getEnv) matcher do throwError "{matcher} is not a matcher"
  let saved ← saveState
  let equations ← Match.getEquationsFor matcher
  let names := equations.eqnNames.push equations.splitterName
  let declarations ← names.mapM getConstInfo
  for name in names do
    let found := auxiliaryMatchers (← getEnv) declarations name
    unless found == #[matcher] do throwError "the matchers of {name} are {found}, not {matcher}"
  saved.restore
  if (← getEnv).contains equations.splitterName then
    throwError "the restored state holds {equations.splitterName}"
  restoreDeclarations declarations
  unless names.all (← getEnv).contains do throwError "{names} are not all declared again"
  let again ← Match.getEquationsFor matcher
  unless again.splitterName == equations.splitterName && again.eqnNames == equations.eqnNames do
    throwError "the equations of {matcher} read again are {again.eqnNames}"
  saved.restore

/-- The matcher of the `pick` private to this module. -/
def localPick : MetaM Name :=
  return mkPrivateName (← getEnv) `TorchLeanTest.RoundoffMatcher.pick.match_1

/-- The matcher of the `pick` private to `TorchLeanTest.Verification.RoundoffMatcherTwin`. -/
def twinPick : Name :=
  mkPrivateNameCore `TorchLeanTest.Verification.RoundoffMatcherTwin
    `TorchLeanTest.RoundoffMatcher.pick.match_1

-- The two matchers of `pick`, whose splitters and equations have the same names.
run_meta do
  let saved ← saveState
  let equations ← Match.getEquationsFor (← localPick)
  saved.restore
  let twins ← Match.getEquationsFor twinPick
  saved.restore
  unless twins.splitterName == equations.splitterName && twins.eqnNames == equations.eqnNames do
    throwError "the splitters are {twins.splitterName} and {equations.splitterName}"
  checkMatcher (← localPick)
  checkMatcher twinPick

-- The splitter alone of the twin, which does not use its matcher, is not restored.
run_meta do
  let saved ← saveState
  let splitter ← getConstInfo (← Match.getEquationsFor twinPick).splitterName
  let found := auxiliaryMatchers (← getEnv) #[splitter] splitter.name
  unless found.size == 2 && found.contains twinPick && found.contains (← localPick) do
    throwError "the matchers of {splitter.name} alone are {found}"
  saved.restore
  let restored ← try
      restoreDeclarations #[splitter]
      pure true
    catch _ => pure false
  if restored then throwError "{splitter.name} alone is restored"
  saved.restore

-- The matcher of `coefficientBits`, private to `TorchLean.Tensor.Numeric.Enclosure`.
run_meta checkMatcher (mkPrivateNameCore `TorchLean.Tensor.Numeric.Enclosure
  `TorchLean.Numeric.coefficientBits.match_1)

end TorchLeanTest.RoundoffMatcher
