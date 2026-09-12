/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.Context.Real

/-!
# What directed endpoint arithmetic is supposed to mean

`NN.MLTheory.CROWN.BoundOps` says what a scalar backend must compute; this file says what those
computations mean, by interpreting an endpoint as a real number and comparing each directed
operation with exact real arithmetic. The enclosure predicates and the two `Lawful` classes are the
form in which CROWN theorems ask for that guarantee, and `ℝ` itself is the instance where the
guarantee is trivial because no rounding happens.

Naming `ℝ` alongside `Real.exp`, `Real.log`, and `Real.sqrt` costs roughly 1700 mathlib modules, so
the split is deliberate: certificate replay, the graph checker, and the trainer only ever run the
executable interface and stop at `BoundOps`, while the soundness proofs continue on to this file.
-/

@[expose] public section

namespace NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Real-semantic enclosure laws for `BoundOps`.

The executable interface above is intentionally available without this class: a backend may be
useful for diagnostics before its arithmetic has been connected to a proof.  Sound CROWN theorems
require `LawfulBoundOps` in addition to `BoundOps`. The interpretation `toReal` says what a scalar
endpoint means mathematically, and the laws compare each directed operation with exact arithmetic
on those real values. This is stronger than merely surrounding the backend's ordinary rounded
operation.

There is a global instance for `ℝ`.  There is deliberately no global instance for Lean `Float` or
for all `IEEE32Exec` bit patterns.  Host `Float` is a trusted runtime boundary, while IEEE-754 NaNs,
infinities, and overflow require finite-path hypotheses; those facts are stated at the IEEE
semantics layer rather than hidden in an invalid ordered-ring instance.
-/
class LawfulBoundOps (α : Type) [TorchLean.Storage α] [Context α] [BoundOps α] where
  /-- Mathematical value represented by an endpoint. -/
  toReal : α → ℝ
  /-- Executable endpoint comparisons agree with the mathematical order. -/
  lt_iff (a b : α) : a < b ↔ toReal a < toReal b
  addDown_le (a b : α) : toReal (BoundOps.addDown a b) ≤ toReal a + toReal b
  le_addUp (a b : α) : toReal a + toReal b ≤ toReal (BoundOps.addUp a b)
  subDown_le (a b : α) : toReal (BoundOps.subDown a b) ≤ toReal a - toReal b
  le_subUp (a b : α) : toReal a - toReal b ≤ toReal (BoundOps.subUp a b)
  mulDown_le (a b : α) : toReal (BoundOps.mulDown a b) ≤ toReal a * toReal b
  le_mulUp (a b : α) : toReal a * toReal b ≤ toReal (BoundOps.mulUp a b)

/--
Soundness predicate for a unary interval transfer.

Returning `none` is always permitted. If the transfer returns endpoints, every real input between
the interpreted input endpoints must map between the interpreted output endpoints.
-/
def UnaryEnclosure [BoundOps α] [LawfulBoundOps α]
    (f : ℝ → ℝ) (transfer : α → α → Option (α × α)) : Prop :=
  ∀ {lo hi outLo outHi : α} {x : ℝ}, transfer lo hi = some (outLo, outHi) →
    LawfulBoundOps.toReal lo ≤ x → x ≤ LawfulBoundOps.toReal hi →
    LawfulBoundOps.toReal outLo ≤ f x ∧ f x ≤ LawfulBoundOps.toReal outHi

/-- Soundness predicate for a binary interval transfer. -/
def BinaryEnclosure [BoundOps α] [LawfulBoundOps α]
    (f : ℝ → ℝ → ℝ) (transfer : α → α → α → α → Option (α × α)) : Prop :=
  ∀ {aLo aHi bLo bHi outLo outHi : α} {x y : ℝ},
    transfer aLo aHi bLo bHi = some (outLo, outHi) →
    LawfulBoundOps.toReal aLo ≤ x → x ≤ LawfulBoundOps.toReal aHi →
    LawfulBoundOps.toReal bLo ≤ y → y ≤ LawfulBoundOps.toReal bHi →
    LawfulBoundOps.toReal outLo ≤ f x y ∧ f x y ≤ LawfulBoundOps.toReal outHi

/--
Real-semantic enclosure laws for `NonlinearBoundOps`.

This class is deliberately separate from the executable transfer table. A backend may implement a
transfer for testing before proving it; sound verification entrypoints can require this class and
therefore cannot silently promote an unchecked implementation into a theorem.
-/
class LawfulNonlinearBoundOps (α : Type) [TorchLean.Storage α] [Context α] [BoundOps α]
    [LawfulBoundOps α] [NonlinearBoundOps α] : Prop where
  divBounds_enclosure :
    BinaryEnclosure (α := α) (· / ·) (NonlinearBoundOps.divBounds (α := α))
  expBounds_enclosure :
    UnaryEnclosure (α := α) Real.exp (NonlinearBoundOps.expBounds (α := α))
  logBounds_enclosure :
    UnaryEnclosure (α := α) Real.log (NonlinearBoundOps.logBounds (α := α))
  sqrtBounds_enclosure :
    UnaryEnclosure (α := α) Real.sqrt (NonlinearBoundOps.sqrtBounds (α := α))
  sigmoidBounds_enclosure :
    UnaryEnclosure (α := α) (fun x : ℝ => 1 / (1 + Real.exp (-x)))
      (NonlinearBoundOps.sigmoidBounds (α := α))
  tanhBounds_enclosure :
    UnaryEnclosure (α := α) Real.tanh (NonlinearBoundOps.tanhBounds (α := α))
  sinBounds_enclosure :
    UnaryEnclosure (α := α) Real.sin (NonlinearBoundOps.sinBounds (α := α))
  cosBounds_enclosure :
    UnaryEnclosure (α := α) Real.cos (NonlinearBoundOps.cosBounds (α := α))
  layerNormAbsBound_sound {n : Nat} {radius : α} :
    NonlinearBoundOps.layerNormAbsBound (α := α) n = some radius →
      Real.sqrt n ≤ LawfulBoundOps.toReal radius
  coupledDerivatives_exact :
    NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) →
      BoundOps.supportsExactAffineReassociation (α := α)

/-!
Exact real arithmetic needs no rounding, so its lower and upper operations coincide.
-/
noncomputable instance instBoundOpsReal : BoundOps ℝ where
  addDown := (· + ·)
  addUp   := (· + ·)
  subDown := (· - ·)
  subUp   := (· - ·)
  mulDown := (· * ·)
  mulUp   := (· * ·)
  supportsExactAffineReassociation := true

/-- Exact real endpoint arithmetic satisfies the directed-operation enclosure laws. -/
noncomputable instance instLawfulBoundOpsReal : LawfulBoundOps ℝ where
  toReal := id
  lt_iff _ _ := Iff.rfl
  addDown_le _ _ := le_rfl
  le_addUp _ _ := le_rfl
  subDown_le _ _ := le_rfl
  le_subUp _ _ := le_rfl
  mulDown_le _ _ := le_rfl
  le_mulUp _ _ := le_rfl

/-- Exact nonlinear interval transfers over the real numbers. -/
noncomputable instance instNonlinearBoundOpsReal : NonlinearBoundOps ℝ where
  divBounds aLo aHi bLo bHi :=
    if bLo > 0 || 0 > bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      some (min (min p1 p2) (min p3 p4), max (max p1 p2) (max p3 p4))
    else
      none
  expBounds lo hi := some (Real.exp lo, Real.exp hi)
  logBounds lo hi :=
    if lo > 0 then some (Real.log lo, Real.log hi) else none
  sqrtBounds lo hi :=
    if hi < 0 then none else some (Real.sqrt (max lo 0), Real.sqrt hi)
  sigmoidBounds lo hi :=
    some ((1 : ℝ) / (1 + Real.exp (-lo)), (1 : ℝ) / (1 + Real.exp (-hi)))
  tanhBounds lo hi := some (Real.tanh lo, Real.tanh hi)
  sinBounds := fun _ _ => some (-1, 1)
  cosBounds := fun _ _ => some (-1, 1)
  layerNormAbsBound n := some (Real.sqrt n)
  supportsIdealCoupledDerivatives := true

end NN.MLTheory.CROWN
