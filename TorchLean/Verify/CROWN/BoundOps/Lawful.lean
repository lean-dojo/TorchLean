/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import TorchLean.Verify.CROWN.BoundOps
public import TorchLean.Tensor.Elementwise.Real

/-!
# Real-number laws for directed endpoint arithmetic

`TorchLean.CROWN.BoundOps` says what a scalar backend computes. This module says what those
computations mean: each endpoint is read as a real number, and each directed operation is compared
with exact real arithmetic. CROWN theorems ask for that guarantee through `LawfulBoundOps`,
`LawfulNonlinearBoundOps` and the enclosure predicates `UnaryEnclosure` and `BinaryEnclosure`.
Over `ℝ` the guarantee holds trivially, since nothing rounds (`instLawfulBoundOpsReal`).

The real-analysis dependencies belong to the soundness proofs. Certificate replay, the graph
checker and the trainer run on the executable `BoundOps` interface and need no `LawfulBoundOps`
instance.
-/

@[expose] public section

namespace TorchLean.CROWN

variable {α : Type} [Storage α] [Context α]

/--
Real-number laws for `BoundOps`.

`BoundOps` is usable without this class, so a backend can serve diagnostics before its arithmetic
is connected to a proof. Sound CROWN theorems require `LawfulBoundOps` as well. `toReal` says which
real number an endpoint denotes, and the laws compare each directed operation with exact
arithmetic on those reals, which demands more than bounding the backend's ordinary rounded result.
The laws also require the context's `max` and `min`, and its IEEE `Context.maximum` and
`Context.minimum`, to be exact on the reals. Negation and absolute value may round on a backend,
so the class assumes nothing about them.

The global instances are for `ℝ` and the rounded-real `FP32` model. There is none for Lean
`Float`, which is a trusted run-time boundary. Nor is there one for every bit pattern of
`ExecFloat.Binary 8 23`: NaNs, infinities and overflow need finite-path hypotheses, which the IEEE
semantics layer states instead of hiding them in an invalid ordered-ring instance.
-/
class LawfulBoundOps (α : Type) [Storage α] [Context α] [BoundOps α] where
  /-- Mathematical value represented by an endpoint. -/
  toReal : α → ℝ
  /-- Executable endpoint comparisons agree with the mathematical order. -/
  lt_iff (a b : α) : a < b ↔ toReal a < toReal b
  /-- The endpoint `0` means the real number `0`. -/
  toReal_zero : toReal 0 = 0
  /-- The endpoint `1` means the real number `1`. -/
  toReal_one : toReal 1 = 1
  /-- The backend maximum is the real maximum. -/
  toReal_max (a b : α) : toReal (max a b) = max (toReal a) (toReal b)
  /-- The backend minimum is the real minimum. -/
  toReal_min (a b : α) : toReal (min a b) = min (toReal a) (toReal b)
  /-- The IEEE maximum `Context.maximum` is the real maximum. -/
  toReal_maximum (a b : α) : toReal (Context.maximum a b) = max (toReal a) (toReal b)
  /-- The IEEE minimum `Context.minimum` is the real minimum. -/
  toReal_minimum (a b : α) : toReal (Context.minimum a b) = min (toReal a) (toReal b)
  /-- Endpoints that test equal denote the same real number. -/
  toReal_eq_of_beq {a b : α} : (a == b) = true → toReal a = toReal b
  /-- `addDown` is at most the real sum. -/
  addDown_le (a b : α) : toReal (BoundOps.addDown a b) ≤ toReal a + toReal b
  /-- `addUp` is at least the real sum. -/
  le_addUp (a b : α) : toReal a + toReal b ≤ toReal (BoundOps.addUp a b)
  /-- `subDown` is at most the real difference. -/
  subDown_le (a b : α) : toReal (BoundOps.subDown a b) ≤ toReal a - toReal b
  /-- `subUp` is at least the real difference. -/
  le_subUp (a b : α) : toReal a - toReal b ≤ toReal (BoundOps.subUp a b)
  /-- `mulDown` is at most the real product. -/
  mulDown_le (a b : α) : toReal (BoundOps.mulDown a b) ≤ toReal a * toReal b
  /-- `mulUp` is at least the real product. -/
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
Real-number laws for `NonlinearBoundOps`. Each transfer encloses its real function in the sense of
`UnaryEnclosure` or `BinaryEnclosure`. A returned bound `layerNormAbsBound n` is at least
`Real.sqrt n`.

The class is kept apart from the executable transfer table, so a backend can implement a transfer
for testing before proving it. Sound verification entry points can require this class, and then
an unchecked implementation cannot be promoted into a theorem unnoticed.
-/
class LawfulNonlinearBoundOps (α : Type) [Storage α] [Context α] [BoundOps α]
    [LawfulBoundOps α] [NonlinearBoundOps α] : Prop where
  /-- `divBounds` encloses real division. -/
  divBounds_enclosure :
    BinaryEnclosure (α := α) (· / ·) (NonlinearBoundOps.divBounds (α := α))
  /-- `expBounds` encloses `Real.exp`. -/
  expBounds_enclosure :
    UnaryEnclosure (α := α) Real.exp (NonlinearBoundOps.expBounds (α := α))
  /-- `logBounds` encloses `Real.log`. -/
  logBounds_enclosure :
    UnaryEnclosure (α := α) Real.log (NonlinearBoundOps.logBounds (α := α))
  /-- `sqrtBounds` encloses `Real.sqrt`. -/
  sqrtBounds_enclosure :
    UnaryEnclosure (α := α) Real.sqrt (NonlinearBoundOps.sqrtBounds (α := α))
  /-- `sigmoidBounds` encloses `Real.sigmoid`. -/
  sigmoidBounds_enclosure :
    UnaryEnclosure (α := α) Real.sigmoid (NonlinearBoundOps.sigmoidBounds (α := α))
  /-- `tanhBounds` encloses `Real.tanh`. -/
  tanhBounds_enclosure :
    UnaryEnclosure (α := α) Real.tanh (NonlinearBoundOps.tanhBounds (α := α))
  /-- `sinBounds` encloses `Real.sin`. -/
  sinBounds_enclosure :
    UnaryEnclosure (α := α) Real.sin (NonlinearBoundOps.sinBounds (α := α))
  /-- `cosBounds` encloses `Real.cos`. -/
  cosBounds_enclosure :
    UnaryEnclosure (α := α) Real.cos (NonlinearBoundOps.cosBounds (α := α))
  /-- A returned bound `layerNormAbsBound n` is at least `Real.sqrt n`. -/
  layerNormAbsBound_sound {n : Nat} {radius : α} :
    NonlinearBoundOps.layerNormAbsBound (α := α) n = some radius →
      Real.sqrt n ≤ LawfulBoundOps.toReal radius

/-- Endpoint arithmetic for `ℝ`, `BoundOps.exact`. Exact real arithmetic needs no rounding, so its
lower and upper operations coincide. -/
noncomputable instance instBoundOpsReal : BoundOps ℝ := .exact ℝ

/-- Exact real endpoint arithmetic satisfies the directed-operation enclosure laws. -/
noncomputable instance instLawfulBoundOpsReal : LawfulBoundOps ℝ where
  toReal := id
  lt_iff _ _ := Iff.rfl
  toReal_zero := rfl
  toReal_one := rfl
  toReal_max _ _ := rfl
  toReal_min _ _ := rfl
  toReal_maximum := Context.maximum_real
  toReal_minimum := Context.minimum_real
  toReal_eq_of_beq h := beq_iff_eq.mp h
  addDown_le _ _ := le_rfl
  le_addUp _ _ := le_rfl
  subDown_le _ _ := le_rfl
  le_subUp _ _ := le_rfl
  mulDown_le _ _ := le_rfl
  le_mulUp _ _ := le_rfl

/-- Nonlinear interval transfers over `ℝ`, computed without rounding. Monotone functions map the
endpoints, and sine and cosine get the range `[-1, 1]`. -/
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
  sigmoidBounds lo hi := some (Real.sigmoid lo, Real.sigmoid hi)
  tanhBounds lo hi := some (Real.tanh lo, Real.tanh hi)
  sinBounds := fun _ _ => some (-1, 1)
  cosBounds := fun _ _ => some (-1, 1)
  layerNormAbsBound n := some (Real.sqrt n)

end TorchLean.CROWN
