/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Movement
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules

/-!
# Broadcast reductions for `roundoff`

`Tensor.reduceBroadcast` sums a tensor over the axes a broadcast at a placement created, one nested
sum per axis, by recursion on the shapes and the placement. For shapes that are not literal the
recursion has no fixed depth, so `roundoff` cannot unfold it. The bound here holds for every pair
of shapes and every placement. It runs the same reduction on `TrackedSum` terms, which carry the
reference of each term and a bound on its error. Adding two of them adds the error bounds and half
an ulp at the computed sum. The budget is the error of the tracked reduction, which depends on the
computed partial sums through their decoded values, and the bound holds when the computed reduction
is finite. The bound is also registered with `roundoff_read`, so the analysis also unfolds a
reduction between literal shapes.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

variable (format plan code) in
/-- A configured partial sum with the exact sum of the references of its terms and a bound on its
error against that sum. -/
structure TrackedSum where
  /-- The computed partial sum. -/
  value : Value
  /-- The exact sum of the references of its terms. -/
  reference : ℝ
  /-- A bound on the distance from `value` to `reference`. -/
  error : ℝ

namespace TrackedSum

/-- The empty sum: its computed value, reference and error bound are all `0`. -/
instance : Zero (TrackedSum format plan code) := ⟨⟨0, 0, 0⟩⟩

/-- The sum of two tracked sums adds their error bounds and half an ulp at the computed sum. -/
noncomputable instance : Add (TrackedSum format plan code) where
  add a b := ⟨a.value + b.value, a.reference + b.reference,
    a.error + b.error + Model.halfUlpAt format
      (ExecFloat.Binary.toReal a.value + ExecFloat.Binary.toReal b.value)⟩

/-- The entries of `tensor` as tracked sums, each with its reference and its error bound. -/
noncomputable def terms [Storage Value] {shape : Shape} (tensor : Tensor Value shape)
    (reference error : shape.Coord → ℝ) : Tensor (TrackedSum format plan code) shape :=
  Tensor.Rep.ofFn fun c => ⟨tensor c, reference c, error c⟩

/-- A tracked sum is sound when, if its value is finite, it is within its error bound of its
reference. -/
def IsSound (sum : TrackedSum format plan code) : Prop :=
  ExecFloat.Binary.isFinite sum.value = true →
    |ExecFloat.Binary.toReal sum.value - sum.reference| ≤ sum.error

/-- The zero tracked sum is sound. -/
theorem isSound_zero : (0 : TrackedSum format plan code).IsSound := fun _ => by
  change |ExecFloat.Binary.toReal (0 : Value) - 0| ≤ 0
  rw [ExecFloat.Binary.toReal_zero, sub_zero, abs_zero]

/-- The sum of two sound tracked sums is sound. -/
theorem IsSound.add (hformat : format.isIEEE = true) {a b : TrackedSum format plan code}
    (ha : a.IsSound) (hb : b.IsSound) : (a + b).IsSound := by
  intro finite
  change ExecFloat.Binary.isFinite (a.value + b.value) = true at finite
  have operands := finite
  unfold ExecFloat.Binary.isFinite at operands
  rw [ExecFloat.Binary.toModel_hAdd a.value b.value] at operands
  obtain ⟨fa, fb⟩ := Model.isFinite_of_isFinite_add hformat operands
  have rounding := ExecFloat.Binary.abs_toReal_add_sub_le finite
  have ea := ha fa
  have eb := hb fb
  change |ExecFloat.Binary.toReal (a.value + b.value) - (a.reference + b.reference)| ≤
    a.error + b.error + Model.halfUlpAt format
      (ExecFloat.Binary.toReal a.value + ExecFloat.Binary.toReal b.value)
  rw [abs_le] at rounding ea eb ⊢
  constructor <;> linarith [rounding.1, rounding.2, ea.1, ea.2, eb.1, eb.2]

end TrackedSum

/-- A finite broadcast reduction at a placement, of terms with known references and error bounds,
differs from the same reduction of the references by at most the error of the tracked reduction. -/
theorem abs_reduceBroadcast_sub_le [Storage Value] (hformat : format.isIEEE = true)
    {s₁ s₂ : Shape} (dims : Fin s₁.rank → Fin s₂.rank)
    (h : Shape.BroadcastDims s₁ s₂ dims)
    (tensor : Tensor Value s₂) (reference error : s₂.Coord → ℝ)
    (hterms : ∀ c, |ExecFloat.Binary.toReal (tensor c) - reference c| ≤ error c) (c : s₁.Coord)
    (hfinite : ExecFloat.Binary.isFinite (Tensor.reduceBroadcast dims h tensor c) = true) :
    |ExecFloat.Binary.toReal (Tensor.reduceBroadcast dims h tensor c) -
        Tensor.reduceBroadcast dims h (Tensor.Rep.ofFn reference) c| ≤
      (Tensor.reduceBroadcast dims h (TrackedSum.terms tensor reference error) c).error := by
  have project {β : Type} [Storage β] [Add β] [Zero β] (f : TrackedSum format plan code → β)
      (zero : f 0 = 0) (add : ∀ a b, f (a + b) = f a + f b) :
      f (Tensor.reduceBroadcast dims h (TrackedSum.terms tensor reference error) c) =
        Tensor.reduceBroadcast dims h (Tensor.map f (TrackedSum.terms tensor reference error)) c :=
    by rw [← Tensor.map_reduceBroadcast f zero add, Tensor.map_apply]
  have values : Tensor.map TrackedSum.value (TrackedSum.terms tensor reference error) = tensor :=
    Tensor.Rep.ext fun c => by
      rw [Tensor.map_apply, TrackedSum.terms, Tensor.Rep.get_ofFn]
  have references : Tensor.map TrackedSum.reference (TrackedSum.terms tensor reference error) =
      Tensor.Rep.ofFn reference :=
    Tensor.Rep.ext fun c => by
      rw [Tensor.map_apply, TrackedSum.terms, Tensor.Rep.get_ofFn,
        Tensor.Rep.get_ofFn]
  have value : (Tensor.reduceBroadcast dims h (TrackedSum.terms tensor reference error) c).value =
      Tensor.reduceBroadcast dims h tensor c := by
    rw [project TrackedSum.value rfl fun _ _ => rfl, values]
  have sum :
      (Tensor.reduceBroadcast dims h (TrackedSum.terms tensor reference error) c).reference =
        Tensor.reduceBroadcast dims h (Tensor.Rep.ofFn reference) c := by
    rw [project TrackedSum.reference rfl fun _ _ => rfl, references]
  have sound := Tensor.reduceBroadcast_induction TrackedSum.IsSound
    (fun _ _ => TrackedSum.IsSound.add hformat) TrackedSum.isSound_zero dims h
    (TrackedSum.terms tensor reference error)
    (fun c _ => by
      rw [TrackedSum.terms, Tensor.Rep.get_ofFn]
      exact hterms c) c (by rw [value]; exact hfinite)
  rw [← value, ← sum]
  exact sound

attribute [roundoff_rule, roundoff_read] abs_reduceBroadcast_sub_le

end TorchLean.Roundoff
