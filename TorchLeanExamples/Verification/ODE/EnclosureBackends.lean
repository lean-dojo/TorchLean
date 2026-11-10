/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.Enclosure
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Topology.Separation.Connected

/-!
# Decoded Backend Trajectories for ODE Enclosures

The corridor theorems of `TorchLeanExamples.Verification.ODE.Enclosure` are stated for real
functions. A backend-valued trajectory reaches them only through a real interpretation whose
continuity, exact derivatives, and corridor inequalities the caller proves. Correct rounding or
finite-error bounds alone do not supply these hypotheses: an ordinary sampled floating-point
trajectory is not automatically a continuous ODE solution.

A directly decoded trajectory of a configured binary format has finite range, so continuity on an
interval forces it to be constant there (`decoded_eq_of_continuousOn`). To reason about sampled
trajectories, use a real-valued interpolant instead. The corridor theorem still needs the
interpolant's clamped-ODE derivative hypothesis, and sample values and interval boxes alone do not
establish that it solves the ODE.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange

namespace TorchLean.Examples.ODE

open Set

/-! ## Continuity of a directly decoded trajectory -/

private theorem finite_range_toReal
    (format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat) :
    (Set.range
      (FloatLib.Floats.Formats.BinaryInterchange.Model.toReal (fmt := format))).Finite := by
  let : Finite (FloatLib.Floats.Formats.BinaryInterchange.Model format) :=
    Finite.of_injective (fun x => x.bits.toFin) (by
      intro left right same
      cases left
      cases right
      have same := congrArg BitVec.ofFin same
      simp only [BitVec.ofFin_toFin] at same
      cases same
      rfl)
  exact Set.finite_range _

/-- A directly decoded trajectory of a configured binary format that is continuous on an interval
is constant there. Use a real-valued interpolant when stating an ODE theorem about discrete samples.
-/
theorem decoded_eq_of_continuousOn {format : FloatFormat} {plan : Configured.StoragePlan format}
    {code : Type} [ExecFloat.ModelCodec plan (Model format) code]
    (u : ℝ → ExecFloat (Configured.Family format code plan)) {a b s t : ℝ}
    (continuous : ContinuousOn (fun t => (ExecFloat.Binary.toModel (u t)).toReal) (Icc a b))
    (hs : s ∈ Icc a b) (ht : t ∈ Icc a b) :
    (ExecFloat.Binary.toModel (u s)).toReal = (ExecFloat.Binary.toModel (u t)).toReal := by
  let f := fun t => (ExecFloat.Binary.toModel (u t)).toReal
  have finite : (f '' Icc a b).Finite := (finite_range_toReal _).subset (by
    rintro _ ⟨x, _, rfl⟩
    exact ⟨ExecFloat.Binary.toModel (u x), rfl⟩)
  have connected := isPreconnected_Icc.image f continuous
  have singleton : (f '' Icc a b).Subsingleton := by
    by_contra nontrivial
    exact (connected.infinite_of_nontrivial (Set.not_subsingleton_iff.mp nontrivial)).not_finite
      finite
  exact singleton ⟨s, hs, rfl⟩ ⟨t, ht, rfl⟩

end TorchLean.Examples.ODE
