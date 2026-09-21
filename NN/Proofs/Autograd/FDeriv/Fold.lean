/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.ContDiff.Operations

/-!
# Smooth finite folds

The traversal order is fixed, while the initial accumulator and each visited value may depend
on the parameters. The update rule can operate on different normed spaces for its accumulator
and entries. No associativity, commutativity, or permutation argument is needed.
-/

public section

open scoped ContDiff

namespace List

variable {𝕜 E F G ι : Type*} [NontriviallyNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E]
  [NormedAddCommGroup F] [NormedSpace 𝕜 F]
  [NormedAddCommGroup G] [NormedSpace 𝕜 G] {n : ℕ∞ω}

/-- A smooth update rule preserves local smoothness along a fixed traversal.
Only the visited entries and initial accumulator need to be smooth near the input. -/
@[fun_prop] theorem contDiffAt_foldl (indices : List ι) {step : F → G → F}
    (hstep : ContDiff 𝕜 n (fun p : F × G => step p.1 p.2))
    {initial : E → F} {term : ι → E → G} {x : E} (hinit : ContDiffAt 𝕜 n initial x)
    (hterm : ∀ i ∈ indices, ContDiffAt 𝕜 n (term i) x) :
    ContDiffAt 𝕜 n (fun y => indices.foldl (fun acc i => step acc (term i y)) (initial y)) x := by
  induction indices generalizing initial with
  | nil => exact hinit
  | cons i indices ih =>
      apply ih
      · exact hstep.contDiffAt.comp x (hinit.prodMk (hterm i (by simp)))
      · intro j hj
        exact hterm j (by simp [hj])

/-- A fixed traversal of a smooth update rule preserves smoothness of the accumulator.
Only entries visited by the list need to be smooth; repetition and empty traversals are allowed. -/
@[fun_prop] theorem contDiff_foldl (indices : List ι) {step : F → G → F}
    (hstep : ContDiff 𝕜 n (fun p : F × G => step p.1 p.2))
    {initial : E → F} {term : ι → E → G} (hinit : ContDiff 𝕜 n initial)
    (hterm : ∀ i ∈ indices, ContDiff 𝕜 n (term i)) :
    ContDiff 𝕜 n (fun x => indices.foldl (fun acc i => step acc (term i x)) (initial x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_foldl indices hstep hinit.contDiffAt
    (fun i hi => (hterm i hi).contDiffAt)

end List
