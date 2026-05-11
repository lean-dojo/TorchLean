/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorBridge
public import NN.Spec.Core.TensorOps

/-!
# `NN.MLTheory.Robustness.Runtime`

Executable Float-specialized utilities for the robustness specifications in
`NN.MLTheory.Robustness.Spec`.
-/

@[expose] public section

open Spec

namespace NN.MLTheory.Robustness.Runtime

/-!
# Robustness runtime utilities (Float)

This file specializes the polymorphic spec in `NN.MLTheory.Robustness.Spec` to `Float` and adds a
few executable helpers used by experiments, examples, and command-line diagnostics.

It includes:

- deterministic **specializations** (norms/distances/balls), and
- small **sampling-based helpers** that compute *empirical* quantities (e.g. “max observed ratio”
  over a chosen sample set).

The sampling helpers return concrete values computed from finite samples. They do not produce
certificate checks or `Prop`-level guarantees.
-/

/--
Runtime `L∞` norm, defined by specializing the polymorphic spec to `Float`.
-/
def tensorLinfNormFloat {s : Shape} (t : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := Float) (s := s) t

/--
Runtime `L2` norm, defined by specializing the polymorphic spec to `Float`.
-/
def tensorL2NormFloat {s : Shape} (t : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorL2Norm (α := Float) (s := s) t

/-- Internal helper: `L2` norm specialization reused by runtime wrappers in this module. -/
def l2Norm : ∀ {s : Shape}, Tensor Float s → Float :=
  fun {s} t => tensorL2NormFloat (s := s) t

/-- Internal helper: `L∞` norm specialization reused by runtime wrappers in this module. -/
def linfNorm : ∀ {s : Shape}, Tensor Float s → Float :=
  fun {s} t => tensorLinfNormFloat (s := s) t

/-- `L2` distance (specialization of `Robustness.Spec.tensor_distance`). -/
def tensorL2DistanceFloat {s : Shape} (t1 t2 : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorDistance (α := Float) (norm := l2Norm) (s := s) t1 t2

/-- `L∞` distance (specialization of `Robustness.Spec.tensor_distance`). -/
def tensorLinfDistanceFloat {s : Shape} (t1 t2 : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorDistance (α := Float) (norm := linfNorm) (s := s) t1 t2

/--
Decide whether `t` lies in the closed `L2`-ball of radius `ε` around `center`.
-/
def inL2BallFloat {s : Shape} (center : Tensor Float s) (ε : Float) (t : Tensor Float s) : Bool
  :=
  tensorL2DistanceFloat center t ≤ ε

/--
Decide whether `t` lies in the closed `L∞`-ball of radius `ε` around `center`.
-/
def inLinfBallFloat {s : Shape} (center : Tensor Float s) (ε : Float) (t : Tensor Float s) : Bool
  :=
  tensorLinfDistanceFloat center t ≤ ε

/-! ## Empirical (sampling-based) helpers -/

namespace Empirical

/--
Maximum observed `L2` Lipschitz ratio over a *given* finite list of input pairs.

This computes:

`max_{(x,y) in pairs} ‖f x - f y‖₂ / ‖x - y‖₂`

with the convention that a pair with `‖x-y‖₂ = 0` contributes `0`.

Factually: this is a maximum over the provided pairs only; it is not a certified global bound.

Edge cases / conventions:

- If `pairs = []`, the result is `0` (the fold's initial value).
- If a particular pair satisfies `x = y` (so the input distance is `0`), that pair contributes `0`.
-/
def maxL2LipschitzRatio {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (pairs : List (Tensor Float s₁ × Tensor Float s₁)) : Float :=
  pairs.foldl (fun maxRatio (p : Tensor Float s₁ × Tensor Float s₁) =>
    let (x, y) := p
    let inputDist := tensorL2DistanceFloat x y
    let outputDist := tensorL2DistanceFloat (f x) (f y)
    let ratio := if inputDist > 0.0 then outputDist / inputDist else 0.0
    max maxRatio ratio) 0.0

end Empirical

namespace Sampling

open TensorBridge

/-!
## Sampling design (why these helpers look “complicated”)

The spec tensor representation `Spec.Tensor α s` is **shape-indexed** and is represented
functionally (`Fin n → ...`). That is great for proofs, but it is not the easiest shape to work
with when you want to build *concrete perturbations* of a fixed length at runtime.

For sampling, we therefore go through a standard interop path:

1. convert the *type-level* shape `s : Shape` into a *runtime* shape list `shapeList s : List Nat`,
2. compute the number of scalar entries `numel s` as the product of those dimensions, then
3. construct a flat list `xs : List Float` of length `numel s`,
4. `unflatten` it back into a tensor of the appropriate shape.

Correctness (what is and is not guaranteed):

- Every point returned by `sampleL2Ball` **provably satisfies** the predicate
  `in_l2_ball_float center ε` because we *filter* candidates using that very predicate.
- The sampler is **deterministic** (no `IO` randomness). You control variability via the `seed`
  input (here derived from the loop index).
- The sampler is **not** intended to approximate a uniform distribution on the ball, and it does
  not attempt to be “complete” in any verification sense: it is purely an empirical exploration
  tool to produce inputs for downstream checks/counterexamples.
-/

/-- Runtime view of a type-level `Shape` (same convention as `TensorBridge.shapeToList`). -/
def shapeList (s : Shape) : List Nat :=
  TensorBridge.shapeToList s

/--
Number of scalar elements (“numel”) in a tensor of shape `s`.

This is the runtime analogue of `Spec.Shape.size`. We compute it via `TensorArray.shapeProd` on the
list view of the shape.
-/
def numel (s : Shape) : Nat :=
  TensorArray.shapeProd (shapeList s)

/--
Cast a tensor with type-level shape `s` into the definitional-equal “list-shaped” view used by
`TensorBridge.flatten/unflatten`.

This cast is purely a type-level transport; it does not change the tensor values.
-/
def castToListShape {s : Shape} :
    Tensor Float s → Tensor Float (TensorBridge.listToShape (shapeList s)) :=
  fun t =>
    cast
      (congrArg (fun sh => Tensor Float sh) (by
        simpa [shapeList] using (TensorBridge.shapeToList_listToShape_involutive s).symm)) t

/--
Inverse cast: transport a “list-shaped” tensor back to the original type-level shape `s`.
-/
def castFromListShape {s : Shape} :
    Tensor Float (TensorBridge.listToShape (shapeList s)) → Tensor Float s :=
  fun t =>
    cast
      (congrArg (fun sh => Tensor Float sh) (by
        simpa [shapeList] using (TensorBridge.shapeToList_listToShape_involutive s))) t

/--
Deterministically generate a length-`n` direction vector from a `seed`.

This is **not** cryptographic and not intended to model a probabilistic distribution; it is a
deterministic way to generate reproducible, varied directions without introducing `IO`.
-/
def perturbDirection (n : Nat) (seed : Nat) : List Float :=
  -- Deterministic “pseudo-random-ish” direction: entry `i` depends on `seed` and `i`.
  -- We avoid `IO` randomness here; callers can treat `seed` as a reproducibility knob.
  (List.finRange n).map (fun i =>
    let a : Float := Float.ofNat (seed + 1) * Float.ofNat (i.1 + 1)
    Float.sin a + 0.5 * Float.cos (a + 1.0))

/-- Euclidean norm of a flat list (helper for normalization). -/
def l2NormList (xs : List Float) : Float :=
  Float.sqrt (xs.foldl (fun acc x => acc + x * x) 0.0)

/--
Normalize a flat list to unit `L2` norm (when possible).

If the list has zero norm, we return it unchanged.
-/
def normalizeList (xs : List Float) : List Float :=
  let n := l2NormList xs
  if n > 0.0 then xs.map (fun x => x / n) else xs

/--
Unflatten a flat list of length `numel s` into a `Spec.Tensor Float s`.

The length proof is part of the interface to avoid “silent truncation/padding”.
-/
def unflattenToTensor {s : Shape} (xs : List Float)
    (h : xs.length = TensorArray.shapeProd (shapeList s)) : Tensor Float s :=
  let tList : Tensor Float (TensorBridge.listToShape (shapeList s)) :=
    TensorBridge.unflatten (shapeList s) xs (by simpa [shapeList] using h)
  castFromListShape (s := s) tList

/--
Build a perturbation tensor of (approximately) the given `radius`, deterministically from `seed`.

Construction:

1. Make a flat “direction” list `base` of length `numel s`.
2. Normalize it to unit norm (when nonzero).
3. Scale by `radius`.
4. Unflatten back into the tensor shape.

This ensures the perturbation has the right shape by construction.
-/
def perturbationTensor {s : Shape} (radius : Float) (seed : Nat) : Tensor Float s :=
  let n := numel s
  let base : List Float := perturbDirection n seed
  let dir : List Float := normalizeList base
  -- Scale direction and unflatten back into the tensor shape.
  let xs : List Float := dir.map (fun x => radius * x)
  have hlen : xs.length = TensorArray.shapeProd (shapeList s) := by
    have hbase : base.length = n := by
      dsimp [base, perturbDirection]
      simp
    have hdir : dir.length = n := by
      dsimp [dir]
      by_cases h : l2NormList base > 0.0
      · simp [normalizeList, h, hbase]
      · simp [normalizeList, h, hbase]
    calc
      xs.length = dir.length := by simp [xs]
      _ = n := hdir
      _ = TensorArray.shapeProd (shapeList s) := by simp [numel, n]
  unflattenToTensor (s := s) xs hlen

/--
Generate candidate samples in the closed `L2` ball around `center` (deterministic sampler).

We generate `numSamples` candidates, each constructed as:

`center + δ_k` where `δ_k` is a direction derived from `k` and then scaled to a radius in `[0, ε]`.

We then **filter** candidates using `in_l2_ball_float center ε` to ensure that every returned
element satisfies the predicate under the same runtime distance function.

This “generate + filter” style is deliberate: it makes the only factual guarantee we claim
extremely clear (“every returned element lies in the ball”), independent of the details of the
direction generator.
-/
def sampleL2Ball {s : Shape} (center : Tensor Float s) (ε : Float) (numSamples : Nat) :
    List (Tensor Float s) :=
  (List.range numSamples).foldl (fun acc k =>
    -- Choose a deterministic radius in `(0, ε)`; we avoid `0` so that, for typical shapes, we
    -- don't only resample the center.
    --
    -- The exact schedule is not semantically important; what matters is that:
    -- - candidates are easy to reproduce, and
    -- - every returned element is checked with `in_l2_ball_float`.
    let radius := ε * (Float.ofNat (k + 1) / Float.ofNat (numSamples + 1))
    let δ := perturbationTensor (s := s) radius k
    let x := Spec.Tensor.addSpec center δ
    if inL2BallFloat center ε x then x :: acc else acc) []
  |>.reverse

/--
Turn a nonempty list `[x₀,x₁,…,x_{m-1}]` into adjacent pairs
`[(x₀,x₁),(x₁,x₂),…,(x_{m-1},x₀)]`.

This is a small combinator that is useful when you want to turn a sample list into a set of
“nearby pairs” for empirical ratio computations.
-/
def adjacentPairs {α : Type} : List α → List (α × α)
  | [] => []
  | [_x] => []
  | x0 :: xs =>
      let ys := xs ++ [x0]
      List.zip (x0 :: xs) ys

end Sampling

/--
Empirical max “gain from the center” on a sampled `L2`-ball neighborhood.

This computes:

`max_{x in samples} ‖f x - f x₀‖₂ / ‖x - x₀‖₂`

where `samples` are generated by `Sampling.sampleL2Ball`. This is a maximum over those samples
only (not a certified global Lipschitz bound).
-/
def empiricalMaxL2GainFromSamples {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (x₀ : Tensor Float s₁) (ε : Float) (numSamples : Nat) : Float :=
  let samples := Sampling.sampleL2Ball (center := x₀) ε numSamples
  -- We compare each sample to the center point `x₀`, because that is the common pattern in
  -- robustness debugging (“how much can the output move within an ε-neighborhood of x₀?”).
  let pairs := samples.map (fun x => (x₀, x))
  Empirical.maxL2LipschitzRatio (s₁ := s₁) (s₂ := s₂) f pairs

end NN.MLTheory.Robustness.Runtime
