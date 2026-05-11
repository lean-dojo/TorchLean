/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.SelfSupervised.JEPA
public import NN.MLTheory.SelfSupervised.MAE
public import NN.MLTheory.SelfSupervised.VICReg
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Real.Basic

/-!
# Predictive-view semantics for self-supervised learning

This file is the objective-algebra layer for finite self-supervised learning.

The guiding split is:

* a **prediction/alignment** term, over compatible finite views; and
* a **geometry/non-collapse** term, such as variance, covariance, redundancy, negative-sample, or
  teacher-dynamics regularization.

The statements here are deliberately finite and method-neutral. They do not claim that SSL
optimization learns good representations. They prove a smaller semantic fact that is useful for
TorchLean and for paper writing: MAE, JEPA, VICReg-style guards, Barlow-style guards, and
autoregressive prediction can share one objective shape.

In this finite layer, a view-prediction contract has:

* `targetIdxs`, the selected masked/target indices;
* a context value;
* a target value at every finite index;
* a target encoder, which chooses the target space;
* a predictor from context and index; and
* a nonnegative geometry guard.

MAE is recovered by choosing the identity target encoder into patch/pixel space. JEPA is recovered
by choosing the target representation itself as the target space. VICReg/Barlow-style objectives are
represented as geometry guards that can be added orthogonally to either prediction objective.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

open scoped BigOperators

/--
A finite predictive-view contract.

`Target` is the raw target-view type, `TargetRep` is the space after the target encoder, and
`Prediction` is the predictor output space. Keeping these three types separate is the whole point:
MAE sets `TargetRep = Target` with an identity encoder; JEPA uses a latent target representation;
contrastive and redundancy-reduction methods can reuse the same contract with different geometry
guards.
-/
structure PredictiveViewContract
    (n : Nat) (Context Target TargetRep Prediction : Type) where
  /-- Selected target/masked indices. -/
  targetIdxs : List (Fin n)
  /-- Context view representation. -/
  context : Context
  /-- Target view before applying the target encoder. -/
  target : Fin n → Target
  /-- Target-space map. MAE uses identity into pixels/patches; JEPA uses a latent target branch. -/
  targetEncoder : Fin n → Target → TargetRep
  /-- Prediction made from the context view for a selected target index. -/
  predict : Context → Fin n → Prediction
  /-- Per-index predictive distance/alignment loss. -/
  distance : TargetRep → Prediction → Nat
  /-- Geometry, spread, redundancy, or anti-collapse guard. -/
  geometryGuard : Nat := 0

/-- Prediction/alignment term of a finite predictive-view SSL objective. -/
def predictiveLoss {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction) : Nat :=
  maskedLoss contract.targetIdxs (fun i =>
    contract.distance (contract.targetEncoder i (contract.target i))
      (contract.predict contract.context i))

/-- Full finite SSL objective: predictive loss plus geometry/non-collapse guard. -/
def predictiveViewObjective {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction) : Nat :=
  predictiveLoss contract + contract.geometryGuard

/-- The generic SSL objective decomposes into prediction/alignment plus geometry guard. -/
theorem predictiveViewObjective_decomposes
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction) :
    predictiveViewObjective contract = predictiveLoss contract + contract.geometryGuard := by
  rfl

/-- If the geometry guard is zero, the full objective is exactly the predictive loss. -/
@[simp] theorem predictiveViewObjective_zero_geometry
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (h : contract.geometryGuard = 0) :
    predictiveViewObjective contract = predictiveLoss contract := by
  simp [predictiveViewObjective, h]

/--
Replace the geometry guard of a predictive contract.

This is how VICReg, Barlow-style redundancy reduction, InfoNCE-style uniformity, or a future
Predictive-Hull coverage guard can be bolted onto the same prediction contract without changing
the view-selection semantics.
-/
def withGeometryGuard {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (guard : Nat) :
    PredictiveViewContract n Context Target TargetRep Prediction :=
  { contract with geometryGuard := guard }

@[simp] theorem predictiveLoss_withGeometryGuard
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (guard : Nat) :
    predictiveLoss (withGeometryGuard contract guard) = predictiveLoss contract := by
  rfl

@[simp] theorem predictiveViewObjective_withGeometryGuard
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (guard : Nat) :
    predictiveViewObjective (withGeometryGuard contract guard) =
      predictiveLoss contract + guard := by
  rfl

/-! ## MAE as predictive-view SSL with identity target encoder -/

/--
MAE as a predictive-view contract.

The context is `Unit` because the finite MAE skeleton already abstracts away the encoder. The
target encoder is identity into patch/pixel space.
-/
def maeAsPredictiveViewContract {n : Nat} {Patch Pred : Type}
    (maskedIdxs : List (Fin n))
    (target : PatchBatch n Patch)
    (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    PredictiveViewContract n Unit Patch Patch Pred where
  targetIdxs := maskedIdxs
  context := ()
  target := target
  targetEncoder := fun _ patch => patch
  predict := fun _ i => pred i
  distance := patchLoss
  geometryGuard := 0

/-- MAE's masked reconstruction loss is the predictive term with identity target encoder. -/
theorem mae_is_predictive_view_loss {n : Nat} {Patch Pred : Type}
    (maskedIdxs : List (Fin n))
    (target : PatchBatch n Patch)
    (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    predictiveLoss (maeAsPredictiveViewContract maskedIdxs target pred patchLoss) =
      maeLoss maskedIdxs target pred patchLoss := by
  rfl

/-- MAE is the zero-geometry predictive-view objective with pixel/patch identity targets. -/
theorem mae_is_predictive_view_objective {n : Nat} {Patch Pred : Type}
    (maskedIdxs : List (Fin n))
    (target : PatchBatch n Patch)
    (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    predictiveViewObjective (maeAsPredictiveViewContract maskedIdxs target pred patchLoss) =
      maeLoss maskedIdxs target pred patchLoss := by
  simp [predictiveViewObjective, predictiveLoss, maeAsPredictiveViewContract, maeLoss]

/-! ## JEPA as predictive-view SSL with latent target representation -/

/--
JEPA as a predictive-view contract when the finite `target` is already a target representation.

This matches `jepaLoss`: target representations are values at the objective boundary, and the
predictor tries to match them at selected target indices.
-/
def jepaAsPredictiveViewContract {n : Nat} {Context Target Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) :
    PredictiveViewContract n Context Target Target Pred where
  targetIdxs := targetIdxs
  context := context
  target := target
  targetEncoder := fun _ targetRep => targetRep
  predict := predict
  distance := repLoss
  geometryGuard := 0

/-- JEPA's finite target-representation loss is the predictive-view loss. -/
theorem jepa_is_predictive_view_loss {n : Nat} {Context Target Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) :
    predictiveLoss (jepaAsPredictiveViewContract targetIdxs context target predict repLoss) =
      jepaLoss targetIdxs context target predict repLoss := by
  rfl

/-- JEPA is the zero-geometry predictive-view objective with latent target values. -/
theorem jepa_is_predictive_view_objective {n : Nat} {Context Target Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (predict : Context → Fin n → Pred)
    (repLoss : Target → Pred → Nat) :
    predictiveViewObjective
        (jepaAsPredictiveViewContract targetIdxs context target predict repLoss) =
      jepaLoss targetIdxs context target predict repLoss := by
  simp [predictiveViewObjective, predictiveLoss, jepaAsPredictiveViewContract, jepaLoss]

/--
More general JEPA/predictive-view contract with a separate target encoder.

This is the paper bridge: changing `targetEncoder` changes the target space while leaving the
finite view-prediction algebra alone. MAE is the special case where this encoder is identity into
pixels/patches; JEPA uses a latent/stopped target branch.
-/
def encodedTargetPredictiveViewContract {n : Nat} {Context Target TargetRep Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (targetEncoder : Fin n → Target → TargetRep)
    (predict : Context → Fin n → Pred)
    (repLoss : TargetRep → Pred → Nat)
    (geometryGuard : Nat := 0) :
    PredictiveViewContract n Context Target TargetRep Pred where
  targetIdxs := targetIdxs
  context := context
  target := target
  targetEncoder := targetEncoder
  predict := predict
  distance := repLoss
  geometryGuard := geometryGuard

theorem encodedTargetPredictiveViewContract_loss_eq_maskedLoss
    {n : Nat} {Context Target TargetRep Pred : Type}
    (targetIdxs : List (Fin n))
    (context : Context)
    (target : Fin n → Target)
    (targetEncoder : Fin n → Target → TargetRep)
    (predict : Context → Fin n → Pred)
    (repLoss : TargetRep → Pred → Nat)
    (geometryGuard : Nat := 0) :
    predictiveLoss
        (encodedTargetPredictiveViewContract targetIdxs context target targetEncoder predict repLoss
          geometryGuard) =
      maskedLoss targetIdxs (fun i => repLoss (targetEncoder i (target i)) (predict context i)) := by
  rfl

/-! ## Geometry guards as reusable SSL modules -/

/-- A VICReg-style geometry guard packaged for predictive-view objectives. -/
structure VICRegGuard where
  /-- Weight for invariance/alignment summary. -/
  lambda : Nat
  /-- Weight for variance/non-collapse summary. -/
  mu : Nat
  /-- Weight for covariance/redundancy summary. -/
  nu : Nat
  /-- Already-computed invariance summary. -/
  invariance : Nat
  /-- Already-computed variance-floor summary. -/
  variance : Nat
  /-- Already-computed covariance/redundancy summary. -/
  covariance : Nat

/-- Evaluate a finite VICReg-style guard through the existing VICReg objective. -/
def VICRegGuard.value (guard : VICRegGuard) : Nat :=
  vicregObjective guard.lambda guard.mu guard.nu guard.invariance guard.variance guard.covariance

/-- A Barlow-Twins-style redundancy guard packaged for predictive-view objectives. -/
structure BarlowGuard where
  /-- Weight for off-diagonal redundancy terms. -/
  lambda : Nat
  /-- Diagonal cross-correlation summaries, ideal value `1`. -/
  diag : List Nat
  /-- Off-diagonal cross-correlation summaries, ideal value `0`. -/
  offDiag : List Nat

/-- Evaluate a finite Barlow-style redundancy guard. -/
def BarlowGuard.value (guard : BarlowGuard) : Nat :=
  redundancyReductionObjective guard.lambda guard.diag guard.offDiag

/-- Adding a VICReg guard gives prediction plus the VICReg geometry value. -/
theorem predictiveViewObjective_with_vicreg_guard
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (guard : VICRegGuard) :
    predictiveViewObjective (withGeometryGuard contract guard.value) =
      predictiveLoss contract + guard.value := by
  rfl

/-- Adding a Barlow-style guard gives prediction plus the redundancy-reduction geometry value. -/
theorem predictiveViewObjective_with_barlow_guard
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (guard : BarlowGuard) :
    predictiveViewObjective (withGeometryGuard contract guard.value) =
      predictiveLoss contract + guard.value := by
  rfl

/--
A pure variance VICReg guard is positive when both the variance weight and variance summary are
positive. This is the finite anti-collapse card used by the generic predictive-view algebra.
-/
theorem vicreg_guard_variance_only_positive {mu variance : Nat}
    (hμ : 0 < mu) (hv : 0 < variance) :
    0 < (VICRegGuard.value
      { lambda := 0, mu := mu, nu := 0, invariance := 0, variance := variance,
        covariance := 0 }) := by
  simpa [VICRegGuard.value] using
    (vicregObjective_variance_positive (μ := mu) (variance := variance) hμ hv)

/-- The ideal Barlow-style guard has zero value. -/
@[simp] theorem barlow_guard_identity_value_zero (lambda d k : Nat) :
    (BarlowGuard.value
      ({ lambda := lambda, diag := List.replicate d 1, offDiag := List.replicate k 0 } :
        BarlowGuard)) = 0 := by
  simp [BarlowGuard.value]

/-- A collapsed diagonal entry pays a positive Barlow-style redundancy guard. -/
theorem barlow_guard_collapsed_diag_positive {lambda d k : Nat} :
    0 < (BarlowGuard.value
      ({ lambda := lambda, diag := 0 :: List.replicate d 1, offDiag := List.replicate k 0 } :
        BarlowGuard)) := by
  simpa [BarlowGuard.value] using
    (redundancyReductionObjective_collapsed_diag_positive (lambda := lambda) (d := d) (k := k))

/-! ## Finite view-graph reading -/

/-- A finite positive-view edge graph over `n` views. -/
structure SSLViewGraph (n : Nat) where
  /-- Positive/compatible view edges. -/
  positiveEdges : List (Fin n × Fin n)

/-- Edge energy for a finite positive-view graph. -/
def viewGraphEnergy {n : Nat}
    (graph : SSLViewGraph n) (edgeLoss : Fin n → Fin n → Nat) : Nat :=
  (graph.positiveEdges.map (fun edge => edgeLoss edge.1 edge.2)).sum

/-! ## Concrete finite Euclidean geometry

The generic objective above is intentionally method-neutral. The definitions below add pressure:
representations are finite real vectors, alignment is squared Euclidean energy on a finite positive
view graph, and non-collapse is expressed as a real variance-floor guard over coordinate-spread
summaries.

These theorems capture an important SSL fact in a checked finite setting:

* graph alignment alone is nonnegative but accepts fully collapsed representations with zero loss;
* a positive variance floor assigns positive penalty to collapsed coordinate-spread summaries.
-/

/-- A finite real embedding vector. -/
abbrev EuclideanRep (d : Nat) := Fin d → ℝ

/-- Squared Euclidean distance between two finite real embeddings. -/
noncomputable def sqDist {d : Nat} (z w : EuclideanRep d) : ℝ :=
  ∑ j : Fin d, (z j - w j) ^ 2

/-- Squared Euclidean distance is nonnegative. -/
theorem sqDist_nonneg {d : Nat} (z w : EuclideanRep d) :
    0 ≤ sqDist z w := by
  unfold sqDist
  exact Finset.sum_nonneg (fun j _ => sq_nonneg (z j - w j))

/-- A vector has zero squared distance from itself. -/
@[simp] theorem sqDist_self {d : Nat} (z : EuclideanRep d) :
    sqDist z z = 0 := by
  simp [sqDist]

/-- Real-valued alignment energy induced by positive edges in a finite view graph. -/
noncomputable def graphAlignmentEnergy {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d) : ℝ :=
  (graph.positiveEdges.map (fun edge => sqDist (rep edge.1) (rep edge.2))).sum

/-- Finite graph alignment energy is nonnegative. -/
theorem graphAlignmentEnergy_nonneg {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d) :
    0 ≤ graphAlignmentEnergy graph rep := by
  unfold graphAlignmentEnergy
  induction graph.positiveEdges with
  | nil => simp
  | cons edge rest ih =>
      simp [List.map, List.sum_cons, add_nonneg (sqDist_nonneg (rep edge.1) (rep edge.2)) ih]

/-- A collapsed representation maps every view to the same finite vector. -/
def CollapsedRep {n d : Nat} (rep : Fin n → EuclideanRep d) : Prop :=
  ∃ z : EuclideanRep d, ∀ i, rep i = z

/--
Alignment alone cannot prevent collapse: any constant representation has zero positive-edge
energy, no matter what the view graph is.
-/
theorem graphAlignmentEnergy_eq_zero_of_collapsed {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d)
    (hcollapsed : CollapsedRep rep) :
    graphAlignmentEnergy graph rep = 0 := by
  rcases hcollapsed with ⟨z, hz⟩
  unfold graphAlignmentEnergy
  induction graph.positiveEdges with
  | nil => simp
  | cons edge rest ih =>
      simp [List.map, List.sum_cons, hz edge.1, hz edge.2, sqDist_self, ih]

/--
Coordinate spread is a finite pairwise squared-difference summary for one embedding coordinate.

This avoids asymptotic probability or population assumptions while still expressing the core
"does this coordinate vary across the batch/views?" question used by finite anti-collapse guards.
-/
noncomputable def coordinateSpread {n d : Nat}
    (rep : Fin n → EuclideanRep d) (j : Fin d) : ℝ :=
  ∑ i : Fin n, ∑ k : Fin n, (rep i j - rep k j) ^ 2

/-- A collapsed representation has zero spread in every coordinate. -/
theorem coordinateSpread_eq_zero_of_collapsed {n d : Nat}
    (rep : Fin n → EuclideanRep d) (hcollapsed : CollapsedRep rep) (j : Fin d) :
    coordinateSpread rep j = 0 := by
  rcases hcollapsed with ⟨z, hz⟩
  simp [coordinateSpread, hz]

/-- Real-valued variance-floor penalty: `max(0, gamma - spread)`. -/
noncomputable def realVarianceFloorPenalty (gamma spread : ℝ) : ℝ :=
  max 0 (gamma - spread)

/-- A finite real variance-floor guard over coordinate-spread summaries. -/
noncomputable def realVarianceFloorGuard {d : Nat}
    (gamma : ℝ) (spread : Fin d → ℝ) : ℝ :=
  ∑ j : Fin d, realVarianceFloorPenalty gamma (spread j)

/-- Zero spread in every coordinate pays exactly `d * gamma` when `gamma` is nonnegative. -/
theorem realVarianceFloorGuard_zero_spread {d : Nat} {gamma : ℝ}
    (hgamma : 0 ≤ gamma) :
    realVarianceFloorGuard (d := d) gamma (fun _ => 0) = d * gamma := by
  simp [realVarianceFloorGuard, realVarianceFloorPenalty, hgamma]

/-- Collapsed coordinate-spread summaries pay a positive variance-floor guard in nonzero dimension. -/
theorem realVarianceFloorGuard_zero_spread_positive {d : Nat} {gamma : ℝ}
    (hd : 0 < d) (hgamma : 0 < gamma) :
    0 < realVarianceFloorGuard (d := d) gamma (fun _ => 0) := by
  rw [realVarianceFloorGuard_zero_spread (d := d) (gamma := gamma) (le_of_lt hgamma)]
  exact mul_pos (Nat.cast_pos.mpr hd) hgamma

/--
The concrete finite alignment-plus-spread objective.

This is the graph-theoretic SSL reading: compatible views should align along positive edges, while
the spread guard prevents the trivial all-views-identical representation from being accepted for
free.
-/
noncomputable def graphSSLObjective {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d) (gamma : ℝ) : ℝ :=
  graphAlignmentEnergy graph rep +
    realVarianceFloorGuard gamma (fun j => coordinateSpread rep j)

/--
For a collapsed representation, the alignment term is zero, so the objective reduces to the
variance-floor guard computed from zero coordinate spread.
-/
theorem graphSSLObjective_eq_guard_of_collapsed {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d) (gamma : ℝ)
    (hcollapsed : CollapsedRep rep) :
    graphSSLObjective graph rep gamma =
      realVarianceFloorGuard (d := d) gamma (fun _ => 0) := by
  simp [graphSSLObjective, graphAlignmentEnergy_eq_zero_of_collapsed graph rep hcollapsed,
    coordinateSpread_eq_zero_of_collapsed rep hcollapsed]

/--
With positive dimension and positive variance floor, a collapsed representation pays positive
finite SSL objective value. This is the concrete theorem version of "alignment needs a spread
guard."
-/
theorem graphSSLObjective_collapsed_positive {n d : Nat}
    (graph : SSLViewGraph n) (rep : Fin n → EuclideanRep d) {gamma : ℝ}
    (hcollapsed : CollapsedRep rep) (hd : 0 < d) (hgamma : 0 < gamma) :
    0 < graphSSLObjective graph rep gamma := by
  rw [graphSSLObjective_eq_guard_of_collapsed graph rep gamma hcollapsed]
  exact realVarianceFloorGuard_zero_spread_positive (d := d) hd hgamma

/--
Any target-index predictive objective can be read as a graph energy from one context anchor to each
selected target index.

The context anchor is represented by the same finite index type. This theorem is intentionally
simple: it is the finite bridge between masked/context-target prediction and graph-style SSL
alignment energy.
-/
theorem predictiveLoss_eq_viewGraphEnergy_from_anchor
    {n : Nat} {Context Target TargetRep Prediction : Type}
    (contract : PredictiveViewContract n Context Target TargetRep Prediction)
    (anchor : Fin n) :
    predictiveLoss contract =
      viewGraphEnergy
        { positiveEdges := contract.targetIdxs.map (fun i => (anchor, i)) }
        (fun _ i =>
          contract.distance (contract.targetEncoder i (contract.target i))
            (contract.predict contract.context i)) := by
  unfold predictiveLoss viewGraphEnergy maskedLoss
  induction contract.targetIdxs with
  | nil => simp
  | cons _ _ ih => simp [ih]

end NN.MLTheory.SelfSupervised
