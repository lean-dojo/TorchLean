/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Group.MinMax
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Data.List.FinRange
public import Mathlib.Data.Real.Sqrt
public import NN.Floats.NeuralFloat.NF
public import NN.Proofs.Gradients.Activation
public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Proofs.RuntimeApprox.Rounding.RoundingApprox
public import NN.Proofs.Utils.List
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Layers.Activation

/-!
# NF Primitive Ops

Forward (runtime→spec) approximation lemmas for the rounded runtime `NF`.

This is the proof-relevant numeric backend: `NF` wraps `ℝ` and applies `neural_round` after each
primitive operation. We prove **forward** bounds for a growing set of primitive tensor ops
(arithmetic + common elementwise nonlinearities), and provide `FwdNode` constructors that can be
composed using
`Proofs.RuntimeApprox.FwdGraph.eval_approx`.

This is the foundation of the NF backend: once a primitive operation has an `approxT_*_spec`
theorem here, it can be packaged as a graph node and composed into larger forward/backward/runtime
approximation results.

Notes:
- These are spec-level statements over `ℝ` (the spec scalar).
- `Float` execution remains a trusted implementation boundary (see `NN/Runtime/Scalar.lean`).
- Coverage is incremental: once per-op forward lemmas exist, any SSA/DAG graph built from those ops
  inherits a global forward bound by composition.
- Some ops cannot be bounded *unconditionally* with a pure `eps`-only interface (e.g. `log`, `inv`,
  `div`): they are ill-conditioned/singular near 0 and require either domain invariants (tracked
  in the approximation state) or safe/clamped spec variants (e.g. `NFBackend.safeLog`).

## PyTorch correspondence / citations
- Elementwise activations: `torch.nn.functional.relu`, `torch.sigmoid`, `torch.tanh`, etc.
  https://pytorch.org/docs/stable/nn.functional.html
- Elementwise math: `torch.exp`, `torch.log`, `torch.sqrt`, etc.
  https://pytorch.org/docs/stable/torch.html
- Autograd background (why we care about composing per-op bounds on a graph):
  https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

/-! ## Tensor Approximation Plumbing -/

/-- `linf_norm` is always nonnegative. -/
lemma linf_norm_nonneg : ∀ {s : Shape} (t : SpecTensor s), 0 ≤ linfNorm t := by
  intro s t
  induction s with
  | scalar =>
      cases t with
      | scalar x =>
          -- `tensor_linf_norm` uses `MathFunctions.abs`; for `ℝ` this is definitional `|·|`.
          dsimp [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm]
          -- Reduce to the usual `abs_nonneg`.
          have : (MathFunctions.abs x : ℝ) = |x| := by rfl
          rw [this]
          exact abs_nonneg x
  | dim n s ih =>
      cases t with
      | dim f =>
          have : (0 : ℝ) ≤
              (List.finRange n).foldl (fun acc i => max acc (tensorLinfNorm (α := ℝ) (f i))) 0 :=
                by
            simpa using (List.le_foldl_max_init (List.finRange n) (fun i => tensorLinfNorm (α :=
              ℝ) (f i)) 0)
          simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this

/--
Componentwise bound for `linf_norm` on a dimensioned tensor.

The norm of any component `t[i]` is bounded by the norm of the whole tensor.
-/
lemma linf_norm_le_get_dim {n : Nat} {s : Shape} (t : SpecTensor (.dim n s)) (i : Fin n) :
    linfNorm (match t with | Tensor.dim f => f i) ≤ linfNorm t := by
  cases t with
  | dim f =>
      have hi : i ∈ List.finRange n := List.mem_finRange i
      have hle :=
        List.le_foldl_max_of_mem (List.finRange n) (fun j => linfNorm (f j)) (acc := (0 : ℝ)) hi
      simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using hle

/--
Scalar characterization of `approxT` on scalar tensors.

This rewrites `approxT (Tensor.scalar x) (Tensor.scalar xR) eps` into the usual absolute-error
inequality `|toSpec xR - x| ≤ eps`.
-/
lemma approxT_scalar_iff {α : Type} {toSpec : α → SpecScalar} {x : SpecScalar} {xR : α} {eps :
  SpecScalar} :
    approxT (α := α) (toSpec := toSpec) (Tensor.scalar x) (Tensor.scalar xR) eps ↔
      abs (toSpec xR - x) ≤ eps := by
  -- `tensor_distance linf_norm` is `|x - toSpec xR|` on scalar tensors.
  constructor
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [approxT, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
        tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
        tensorLinfNorm, Spec.mapTensor] using h
    simpa [abs_sub_comm] using h'
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [abs_sub_comm] using h
    simpa [approxT, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
      tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
      tensorLinfNorm, Spec.mapTensor] using h'

/--
Projection lemma for `approxT` on dimensioned tensors.

If `xS` approximates `xR` within `eps`, then each component `xS[i]` approximates `xR[i]` within
  `eps`.
-/
lemma approxT_dim_get {α : Type} {toSpec : α → SpecScalar} {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor α (.dim n s)} {eps : SpecScalar}
    (h : approxT (α := α) (toSpec := toSpec) xS xR eps) (i : Fin n) :
    approxT (α := α) (toSpec := toSpec)
      (match xS with | Tensor.dim f => f i)
      (match xR with | Tensor.dim f => f i)
      eps := by
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          -- Unfold `approxT` at dimension shape: it is a `foldl max` over component distances.
          have hi : i ∈ List.finRange n := List.mem_finRange i
          have hComp :
              tensorDistance (α := SpecScalar) linfNorm (xSf i)
                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                ≤ tensorDistance (α := SpecScalar) linfNorm (Tensor.dim xSf)
                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)) := by
            -- Component distance is bounded by the max fold used in `linf_norm`.
            -- We unfold the RHS into the `foldl max` and use `le_foldl_max_of_mem`.
            have hle :=
              List.le_foldl_max_of_mem (List.finRange n)
                (fun j =>
                  tensorDistance (α := SpecScalar) linfNorm (xSf j)
                    (tensorToSpec (α := α) (toSpec := toSpec) (xRf j)))
                (acc := (0 : SpecScalar)) hi
            -- Now rewrite the unfolded RHS back to `tensor_distance`.
            simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
              linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec, Spec.mapTensor]
                using hle
          have := le_trans hComp h
          simpa [approxT, approxWith] using this

-- ---------------------------------------------------------------------------
-- Generic lifting lemmas for elementwise ops (`map_spec`, `map2_spec`)
-- ---------------------------------------------------------------------------

/--
Lift a scalar approximation bound to an elementwise `map_spec`.

Given a scalar bound of the form
`|toSpec (fR xR) - fS x| ≤ bnd (toSpec xR) eps`
and an input approximation `approxT xS xR eps`, this produces an approximation bound for
`map_spec fS xS` vs `map_spec fR xR`, with an output epsilon computed by taking the `linf_norm` of
the pointwise bound.
-/
theorem approxT_map_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar) (fR : α → α) (bnd : SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar},
      approxT (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : SpecScalar} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        approxT (α := α) (toSpec := toSpec)
          (mapSpec (s := s) fS xS)
          (mapSpec (s := s) fR xR)
          (linfNorm
            (mapSpec (s := s) (fun a => bnd a eps) (tensorToSpec (α := α) (toSpec := toSpec)
              xR))) := by
  intro xS xR eps hx hscalar
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxT_scalar_iff (α := α) (toSpec := toSpec)
                  (x := x) (xR := xR) (eps := eps)).1 hx
              have herr :
                  abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps :=
                hscalar hx'
              have herr' : abs (toSpec (fR xR) - fS x) ≤ abs (bnd (toSpec xR) eps) :=
                le_trans herr (le_abs_self _)
              exact
                (approxT_scalar_iff (α := α) (toSpec := toSpec)
                  (x := fS x) (xR := fR xR)
                  (eps := linfNorm
                    (mapSpec (s := Shape.scalar) (fun a => bnd a eps)
                      (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))))).2 (by
                        simpa [tensorToSpec, Spec.mapTensor, mapSpec,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using herr')
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm
                  (mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                    (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t :=
                    mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                      (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxT_dim_get (α := α) (toSpec := toSpec)
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm
                        (mapSpec (s := s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (xRf i)))
                      ≤ B := by
                  simpa [B, tensorToSpec, Spec.mapTensor, mapSpec] using
                    (linf_norm_le_get_dim
                      (t :=
                        mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf)))
                      i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤
                      linfNorm
                        (mapSpec (s := s) (fun a => bnd a eps)
                          (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge

              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) fS (xSf i))
                        (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) fS (xSf i))
                      (tensorToSpec (α := α) (toSpec := toSpec) (mapSpec (s := s) fR (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) fS (Tensor.dim xSf))
                      (tensorToSpec (α := α) (toSpec := toSpec)
                        (mapSpec (s := Shape.dim n s) fR (Tensor.dim xRf)))
                    ≤ B := by
                -- Rewrite back into `tensor_distance`.
                simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                    Spec.mapTensor, mapSpec] using hfold
              simpa [approxT, approxWith, B] using this

/--
Lift a scalar approximation bound to an elementwise `map2_spec`.

This is the binary analogue of `approxT_map_spec_of_scalar_bound`, used for elementwise arithmetic
(`add`, `sub`, `mul_elem`, etc.).
-/
theorem approxT_map2_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar → SpecScalar) (fR : α → α → α)
    (bnd : SpecScalar → SpecScalar → SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor α s} {epsx epsy : SpecScalar},
      approxT (α := α) (toSpec := toSpec) xS xR epsx →
      approxT (α := α) (toSpec := toSpec) yS yR epsy →
        (∀ {x y : SpecScalar} {xR yR : α},
          abs (toSpec xR - x) ≤ epsx →
          abs (toSpec yR - y) ≤ epsy →
            abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy) →
        approxT (α := α) (toSpec := toSpec)
          (map2Spec fS xS yS)
          (map2Spec fR xR yR)
          (linfNorm
            (map2Spec (fun a b => bnd a b epsx epsy)
              (tensorToSpec (α := α) (toSpec := toSpec) xR)
              (tensorToSpec (α := α) (toSpec := toSpec) yR))) := by
  intro xS yS xR yR epsx epsy hx hy hscalar
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      have hx' :=
                        (approxT_scalar_iff (α := α) (toSpec := toSpec)
                          (x := x) (xR := xR) (eps := epsx)).1 hx
                      have hy' :=
                        (approxT_scalar_iff (α := α) (toSpec := toSpec)
                          (x := y) (xR := yR) (eps := epsy)).1 hy
                      have herr :
                          abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy
                            :=
                        hscalar hx' hy'
                      have herr' :
                          abs (toSpec (fR xR yR) - fS x y) ≤ abs (bnd (toSpec xR) (toSpec yR) epsx
                            epsy) :=
                        le_trans herr (le_abs_self _)
                      exact
                        (approxT_scalar_iff (α := α) (toSpec := toSpec)
                          (x := fS x y) (xR := fR xR yR)
                          (eps := linfNorm
                            (map2Spec (fun a b => bnd a b epsx epsy)
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR))
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar yR))))).2
                                (by
                                simpa [tensorToSpec, Spec.mapTensor, map2Spec,
                                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using herr')
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      let B : ℝ :=
                        linfNorm
                          (map2Spec (fun a b => bnd a b epsx epsy)
                            (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                            (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf)))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t :=
                            map2Spec (fun a b => bnd a b epsx epsy)
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf))))
                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤ B := by
                        intro i
                        have hx_i :=
                          approxT_dim_get (α := α) (toSpec := toSpec)
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxT_dim_get (α := α) (toSpec := toSpec)
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i
                        have hih := ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i
                          hy_i
                        have hB_ge :
                            linfNorm
                                (map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (yRf i)))
                              ≤ B := by
                          simpa [B, tensorToSpec, Spec.mapTensor, map2Spec] using
                            (linf_norm_le_get_dim
                              (t :=
                                map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim xRf))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.dim yRf)))
                              i)
                        have hdist :
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤
                              linfNorm
                                (map2Spec (fun a b => bnd a b epsx epsy)
                                  (tensorToSpec (α := α) (toSpec := toSpec) (xRf i))
                                  (tensorToSpec (α := α) (toSpec := toSpec) (yRf i))) := by
                          simpa [approxT, approxWith] using hih
                        exact le_trans hdist hB_ge

                      have hf :
                          ∀ i ∈ List.finRange n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (map2Spec fS (xSf i) (ySf i))
                                (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i)
                                  (yRf i)))
                              ≤ B := by
                        intro i _hi
                        exact hcomp i
                      have hfold :=
                        List.foldl_max_le_of_le (List.finRange n)
                          (fun i =>
                            tensorDistance (α := SpecScalar) linfNorm
                              (map2Spec fS (xSf i) (ySf i))
                              (tensorToSpec (α := α) (toSpec := toSpec) (map2Spec fR (xRf i) (yRf
                                i))))
                          (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                      have :
                          tensorDistance (α := SpecScalar) linfNorm
                              (map2Spec fS (Tensor.dim xSf) (Tensor.dim ySf))
                              (tensorToSpec (α := α) (toSpec := toSpec)
                                (map2Spec fR (Tensor.dim xRf) (Tensor.dim yRf)))
                            ≤ B := by
                        simpa [tensorDistance,
                          NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                            Spec.mapTensor,
                          map2Spec] using hfold
                      simpa [approxT, approxWith, B] using this

-- ---------------------------------------------------------------------------
-- NF backend: forward bounds for primitive tensor ops
-- ---------------------------------------------------------------------------

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

/-- Interpret a runtime `NF` scalar as a spec scalar (`ℝ`) by forgetting rounding metadata. -/
@[inline] abbrev toSpec (x : R) : SpecScalar := TorchLean.Floats.NF.toReal x

/-!
## NF → ℝ bridge lemmas

Most approximation statements in this file are phrased over the spec scalar `ℝ`, but the runtime
backend is `NF β fexp rnd`. The following lemmas are small bridge facts that let us rewrite
runtime expressions into:

- an exact real expression in terms of `toSpec`, plus
- an explicit rounding operator `roundR` applied at the outermost step.

Keeping these as named lemmas (instead of repeating huge `simp [...]` lists) makes the later
forward-approx proofs much easier to read.
-/

/-- Rounding `0` is `0` for any valid rounding mode. -/
private lemma roundR_zero : Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
  (0 : ℝ) = 0 := by
  -- For `x = 0`, the scaled mantissa is `0`, so `rnd` returns `0` by `NeuralValidRnd.id`,
  -- and `neural_to_real` is `0` regardless of exponent.
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    -- `rnd` is exact on integers.
    simpa using (NeuralValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  simp [Proofs.RuntimeRoundingApprox.roundR, TorchLean.Floats.neuralRound,
    TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.neuralToReal, hrnd0]

/-- The `NF.roundR` wrapper also rounds `0` to `0`. -/
private lemma NF_roundR_zero : TorchLean.Floats.NF.roundR (β := β) (fexp := fexp) (rnd := rnd) (0 :
  ℝ) = 0 := by
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    simpa using (NeuralValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  simp [TorchLean.Floats.NF.roundR, TorchLean.Floats.neuralRound,
    TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.neuralToReal, hrnd0]

/-- `toSpec` of runtime `0` is the spec scalar `0`. -/
@[simp] lemma toSpec_zero : toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) = (0 : ℝ) := by
  -- `0 : R` is `NF.ofReal 0`, so `toSpec 0` is `NF.roundR 0`.
  simpa [toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.instZero,
    TorchLean.Floats.NF.ofReal] using
    (NF_roundR_zero (β := β) (fexp := fexp) (rnd := rnd))

omit [NeuralValidRndToNearest rnd] in
/--
`toSpec` respects runtime addition, up to an explicit rounding step.

This is the defining `NF` semantics: compute in `ℝ` and then apply `roundR`.
-/
private lemma toSpec_add (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x + y) =
      roundedAdd (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime multiplication, up to an explicit rounding step. -/
private lemma toSpec_mul (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x * y) =
      roundedMul (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime division, up to an explicit rounding step. -/
private lemma toSpec_div (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x / y) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x /
          toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime subtraction, up to an explicit rounding step. -/
private lemma toSpec_sub (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x - y) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x -
          toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime negation, up to an explicit rounding step. -/
private lemma toSpec_neg (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (-x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (-toSpec (β := β) (fexp := fexp) (rnd := rnd) x) := by
  simp [toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, Neg.neg]

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime `exp`, up to an explicit rounding step. -/
private lemma toSpec_exp (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, MathFunctions.exp,
    ]

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime `tanh`, up to an explicit rounding step. -/
private lemma toSpec_tanh (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.tanh x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.tanh (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, MathFunctions.tanh,
    ]

omit [NeuralValidRndToNearest rnd] in
/-- `toSpec` respects runtime `sqrt`, up to an explicit rounding step. -/
private lemma toSpec_sqrt (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, MathFunctions.sqrt,
    ]

-- ---------------------------------------------------------------------------
-- Sqrt (clamped) approximation
-- ---------------------------------------------------------------------------

private lemma abs_sqrt_sub_sqrt_le_div_sqrt_of_le {a b η : ℝ} (ha : 0 ≤ a) (hη : 0 < η) (hb : η ≤ b)
  :
    abs (Real.sqrt a - Real.sqrt b) ≤ abs (a - b) / Real.sqrt η := by
  have hb0 : 0 < b := lt_of_lt_of_le hη hb
  have hsb_pos : 0 < Real.sqrt b := Real.sqrt_pos.2 hb0
  have hsa_nonneg : 0 ≤ Real.sqrt a := Real.sqrt_nonneg a
  have hden_pos : 0 < Real.sqrt a + Real.sqrt b := add_pos_of_nonneg_of_pos hsa_nonneg hsb_pos
  have hden_ne : Real.sqrt a + Real.sqrt b ≠ 0 := ne_of_gt hden_pos
  have hprod :
      (Real.sqrt a - Real.sqrt b) * (Real.sqrt a + Real.sqrt b) = a - b := by
    -- `(√a - √b) * (√a + √b) = (√a)^2 - (√b)^2 = a - b`
    have ha' : Real.sqrt a ^ 2 = a := Real.sq_sqrt ha
    have hb' : Real.sqrt b ^ 2 = b := Real.sq_sqrt (le_of_lt hb0)
    ring_nf
    simp [ha', hb']
  have hdiv : Real.sqrt a - Real.sqrt b = (a - b) / (Real.sqrt a + Real.sqrt b) :=
    (eq_div_iff hden_ne).2 hprod
  have hden_ge : Real.sqrt η ≤ Real.sqrt a + Real.sqrt b := by
    have hη0 : 0 ≤ η := le_of_lt hη
    have hsqrt : Real.sqrt η ≤ Real.sqrt b := Real.sqrt_le_sqrt hb
    have : Real.sqrt b ≤ Real.sqrt a + Real.sqrt b := by
      simp
    exact le_trans hsqrt this
  have hquot :=
    div_le_div_of_nonneg_left (abs_nonneg (a - b)) (Real.sqrt_pos.2 hη) hden_ge
  calc
    abs (Real.sqrt a - Real.sqrt b)
        = abs ((a - b) / (Real.sqrt a + Real.sqrt b)) := by simp [hdiv]
    _ = abs (a - b) / abs (Real.sqrt a + Real.sqrt b) := by simp [abs_div]
    _ = abs (a - b) / (Real.sqrt a + Real.sqrt b) := by
          simp [abs_of_pos hden_pos]
    _ ≤ abs (a - b) / Real.sqrt η := hquot

/--
Forward approximation bound for `sqrt (max · 0)` under a positive lower bound.

This is a *clamped* sqrt bound: we work with `sqrt (max x 0)` to avoid the `sqrt` domain issue, but
still require a *strict* lower bound `η > 0` on `max x 0` to control conditioning via
`|√a - √b| ≤ |a-b| / √η`.
-/
lemma approx_sqrt_clamp_nf_of_lb {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ max x 0)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)) -
          Real.sqrt (max x 0)) ≤
      eps / Real.sqrt η +
        neuralUlp β fexp (Real.sqrt (max (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) 0))
            TrainingPhase.forward / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx
  have hmax : abs (max xhat 0 - max x 0) ≤ eps := by
    have h1 : abs (max xhat 0 - max x 0) ≤ abs (xhat - x) := by
      simpa using (abs_max_sub_max_le_abs xhat x (0 : ℝ))
    exact le_trans h1 hxhat
  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max xhat 0)) ≤
        neuralUlp β fexp (Real.sqrt (max xhat 0)) TrainingPhase.forward / 2 := by
    -- `sqrt` on NF is a single rounding of the real `sqrt`.
    have :
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)) =
          Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
            (Real.sqrt (max xhat 0)) := by
      -- `max xR 0` is either `xR` or `0`; `toSpec` commutes with `max`.
      have hxmax :
          toSpec (β := β) (fexp := fexp) (rnd := rnd) (max xR (0 : R)) = max xhat 0 := by
        by_cases h0 : (0 : R) ≤ xR
        · have hxhat0 : 0 ≤ xhat := by
            have h0' : (0 : R).val ≤ xR.val := by
              simpa [LE.le, TorchLean.Floats.NF.instLE] using h0
            have h0z : (0 : R).val = (0 : ℝ) := by
              simpa [toSpec, TorchLean.Floats.NF.toReal] using (toSpec_zero (β := β) (fexp := fexp)
                (rnd := rnd))
            simpa [xhat, toSpec, TorchLean.Floats.NF.toReal, h0z] using h0'
          have hmaxR : max xR (0 : R) = xR := by
            -- `max` on `NF` is a pure selection.
            have : xR ≥ (0 : R) := h0
            simp [Max.max, this]
          have hmaxS : max xhat 0 = xhat := max_eq_left hxhat0
          simpa [hmaxR, hmaxS, xhat, toSpec, TorchLean.Floats.NF.toReal]
        · have hxhat0 : xhat ≤ 0 := by
            have h0' : ¬ (0 : R).val ≤ xR.val := by
              simpa [LE.le, TorchLean.Floats.NF.instLE] using h0
            have : ¬ (0 : ℝ) ≤ xhat := by
              have h0z : (0 : R).val = (0 : ℝ) := by
                simpa [toSpec, TorchLean.Floats.NF.toReal] using (toSpec_zero (β := β) (fexp :=
                  fexp) (rnd := rnd))
              simpa [xhat, toSpec, TorchLean.Floats.NF.toReal, h0z] using h0'
            exact le_of_not_ge this
          have hmaxR : max xR (0 : R) = (0 : R) := by
            have : ¬ xR ≥ (0 : R) := by
              -- `xR ≥ 0` is definitionally `0 ≤ xR`.
              simpa [ge_iff_le] using h0
            simp [Max.max, this]
          have hmaxS : max xhat 0 = 0 := max_eq_right hxhat0
          simp [hmaxR, hmaxS, toSpec_zero]
      simpa [xhat, hxmax] using
        (toSpec_sqrt (β := β) (fexp := fexp) (rnd := rnd) (max xR 0))
    -- Now apply the generic rounding error lemma.
    simpa [this, Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt
        (max xhat 0)))

  have hdiff : abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) ≤ eps / Real.sqrt η := by
    have ha : 0 ≤ max xhat 0 := le_max_right _ _
    exact le_trans
      (abs_sqrt_sub_sqrt_le_div_sqrt_of_le (a := max xhat 0) (b := max x 0) (η := η) ha hη hdom)
      (by
        -- monotonicity in the numerator
        have : abs (max xhat 0 - max x 0) / Real.sqrt η ≤ eps / Real.sqrt η := by
          exact div_le_div_of_nonneg_right hmax (Real.sqrt_nonneg η)
        simpa [abs_sub_comm] using this)

  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max x 0))
          ≤ abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)) -
                Real.sqrt (max xhat 0)) +
              abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.sqrt (max xR 0)))
                    (Real.sqrt (max xhat 0))
                    (Real.sqrt (max x 0))
      _ ≤ neuralUlp β fexp (Real.sqrt (max xhat 0)) TrainingPhase.forward / 2 + eps / Real.sqrt η
        := by
            exact add_le_add hround hdiff
      _ = eps / Real.sqrt η +
            neuralUlp β fexp (Real.sqrt (max xhat 0)) TrainingPhase.forward / 2 := by
            ring
  simpa [xhat, add_comm, add_left_comm, add_assoc] using this

private lemma abs_tanh_le_one (x : ℝ) : abs (Real.tanh x) ≤ 1 := by
  -- `tanh x = (exp x - exp (-x)) / (exp x + exp (-x))`, so `|tanh x| ≤ 1` by `|a-b| ≤ a+b`.
  have htanh :
      Real.tanh x =
        (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x)) := by
    -- Reduce to `sinh/cosh` and then to the `exp` definitions.
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    -- Cancel the common `/2`.
    field_simp [two_ne_zero]

  have hden_pos : 0 < Real.exp x + Real.exp (-x) :=
    add_pos (Real.exp_pos x) (Real.exp_pos (-x))
  have hden_ne : Real.exp x + Real.exp (-x) ≠ 0 := ne_of_gt hden_pos

  have hnum :
      abs (Real.exp x - Real.exp (-x)) ≤ Real.exp x + Real.exp (-x) := by
    -- `|a-b| ≤ |a| + |b|` and `exp` is nonnegative.
    have := abs_add_le (Real.exp x) (-Real.exp (-x))
    -- `abs (a + (-b)) ≤ abs a + abs (-b)`.
    simpa [sub_eq_add_neg, abs_neg, abs_of_nonneg (Real.exp_nonneg _),
      abs_of_nonneg (Real.exp_nonneg _)] using this

  -- Divide by the positive denominator.
  have hdiv :=
    div_le_div_of_nonneg_right hnum (le_of_lt hden_pos)
  -- `|(a-b)/(a+b)| ≤ (a+b)/(a+b) = 1`.
  simpa [htanh, abs_div, abs_of_pos hden_pos, div_self hden_ne] using hdiv

/--
Forward approximation bound for addition in `NF`.

In words: if `xR` approximates `x` within `epsx` and `yR` approximates `y` within `epsy`,
then `xR + yR` approximates `x + y` within `epsx + epsy + ulp(toSpec xR + toSpec yR)/2`.
-/
lemma approx_add_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR + yR) - (x + y)) ≤
      epsx + epsy +
        neuralUlp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR +
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
            TrainingPhase.forward / 2 := by
  have hx' :
      Proofs.RuntimeRoundingApprox.scalarApprox x
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) epsx := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hx
  have hy' :
      Proofs.RuntimeRoundingApprox.scalarApprox y
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) epsy := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hy
  have h := scalarApprox_roundedAdd (β := β) (fexp := fexp) (rnd := rnd) hx' hy'
  -- Rewrite the runtime result as `toSpec (xR + yR)`.
  simpa [Proofs.RuntimeRoundingApprox.scalarApprox,
    toSpec_add (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/--
Forward approximation bound for subtraction in `NF`.

This is proved by reducing subtraction to addition with a negation and applying `approx_add_nf`.
-/
lemma approx_sub_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR - yR) - (x - y)) ≤
      epsx + epsy +
        neuralUlp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR -
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
            TrainingPhase.forward / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  let yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat - yhat) -
            (xhat - yhat)) ≤
        neuralUlp β fexp (xhat - yhat) TrainingPhase.forward / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (xhat -
        yhat))
  have hdiff :
      abs ((xhat - yhat) - (x - y)) ≤ epsx + epsy := by
    have hrewrite : (xhat - yhat) - (x - y) = (xhat - x) - (yhat - y) := by ring
    have htri : abs ((xhat - x) - (yhat - y)) ≤ abs (xhat - x) + abs (yhat - y) := by
      simpa [abs_sub_comm] using (abs_sub_le (xhat - x) 0 (yhat - y))
    have hsum : abs (xhat - x) + abs (yhat - y) ≤ epsx + epsy := add_le_add hx hy
    exact le_trans (by simpa [hrewrite] using htri) hsum
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR - yR) - (x - y))
          =
          abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat - yhat)
              -
              (x - y)) := by
              simp [xhat, yhat, toSpec_sub (β := β) (fexp := fexp) (rnd := rnd) xR yR]
      _ ≤
          abs
              (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat -
                yhat) -
                (xhat - yhat)) +
            abs ((xhat - yhat) - (x - y)) := by
              simpa [sub_eq_add_neg, add_assoc] using
                abs_sub_le
                  (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat -
                    yhat))
                  (xhat - yhat) (x - y)
      _ ≤ neuralUlp β fexp (xhat - yhat) TrainingPhase.forward / 2 + (epsx + epsy) := by
            exact add_le_add hround hdiff
      _ = epsx + epsy + neuralUlp β fexp (xhat - yhat) TrainingPhase.forward / 2 := by ring
  simpa [xhat, yhat, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this

/-- Forward approximation bound for negation in `NF` (rounding error on `-toSpec xR`). -/
lemma approx_neg_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-x)) ≤
      eps +
        neuralUlp β fexp
            (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
            TrainingPhase.forward / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-xhat)) ≤
        neuralUlp β fexp (-xhat) TrainingPhase.forward / 2 := by
    -- `toSpec (-xR)` is a single rounding of `-xhat`.
    simpa [xhat, toSpec_neg (β := β) (fexp := fexp) (rnd := rnd) xR,
      Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (-xhat))
  have hdiff : abs (-xhat - (-x)) ≤ eps := by
    have hxhat : abs (xhat - x) ≤ eps := by
      simpa [xhat] using hx
    have hx' : abs (-xhat + x) ≤ eps := by
      have habs : abs (-xhat + x) = abs (xhat - x) := by
        calc
          abs (-xhat + x) = abs (x - xhat) := by
            simp [sub_eq_add_neg, add_comm]
          _ = abs (xhat - x) := by
            simp [abs_sub_comm]
      simpa [habs] using hxhat
    -- `-xhat - (-x)` is definitional `-xhat + x`.
    simpa [sub_eq_add_neg] using hx'
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-x))
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-xhat)) + abs (-xhat - (-x))
            := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR))
                    (-xhat) (-x)
      _ ≤ neuralUlp β fexp (-xhat) TrainingPhase.forward / 2 + eps := by
            exact add_le_add hround hdiff
      _ = eps + neuralUlp β fexp (-xhat) TrainingPhase.forward / 2 := by ring
  simpa [xhat, add_assoc, add_left_comm, add_comm] using this

/-- Forward approximation bound for absolute value in `NF` (`abs` is pure + a final rounding). -/
lemma approx_abs_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.abs xR) - abs x) ≤
      eps +
        neuralUlp β fexp
            (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
            TrainingPhase.forward / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.abs xR) - abs xhat) ≤
        neuralUlp β fexp (abs xhat) TrainingPhase.forward / 2 := by
    -- `toSpec (abs xR)` is a single rounding of `|xhat|`.
    simpa [xhat, toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
      TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (abs
        xhat))
  have habs : abs (abs xhat - abs x) ≤ abs (xhat - x) := by
    simpa [abs_sub_comm] using (abs_abs_sub_abs_le_abs_sub xhat x)
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.abs xR) - abs x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.abs xR) - abs xhat) +
              abs (abs xhat - abs x) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.abs xR))
                    (abs xhat) (abs x)
      _ ≤ neuralUlp β fexp (abs xhat) TrainingPhase.forward / 2 + abs (xhat - x) := by
            exact add_le_add hround habs
      _ ≤ neuralUlp β fexp (abs xhat) TrainingPhase.forward / 2 + eps := by
            linarith [hxhat]
      _ = eps + neuralUlp β fexp (abs xhat) TrainingPhase.forward / 2 := by ring
  simpa [xhat, add_assoc, add_left_comm, add_comm] using this

/--
Forward approximation bound for `exp` in `NF`.

Uses the mean value theorem for `Real.exp` to bound the propagation of input error, then adds one
rounding-ULP term for the final `NF` rounding.
-/
lemma approx_exp_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp xR) - Real.exp x) ≤
      Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) +
        Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR + eps) +
        neuralUlp β fexp
            (Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
            TrainingPhase.forward / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR

  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp xhat)
            -
            Real.exp xhat) ≤
        neuralUlp β fexp (Real.exp xhat) TrainingPhase.forward / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
        xhat))

  have hx_le : x ≤ xhat + eps := by
    have hx' := (abs_sub_le_iff.1 (by simpa [xhat] using hx))
    have h : x - xhat ≤ eps := hx'.2
    have : x ≤ eps + xhat := (sub_le_iff_le_add).1 h
    simpa [add_comm, add_left_comm, add_assoc] using this

  have hexp_le : Real.exp x ≤ Real.exp (xhat + eps) :=
    Real.exp_monotone hx_le

  have hdiff : abs (Real.exp xhat - Real.exp x) ≤ Real.exp xhat + Real.exp (xhat + eps) := by
    have h' : abs (Real.exp xhat - Real.exp x) ≤ Real.exp xhat + Real.exp x := by
      have := abs_add_le (Real.exp xhat) (-Real.exp x)
      simpa [sub_eq_add_neg, abs_neg, abs_of_nonneg (Real.exp_nonneg _),
        abs_of_nonneg (Real.exp_nonneg _)] using this
    exact le_trans h' (by linarith [hexp_le])

  have htotal :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp xhat)
            -
            Real.exp x) ≤
        Real.exp xhat + Real.exp (xhat + eps) + neuralUlp β fexp (Real.exp xhat)
          TrainingPhase.forward / 2 := by
    have :=
      calc
        abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
              xhat) -
              Real.exp x)
            ≤ abs
                (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
                  xhat) -
                  Real.exp xhat) +
                abs (Real.exp xhat - Real.exp x) := by
                  simpa [sub_eq_add_neg, add_assoc] using
                    abs_sub_le
                      (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
                        (Real.exp xhat))
                      (Real.exp xhat) (Real.exp x)
        _ ≤ neuralUlp β fexp (Real.exp xhat) TrainingPhase.forward / 2 +
              (Real.exp xhat + Real.exp (xhat + eps)) := by
              exact add_le_add hround hdiff
        _ = Real.exp xhat + Real.exp (xhat + eps) +
              neuralUlp β fexp (Real.exp xhat) TrainingPhase.forward / 2 := by ring
    exact this

  simpa [xhat, toSpec_exp (β := β) (fexp := fexp) (rnd := rnd) xR, add_assoc, add_left_comm,
    add_comm]
    using htotal

/--
Forward approximation bound for `tanh` in `NF` (coarse but unconditional).

Because `tanh` is bounded in `[-1, 1]`, we always have `|tanh(toSpec xR) - tanh(x)| ≤ 2`, and then
we add one rounding-ULP term for the final `NF` rounding step.
-/
lemma approx_tanh_nf {x : ℝ} {xR : R} {eps : ℝ}
    (_hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.tanh xR) - Real.tanh x) ≤
      2 +
        neuralUlp β fexp
            (Real.tanh (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
            TrainingPhase.forward / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR

  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh xhat)
            -
            Real.tanh xhat) ≤
        neuralUlp β fexp (Real.tanh xhat) TrainingPhase.forward / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
        xhat))

  have hdiff : abs (Real.tanh xhat - Real.tanh x) ≤ 2 := by
    have h' : abs (Real.tanh xhat - Real.tanh x) ≤ abs (Real.tanh xhat) + abs (Real.tanh x) := by
      have := abs_add_le (Real.tanh xhat) (-Real.tanh x)
      simpa [sub_eq_add_neg, abs_neg] using this
    have hxhat_le : abs (Real.tanh xhat) ≤ 1 := abs_tanh_le_one xhat
    have hx_le : abs (Real.tanh x) ≤ 1 := abs_tanh_le_one x
    have : abs (Real.tanh xhat) + abs (Real.tanh x) ≤ (1 : ℝ) + 1 := add_le_add hxhat_le hx_le
    have := le_trans h' this
    simpa [one_add_one_eq_two] using this

  have htotal :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh xhat)
            -
            Real.tanh x) ≤
        2 + neuralUlp β fexp (Real.tanh xhat) TrainingPhase.forward / 2 := by
    have :=
      calc
        abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
              xhat) -
              Real.tanh x)
            ≤ abs
                (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
                  xhat) -
                  Real.tanh xhat) +
                abs (Real.tanh xhat - Real.tanh x) := by
                  simpa [sub_eq_add_neg, add_assoc] using
                    abs_sub_le
                      (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
                        (Real.tanh xhat))
                      (Real.tanh xhat) (Real.tanh x)
        _ ≤ neuralUlp β fexp (Real.tanh xhat) TrainingPhase.forward / 2 + 2 := by
              exact add_le_add hround hdiff
        _ = 2 + neuralUlp β fexp (Real.tanh xhat) TrainingPhase.forward / 2 := by ring
    exact this

  -- `hx` is not needed for the range-based bound, but kept for uniformity with other unary lemmas.
  simpa [xhat, toSpec_tanh (β := β) (fexp := fexp) (rnd := rnd) xR, add_assoc, add_left_comm,
    add_comm]
    using htotal

-- ---------------------------------------------------------------------------
-- Safe log: `log (max x ε)` (needed for unconditional forward bounds)
-- ---------------------------------------------------------------------------

/--
Clamped log on spec scalars: `log (max x ε)`.

This is used to obtain unconditional forward bounds for `log` by avoiding the singularity at `0`.
-/
def safeLog (ε : ℝ) (x : ℝ) : ℝ :=
  Real.log (max x ε)

/--
Clamped log on runtime `NF` scalars (implemented as `NF.ofReal (safeLog (toSpec xR))`).

This definition keeps the semantic spec function explicit (so proofs can reason about it) while
still producing an executable runtime scalar.
-/
def safeLogR (ε : ℝ) (xR : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeLog (ε := ε) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))

private lemma abs_log_sub_log_le_one_div_mul_abs_sub {ε u v : ℝ}
    (hε : 0 < ε) (hu : ε ≤ u) (hv : ε ≤ v) :
    abs (Real.log u - Real.log v) ≤ (1 / ε) * abs (u - v) := by
  -- Mean value theorem on `s = Ici ε` (derivative bounded by `1/ε`).
  have hf : ∀ x ∈ Set.Ici ε, HasDerivWithinAt Real.log (x⁻¹) (Set.Ici ε) x := by
    intro x hx
    have hx0 : x ≠ 0 := ne_of_gt (lt_of_lt_of_le hε hx)
    simpa using (Real.hasDerivAt_log (x := x) hx0).hasDerivWithinAt

  have hbound : ∀ x ∈ Set.Ici ε, ‖x⁻¹‖ ≤ (1 / ε) := by
    intro x hx
    have hxpos : 0 < x := lt_of_lt_of_le hε hx
    have hxinv : ‖x⁻¹‖ = (1 : ℝ) / x := by
      have hxinvpos : 0 < x⁻¹ := inv_pos.2 hxpos
      calc
        ‖x⁻¹‖ = |x⁻¹| := Real.norm_eq_abs (r := x⁻¹)
        _ = x⁻¹ := abs_of_pos hxinvpos
        _ = (1 : ℝ) / x := (one_div x).symm
    have hle : (1 : ℝ) / x ≤ (1 : ℝ) / ε := by
      simpa using (one_div_le_one_div_of_le hε hx)
    simpa [hxinv] using hle

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (f := Real.log) (f' := fun x : ℝ => x⁻¹)
      (s := Set.Ici ε) (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_Ici ε) hu hv

  -- Unwrap norms on `ℝ`.
  simpa [Real.norm_eq_abs, abs_sub_comm] using hmv

/--
Forward approximation bound for `safeLog` in `NF`.

On the clamped domain `u,v ≥ ε > 0`, `log` is `(1/ε)`-Lipschitz. We use that to propagate the input
error and then add one rounding-ULP term for the final `NF` rounding.
-/
lemma approx_safeLog_nf {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (safeLogR (β := β) (fexp := fexp) (rnd := rnd)
      ε xR) -
          safeLog (ε := ε) x) ≤
      (1 / ε) * eps +
        neuralUlp β fexp (safeLog (ε := ε) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
          TrainingPhase.forward / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := max xhat ε
  set y : ℝ := max x ε

  have hyhat : ε ≤ yhat := le_max_right _ _
  have hy : ε ≤ y := le_max_right _ _

  have hmax : abs (yhat - y) ≤ eps := by
    have hmax' : abs (max xhat ε - max x ε) ≤ abs (xhat - x) := by
      simpa using (abs_max_sub_max_le_abs xhat x ε)
    have hx' : abs (xhat - x) ≤ eps := by
      simpa [xhat, abs_sub_comm] using hx
    -- Rewrite `yhat,y` and chain.
    simpa [yhat, y] using le_trans hmax' hx'

  have hdiff :
      abs (Real.log yhat - Real.log y) ≤ (1 / ε) * eps := by
    have hlog :
        abs (Real.log yhat - Real.log y) ≤ (1 / ε) * abs (yhat - y) := by
      simpa [one_div] using (abs_log_sub_log_le_one_div_mul_abs_sub (ε := ε) (u := yhat) (v := y) hε
        hyhat hy)
    have hεinv_nonneg : 0 ≤ (1 / ε) := by
      exact one_div_nonneg.2 (le_of_lt hε)
    exact le_trans hlog (mul_le_mul_of_nonneg_left hmax hεinv_nonneg)

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Real.log yhat) ≤
        neuralUlp β fexp (Real.log yhat) TrainingPhase.forward / 2 := by
    -- `safeLogR` rounds the real `log (max x̂ ε)`.
    have :
        toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) =
          Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.log yhat)
            := by
      simp [safeLogR, safeLog, toSpec, xhat, yhat, Proofs.RuntimeRoundingApprox.roundR,
        TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal]
    simpa [this] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.log
        yhat))

  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            safeLog (ε := ε) x)
          ≤ abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
                Real.log yhat) +
              abs (Real.log yhat - safeLog (ε := ε) x) := by
                simpa [safeLog, y, sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                      (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR))
                    (Real.log yhat)
                    (safeLog (ε := ε) x)
      _ ≤ neuralUlp β fexp (Real.log yhat) TrainingPhase.forward / 2 + (1 / ε) * eps := by
            -- second term is the `log` perturbation
            have : abs (Real.log yhat - safeLog (ε := ε) x) = abs (Real.log yhat - Real.log y) := by
              simp [safeLog, y]
            simpa [this, add_comm, add_left_comm, add_assoc] using add_le_add hround hdiff
      _ = (1 / ε) * eps + neuralUlp β fexp (safeLog (ε := ε) xhat) TrainingPhase.forward / 2 := by
            simp [safeLog, xhat, yhat, add_comm]
  simpa [xhat] using this

/--
Forward approximation bound for multiplication in `NF`.

This has the standard "first-order" shape:
terms proportional to `|toSpec xR| * epsy` and `|toSpec yR| * epsx`, plus an `ulp` term for the
  final
rounding. (For classical background, see Higham, *Accuracy and Stability of Numerical Algorithms*.)
-/
lemma approx_mul_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * yR) - (x * y)) ≤
      ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * epsy +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) + epsy) * epsx +
        neuralUlp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
            TrainingPhase.forward / 2) := by
  have hx' :
      Proofs.RuntimeRoundingApprox.scalarApprox x
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) epsx := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hx
  have hy' :
      Proofs.RuntimeRoundingApprox.scalarApprox y
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) epsy := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hy
  have h := scalarApprox_roundedMul (β := β) (fexp := fexp) (rnd := rnd) hx' hy'
  simpa [Proofs.RuntimeRoundingApprox.scalarApprox,
    toSpec_mul (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/--
Forward approximation bound for division under a coarse denominator lower bound (`y ≥ 1`).

Division is ill-conditioned near `0`, so we need a domain condition. This lemma is tailored for the
simple case `y ≥ 1` to keep constants small.
-/
lemma approx_div_nf_of_one_le {x y : ℝ} {xR yR : R} {epsx : ℝ}
    (hy : (1 : ℝ) ≤ y)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - (x / y)) ≤
      neuralUlp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR /
            toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
          TrainingPhase.forward / 2
        + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) *
            abs (1 / toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
        + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
        + epsx := by
  -- Notation for the embedded runtime values.
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  set qhat : ℝ := xhat / yhat

  have hy_pos : 0 < y := lt_of_lt_of_le (by norm_num) hy

  -- Bound `|x|` by the rounded magnitude plus epsilon.
  have hx_abs : abs x ≤ abs xhat + epsx := by
    have hx' : abs (x - xhat) ≤ epsx := by
      -- `hx` is stated using `(xhat - x)`.
      simpa [xhat, abs_sub_comm] using hx
    calc
      abs x = abs ((x - xhat) + xhat) := by
        simp [sub_add_cancel]
      _ ≤ abs (x - xhat) + abs xhat := by
        simpa using abs_add_le (x - xhat) xhat
      _ ≤ epsx + abs xhat := by
        exact add_le_add_left hx' (abs xhat)
      _ = abs xhat + epsx := by
        simp [add_comm]

  -- `|1/y| ≤ 1` since `y ≥ 1`.
  have hy_inv_le_one : abs (1 / y) ≤ (1 : ℝ) := by
    have h : (1 : ℝ) / y ≤ 1 / (1 : ℝ) := by
      simpa using (one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < (1 : ℝ)) hy)
    have h' : (1 : ℝ) / y ≤ (1 : ℝ) := by simpa using h
    have hy_div_pos : 0 < (1 : ℝ) / y := by
      simpa [div_eq_mul_inv] using (div_pos (show (0 : ℝ) < (1 : ℝ) by norm_num) hy_pos)
    calc
      abs (1 / y) = (1 : ℝ) / y := abs_of_pos hy_div_pos
      _ ≤ 1 := h'

  -- Unrounded quotient error: `|q̂ - x/y| ≤ |q̂| + |x/y|`.
  have hquot :
      abs (qhat - x / y) ≤ abs xhat * abs (1 / yhat) + abs xhat + epsx := by
    have hsub : abs (qhat - x / y) ≤ abs qhat + abs (x / y) := by
      -- `|a-b| ≤ |a| + |b|`.
      simpa using (abs_sub_le qhat 0 (x / y))
    have hq : abs qhat = abs xhat * abs (1 / yhat) := by
      simp [qhat, div_eq_mul_inv, abs_mul, abs_inv]
    have hx_over : abs (x / y) ≤ abs xhat + epsx := by
      have : abs (x / y) = abs x * abs (1 / y) := by
        simp [div_eq_mul_inv, abs_mul, abs_inv]
      calc
        abs (x / y) = abs x * abs (1 / y) := this
        _ ≤ abs x * 1 := by
              exact mul_le_mul_of_nonneg_left hy_inv_le_one (abs_nonneg x)
        _ = abs x := by simp
        _ ≤ abs xhat + epsx := hx_abs
    calc
      abs (qhat - x / y) ≤ abs qhat + abs (x / y) := hsub
      _ = abs xhat * abs (1 / yhat) + abs (x / y) := by simp [hq, add_comm]
      _ ≤ abs xhat * abs (1 / yhat) + (abs xhat + epsx) := by
            linarith [hx_over]
      _ = abs xhat * abs (1 / yhat) + abs xhat + epsx := by simp [add_assoc]

  -- Rounding error of the final division.
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - qhat) ≤
        neuralUlp β fexp qhat TrainingPhase.forward / 2 := by
    have : toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) =
        Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) qhat := by
      simpa [qhat, xhat, yhat] using (toSpec_div (β := β) (fexp := fexp) (rnd := rnd) xR yR)
    simpa [this] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) qhat)

  -- Combine rounding + perturbation.
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - x / y)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - qhat) +
              abs (qhat - x / y) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR)) qhat (x / y)
      _ ≤ neuralUlp β fexp qhat TrainingPhase.forward / 2 +
            (abs xhat * abs (1 / yhat) + abs xhat + epsx) := by
            exact add_le_add hround hquot
      _ = neuralUlp β fexp qhat TrainingPhase.forward / 2 +
            abs xhat * abs (1 / yhat) + abs xhat + epsx := by simp [add_assoc]
  simpa [qhat, xhat, yhat, add_assoc, add_left_comm, add_comm] using this

/--
Forward approximation bound for division under a general positive lower bound (`η ≤ y` with `η >
  0`).

This is the more general variant of `approx_div_nf_of_one_le`, making the conditioning explicit via
the factor `(1/η)`.
-/
lemma approx_div_nf_of_lb {x y : ℝ} {xR yR : R} {epsx η : ℝ}
    (hη : 0 < η) (hy : η ≤ y)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - (x / y)) ≤
      neuralUlp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR /
            toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
          TrainingPhase.forward / 2
        + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) *
            abs (1 / toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
        + (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * (1 / η) := by
  -- Notation for the embedded runtime values.
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  set qhat : ℝ := xhat / yhat

  have hy_pos : 0 < y := lt_of_lt_of_le hη hy
  have hy_ne : y ≠ 0 := ne_of_gt hy_pos

  -- Bound `|x|` by the rounded magnitude plus epsilon.
  have hx_abs : abs x ≤ abs xhat + epsx := by
    have hx' : abs (xhat - x) ≤ epsx := by
      simpa [xhat, abs_sub_comm] using hx
    have h0 : abs x ≤ abs (x - xhat) + abs xhat := by
      simpa using (abs_sub_le x xhat 0)
    have h1 : abs (x - xhat) = abs (xhat - x) := by simp [abs_sub_comm]
    have := le_trans h0 (by
      simpa [h1, add_assoc, add_left_comm, add_comm] using add_le_add_right hx' _)
    simpa [add_assoc, add_left_comm, add_comm] using this

  -- `|1/y| ≤ 1/η` since `η ≤ y` and `η > 0`.
  have hy_inv_le : abs (1 / y) ≤ (1 / η) := by
    have hdiv_pos : 0 < (1 : ℝ) / y := by
      simpa [div_eq_mul_inv] using (div_pos (show (0 : ℝ) < (1 : ℝ) by norm_num) hy_pos)
    have hdiv : (1 : ℝ) / y ≤ (1 : ℝ) / η := by
      simpa using (one_div_le_one_div_of_le hη hy)
    calc
      abs (1 / y) = (1 : ℝ) / y := abs_of_pos hdiv_pos
      _ ≤ (1 : ℝ) / η := hdiv

  have hquot :
      abs (qhat - x / y) ≤ abs xhat * abs (1 / yhat) + (abs xhat + epsx) * (1 / η) := by
    have hsub : abs (qhat - x / y) ≤ abs qhat + abs (x / y) := by
      simpa using (abs_sub_le qhat 0 (x / y))
    have hq : abs qhat = abs xhat * abs (1 / yhat) := by
      simp [qhat, div_eq_mul_inv, abs_mul, abs_inv]
    have hepsx : 0 ≤ epsx := le_trans (abs_nonneg _) hx
    have hx_over : abs (x / y) ≤ (abs xhat + epsx) * (1 / η) := by
      have : abs (x / y) = abs x * abs (1 / y) := by
        simp [div_eq_mul_inv, abs_mul, abs_inv]
      calc
        abs (x / y) = abs x * abs (1 / y) := this
        _ ≤ (abs xhat + epsx) * abs (1 / y) := by
              exact mul_le_mul_of_nonneg_right hx_abs (abs_nonneg _)
        _ ≤ (abs xhat + epsx) * (1 / η) := by
              exact mul_le_mul_of_nonneg_left hy_inv_le (add_nonneg (abs_nonneg _) hepsx)
    calc
      abs (qhat - x / y) ≤ abs qhat + abs (x / y) := hsub
      _ = abs xhat * abs (1 / yhat) + abs (x / y) := by
              simp [hq, add_comm]
      _ ≤ abs xhat * abs (1 / yhat) + (abs xhat + epsx) * (1 / η) := by
            linarith [hx_over]

  -- Rounding error of the final division.
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - qhat) ≤
        neuralUlp β fexp qhat TrainingPhase.forward / 2 := by
    have : toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) =
        Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) qhat := by
      simpa [qhat, xhat, yhat] using (toSpec_div (β := β) (fexp := fexp) (rnd := rnd) xR yR)
    simpa [this] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) qhat)

  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - x / y)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - qhat) +
              abs (qhat - x / y) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR)) qhat (x / y)
      _ ≤ neuralUlp β fexp qhat TrainingPhase.forward / 2 +
            (abs xhat * abs (1 / yhat) + (abs xhat + epsx) * (1 / η)) := by
            exact add_le_add hround hquot
      _ = neuralUlp β fexp qhat TrainingPhase.forward / 2 +
            abs xhat * abs (1 / yhat) + (abs xhat + epsx) * (1 / η) := by simp [add_assoc]
  simpa [qhat, xhat, yhat, add_assoc, add_left_comm, add_comm, mul_assoc] using this

/-- Forward approximation bound for scaling (elementwise multiply by a runtime constant `c`). -/
lemma approx_scale_nf {x : ℝ} {xR : R} {eps : ℝ} (c : R)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * c) - (x * toSpec (β := β) (fexp := fexp)
      (rnd := rnd) c)) ≤
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
        neuralUlp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
              toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
            TrainingPhase.forward / 2 := by
  have hc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c -
        toSpec (β := β) (fexp := fexp) (rnd := rnd) c) ≤ (0 : ℝ) := by
    simp
  have h :=
    approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
      (xR := xR) (yR := c) (epsx := eps) (epsy := (0 : ℝ)) hx hc
  -- Simplify away the `* 0` and `+ 0` terms.
  simpa [mul_assoc, add_assoc, add_left_comm, add_comm] using h

-- ---------------------------------------------------------------------------
-- Sum reduction bound (fold with rounded addition)
-- ---------------------------------------------------------------------------

/--
One fold step for `sum_spec` that tracks an explicit forward error budget.

State is `(accR, epsAcc)` where `accR` is the runtime accumulator and `epsAcc` bounds the absolute
error `|toSpec accR - accS|` for the corresponding spec accumulator `accS`. Each step adds:
- the incoming per-element budget `epsElem`;
- one rounding-ULP term for the addition.
-/
def sumStep (epsElem : ℝ) : (R × ℝ) → R → (R × ℝ)
  | (accR, epsAcc), xR =>
      let epsAcc' : ℝ :=
        epsAcc + epsElem +
          neuralUlp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
              TrainingPhase.forward / 2
      (accR + xR, epsAcc')

/--
Fold `sum_step` over a tensor via `tensor_foldl_spec`.

This is the shared helper behind `sum_bound` and `approxT_sum_spec`: it simultaneously computes the
runtime sum (in `.1`) and the accumulated error bound (in `.2`).
-/
def sumFoldState {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) : (R × ℝ) :=
  tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st tR

/--
Forward absolute-error bound for `sum_spec`.

`sum_bound epsElem tR` is the `.2` component of `sum_fold_state` started at 0, assuming each element
is approximated within `epsElem`. This corresponds to naive sequential summation with a rounding
term added at each step (cf. standard floating-point summation analyses).
-/
def sumBound {s : Shape} (epsElem : ℝ) (tR : Tensor R s) : ℝ :=
  (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem
    ((0 : R), neuralUlp β fexp 0 TrainingPhase.forward / 2) tR).2

-- ---------------------------------------------------------------------------
-- `tensor_foldl_spec.go` one-step unfoldings
-- ---------------------------------------------------------------------------

/--
One-step unfolding of `tensor_foldl_spec.go` when `k < n`.

We keep this as a private lemma so that downstream proofs can peel the `go` loop without using
`unfold` directly (which is brittle and can trigger simp loops).
-/
private lemma tensor_foldl_spec_go_of_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    tensorFoldlSpec.go f n s values k acc =
      tensorFoldlSpec.go f n s values (k + 1) (tensorFoldlSpec f acc (values ⟨k, hk⟩)) := by
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

/-- One-step unfolding of `tensor_foldl_spec.go` when the loop terminates (`¬ k < n`). -/
private lemma tensor_foldl_spec_go_of_not_lt {α β : Type} (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    tensorFoldlSpec.go f n s values k acc = acc := by
  rw [tensorFoldlSpec.go.eq_1]
  simp [hk]

omit [NeuralValidRndToNearest rnd] in
/--
The accumulator component of `sum_fold_state` matches the plain spec fold.

Informal: `sum_fold_state` only adds bookkeeping to `.2`; `.1` is exactly `tensor_foldl_spec (·+·)`.
-/
private lemma sum_fold_state_fst_eq {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) :
    (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st tR).1 =
      tensorFoldlSpec (· + ·) st.1 tR := by
  induction s generalizing st with
  | scalar =>
      cases tR with
      | scalar xR =>
          cases st with
          | mk accR epsAcc =>
              simp [sumFoldState, sumStep, tensorFoldlSpec]
  | dim n s ih =>
      cases tR with
      | dim valuesR =>
          -- Compare the `go` loops for the pair-valued fold vs the scalar fold.
          have go_fst :
              ∀ k (st : R × ℝ), k ≤ n →
                (tensorFoldlSpec.go (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s
                  valuesR k st).1 =
                  tensorFoldlSpec.go (· + ·) n s valuesR k st.1 := by
            intro k st hk
            induction hn : n - k generalizing k st with
            | zero =>
                have hk' : k = n := by
                  have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
                  exact Nat.le_antisymm hk this
                subst k
                simp [tensor_foldl_spec_go_of_not_lt]
            | succ m ih_go =>
                have hlt : k < n := by
                  have : 0 < n - k := by simp [hn]
                  exact Nat.sub_pos_iff_lt.mp this
                have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
                rw [tensor_foldl_spec_go_of_lt (f := sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                  (values := valuesR) (k := k) (acc := st) hlt]
                rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesR) (k := k) (acc := st.1) hlt]
                have h_next : n - (k + 1) = m := by
                  rw [Nat.sub_succ, hn]
                  rfl
                -- The recursive fold over the sub-tensor updates only the accumulator in `.1`.
                have h_step :
                    (tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st
                        (valuesR ⟨k, hlt⟩)).1 =
                      tensorFoldlSpec (· + ·) st.1 (valuesR ⟨k, hlt⟩) := by
                  simpa [sumFoldState] using
                    ih (st := st) (tR := valuesR ⟨k, hlt⟩)
                have := ih_go (k := k + 1)
                  (st := tensorFoldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                    st
                    (valuesR ⟨k, hlt⟩)) hk1
                simpa [h_next, h_step] using this
          have h0 := go_fst (k := 0) (st := st) (by exact Nat.zero_le n)
          simpa [sumFoldState, tensorFoldlSpec] using h0

/--
Core summation induction: `sum_fold_state` preserves a forward bound.

In words: if the current accumulator `st.1` approximates a spec value `accS` within
  `st.2`,
and each tensor entry is approximated within `epsElem`, then folding `sum_fold_state` over the
  tensor
produces an accumulator whose spec value is within the final `.2` budget of the corresponding spec
fold.
-/
private theorem approx_sum_fold_state {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {accS : ℝ} {st : R × ℝ} {epsElem : ℝ},
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsElem →
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).1 -
              tensorFoldlSpec (· + ·) accS xS) ≤
          (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).2 := by
  intro xS xR accS st epsElem hAcc hx
  induction s generalizing accS st with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              cases st with
              | mk accR epsAcc =>
                  have hx' :=
                    (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                      rnd))
                      (x := x) (xR := xR) (eps := epsElem)).1 hx
                  have h :=
                    approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
                      (x := accS) (y := x) (xR := accR) (yR := xR)
                      (epsx := epsAcc) (epsy := epsElem) hAcc hx'
                  simpa [sumFoldState, sumStep, tensorFoldlSpec, add_assoc, add_left_comm,
                    add_comm] using h
  | dim n s ih =>
      cases xS with
      | dim valuesS =>
          cases xR with
          | dim valuesR =>
              -- Prove the accumulator/error invariant for the internal `go` loops.
              have go_sound :
                  ∀ k (accS : ℝ) (st : R × ℝ), k ≤ n →
                    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
                      abs
                          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                              (tensorFoldlSpec.go
                                  (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s
                                    valuesR k st).1 -
                            tensorFoldlSpec.go (· + ·) n s valuesS k accS) ≤
                        (tensorFoldlSpec.go
                            (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) n s valuesR k
                              st).2 := by
                intro k accS st hk hAcc
                induction hn : n - k generalizing k accS st with
                | zero =>
                    have hk' : k = n := by
                      have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
                      exact Nat.le_antisymm hk this
                    subst k
                    simpa [tensor_foldl_spec_go_of_not_lt] using hAcc
                | succ m ih_go =>
                    have hlt : k < n := by
                      have : 0 < n - k := by simp [hn]
                      exact Nat.sub_pos_iff_lt.mp this
                    have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
                    rw [tensor_foldl_spec_go_of_lt (f := sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem)
                      (values := valuesR) (k := k) (acc := st) hlt]
                    rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesS) (k := k) (acc := accS) hlt]
                    have h_next : n - (k + 1) = m := by
                      rw [Nat.sub_succ, hn]
                      rfl
                    -- Apply the shape IH to fold over the current slice `valuesR ⟨k, hlt⟩`.
                    have hx_k :=
                      approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (xS := Tensor.dim valuesS) (xR := Tensor.dim valuesR) (eps := epsElem) hx
                          ⟨k, hlt⟩
                    have h_step :=
                      ih (xS := valuesS ⟨k, hlt⟩) (xR := valuesR ⟨k, hlt⟩) (accS := accS) (st := st)
                        hAcc hx_k
                    -- Use IH on the tail of the outer `go`.
                    have htail :=
                      ih_go (k := k + 1)
                        (accS := tensorFoldlSpec (· + ·) accS (valuesS ⟨k, hlt⟩))
                        (st := tensorFoldlSpec
                          (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st (valuesR ⟨k,
                            hlt⟩))
                        hk1 (by simpa [h_next] using h_step)
                    simpa [h_next] using htail
              have h0 := go_sound (k := 0) (accS := accS) (st := st) (by exact Nat.zero_le n) hAcc
              simpa [sumFoldState, tensorFoldlSpec] using h0

/--
Forward approximation bound for `sum_spec` over an arbitrary tensor shape.

If `xR` approximates `xS` elementwise within `eps`, then the scalar sums `sum_spec xR` and
`sum_spec xS` differ by at most `sum_bound eps xR`.
-/
theorem approxT_sum_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Tensor.scalar (sumSpec (α := ℝ) (s := s) xS))
          (Tensor.scalar (sumSpec (α := R) (s := s) xR))
          (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR) := by
  intro xS xR eps hx
  -- Start from accumulator 0 with a conservative rounding bound.
  let initEps : ℝ := neuralUlp β fexp 0 TrainingPhase.forward / 2
  have hAcc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : ℝ)) ≤ initEps := by
    -- `toSpec (0 : R)` is `roundR 0`, so this is exactly the single-step rounding bound.
    simpa [initEps, NFBackend.toSpec, TorchLean.Floats.NF.toReal,
      Proofs.RuntimeRoundingApprox.roundR,
      TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ))
  have h :=
    approx_sum_fold_state (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := xS) (xR := xR) (accS := (0 : ℝ)) (st := ((0 : R), initEps)) (epsElem := eps) hAcc hx
  -- Relate the accumulator component to `sum_spec`.
  have hfst :
      (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).1 =
        sumSpec (α := R) (s := s) xR := by
    simpa [sumSpec] using
      (sum_fold_state_fst_eq (β := β) (fexp := fexp) (rnd := rnd) (s := s) (epsElem := eps)
        (st := ((0 : R), initEps)) (tR := xR))
  -- Wrap back into `approxT` on scalar tensors.
  refine (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (x := sumSpec (α := ℝ) (s := s) xS) (xR := sumSpec (α := R) (s := s) xR)
      (eps := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)).2 ?_
  have h' :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := s) xR) -
            sumSpec (α := ℝ) (s := s) xS) ≤
        (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).2
          := by
    simpa [hfst, sumSpec] using h
  simpa [sumBound, sumFoldState, initEps] using h'

-- Elementwise bounds lifted to tensors via `linf_norm`.

/--
Per-entry bound tensor for addition.

`add_bound_tensor epsx epsy xR yR` computes an elementwise error budget for `xR + yR`. Its
  `linf_norm`
is used as the output epsilon in `approxT_add_spec`.
-/
def addBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + neuralUlp β fexp (a + b) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for subtraction.

Analogous to `add_bound_tensor`, but for `xR - yR` (and the corresponding spec subtraction).
-/
def subBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + neuralUlp β fexp (a - b) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for multiplication.

This is the elementwise lifting of the scalar bound `approx_mul_nf`, tracking first-order error
propagation plus one rounding term.
-/
def mulBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (abs a + epsx) * epsy + (abs b + epsy) * epsx + neuralUlp β fexp (a * b)
        TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for scaling by a runtime constant.

`scale_bound_tensor eps c xR` bounds the error of `xR * c` assuming the input is approximated within
`eps` and treating `c` as exact (relative to its own `toSpec` value).
-/
def scaleBoundTensor {s : Shape} (eps : ℝ) (c : R) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
        neuralUlp β fexp (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c) TrainingPhase.forward
          / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for negation. -/
def negBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + neuralUlp β fexp (-a) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for absolute value. -/
def absBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + neuralUlp β fexp (abs a) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for exponentiation (`exp`).

This matches `approx_exp_nf`: a mean-value-theorem bound on the real `exp` plus one rounding term.
-/
def expBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      Real.exp a + Real.exp (a + eps) + neuralUlp β fexp (Real.exp a) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for hyperbolic tangent (`tanh`).

Currently uses the coarse unconditional bound from `approx_tanh_nf` (boundedness of `tanh`).
-/
def tanhBoundTensor {s : Shape} (_eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => (2 : ℝ) + neuralUlp β fexp (Real.tanh a) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

-- Safe log (clamped) bound.

/--
Per-entry bound tensor for `safeLog`.

`safeLog_bound_tensor ε eps xR` is the elementwise bound used by `approxT_safeLog_spec`, combining a
`(1/ε)` Lipschitz propagation term with one rounding-ULP term.
-/
def safeLogBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (1 / ε) * eps +
        neuralUlp β fexp (safeLog (ε := ε) a) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxT` bound for clamped log (`safeLog`) lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeLog_nf`, using
  `approxT_map_spec_of_scalar_bound`.
-/
theorem approxT_safeLog_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (safeLog (ε := ε)) xS)
          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps xR))
            := by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := x) (xR := xR) (eps := eps)).1 hx
              have h :=
                approx_safeLog_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps :=
                  eps) (ε := ε)
                  hε hx'
              have hle :
                  abs
                      (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
                        safeLog (ε := ε) x) ≤
                    linfNorm
                      (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.scalar)
                        ε eps
                        (Tensor.scalar xR)) := by
                refine le_trans h ?_
                -- `linf_norm` of a scalar tensor is the `abs` of its entry.
                simpa [safeLogBoundTensor, tensorToSpec, Spec.mapTensor, mapSpec, linfNorm,
                  RuntimeApprox.linfNorm, tensorDistance,
                    NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  tensorLinfNorm, safeLog] using le_abs_self _
              -- Wrap back into `approxT`.
              exact
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := safeLog (ε := ε) x)
                  (xR := safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR)
                  (eps := linfNorm
                    (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.scalar) ε
                      eps
                      (Tensor.scalar xR)))).2 hle
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm
                  (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim n s) ε
                    eps
                    (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using
                  (linf_norm_nonneg (t := safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) ε eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ B := by
                intro i
                have hx_i := approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                  := rnd)) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm
                        (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps
                          (xRf i)) ≤
                      B := by
                  simpa [B, safeLogBoundTensor, Spec.mapTensor] using
                    (linf_norm_le_get_dim
                      (t := safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) ε eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ linfNorm
                        (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps
                          (xRf i)) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                          i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (safeLog (ε := ε)) (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (safeLogR (β := β) (fexp := fexp) (rnd :=
                          rnd) ε)
                          (Tensor.dim xRf)))
                    ≤ B := by
                simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                    Spec.mapTensor, mapSpec] using hfold
              simpa [approxT, approxWith, B] using this

-- ---------------------------------------------------------------------------
-- Softplus / safe_log (smooth) via `safeLog`
-- ---------------------------------------------------------------------------

/-!
`NFBackend.safeLog` is a clamped log surrogate `log (max x ε)` with an unconditional forward bound.

For smooth activations that *use* `log` (notably `softplus` and `safe_log`), we route the outer log
through `safeLog` at a known lower bound:

* `softplus(x) = log(1 + exp x)` and `1 + exp x ≥ 1`, so `softplus = safeLog 1 (1 + exp x)`;
* `safe_log(x) = log(softplus(x) + ε)` and `softplus(x) + ε ≥ ε`, so `safe_log = safeLog ε
  (softplus(x) + ε)`.

This avoids needing a separate `log` approximation lemma while remaining extensionally equal on
`ℝ` for the intended arguments.
-/

/-- Half-ULP rounding budget at the scalar value `1`, used in the softplus helper bounds. -/
def oneEps : ℝ :=
  neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2

/--
Scalar forward-error envelope for `exp`.

This packages the exact value, the perturbed value at `a + eps`, and the final rounding budget
into one reusable bound for the later softplus/safe-log proofs.
-/
def expBoundScalar (a eps : ℝ) : ℝ :=
  Real.exp a + Real.exp (a + eps) +
    neuralUlp β fexp (Real.exp a) TrainingPhase.forward / 2

/-- Rounded representation of the scalar constant `1` at the `NF` backend. -/
def oneHat : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ)

/-- Rounded representation of `exp a` at the `NF` backend. -/
def expHat (a : ℝ) : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp a)

/-- Rounded surrogate for the inner `1 + exp a` term appearing in `softplus`. -/
def addHatSoftplus (a : ℝ) : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (oneHat (β := β) (fexp
    := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd) a)

/-- Forward-error envelope for the rounded `1 + exp a` subexpression used by `softplus`. -/
def addBoundSoftplus (a eps : ℝ) : ℝ :=
  oneEps (β := β) (fexp := fexp) + expBoundScalar (β := β) (fexp := fexp) a eps +
    neuralUlp β fexp
        (oneHat (β := β) (fexp := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd)
          a)
        TrainingPhase.forward / 2

/-- Unconditional scalar forward-error bound for `softplus`, via the `safeLog` factorization. -/
def softplusBoundScalar (a eps : ℝ) : ℝ :=
  addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd) a eps +
    neuralUlp β fexp
        (safeLog (ε := (1 : ℝ)) (addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd) a))
        TrainingPhase.forward / 2

/-- `softplus` implemented by `safeLog 1 (1 + exp x)` at the `NF` backend. -/
def softplusR (xR : R) : R :=
  let yR : R := (1 : R) + MathFunctions.exp xR
  safeLogR (β := β) (fexp := fexp) (rnd := rnd) (ε := (1 : ℝ)) yR

/--
Forward approximation bound for `softplus` in `NF`.

We treat `softplus(x) = log(1 + exp x)` as `safeLog 1 (1 + exp x)` (since `1 + exp x ≥ 1`) and then
compose the scalar bounds for `exp`, `+`, and `safeLog`.
-/
  lemma approx_softplus_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (softplusR (β := β) (fexp := fexp) (rnd := rnd)
      xR) -
          Activation.Math.softplusSpec (α := ℝ) x) ≤
      softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps := by
  -- Step 1: exp approximation.
  have hexp :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp xR) - Real.exp x) ≤
        expBoundScalar (β := β) (fexp := fexp) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
          eps := by
    simpa [expBoundScalar] using (approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR
      := xR) (eps := eps) hx)

  -- Step 2: `1 + exp x` approximation.
  let oneR : R := (1 : R)
  have hone :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR - (1 : ℝ)) ≤
        oneEps (β := β) (fexp := fexp) := by
    simpa [oneEps, NFBackend.toSpec, TorchLean.Floats.NF.toReal,
      Proofs.RuntimeRoundingApprox.roundR,
      TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, TorchLean.Floats.NF.instOne] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))

  let yR : R := oneR + MathFunctions.exp xR
  let y : ℝ := (1 : ℝ) + Real.exp x

  have hy :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤
        addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps := by
    have hadd :=
      approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
        (x := (1 : ℝ)) (y := Real.exp x)
        (xR := oneR) (yR := MathFunctions.exp xR)
        (epsx := oneEps (β := β) (fexp := fexp))
        (epsy := expBoundScalar (β := β) (fexp := fexp) (toSpec (β := β) (fexp := fexp) (rnd :=
          rnd) xR) eps)
        hone hexp
    -- Rewrite to match the local definitions.
    simpa [yR, y, addBoundSoftplus, expBoundScalar, oneHat, expHat,
      NFBackend.toSpec, TorchLean.Floats.NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
      TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal, TorchLean.Floats.NF.instOne,
      toSpec_exp (β := β) (fexp := fexp) (rnd := rnd) xR] using hadd

  -- Step 3: `safeLog 1` turns this into `softplus`.
  have hlog :=
    approx_safeLog_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := y) (xR := yR) (eps := addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd) (toSpec (β
        := β) (fexp := fexp) (rnd := rnd) xR) eps)
      (ε := (1 : ℝ)) (hε := by norm_num) hy

  have hy_ge : (1 : ℝ) ≤ y := by
    have : 0 ≤ Real.exp x := by simpa using (Real.exp_nonneg x)
    linarith

  have hsimp : safeLog (ε := (1 : ℝ)) y = Activation.Math.softplusSpec (α := ℝ) x := by
    calc
      safeLog (ε := (1 : ℝ)) y = Real.log y := by
        simp [NFBackend.safeLog, hy_ge]
      _ = Real.log ((1 : ℝ) + Real.exp x) := by
        simp [y]
      _ = Activation.Math.softplusSpec (α := ℝ) x := rfl

  -- `softplusR` is exactly `safeLogR 1 (1 + exp xR)`.
  simpa [softplusR, yR, hsimp, softplusBoundScalar, addHatSoftplus] using hlog

/--
Per-entry bound tensor for `softplusR`.

This is the elementwise lifting of `softplus_bound_scalar`, used with `linf_norm` in
`approxT_softplus_spec`.
-/
def softplusBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec (fun a => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxT` bound for `softplus` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_softplus_nf`, built via
  `approxT_map_spec_of_scalar_bound`.
-/
theorem approxT_softplus_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) xS)
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR))
            := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Activation.Math.softplusSpec (α := ℝ))
      (fR := softplusR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun a eps => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_softplus_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR)
          (eps := eps) hx))
  simpa [softplusBoundTensor] using h

-- ---------------------------------------------------------------------------
-- safe_log (smooth activation): `log(softplus(x) + ε)`
-- ---------------------------------------------------------------------------

/-- Runtime implementation of `safe_log` as a single rounded primitive. -/
def safeLogSoftplusR (ε : ℝ) (xR : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (Activation.Math.safeLogSpec (α := ℝ)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) ε)

private lemma sigmoid_spec_nonneg (x : ℝ) :
    0 ≤ Activation.Math.sigmoidSpec (α := ℝ) x := by
  unfold Activation.Math.sigmoidSpec
  -- `MathFunctions.exp` is `Real.exp` on `ℝ`.
  rw [Proofs.mathfunc_exp_eq_rexp (-x)]
  have hden : 0 < (1 : ℝ) + Real.exp (-x) := by
    linarith [Real.exp_pos (-x)]
  have : 0 < (1 : ℝ) / ((1 : ℝ) + Real.exp (-x)) :=
    div_pos (by norm_num) hden
  simpa using le_of_lt this

private lemma sigmoid_spec_le_one (x : ℝ) :
    Activation.Math.sigmoidSpec (α := ℝ) x ≤ 1 := by
  unfold Activation.Math.sigmoidSpec
  simp [Proofs.mathfunc_exp_eq_rexp]
  have hden : (1 : ℝ) ≤ (1 : ℝ) + Real.exp (-x) := by
    have : 0 ≤ Real.exp (-x) := le_of_lt (Real.exp_pos (-x))
    linarith
  have := one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < (1 : ℝ)) hden
  simpa using this

private lemma softplus_spec_nonneg (x : ℝ) :
    0 ≤ Activation.Math.softplusSpec (α := ℝ) x := by
  -- `softplus(x) = log(1 + exp(x)) ≥ 0`.
  unfold Activation.Math.softplusSpec
  simp [Proofs.mathfunc_exp_eq_rexp]
  have h1 : (1 : ℝ) ≤ (1 : ℝ) + Real.exp x := by
    linarith [Real.exp_pos x]
  simpa using (Real.log_nonneg h1)

/--
`safe_log_spec` is `(1/ε)`-Lipschitz for `ε > 0`.

This is the analytic heart of `approx_safe_log_nf`: it bounds how much the spec `safe_log` output
  can
change when the input changes by `|u - v|`.
-/
private lemma abs_safe_log_sub_safe_log_le_one_div_mul_abs_sub {ε u v : ℝ}
    (hε : 0 < ε) :
    abs (Activation.Math.safeLogSpec (α := ℝ) u ε - Activation.Math.safeLogSpec (α := ℝ) v ε) ≤
      (1 / ε) * abs (u - v) := by
  -- Mean value theorem on `ℝ` (derivative bounded by `1/ε` everywhere).
  have hf :
      ∀ x ∈ (Set.univ : Set ℝ),
        HasDerivWithinAt (fun y : ℝ => Activation.Math.safeLogSpec (α := ℝ) y ε)
          (Activation.Math.safeLogDerivSpec (α := ℝ) x ε) (Set.univ : Set ℝ) x := by
    intro x _hx
    exact (Proofs.safe_log_deriv_correct (x := x) (ε := ε) hε).hasDerivWithinAt (s := (Set.univ :
      Set ℝ))

  have hbound :
      ∀ x ∈ (Set.univ : Set ℝ),
        ‖Activation.Math.safeLogDerivSpec (α := ℝ) x ε‖ ≤ (1 / ε) := by
    intro x _hx
    have hsig0 : 0 ≤ Activation.Math.sigmoidSpec (α := ℝ) x := sigmoid_spec_nonneg x
    have hsig1 : Activation.Math.sigmoidSpec (α := ℝ) x ≤ 1 := sigmoid_spec_le_one x
    have hsoft0 : 0 ≤ Activation.Math.softplusSpec (α := ℝ) x := softplus_spec_nonneg x
    have hden_ge : ε ≤ Activation.Math.softplusSpec (α := ℝ) x + ε := by linarith
    have hden_pos : 0 < Activation.Math.softplusSpec (α := ℝ) x + ε :=
      lt_of_lt_of_le hε hden_ge

    -- Unfold the derivative and bound it by `1/ε`.
    have hderiv_le :
        abs (Activation.Math.safeLogDerivSpec (α := ℝ) x ε) ≤ (1 / ε) := by
      -- `safe_log'(x) = sigmoid(x) / (softplus(x) + ε)` and `0 ≤ sigmoid ≤ 1`, `softplus+ε ≥ ε`.
      have hone_div :
          (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε hden_ge)
      have hsig_div :
          Activation.Math.sigmoidSpec (α := ℝ) x / (Activation.Math.softplusSpec (α := ℝ) x + ε) ≤
            (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        exact div_le_div_of_nonneg_right hsig1 (le_of_lt hden_pos)
      have hpos :
          0 ≤
            Activation.Math.sigmoidSpec (α := ℝ) x /
              (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        exact div_nonneg hsig0 (le_of_lt hden_pos)
      have habs :
          abs
              (Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε))
            =
          Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε) :=
        abs_of_nonneg hpos
      -- Rewrite the derivative to match this form.
      have hsimp :
          Activation.Math.safeLogDerivSpec (α := ℝ) x ε =
            Activation.Math.sigmoidSpec (α := ℝ) x /
              (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        simp [Activation.Math.safeLogDerivSpec, Activation.Math.softplusDerivSpec]
      -- Combine the inequalities.
      calc
        abs (Activation.Math.safeLogDerivSpec (α := ℝ) x ε)
            =
          abs
              (Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε)) := by
              simp [hsimp]
        _ = Activation.Math.sigmoidSpec (α := ℝ) x / (Activation.Math.softplusSpec (α := ℝ) x + ε)
          := habs
        _ ≤ (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) := hsig_div
        _ ≤ (1 : ℝ) / ε := hone_div
        _ = (1 / ε) := by ring

    -- `‖·‖` on `ℝ` is `abs`.
    simpa [Real.norm_eq_abs] using hderiv_le

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun y : ℝ => Activation.Math.safeLogSpec (α := ℝ) y ε)
      (f' := fun y : ℝ => Activation.Math.safeLogDerivSpec (α := ℝ) y ε)
      (s := (Set.univ : Set ℝ))
      (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_univ : Convex ℝ (Set.univ : Set ℝ))
      (by trivial) (by trivial)

  simpa [Real.norm_eq_abs, abs_sub_comm] using hmv

/--
Forward approximation bound for the smooth `safe_log` activation in `NF`.

`safe_log` is defined as `log(softplus(x) + ε)`, which is globally well-defined for `ε > 0`. The
proof combines:
- one rounding step for `safe_logR` (defined as `NF.ofReal (safe_log_spec ...)`);
- a `(1/ε)` Lipschitz bound for the spec function (via mean value theorem + derivative bound).
-/
lemma approx_safe_log_nf {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
          Activation.Math.safeLogSpec (α := ℝ) x ε) ≤
      (1 / ε) * eps +
        neuralUlp β fexp
            (Activation.Math.safeLogSpec (α := ℝ)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) ε)
            TrainingPhase.forward / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hx' : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.Math.safeLogSpec (α := ℝ) xhat ε) ≤
        neuralUlp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) TrainingPhase.forward / 2
          := by
    -- `safe_logR` rounds the real `safe_log_spec`.
    simpa [safeLogSoftplusR, xhat, toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd)
          (Activation.Math.safeLogSpec (α := ℝ) xhat ε))

  have hdiff :
      abs (Activation.Math.safeLogSpec (α := ℝ) xhat ε - Activation.Math.safeLogSpec (α := ℝ) x
        ε) ≤
        (1 / ε) * eps := by
    have hL :=
      abs_safe_log_sub_safe_log_le_one_div_mul_abs_sub (ε := ε) (u := xhat) (v := x) hε
    have hcoef : 0 ≤ (1 / ε) := by
      exact le_of_lt (one_div_pos.2 hε)
    have hscale : (1 / ε) * abs (xhat - x) ≤ (1 / ε) * eps :=
      mul_le_mul_of_nonneg_left hx' hcoef
    exact le_trans hL hscale

  -- Triangle inequality + reorder.
  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.Math.safeLogSpec (α := ℝ) x ε)
          ≤
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
              Activation.Math.safeLogSpec (α := ℝ) xhat ε) +
          abs
              (Activation.Math.safeLogSpec (α := ℝ) xhat ε -
                Activation.Math.safeLogSpec (α := ℝ) x ε) := by
            simpa [sub_eq_add_neg, add_assoc] using
              abs_sub_le
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR))
                (Activation.Math.safeLogSpec (α := ℝ) xhat ε)
                (Activation.Math.safeLogSpec (α := ℝ) x ε)
      _ ≤
        neuralUlp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) TrainingPhase.forward / 2
          +
          (1 / ε) * eps := by
            exact add_le_add hround hdiff
      _ = (1 / ε) * eps +
        neuralUlp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) TrainingPhase.forward / 2
          := by
            ring
  simpa [xhat] using this

/--
Per-entry bound tensor for `safe_log`.

This is the elementwise lifting of `approx_safe_log_nf`'s bound.
-/
def safeLogSoftplusBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (1 / ε) * eps +
        neuralUlp β fexp (Activation.Math.safeLogSpec (α := ℝ) a ε) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxT` bound for `safe_log` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safe_log_nf`, built via
  `approxT_map_spec_of_scalar_bound`.
-/
theorem approxT_safe_log_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε) xS)
          (mapSpec (s := s) (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm (safeLogSoftplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps xR))
            := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := fun x => Activation.Math.safeLogSpec (α := ℝ) x ε)
      (fR := safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a eps =>
        (1 / ε) * eps +
          neuralUlp β fexp (Activation.Math.safeLogSpec (α := ℝ) a ε) TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_safe_log_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (xR := xR) (eps := eps) (ε := ε) hε hx))
  simpa [safeLogSoftplusBoundTensor] using h

-- ---------------------------------------------------------------------------
-- Safe division (clamped): `x / max y ε`
-- ---------------------------------------------------------------------------

/-- Spec-side safe division with a clamped denominator. -/
def safeDiv (ε : ℝ) (x y : ℝ) : ℝ :=
  x / max y ε

/-- Runtime implementation of `safeDiv` as a single rounded primitive. -/
def safeDivR (ε : ℝ) (xR yR : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeDiv (ε := ε)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR))

/--
Forward approximation bound for `safeDiv` in `NF`.

`safeDiv ε x y = x / max y ε` clamps the denominator away from 0. For `ε > 0`, this yields an
unconditional bound with explicit `(1/ε)` and `(1/ε^2)` sensitivity terms plus one rounding-ULP
  term.
-/
lemma approx_safeDiv_nf {x y : ℝ} {xR yR : R} {epsx epsy ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
          safeDiv (ε := ε) x y) ≤
      (1 / ε) * epsx +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp
            (safeDiv (ε := ε)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR))
            TrainingPhase.forward / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  set uhat : ℝ := max yhat ε
  set u : ℝ := max y ε

  have uhat_ge : ε ≤ uhat := le_max_right _ _
  have u_ge : ε ≤ u := le_max_right _ _
  have uhat_pos : 0 < uhat := lt_of_lt_of_le hε uhat_ge
  have u_pos : 0 < u := lt_of_lt_of_le hε u_ge

  have hx' : abs (xhat - x) ≤ epsx := by
    simpa [xhat, abs_sub_comm] using hx
  have hy' : abs (yhat - y) ≤ epsy := by
    simpa [yhat, abs_sub_comm] using hy

  have hx_abs : abs x ≤ abs xhat + epsx := by
    have h0 : abs x ≤ abs (x - xhat) + abs xhat := by
      simpa using (abs_sub_le x xhat 0)
    have h1 : abs (x - xhat) = abs (xhat - x) := by simp [abs_sub_comm]
    have h2 : abs (x - xhat) ≤ epsx := by simpa [h1] using hx'
    have := le_trans h0 (by
      simpa [add_assoc, add_left_comm, add_comm] using add_le_add_right h2 (abs xhat))
    simpa [add_assoc, add_left_comm, add_comm] using this

  have hmax : abs (uhat - u) ≤ epsy := by
    have hLip : abs (max yhat ε - max y ε) ≤ abs (yhat - y) := by
      simpa [abs_sub_comm] using (abs_max_sub_max_le_abs yhat y ε)
    exact le_trans (by simpa [uhat, u, abs_sub_comm] using hLip) hy'

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) xhat yhat) ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) TrainingPhase.forward / 2 := by
    simpa [safeDivR, safeDiv, xhat, yhat, toSpec, TorchLean.Floats.NF.toReal,
      TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd)
          (safeDiv (ε := ε) xhat yhat))

  have hdiff :
      abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) ≤
        (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
    -- Split numerator and denominator effects.
    have hsplit :
        abs (xhat / uhat - x / u) ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := by
      -- `|a-c| ≤ |a-b| + |b-c|` with `b = x/uhat`.
      simpa [sub_eq_add_neg, add_assoc] using
        abs_sub_le (xhat / uhat) (x / uhat) (x / u)

    have hnum :
        abs (xhat / uhat - x / uhat) ≤ (1 / ε) * epsx := by
      have hsub : xhat / uhat - x / uhat = (xhat - x) / uhat := by
        simpa using (sub_div xhat x uhat).symm
      have hinv : (1 : ℝ) / uhat ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε uhat_ge)
      have hcoef : 0 ≤ (1 : ℝ) / ε := by exact le_of_lt (one_div_pos.2 hε)
      calc
        abs (xhat / uhat - x / uhat)
            = abs ((xhat - x) / uhat) := by simp [hsub]
        _ = abs (xhat - x) * ((1 : ℝ) / uhat) := by
                simp [div_eq_mul_inv, abs_mul, abs_inv, abs_of_pos uhat_pos]
        _ ≤ abs (xhat - x) * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_left hinv (abs_nonneg _)
        _ ≤ epsx * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_right hx' hcoef
        _ = (1 / ε) * epsx := by ring

    have hden :
        abs (x / uhat - x / u) ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
      have hsub : x / uhat - x / u = x * ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
        simp [div_eq_mul_inv, sub_eq_add_neg, mul_add, mul_comm]
      -- Bound `|1/uhat - 1/u|` using algebra and the `max` Lipschitz bound.
      have h_inv :
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u) ≤ epsy / (ε * ε) := by
        have hu0 : uhat ≠ 0 := ne_of_gt uhat_pos
        have hv0 : u ≠ 0 := ne_of_gt u_pos
        have hiden :
            (1 : ℝ) / uhat - (1 : ℝ) / u = (u - uhat) / (uhat * u) := by
          field_simp [hu0, hv0]
        have hprod_ge : (ε * ε) ≤ uhat * u := by
          have : ε ≤ uhat := uhat_ge
          have : ε ≤ u := u_ge
          nlinarith
        have hprod_pos : 0 < uhat * u := mul_pos uhat_pos u_pos
        have hprod_inv :
            (1 : ℝ) / (uhat * u) ≤ (1 : ℝ) / (ε * ε) := by
          simpa [one_div] using (one_div_le_one_div_of_le (mul_pos hε hε) hprod_ge)
        have hprod_inv_nonneg : 0 ≤ (1 : ℝ) / (uhat * u) := by
          exact le_of_lt (one_div_pos.2 hprod_pos)
        calc
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u)
              = abs ((u - uhat) / (uhat * u)) := by
                  simpa using congrArg abs hiden
          _ = abs (u - uhat) / (uhat * u) := by
                  simpa [abs_of_pos hprod_pos] using (abs_div (u - uhat) (uhat * u))
          _ = abs (u - uhat) * ((1 : ℝ) / (uhat * u)) := by
                  simp [div_eq_mul_inv]
          _ ≤ abs (u - uhat) * ((1 : ℝ) / (ε * ε)) := by
                exact mul_le_mul_of_nonneg_left hprod_inv (abs_nonneg _)
          _ ≤ epsy * ((1 : ℝ) / (ε * ε)) := by
                have : abs (u - uhat) ≤ epsy := by simpa [abs_sub_comm] using hmax
                exact mul_le_mul_of_nonneg_right this (by
                  have : 0 < (1 : ℝ) / (ε * ε) := by
                    exact one_div_pos.2 (mul_pos hε hε)
                  exact le_of_lt this)
          _ = epsy / (ε * ε) := by
                simp [div_eq_mul_inv, mul_comm]

      calc
        abs (x / uhat - x / u)
            = abs (x * ((1 : ℝ) / uhat - (1 : ℝ) / u)) := by simp [hsub]
        _ = abs x * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              simp [abs_mul]
        _ ≤ (abs xhat + epsx) * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              exact mul_le_mul_of_nonneg_right hx_abs (abs_nonneg _)
        _ ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
              have epsx_nonneg : 0 ≤ epsx := le_trans (abs_nonneg _) hx'
              have hsum_nonneg : 0 ≤ abs xhat + epsx := add_nonneg (abs_nonneg _) epsx_nonneg
              exact mul_le_mul_of_nonneg_left h_inv hsum_nonneg

    -- Combine.
    have hadd :=
      calc
        abs (xhat / uhat - x / u)
            ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := hsplit
        _ ≤ (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
              exact add_le_add hnum hden
        _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by rfl
    -- Rewrite `safeDiv`.
    simpa [safeDiv, uhat, u] using hadd

  -- Final triangle inequality: rounding + input sensitivity.
  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) x y)
          ≤
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
              safeDiv (ε := ε) xhat yhat) +
          abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) := by
            simpa [sub_eq_add_neg, add_assoc] using
              abs_sub_le
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR))
                (safeDiv (ε := ε) xhat yhat)
                (safeDiv (ε := ε) x y)
      _ ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) TrainingPhase.forward / 2 +
          ((1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε))) := by
            exact add_le_add hround hdiff
      _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) TrainingPhase.forward / 2 := by
            ring

  simpa [xhat, yhat] using this

/--
Per-entry bound tensor for `safeDiv`.

This is the elementwise lifting of `approx_safeDiv_nf`'s bound (with a max-clamped denominator).
-/
def safeDivBoundTensor {s : Shape} (ε epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (1 / ε) * epsx +
        (abs a + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) a b) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
`approxT` bound for `safeDiv` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeDiv_nf`, built via
  `approxT_map2_spec_of_scalar_bound`.
-/
theorem approxT_safeDiv_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (map2Spec (s := s) (safeDiv (ε := ε)) xS yS)
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) xR yR)
          (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε epsx epsy
            xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxT_map2_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := safeDiv (ε := ε))
      (fR := safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a b epsx epsy =>
        (1 / ε) * epsx +
          (abs a + epsx) * (epsy / (ε * ε)) +
          neuralUlp β fexp (safeDiv (ε := ε) a b) TrainingPhase.forward / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_safeDiv_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) (ε := ε) hε hx
              hy))
  simpa [safeDivBoundTensor] using h

-- Sigmoid / Softmax (elementwise logistic) bounds.

/--
Scalar forward bound for `sigmoid` in `NF`.

`sigmoid(x) = 1 / (1 + exp(-x))` is implemented using the existing bounds for `exp`, `+`, and
division (with the denominator lower-bounded by 1).
-/
def sigmoidBoundScalar (xR : R) : ℝ :=
  let oneR : R := (1 : R)
  let denomR : R := oneR + MathFunctions.exp (-xR)
  let oneHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR
  let denomHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR
  let qhat : ℝ := oneHat / denomHat
  neuralUlp β fexp qhat TrainingPhase.forward / 2 +
    abs oneHat * abs (1 / denomHat) + abs oneHat + oneEps (β := β) (fexp := fexp)

/-- Per-entry bound tensor for `sigmoid`. -/
def sigmoidBoundTensor {s : Shape} (_eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Spec.mapTensor (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd)) xR

/--
`approxT` bound for `sigmoid` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `sigmoid_bound_scalar` (scalar case) and the usual
componentwise `linf_norm` lifting (dimension case).
-/
theorem approxT_sigmoid_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
            by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              -- `sigmoid` is just a division `1 / (1 + exp (-x))`.
              let oneR : R := (1 : R)
              let denomR : R := oneR + MathFunctions.exp (-xR)
              let y : ℝ := (1 : ℝ) + Real.exp (-x)
              have hy : (1 : ℝ) ≤ y := by
                have : 0 ≤ Real.exp (-x) := by simpa using (Real.exp_nonneg (-x))
                linarith
              have hone :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR - (1 : ℝ)) ≤
                    oneEps (β := β) (fexp := fexp) := by
                simpa [oneEps, NFBackend.toSpec, TorchLean.Floats.NF.toReal,
                  Proofs.RuntimeRoundingApprox.roundR,
                  TorchLean.Floats.NF.roundR, TorchLean.Floats.NF.ofReal,
                    TorchLean.Floats.NF.instOne] using
                  (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd :=
                    rnd) (1 : ℝ))
              have hdiv :=
                approx_div_nf_of_one_le (β := β) (fexp := fexp) (rnd := rnd)
                  (x := (1 : ℝ)) (y := y) (xR := oneR) (yR := denomR)
                  (epsx := oneEps (β := β) (fexp := fexp)) hy hone
              have hb :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.sigmoidSpec (α := R) xR) -
                        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
                    sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) xR := by
                simpa [Activation.Math.sigmoidSpec, sigmoidBoundScalar, oneEps, oneR, denomR, y]
                  using hdiv
              have hle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.sigmoidSpec (α := R) xR) -
                        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
                    linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := Shape.scalar) eps (Tensor.scalar xR)) := by
                refine le_trans hb ?_
                -- `linf_norm` of a scalar tensor is `abs` of its entry.
                simpa [sigmoidBoundTensor, Spec.mapTensor, linfNorm, RuntimeApprox.linfNorm,
                  tensorLinfNorm] using
                  (le_abs_self (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) xR))
              exact
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := Activation.Math.sigmoidSpec (α := ℝ) x)
                  (xR := Activation.Math.sigmoidSpec (α := R) xR)
                  (eps := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.scalar) eps (Tensor.scalar xR)))).2 (by
                      simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (s := Shape.dim n s) eps (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t := sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) ≤ B := by
                  simpa [B, sigmoidBoundTensor, Spec.mapTensor] using
                    (linf_norm_le_get_dim
                      (t := sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (Activation.Math.sigmoidSpec (α := ℝ))
                        (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (Activation.Math.sigmoidSpec (α := R))
                          (Tensor.dim xRf)))
                    ≤ B := by
                simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                    Spec.mapTensor, mapSpec] using hfold
              simpa [approxT, approxWith, B] using this

/--
Scalar forward bound for the scalar logistic-compatible NF `softmax` node.

Here the node computes the scalar logistic-like function `exp(x) / (exp(x) + 1)`, implemented using
`exp`, `+`, and division (denominator is ≥ 1). We keep the public node name stable for the NF graph,
but the mathematical function is `Activation.Math.logisticSpec`, not axis softmax.
-/
def softmaxBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  let numR : R := MathFunctions.exp xR
  let denomR : R := numR + (1 : R)
  let numHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) numR
  let denomHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR
  let qhat : ℝ := numHat / denomHat
  let epsNum : ℝ :=
    Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) +
      Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR + eps) +
      neuralUlp β fexp (Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
        TrainingPhase.forward / 2
  neuralUlp β fexp qhat TrainingPhase.forward / 2 +
    abs numHat * abs (1 / denomHat) + abs numHat + epsNum

/-- Per-entry bound tensor for the scalar logistic NF node. -/
def softmaxBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Spec.mapTensor (softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/--
`approxT` bound for the scalar logistic NF node lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around the scalar `exp`/`+`/`div` bound, using the usual
  `linf_norm`
lifting for dimensioned tensors.
-/
theorem approxT_softmax_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) xR)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
            by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := x) (xR := xR) (eps := eps)).1 hx
              let numR : R := MathFunctions.exp xR
              let denomR : R := numR + (1 : R)
              let y : ℝ := Real.exp x + 1
              have hy : (1 : ℝ) ≤ y := by
                have : 0 ≤ Real.exp x := by simpa using (Real.exp_nonneg x)
                linarith
              have hnum :=
                approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps)
                  hx'
              -- Apply the division bound with numerator = exp and denominator = exp + 1.
              have hdiv :=
                approx_div_nf_of_one_le (β := β) (fexp := fexp) (rnd := rnd)
                  (x := Real.exp x) (y := y) (xR := numR) (yR := denomR)
                  (epsx :=
                    Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) +
                      Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR + eps) +
                      neuralUlp β fexp
                          (Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))
                          TrainingPhase.forward / 2)
                  hy hnum
              have hb :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.logisticSpec (α := R) xR) -
                        Activation.Math.logisticSpec (α := ℝ) x) ≤
                    softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
                simpa [Activation.Math.logisticSpec, softmaxBoundScalar, numR, denomR, y] using
                  hdiv
              have hle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.logisticSpec (α := R) xR) -
                        Activation.Math.logisticSpec (α := ℝ) x) ≤
                    linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := Shape.scalar) eps (Tensor.scalar xR)) := by
                refine le_trans hb ?_
                simpa [softmaxBoundTensor, Spec.mapTensor, linfNorm, RuntimeApprox.linfNorm,
                  tensorLinfNorm] using
                  (le_abs_self (softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR))
              exact
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := Activation.Math.logisticSpec (α := ℝ) x)
                  (xR := Activation.Math.logisticSpec (α := R) xR)
                  (eps := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.scalar) eps (Tensor.scalar xR)))).2 (by
                      simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (s := Shape.dim n s) eps (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) ≤ B := by
                  simpa [B, softmaxBoundTensor, Spec.mapTensor] using
                    (linf_norm_le_get_dim
                      (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (Activation.Math.logisticSpec (α := ℝ))
                        (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (Activation.Math.logisticSpec (α := R))
                          (Tensor.dim xRf)))
                    ≤ B := by
                simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                    Spec.mapTensor, mapSpec] using hfold
              simpa [approxT, approxWith, B] using this

omit [NeuralValidRndToNearest rnd] in
/--
`approxT` bound for elementwise addition (`add_spec`) over arbitrary tensor shapes.

The output epsilon is computed as `linf_norm (add_bound_tensor epsx epsy xR yR)`, which combines the
input epsilons and one rounding-ULP term per element.
-/
theorem approxT_add_spec {s : Shape} [NeuralValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec xS yS) (addSpec xR yR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      -- Reduce to the scalar rounding lemma and wrap back into `approxT`.
                      have hx' :=
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x) (xR := xR) (eps := epsx)).1 hx
                      have hy' :=
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := y) (xR := yR) (eps := epsy)).1 hy
                      have hxy := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y :=
                        y) hx' hy'
                      have hle :
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR + yR) - (x + y)) ≤
                            linfNorm
                              (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar yR))
                                  := by
                        -- The RHS is `abs` of the scalar bound; widen using `le_abs_self`.
                        refine le_trans hxy ?_
                        -- `linf_norm` of the scalar bound tensor is `abs` of its scalar entry.
                        simpa [addBoundTensor, tensorToSpec, Spec.mapTensor, map2Spec,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using
                          (le_abs_self (epsx + epsy +
                            neuralUlp β fexp
                              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR +
                                toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
                              TrainingPhase.forward / 2))
                      exact
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x + y) (xR := xR + yR)
                          (eps := linfNorm
                            (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar
                                yR)))).2 (by
                                simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      -- Let `B` be the global bound (max over component bounds).
                      let B : ℝ :=
                        linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t := addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                            (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf)))

                      -- Show each component output distance is ≤ B, then take the max fold.
                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (addSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (addSpec (xRf i) (yRf i)))
                              ≤ B := by
                        intro i
                        -- Project input approximations to the component.
                        have hx_i :=
                          approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i

                        have hih :=
                          ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i hy_i

                        -- The component bound is ≤ the global bound `B`.
                        have hB_ge :
                            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) ≤ B := by
                          -- `add_bound_tensor` is shape-preserving, so this is a
                          -- max-over-components inequality.
                          simpa [B, addBoundTensor, tensorToSpec, Spec.mapTensor, map2Spec]
                            using
                            (linf_norm_le_get_dim
                              (t :=
                                addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                  (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                              i)

                        -- Convert the IH (an `approxT` statement) into a `tensor_distance`
                        -- inequality and weaken the bound.
                        have hdist : tensorDistance (α := SpecScalar) linfNorm
                            (addSpec (xSf i) (ySf i))
                            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                              := rnd))
                              (addSpec (xRf i) (yRf i)))
                          ≤ linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) := by
                          simpa [approxT, approxWith] using hih
                        exact le_trans hdist hB_ge

                      -- Fold the component distances with `max` and bound by `B`.
                      have : tensorDistance (α := SpecScalar) linfNorm
                          (Tensor.dim fun i => addSpec (xSf i) (ySf i))
                          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (Tensor.dim fun i => addSpec (xRf i) (yRf i)))
                        ≤ B := by
                        -- Unfold `tensor_distance` on `.dim`: it becomes a `foldl max` over
                        -- component distances.
                        have hf :
                            ∀ i ∈ List.finRange n,
                              tensorDistance (α := SpecScalar) linfNorm
                                  (addSpec (xSf i) (ySf i))
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (addSpec (xRf i) (yRf i)))
                                ≤ B := by
                          intro i _hi
                          exact hcomp i
                        -- Apply the list lemma.
                        have hfold :=
                          List.foldl_max_le_of_le (List.finRange n)
                            (fun i =>
                              tensorDistance (α := SpecScalar) linfNorm
                                (addSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (addSpec (xRf i) (yRf i))))
                            (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                        -- Rewrite back into `tensor_distance`.
                        simpa [tensorDistance,
                          NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                            Spec.mapTensor] using hfold

                      -- Conclude `approxT` for the whole tensor.
                      simpa [approxT, approxWith, B, addSpec, map2Spec] using this

/--
`approxT` bound for elementwise subtraction (`sub_spec`) over arbitrary tensor shapes.

This is obtained by lifting the scalar subtraction bound `approx_sub_nf` via
`approxT_map2_spec_of_scalar_bound`.
-/
theorem approxT_sub_spec {s : Shape} :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (subSpec xS yS) (subSpec xR yR)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxT_map2_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := fun a b => a - b) (fR := fun a b => a - b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + neuralUlp β fexp (a - b) TrainingPhase.forward / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_sub_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y := y) (xR := xR) (yR :=
            yR) hx hy))
  simpa [subSpec, subBoundTensor] using h

omit [NeuralValidRndToNearest rnd] in
/--
`approxT` bound for elementwise multiplication (`mul_spec`) over arbitrary tensor shapes.

The scalar core is `approx_mul_nf`, lifted componentwise; the resulting bound is packaged as
`mul_bound_tensor` and reduced with `linf_norm`.
-/
theorem approxT_mul_spec {s : Shape} [NeuralValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec xS yS) (mulSpec xR yR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      have hx' :=
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x) (xR := xR) (eps := epsx)).1 hx
                      have hy' :=
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := y) (xR := yR) (eps := epsy)).1 hy
                      have hxy := approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y :=
                        y) hx' hy'
                      have hle :
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * yR) - (x * y)) ≤
                            linfNorm
                              (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar yR))
                                  := by
                        refine le_trans hxy ?_
                        simpa [mulBoundTensor, tensorToSpec, Spec.mapTensor, map2Spec,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using
                          (le_abs_self
                            ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * epsy +
                              (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) + epsy) * epsx +
                              neuralUlp β fexp
                                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
                                    toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)
                                  TrainingPhase.forward / 2))
                      exact
                        (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                          := rnd))
                          (x := x * y) (xR := xR * yR)
                          (eps := linfNorm
                            (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := Shape.scalar) epsx epsy (Tensor.scalar xR) (Tensor.scalar
                                yR)))).2 (by
                                simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases yS with
          | dim ySf =>
              cases xR with
              | dim xRf =>
                  cases yR with
                  | dim yRf =>
                      let B : ℝ :=
                        linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                      have hB_nonneg : 0 ≤ B := by
                        simpa [B] using (linf_norm_nonneg
                          (t := mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                            (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf)))

                      have hcomp :
                          ∀ i : Fin n,
                            tensorDistance (α := SpecScalar) linfNorm
                                (mulSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (mulSpec (xRf i) (yRf i)))
                              ≤ B := by
                        intro i
                        have hx_i :=
                          approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := epsx) hx i
                        have hy_i :=
                          approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (xS := Tensor.dim ySf) (xR := Tensor.dim yRf) (eps := epsy) hy i

                        have hih :=
                          ih (xS := xSf i) (yS := ySf i) (xR := xRf i) (yR := yRf i) hx_i hy_i

                        have hB_ge :
                            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) ≤ B := by
                          simpa [B, mulBoundTensor, tensorToSpec, Spec.mapTensor, map2Spec]
                            using
                            (linf_norm_le_get_dim
                              (t :=
                                mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                  (s := Shape.dim n s) epsx epsy (Tensor.dim xRf) (Tensor.dim yRf))
                              i)

                        have hdist : tensorDistance (α := SpecScalar) linfNorm
                            (mulSpec (xSf i) (ySf i))
                            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                              := rnd))
                              (mulSpec (xRf i) (yRf i)))
                          ≤ linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := s) epsx epsy (xRf i) (yRf i)) := by
                          simpa [approxT, approxWith] using hih
                        exact le_trans hdist hB_ge

                      have : tensorDistance (α := SpecScalar) linfNorm
                          (Tensor.dim fun i => mulSpec (xSf i) (ySf i))
                          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                            rnd))
                            (Tensor.dim fun i => mulSpec (xRf i) (yRf i)))
                        ≤ B := by
                        have hf :
                            ∀ i ∈ List.finRange n,
                              tensorDistance (α := SpecScalar) linfNorm
                                  (mulSpec (xSf i) (ySf i))
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (mulSpec (xRf i) (yRf i)))
                                ≤ B := by
                          intro i _hi
                          exact hcomp i
                        have hfold :=
                          List.foldl_max_le_of_le (List.finRange n)
                            (fun i =>
                              tensorDistance (α := SpecScalar) linfNorm
                                (mulSpec (xSf i) (ySf i))
                                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                  (rnd := rnd))
                                  (mulSpec (xRf i) (yRf i))))
                            (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
                        simpa [tensorDistance,
                          NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                            Spec.mapTensor] using hfold

                      simpa [approxT, approxWith, B, mulSpec, map2Spec] using this

/--
`approxT` bound for scaling by a runtime constant (`scale_spec`) over arbitrary tensor shapes.

This is the tensor-level wrapper around the scalar scaling lemma `approx_scale_nf`.
-/
theorem approxT_scale_spec {s : Shape} (c : R) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (scaleSpec (α := SpecScalar) (s := s) xS (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
          (scaleSpec (α := R) (s := s) xR c)
          (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps c xR)) :=
            by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := fun x => x * toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
      (fR := fun xR => xR * c)
      (bnd := fun a eps =>
        abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
          neuralUlp β fexp (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
            TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_scale_nf (β := β) (fexp := fexp) (rnd := rnd) (c := c)
          (x := x) (xR := xR) (eps := eps) hx))
  simpa [scaleSpec, scaleBoundTensor] using h

/-- `approxT` bound for elementwise negation (`neg_spec`) over arbitrary tensor shapes. -/
theorem approxT_neg_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (negSpec xS) (negSpec xR)
          (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Neg.neg) (fR := Neg.neg)
      (bnd := fun a eps =>
        eps + neuralUlp β fexp (-a) TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_neg_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [negSpec, negBoundTensor] using h

/-- `approxT` bound for elementwise absolute value (`abs_spec`) over arbitrary tensor shapes. -/
theorem approxT_abs_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (absSpec xS) (absSpec xR)
          (linfNorm (absBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := MathFunctions.abs) (fR := MathFunctions.abs)
      (bnd := fun a eps =>
        eps + neuralUlp β fexp (abs a) TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        -- `MathFunctions.abs` is definitional `abs` on `ℝ`.
        simpa using
          (approx_abs_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [absSpec, absBoundTensor] using h

/--
`approxT` bound for elementwise exponentiation (`exp_spec`) over arbitrary tensor shapes.

This lifts the scalar mean-value-theorem bound `approx_exp_nf`.
-/
theorem approxT_exp_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (expSpec xS) (expSpec xR)
          (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := MathFunctions.exp) (fR := MathFunctions.exp)
      (bnd := fun a eps =>
        Real.exp a + Real.exp (a + eps) + neuralUlp β fexp (Real.exp a) TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [expSpec, expBoundTensor] using h

/--
`approxT` bound for elementwise hyperbolic tangent (`tanh`) over arbitrary tensor shapes.

Currently uses the coarse unconditional scalar bound `approx_tanh_nf` (boundedness of `tanh`).
-/
theorem approxT_tanh_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) MathFunctions.tanh xS)
          (mapSpec (s := s) MathFunctions.tanh xR)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxT_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := MathFunctions.tanh) (fR := MathFunctions.tanh)
      (bnd := fun a _eps => (2 : ℝ) + neuralUlp β fexp (Real.tanh a) TrainingPhase.forward / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_tanh_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [tanhBoundTensor] using h

-- ReLU (via `max`) is non-expansive: it does not add rounding error in `NF` (it selects an input).

private lemma abs_max0_sub_max0_le (x y : ℝ) : abs (max x 0 - max y 0) ≤ abs (x - y) := by
  simpa using (abs_max_sub_max_le_abs x y (0 : ℝ))

/-- Rounded ReLU scalar op for `NF`: apply `max · 0` then round. -/
noncomputable def reluR (x : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (max (toSpec (β := β) (fexp := fexp) (rnd := rnd) x) 0)

/--
Per-entry bound tensor for ReLU (`max · 0`).

ReLU is 1-Lipschitz (`|max x 0 - max y 0| ≤ |x - y|`), so the only new error is the final rounding
step in `reluR`.
-/
def reluBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + neuralUlp β fexp (max a 0) TrainingPhase.forward / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxT` bound for elementwise ReLU (`max · 0`) over arbitrary tensor shapes.

Combines the 1-Lipschitz property of `max` with one rounding step for `reluR`.
-/
theorem approxT_relu_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (fun x => max x 0) xS)
          (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := x) (xR := xR) (eps := eps)).1 hx
              let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
              have hround :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp := fexp)
                    (rnd := rnd) xR) - max xhat 0) ≤
                    neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2 := by
                -- `reluR` is `ofReal (max xhat 0)` so this is a single rounding step.
                simpa [reluR, xhat, toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
                  TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
                  (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd :=
                    rnd) (max xhat 0))
              have hmax :
                  abs (max xhat 0 - max x 0) ≤ abs (xhat - x) := by
                simpa [xhat, abs_sub_comm] using abs_max0_sub_max0_le xhat x
              have htriangle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp := fexp)
                    (rnd := rnd) xR) - max x 0) ≤
                    eps + neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2 := by
                have hxhat : abs (xhat - x) ≤ eps := by simpa [xhat] using hx'
                -- triangle inequality: (rounded - relu x) = (rounded - relu xhat) + (relu xhat -
                -- relu x)
                have :=
                  calc
                    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp := fexp)
                      (rnd := rnd) xR) - max x 0)
                        ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp :=
                          fexp) (rnd := rnd) xR) - max xhat 0)
                            + abs (max xhat 0 - max x 0) := by
                              simpa [sub_eq_add_neg, add_assoc] using
                                abs_sub_le
                                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp
                                    := fexp) (rnd := rnd) xR))
                                  (max xhat 0) (max x 0)
                    _ ≤ neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2 + abs (xhat - x) :=
                      by
                          exact add_le_add hround (le_trans hmax (le_rfl))
                    _ ≤ neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2 + eps := by
                          linarith [hxhat]
                    _ = eps + neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2 := by ring
                simpa [xhat] using this
              have hle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reluR (β := β) (fexp := fexp)
                    (rnd := rnd) xR) - max x 0) ≤
                    linfNorm
                      (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.scalar) eps (Tensor.scalar xR)) := by
                -- The RHS is `abs (eps + ulp(max xhat 0)/2)`; widen via `le_abs_self`.
                refine le_trans htriangle ?_
                simpa [reluBoundTensor, tensorToSpec, Spec.mapTensor, mapSpec, linfNorm,
                  RuntimeApprox.linfNorm,
                  tensorLinfNorm, xhat] using
                  (le_abs_self (eps + neuralUlp β fexp (max xhat 0) TrainingPhase.forward / 2))
              exact
                (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := max x 0) (xR := reluR (β := β) (fexp := fexp) (rnd := rnd) xR)
                  (eps := linfNorm
                    (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := Shape.scalar) eps (Tensor.scalar xR)))).2 (by
                        simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (s := Shape.dim n s) eps (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t := reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (fun x => max x 0) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := s) eps (xRf i)) ≤ B := by
                  simpa [B, reluBoundTensor, tensorToSpec, Spec.mapTensor, mapSpec] using
                    (linf_norm_le_get_dim
                      (t := reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (fun x => max x 0) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (xRf i)))
                      ≤ linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) := by
                  simpa [approxT, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (fun x => max x 0) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (fun x => max x 0) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have : tensorDistance (α := SpecScalar) linfNorm
                  (Tensor.dim fun i => mapSpec (fun x => max x 0) (xSf i))
                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (Tensor.dim fun i => mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (xRf
                      i)))
                ≤ B := by
                simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                    Spec.mapTensor] using hfold
              simpa [approxT, approxWith, B, mapSpec] using this

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for `NF` ops (for building SSA/DAG forward bounds)
-- ---------------------------------------------------------------------------

/--
`FwdNode` for elementwise addition.

This packages `approxT_add_spec` so addition can be used inside larger verified `FwdGraph`s.
-/
def addNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        addSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        addSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/-- `FwdNode` for elementwise subtraction (wraps `approxT_sub_spec`). -/
def subNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        subSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        subSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/-- `FwdNode` for elementwise multiplication (wraps `approxT_mul_spec`). -/
def mulNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mulSpec (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        mulSpec (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/--
`FwdNode` for clamped division `safeDiv`.

Requires a proof `hε : 0 < ε` and uses `approxT_safeDiv_spec` to obtain an unconditional bound.
-/
def safeDivNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        map2Spec (s := s) (safeDiv (ε := ε))
          (getIdx (α := SpecScalar) ctx a) (getIdx (α := SpecScalar) ctx b)
    , forwardRuntime := fun ctx =>
        map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b)
    , bound := fun eps ctx =>
        linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          ε
          (getIdxEps (Γ := Γ) (s := s) eps a)
          (getIdxEps (Γ := Γ) (s := s) eps b)
          (getIdx (α := R) ctx a) (getIdx (α := R) ctx b))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  have hb := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    b
  simpa using
    (approxT_safeDiv_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (yS := getIdx (α := SpecScalar) xS b)
      (xR := getIdx (α := R) xR a) (yR := getIdx (α := R) xR b)
      (epsx := getIdxEps (Γ := Γ) (s := s) eps a) (epsy := getIdxEps (Γ := Γ) (s := s) eps b)
      ha hb)

/--
`FwdNode` for scaling by a runtime constant `c`.

Wraps `approxT_scale_spec`.
-/
def scaleNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (c : R) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        scaleSpec (α := SpecScalar) (s := s) (getIdx (α := SpecScalar) ctx a)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
    , forwardRuntime := fun ctx =>
        scaleSpec (α := R) (s := s) (getIdx (α := R) ctx a) c
    , bound := fun eps ctx =>
        linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) c (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_scale_spec (β := β) (fexp := fexp) (rnd := rnd) (c := c)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise negation (wraps `approxT_neg_spec`). -/
def negNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        negSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        negSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise absolute value (wraps `approxT_abs_spec`). -/
def absNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        absSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        absSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (absBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_abs_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise exponentiation (wraps `approxT_exp_spec`). -/
def expNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        expSpec (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        expSpec (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_exp_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise softplus (wraps `approxT_softplus_spec`). -/
def softplusNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_softplus_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for clamped log `safeLog`.

Requires a proof `hε : 0 < ε` and wraps `approxT_safeLog_spec`.
-/
def safeLogNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (safeLog (ε := ε)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_safeLog_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for the smooth `safe_log` activation.

Requires `hε : 0 < ε` and wraps `approxT_safe_log_spec`.
-/
def safeLogSoftplusNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε) (getIdx (α :=
          SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (safeLogSoftplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_safe_log_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise `tanh` (wraps `approxT_tanh_spec`). -/
def tanhNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_tanh_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise sigmoid (wraps `approxT_sigmoid_spec`). -/
def sigmoidNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for elementwise ReLU (`max · 0`, wraps `approxT_relu_spec`). -/
def reluNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (fun x => max x 0) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_relu_spec (β := β) (fexp := fexp) (rnd := rnd)
      (s := s) (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/-- `FwdNode` for the scalar logistic-compatible `softmax` node (wraps `approxT_softmax_spec`). -/
def softmaxNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctx a)
    , forwardRuntime := fun ctx =>
        mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctx a)
    , bound := fun eps ctx =>
        linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a))
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_softmax_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

/--
`FwdNode` for sum reduction (`sum_spec`).

This reduces a tensor to a scalar and uses `approxT_sum_spec` with the accumulated `sum_bound`.
-/
def sumNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ Shape.scalar :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Tensor.scalar (sumSpec (α := ℝ) (s := s) (getIdx (α := SpecScalar) ctx a))
    , forwardRuntime := fun ctx =>
        Tensor.scalar (sumSpec (α := R) (s := s) (getIdx (α := R) ctx a))
    , bound := fun eps ctx =>
        sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s)
          (getIdxEps (Γ := Γ) (s := s) eps a) (getIdx (α := R) ctx a)
    , sound := ?_ }
  intro xS xR eps hctx
  have ha := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    a
  simpa using
    (approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := getIdx (α := SpecScalar) xS a) (xR := getIdx (α := R) xR a)
      (eps := getIdxEps (Γ := Γ) (s := s) eps a) ha)

end NFBackend

end

end RuntimeApprox
end Proofs
