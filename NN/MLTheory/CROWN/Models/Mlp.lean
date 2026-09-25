/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Activations
public import NN.MLTheory.CROWN.Proofs.ReLUUpperBound
public import NN.MLTheory.CROWN.Graph.Engine.BackwardObjective
public import NN.Spec.Layers.Linear
import NN.Proofs.Tensor.Algebra
public import NN.MLTheory.CROWN.BoundOps.Lawful

/-!
# Mlp

CROWN/DeepPoly-style propagation for MLPs (vector in/out) using TorchLean tensors.

This file is a compact implementation that sits on top of:
- `NN.MLTheory.CROWN.Core` (`Box`, `AffineVec`, and `IBP.linear`), and
- TorchLean’s typed tensor layer (`TorchLean.Tensor`).

What is implemented:
- Per-neuron ReLU linear relaxations derived from pre-activation bounds using the canonical
  `Runtime.Ops.ReLU.relaxScalar` and `Runtime.Ops.ReLU.relaxScalarLower` definitions.
- IBP forward rules for ReLU, sigmoid, tanh, and Leaky ReLU.
- A two-layer ReLU MLP wrapper `TwoLayerMLP` with a simple end-to-end bounding API.

Scope boundaries in this MLP-focused module:
- General computation graphs. Use `NN.MLTheory.CROWN.Graph` for the
  graph-level certificate checker and `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness` for the
  corresponding end-to-end soundness theorem.
- Objective-dependent / backward CROWN slope optimization. The
  certificate infrastructure for alpha-CROWN and alpha/beta-CROWN artifacts lives under
  `NN.MLTheory.CROWN.Cert.AlphaCROWN` and `NN.MLTheory.CROWN.Cert.AlphaBetaCROWN`.

References:
- CROWN: Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions",
  arXiv:1811.00866.
- auto_LiRPA: Xu et al.,
  "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920.

PyTorch analogues:
- `torch.nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `torch.nn.ReLU`: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN.Runtime.Ops

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Column-wise scaling of a matrix by a vector: scale each column `j` by `v[j]`. -/
def matColScaleSpec
  {m n : Nat} (A : Tensor α [m, n])
  (v : Tensor α [n]) : Tensor α [m, n] :=
  Tensor.matrix (fun i j => get2 A i j * Tensor.getScalar v j)

/-- Elementwise positive part of a matrix: replace negative entries by `0`. -/
abbrev matPosSpec {m n : Nat}
  (A : Tensor α [m, n]) : Tensor α [m, n] :=
  IBP.matPos A

/-- Elementwise negative part of a matrix: replace positive entries by `0`. -/
abbrev matNegSpec {m n : Nat}
  (A : Tensor α [m, n]) : Tensor α [m, n] :=
  IBP.matNeg A

/-- Extract the slope vector from a tensor of ReLU relaxations. -/
def reluRelaxSlopeVec {n : Nat}
  (relax : Tensor (ReLURelax α) [n]) : Tensor α [n] :=
  Tensor.ofFn (fun i => (Tensor.getScalar relax i).slope)

/-- Extract the bias vector from a tensor of ReLU relaxations. -/
def reluRelaxBiasVec {n : Nat}
  (relax : Tensor (ReLURelax α) [n]) : Tensor α [n] :=
  Tensor.ofFn (fun i => (Tensor.getScalar relax i).bias)

/- Interval forward (IBP) for MLP layer + ReLU -/
namespace IBP

/-!
Interval Bound Propagation (IBP) utilities for vector-shaped activations.

These are executable transfer functions. Their soundness theorems instantiate the scalar type with
`ℝ` and use the corresponding monotonicity or activation-specific proof.

The linear-layer bound helper `IBP.linear` lives in `NN.MLTheory.CROWN.Core`.
-/

/--
Interval bounds for ReLU on a vector.

This is the standard elementwise interval evaluation:
`relu([l,u]) = [relu(l), relu(u)]`.
-/
def relu {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  { lo := Tensor.map (fun l => if l > 0 then l else 0) xB.lo
    hi := Tensor.map (fun u => if u > 0 then u else 0) xB.hi }

/--
Re-export of the runtime-only monotone-activation IBP helper.

We keep this file compact and Mathlib-friendly for proofs, but we do not want to
maintain two copies of the same computational rule. Canonical implementation lives in:
`NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax`.

Semantics (per component): given an interval `[l,u]`, this returns
`[min(f(l), f(u)), max(f(l), f(u))]` (intended for monotone `f`).
-/
abbrev mapMinmax {n : Nat} (f : α → α) (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  Runtime.Ops.IBP.mapMinmax (α := α) f xB

/-- Interval bounds for `sigmoid`. -/
abbrev sigmoid {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  Runtime.Ops.IBP.sigmoid (α := α) xB

/-- Interval bounds for `tanh`. -/
abbrev tanh {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  Runtime.Ops.IBP.tanh (α := α) xB

/-- Interval bounds for leaky ReLU, including the zero kink on crossing intervals. -/
def leakyRelu {n : Nat} (αₗ : α) (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  Operators.Activations.ibpLeakyRelu n αₗ xB

end IBP

/--
Two-layer MLP payload used by this file.

Semantics: `y = outputWeight * relu(hiddenWeight * x + hiddenBias) + outputBias`.

PyTorch analogue: `torch.nn.Sequential(Linear(inDim,hidDim), ReLU(), Linear(hidDim,outDim))`.
-/
structure TwoLayerMLP (α : Type) [TorchLean.Storage α] (inDim hidDim outDim : Nat) where
  /-- First layer weight matrix. -/
  hiddenWeight : Tensor α [hidDim, inDim]
  /-- First layer bias vector. -/
  hiddenBias : Tensor α [hidDim]
  /-- Second layer weight matrix. -/
  outputWeight : Tensor α [outDim, hidDim]
  /-- Second layer bias vector. -/
  outputBias : Tensor α [outDim]

/-- Forward semantics for `TwoLayerMLP` (used to state soundness theorems). -/
def forward {inDim hidDim outDim : Nat}
  (net : TwoLayerMLP α inDim hidDim outDim)
  (x : Tensor α [inDim]) : Tensor α [outDim] :=
  let hiddenLayer : Spec.LinearSpec α inDim hidDim :=
    { weights := net.hiddenWeight, bias := net.hiddenBias }
  let outputLayer : Spec.LinearSpec α hidDim outDim :=
    { weights := net.outputWeight, bias := net.outputBias }
  let hiddenPreactivation := Spec.linearSpec (α:=α) hiddenLayer x
  let hiddenActivation := Activation.reluSpec (α:=α) hiddenPreactivation
  Spec.linearSpec (α:=α) outputLayer hiddenActivation

/-- Build a `TwoLayerMLP` from two `LinearSpec` records. -/
def ofLinearSpecs {inDim hidDim outDim : Nat}
  (hiddenLayer : Spec.LinearSpec α inDim hidDim) (outputLayer : Spec.LinearSpec α hidDim outDim) :
  TwoLayerMLP α inDim hidDim outDim :=
  { hiddenWeight := hiddenLayer.weights
    hiddenBias := hiddenLayer.bias
    outputWeight := outputLayer.weights
    outputBias := outputLayer.bias }

/--
Compute an output interval box via pure IBP.

This is fast and, when `BoundOps α` supplies sound outward endpoint operations, typically looser
than CROWN/DeepPoly affine bounds.
-/
def boundIbp {inDim hidDim outDim : Nat} [BoundOps α]
  (net : TwoLayerMLP α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  -- z1 = hiddenWeight x + hiddenBias
  let b1B : Box α (.dim hidDim .scalar) := { lo := net.hiddenBias, hi := net.hiddenBias }
  let z1B := IBP.linear net.hiddenWeight xB b1B
  let a1B := IBP.relu (n:=hidDim) z1B
  let b2B : Box α (.dim outDim .scalar) := { lo := net.outputBias, hi := net.outputBias }
  IBP.linear net.outputWeight a1B b2B

/--
The lower and upper affine CROWN forms for this two-layer ReLU MLP.

The returned pair is `(lower, upper)`. `boundAffineCrown` evaluates these forms on the input box and
takes the lower and upper endpoints. Exact backends use the two-layer algebraic formula;
rounded backends share the graph engine's directed coefficient propagation.
-/
def affineCrownForms {inDim hidDim outDim : Nat} [BoundOps α]
  (net : TwoLayerMLP α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : AffineVec α inDim outDim × AffineVec α inDim outDim :=
  if !BoundOps.supportsExactAffineReassociation (α := α) then
    let graph : NN.IR.Graph := ⟨#[
      { id := 0, parents := #[], kind := .input, outShape := [inDim] },
      { id := 1, parents := #[0], kind := .linear, outShape := [hidDim] },
      { id := 2, parents := #[1], kind := .relu, outShape := [hidDim] },
      { id := 3, parents := #[2], kind := .linear, outShape := [outDim] }]⟩
    let params : Graph.ParamStore α :=
      { inputBoxes := ({} : Std.HashMap Nat (FlatBox α)).insert 0
          { dim := inDim, lo := xB.lo, hi := xB.hi }
        linearWB := ({} : Std.HashMap Nat (Graph.LinParams α)).insert 1
          { m := hidDim, n := inDim, w := net.hiddenWeight, b := net.hiddenBias } |>.insert 3
          { m := outDim, n := hidDim, w := net.outputWeight, b := net.outputBias } }
    let fallback :=
      let box := boundIbp net xB
      let zero := Tensor.full [outDim, inDim] 0
      ({ A := zero, c := box.lo }, { A := zero, c := box.hi })
    match Graph.directedNodeBounds? graph params { inputId := 0, inputDim := inDim }
        (Graph.runIBP graph params) 3 with
    | some bounds =>
      if hIn : bounds.inDim = inDim then
        if hOut : bounds.outDim = outDim then
          let castForm (aff : AffineVec α bounds.inDim bounds.outDim) :
              AffineVec α inDim outDim :=
            Graph.castAffineIn hIn (Graph.castAffineOut hOut aff)
          (castForm bounds.loAff, castForm bounds.hiAff)
        else fallback
      else fallback
    | none => fallback
  else
    -- First get the ReLU intervals. Then outputWeight's sign tells us which relaxation feeds the
    -- lower or upper affine form.
    let b1B : Box α (.dim hidDim .scalar) := { lo := net.hiddenBias, hi := net.hiddenBias }
    let z1B := IBP.linear (α:=α) net.hiddenWeight xB b1B
    let relaxU := ReLU.relaxVector (α:=α) (n:=hidDim) z1B.lo z1B.hi
    let relaxL := ReLU.relaxVectorLower (α:=α) (n:=hidDim) z1B.lo z1B.hi
    let slopeU := reluRelaxSlopeVec (α:=α) (n:=hidDim) relaxU
    let biasU  := reluRelaxBiasVec  (α:=α) (n:=hidDim) relaxU
    let slopeL := reluRelaxSlopeVec (α:=α) (n:=hidDim) relaxL
    let biasL  := reluRelaxBiasVec  (α:=α) (n:=hidDim) relaxL

    let W2pos := matPosSpec (α:=α) (m:=outDim) (n:=hidDim) net.outputWeight
    let W2neg := matNegSpec (α:=α) (m:=outDim) (n:=hidDim) net.outputWeight

    -- Upper affine: W2pos uses ReLU upper, W2neg uses ReLU lower.
    let W2posU := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2pos slopeU
    let W2negL := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2neg slopeL
    let AU := Spec.matMulSpec (α:=α) (Tensor.addSpec W2posU W2negL) net.hiddenWeight
    let innerUPos := Tensor.addSpec (Tensor.mulSpec slopeU net.hiddenBias) biasU
    let innerLNeg := Tensor.addSpec (Tensor.mulSpec slopeL net.hiddenBias) biasL
    let cU :=
      Tensor.addSpec
        (Tensor.addSpec
          (Spec.matVecMulSpec (α:=α) W2pos innerUPos)
          (Spec.matVecMulSpec (α:=α) W2neg innerLNeg))
        net.outputBias

    -- Lower affine: W2pos uses ReLU lower, W2neg uses ReLU upper.
    let W2posL := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2pos slopeL
    let W2negU := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2neg slopeU
    let AL := Spec.matMulSpec (α:=α) (Tensor.addSpec W2posL W2negU) net.hiddenWeight
    let innerLPos := Tensor.addSpec (Tensor.mulSpec slopeL net.hiddenBias) biasL
    let innerUNeg := Tensor.addSpec (Tensor.mulSpec slopeU net.hiddenBias) biasU
    let cL :=
      Tensor.addSpec
        (Tensor.addSpec
          (Spec.matVecMulSpec (α:=α) W2pos innerLPos)
          (Spec.matVecMulSpec (α:=α) W2neg innerUNeg))
        net.outputBias

    let affU : AffineVec α inDim outDim := AffineVec.ofLinear (α:=α) AU cU
    let affL : AffineVec α inDim outDim := AffineVec.ofLinear (α:=α) AL cL
    (affL, affU)

/--
Affine (CROWN/DeepPoly-style) bounds for the 2-layer ReLU MLP.

Rounded scalar backends use the same directed arithmetic as the general graph CROWN API.
-/
def boundAffineCrown {inDim hidDim outDim : Nat} [BoundOps α]
  (net : TwoLayerMLP α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  let forms := affineCrownForms (α:=α) net xB
  let BL := AffineVec.evalOnBox (α:=α) forms.1 xB
  let BU := AffineVec.evalOnBox (α:=α) forms.2 xB
  { lo := BL.lo, hi := BU.hi }

/--
End-to-end bound API exposed by this file.

This API returns the IBP bound. Its enclosure guarantee depends on the selected `BoundOps`
implementation; `boundAffineCrown` additionally retains affine dependence on exact backends
and uses directed graph propagation on rounded backends.
-/
def boundAffine {inDim hidDim outDim : Nat} [BoundOps α]
  (net : TwoLayerMLP α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  boundIbp (α:=α) net xB

/-!
Theorems inspired by CROWN (Zhang et al., 2018, arXiv:1811.00866)

We record soundness properties of the relaxations and bound propagation.
Proofs below require elementary order reasoning and case splits on signs, plus
properties of mat-vec interval arithmetic.
-/

namespace Theorems

open NN.MLTheory.CROWN

/--
Scalar ReLU relaxation soundness over `ℝ` (upper bound).

If `x ∈ [l, u]` and `rp := ReLU.relaxScalar l u`, then:
`relu(x) <= rp.slope * x + rp.bias`.

This is the standard CROWN/DeepPoly upper chord construction (arXiv:1811.00866).
-/
theorem relu_relax_scalar_upper_real
  (l u x : ℝ)
  (hlx : l ≤ x) (hxu : x ≤ u) :
  let rp := ReLU.relaxScalar (α:=ℝ) l u
  Activation.Math.reluSpec (α:=ℝ) x ≤ rp.slope * x + rp.bias := by
  exact NN.MLTheory.CROWN.Proofs.relu_relax_scalar_upper_real_runtime l u x hlx hxu

/--
Vectorized ReLU relaxation (pointwise upper bound) over `ℝ`.

If `x ∈ [lo, hi]` and `rp := ReLU.relax_vector lo hi`, then for every component `i` we have
`relu(xᵢ) ≤ rpᵢ.slope * xᵢ + rpᵢ.bias`.
-/
theorem relu_relax_vector_pointwise_upper_real {n : Nat}
  (lo hi x : Tensor ℝ [n])
  (hIn : Box.contains (α:=ℝ) { lo := lo, hi := hi } x) :
  ∀ i : Fin n,
    let li := Tensor.getScalar lo i
    let ui := Tensor.getScalar hi i
    let xi := Tensor.getScalar x i
    let rp := ReLU.relaxScalar (α:=ℝ) li ui
    Activation.Math.reluSpec (α:=ℝ) xi ≤ rp.slope * xi + rp.bias :=
  by
  intro i
  have hcoord := hIn i
  exact relu_relax_scalar_upper_real
    (l := Tensor.getScalar lo i) (u := Tensor.getScalar hi i)
    (x := Tensor.getScalar x i) hcoord.1 hcoord.2

/- Pure IBP soundness for the 2-layer MLP. -/
/--
Soundness of `IBP.linear` over `ℝ`.

If `x ∈ xB` and `b ∈ bB`, then `W*x + b` lies in the interval box computed by
`IBP.linear W xB bB`.
-/
theorem ibp_linear_sound_real {m n : Nat}
  (W : Tensor ℝ [m, n])
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar))
  (x : Tensor ℝ [n]) (b : Tensor ℝ [m])
  (hx : Box.contains (α:=ℝ) xB x) (hb : Box.contains (α:=ℝ) bB b) :
  Box.contains (α:=ℝ) (IBP.linear (α:=ℝ) W xB bB)
    (Spec.linearSpec (α:=ℝ) { weights := W, bias := b } x) := by
  classical
  intro i
  change
    getScalar (IBP.linear (α := ℝ) W xB bB).lo i ≤
        getScalar (Spec.linearSpec (α := ℝ) { weights := W, bias := b } x) i ∧
      getScalar (Spec.linearSpec (α := ℝ) { weights := W, bias := b } x) i ≤
        getScalar (IBP.linear (α := ℝ) W xB bB).hi i
  simp only [IBP.linear, getScalar_dim, Spec.linearSpec]
  rw [show
    getScalar (addSpec (matVecMulSpec W x) b) i =
      getScalar (matVecMulSpec W x) i + getScalar b i by
        simp [getScalar_eq_apply, addSpec, map2Spec]]
  rw [Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec]
  simp only [BoundOps.addDown, BoundOps.addUp, BoundOps.mulDown, BoundOps.mulUp]
  have min2_eq_min (a c : ℝ) : BoundOps.min2 a c = min a c := by
    by_cases h : a > c
    · simp [BoundOps.min2, h, min_eq_right (le_of_lt h)]
    · simp [BoundOps.min2, h, min_eq_left (le_of_not_gt h)]
  have max2_eq_max (a c : ℝ) : BoundOps.max2 a c = max a c := by
    by_cases h : a > c
    · simp [BoundOps.max2, h, max_eq_left (le_of_lt h)]
    · simp [BoundOps.max2, h, max_eq_right (le_of_not_gt h)]
  let lower : Fin n → ℝ := fun j =>
    min (get2 W i j * getScalar xB.lo j) (get2 W i j * getScalar xB.hi j)
  let middle : Fin n → ℝ := fun j => get2 W i j * getScalar x j
  let upper : Fin n → ℝ := fun j =>
    max (get2 W i j * getScalar xB.lo j) (get2 W i j * getScalar xB.hi j)
  have termBounds (j : Fin n) : lower j ≤ middle j ∧ middle j ≤ upper j := by
    have hj := hx j
    change getScalar xB.lo j ≤ getScalar x j ∧
      getScalar x j ≤ getScalar xB.hi j at hj
    by_cases hw : 0 ≤ get2 W i j
    · exact ⟨
        le_trans (min_le_left _ _)
          (mul_le_mul_of_nonneg_left hj.1 hw),
        le_trans (mul_le_mul_of_nonneg_left hj.2 hw)
          (le_max_right _ _)⟩
    · have hw' : get2 W i j ≤ 0 := le_of_not_ge hw
      exact ⟨
        le_trans (min_le_right _ _)
          (mul_le_mul_of_nonpos_left hj.2 hw'),
        le_trans (mul_le_mul_of_nonpos_left hj.1 hw')
          (le_max_left _ _)⟩
  have foldLower :
      ∀ (indices : List (Fin n)) (a c : ℝ), a ≤ c →
        indices.foldl (fun acc j => acc + lower j) a ≤
          indices.foldl (fun acc j => acc + middle j) c := by
    intro indices
    induction indices with
    | nil => simp
    | cons j js ih =>
        intro a c hac
        apply ih
        exact add_le_add hac (termBounds j).1
  have foldUpper :
      ∀ (indices : List (Fin n)) (a c : ℝ), a ≤ c →
        indices.foldl (fun acc j => acc + middle j) a ≤
          indices.foldl (fun acc j => acc + upper j) c := by
    intro indices
    induction indices with
    | nil => simp
    | cons j js ih =>
        intro a c hac
        apply ih
        exact add_le_add hac (termBounds j).2
  have hLower :
      (List.finRange n).foldl (fun acc j => acc + lower j) 0 ≤
        ∑ j : Fin n, middle j := by
    rw [← List.finRange_foldl_add_eq_finset_sum]
    exact foldLower _ _ _ le_rfl
  have hUpper :
      (∑ j : Fin n, middle j) ≤
        (List.finRange n).foldl (fun acc j => acc + upper j) 0 := by
    rw [← List.finRange_foldl_add_eq_finset_sum]
    exact foldUpper _ _ _ le_rfl
  have hbI := hb i
  change getScalar bB.lo i ≤ getScalar b i ∧
    getScalar b i ≤ getScalar bB.hi i at hbI
  simpa [lower, middle, upper, min2_eq_min, max2_eq_max] using
    And.intro (add_le_add hLower hbI.1) (add_le_add hUpper hbI.2)

/- Helper: soundness of IBP.relu over ℝ -/
private theorem ibp_relu_sound_real {n : Nat}
  (zB : Box ℝ (.dim n .scalar))
  (z : Tensor ℝ [n])
  (hz : Box.contains (α:=ℝ) zB z) :
  Box.contains (α:=ℝ) (IBP.relu (α:=ℝ) zB) (Activation.reluSpec (α:=ℝ) z) := by
  have relu_eq_max (a : ℝ) : (if a > 0 then a else 0) = max a 0 := by
    by_cases ha : a > 0
    · simp [ha, max_eq_left (le_of_lt ha)]
    · simp [ha, max_eq_right (le_of_not_gt ha)]
  intro i
  have hcoord := hz i
  change
    getScalar (IBP.relu (α := ℝ) zB).lo i ≤
        getScalar (Activation.reluSpec (α := ℝ) z) i ∧
      getScalar (Activation.reluSpec (α := ℝ) z) i ≤
        getScalar (IBP.relu (α := ℝ) zB).hi i
  simp only [IBP.relu, Tensor.getScalar_map, Activation.reluSpec, getScalar_mapSpec,
    Activation.Math.reluSpec_eq_max, relu_eq_max]
  constructor
  · exact max_le_max hcoord.1 (le_refl 0)
  · exact max_le_max hcoord.2 (le_refl 0)

/-- Soundness of pure IBP bounds for a 2-layer MLP over `ℝ`. -/
theorem bound_ibp_sound {inDim hidDim outDim : Nat}
  (net : TwoLayerMLP ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar))
  (x : Tensor ℝ [inDim])
  (hx : Box.contains (α:=ℝ) xB x) :
  Box.contains (α:=ℝ) (boundIbp (α:=ℝ) net xB) (forward (α:=ℝ) net x) := by
  classical
  -- Unfold bound_ibp and forward
  -- Step 1: z1 ∈ IBP.linear(hiddenWeight, xB, hiddenBias)
  -- Bias box is dirac at hiddenBias
  -- pointwise containment is trivial when lo=hi=b
  have hb1 : Box.contains (α:=ℝ) { lo := net.hiddenBias, hi := net.hiddenBias } net.hiddenBias := by
    intro i
    exact ⟨le_rfl, le_rfl⟩
  -- z1 containment
  have hz1 : Box.contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias })
      (Spec.linearSpec (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x) := by
    exact ibp_linear_sound_real net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias }
      x net.hiddenBias hx hb1
  -- Step 2: a1 ∈ IBP.relu(z1B)
  have ha1 : Box.contains (α:=ℝ)
      (IBP.relu (α:=ℝ)
        (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias }))
      (Activation.reluSpec (α:=ℝ)
        (Spec.linearSpec (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x)) := by
    exact ibp_relu_sound_real _ _ hz1
  -- Step 3: y ∈ IBP.linear(outputWeight, a1B, outputBias)
  -- Build a1B and b2B as in bound_ibp
  have hb2 : Box.contains (α:=ℝ) { lo := net.outputBias, hi := net.outputBias } net.outputBias := by
    intro i
    exact ⟨le_rfl, le_rfl⟩
  have hy : Box.contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.outputWeight
        (IBP.relu (α:=ℝ)
          (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias }))
        { lo := net.outputBias, hi := net.outputBias })
      (Spec.linearSpec (α:=ℝ) { weights := net.outputWeight, bias := net.outputBias }
        (Activation.reluSpec (α:=ℝ)
          (Spec.linearSpec (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x))) := by
    exact ibp_linear_sound_real net.outputWeight
      (IBP.relu (α:=ℝ)
        (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias }))
      { lo := net.outputBias, hi := net.outputBias }
      (Activation.reluSpec (α:=ℝ)
        (Spec.linearSpec (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x))
      net.outputBias
      ha1 hb2
  -- Combine: bound_ibp is exactly the composition of the above boxes
  -- Unfold bound_ibp and forward to match hy
  simpa [boundIbp, forward]

/--
Soundness of the affine-bound wrapper for a 2-layer MLP over `ℝ`.

In this module `boundAffine` delegates to the IBP implementation, so this theorem is a direct
corollary of `bound_ibp_sound`.
-/
theorem bound_affine_sound {inDim hidDim outDim : Nat}
  (net : TwoLayerMLP ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar))
  (x : Tensor ℝ [inDim])
  (hx : Box.contains (α:=ℝ) xB x) :
  Box.contains (α:=ℝ) (boundAffine (α:=ℝ) net xB) (forward (α:=ℝ) net x) := by
  -- `boundAffine` delegates to pure IBP bounds in this module.
  simpa [boundAffine] using bound_ibp_sound (net := net) (xB := xB) (x := x) hx

end Theorems

/- Public API -/
namespace Examples

/--
Compute both IBP bounds and affine-CROWN bounds for a two-layer MLP around an `ε`-box.

The input set is the axis-aligned box centered at `xCenter` with radius `eps` in each coordinate.
-/
def crownTwoLayerMlpBounds {inDim hidDim outDim : Nat} [BoundOps α]
  (hiddenLayer : Spec.LinearSpec α inDim hidDim)
  (outputLayer : Spec.LinearSpec α hidDim outDim)
  (xCenter : Tensor α [inDim]) (eps : α) :
  Box α (.dim outDim .scalar) × Box α (.dim outDim .scalar) :=
  let net := ofLinearSpecs (α:=α) hiddenLayer outputLayer
  let xB : Box α (.dim inDim .scalar) :=
    let rad := Tensor.scaleSpec (Tensor.full (α:=α) (.dim inDim .scalar) eps) 1
    { lo := Tensor.subSpec xCenter rad, hi := Tensor.addSpec xCenter rad }
  (boundIbp (α:=α) net xB, boundAffineCrown (α:=α) net xB)

end Examples

/- Classification helpers based on logit bounds -/
namespace Classify

open NN.MLTheory.CROWN

/-- Lower endpoint at index `i` from a vector box. -/
def lowerAt {n : Nat} (B : Box α (.dim n .scalar)) (i : Fin n) : α :=
  Tensor.getScalar B.lo i

/-- Upper endpoint at index `i` from a vector box. -/
def upperAt {n : Nat} (B : Box α (.dim n .scalar)) (i : Fin n) : α :=
  Tensor.getScalar B.hi i

/-- Maximum upper bound among competitors `k ≠ c`. -/
def maxCompetitorUpper {n : Nat} (B : Box α (.dim n .scalar)) (c : Fin n) : α :=
  let init : α := upperAt B c
  (List.finRange n).foldl (fun acc k =>
    if k ≠ c then
      let uk := upperAt B k
      if uk > acc then uk else acc
    else acc) init

/-- Certified margin lower bound: `lowerAt c - maxCompetitorUpper c`. -/
def certifiedMargin {n : Nat} (B : Box α (.dim n .scalar)) (c : Fin n) : α :=
  lowerAt B c - maxCompetitorUpper B c

/-- Decide whether class `c` is certified by a positive margin. -/
def isCertifiedClass {n : Nat} (B : Box α (.dim n .scalar)) (c : Fin n) : Bool :=
  decide (certifiedMargin (α:=α) B c > 0)

end Classify

end NN.MLTheory.CROWN
