/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Linear
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Mlp

CROWN/DeepPoly-style propagation for MLPs (vector in/out) using TorchLean tensors.

This file is a compact implementation that sits on top of:
- `NN.MLTheory.CROWN.Core` (`Box`, `AffineVec`, and `IBP.linear`), and
- TorchLean’s typed tensor layer (`Spec.Tensor`).

What is implemented:
- Per-neuron ReLU linear relaxations derived from pre-activation bounds (`ReLU.relax_scalar*`).
- A basic IBP forward pass for common activations (ReLU/sigmoid/tanh/etc.).
- A 2-layer ReLU MLP wrapper `MLP2` with a simple end-to-end bounding API.

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

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α]

/- ReLU relaxation parameters per neuron -/
/--
Parameters of a scalar linear relaxation used for bounding ReLU.

Typical shape (for crossing intervals `l <= 0 < u`) is an upper line `y <= slope * x + bias` with
`slope ∈ [0, 1]` and `bias >= 0`.
-/
structure ReLURelax (α : Type) where
  /-- Linear coefficient (often written `α` in the CROWN literature). -/
  slope : α
  /-- Constant offset (often written `β`). -/
  bias  : α

namespace ReLU

/--
Compute a standard (upper) linear relaxation for scalar ReLU on an interval `[l, u]`.

This matches the classic CROWN/DeepPoly choice:
- If `u <= 0`, ReLU is identically 0.
- If `l >= 0`, ReLU is the identity.
- If `l < 0 < u`, use the upper chord through `(l, 0)` and `(u, u)`.
-/
def relaxScalar (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      -- crossing zero: use upper line through (l,0),(u,u): y = α (x - l)
      let denom := (u - l)
      let αs := u / denom
      let β := -αs * l
      { slope := αs, bias := β }
  else
    { slope := 0, bias := 0 }

/--
Compute a basic (lower) linear relaxation for scalar ReLU on `[l, u]`.

For the crossing case `l <= 0 < u`, we choose either:
- `y >= 0` (slope 0), or
- `y >= x` (slope 1),
depending on which yields the tighter lower bound for the downstream objective.
-/
def relaxScalarLower (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      let slope := if u > (-l) then Numbers.one else Numbers.zero
      { slope := slope, bias := 0 }
  else
    { slope := 0, bias := 0 }

/-- Apply `relax_scalar` elementwise to vector bounds `(lo, hi)`. -/
def relaxVector {n : Nat} (lo hi : Tensor α (.dim n .scalar)) : Tensor (ReLURelax α) (.dim n
  .scalar) :=
  match lo, hi with
  | Tensor.dim l, Tensor.dim u =>
    Tensor.dim (fun i => match l i, u i with
      | Tensor.scalar li, Tensor.scalar ui => Tensor.scalar (relaxScalar li ui))

/-- Apply `relax_scalar_lower` elementwise to vector bounds `(lo, hi)`. -/
def relaxVectorLower {n : Nat} (lo hi : Tensor α (.dim n .scalar)) : Tensor (ReLURelax α) (.dim n
  .scalar) :=
  match lo, hi with
  | Tensor.dim l, Tensor.dim u =>
    Tensor.dim (fun i => match l i, u i with
      | Tensor.scalar li, Tensor.scalar ui => Tensor.scalar (relaxScalarLower li ui))

/--
Propagate an affine form through ReLU using a fixed per-neuron relaxation.

This is the "forward" DeepPoly-style propagation: given an affine bound on `z`, we produce an
affine bound on `relu(z)` by scaling rows and adjusting the constant term.
-/
def propagateAffine {inDim hidDim : Nat}
  (relax : Tensor (ReLURelax α) (.dim hidDim .scalar))
  (aff : AffineVec α inDim hidDim) : AffineVec α inDim hidDim :=
  match relax, aff.A, aff.c with
  | Tensor.dim r, Tensor.dim rows, Tensor.dim bias =>
    let A' := Tensor.dim (fun i =>
      match rows i, r i with
      | Tensor.dim cols, Tensor.scalar rp =>
        Tensor.dim (fun j =>
          match cols j with
          | Tensor.scalar aij => Tensor.scalar (aij * rp.slope)))
    let c' := Tensor.dim (fun i =>
      match bias i, r i with
      | Tensor.scalar ci, Tensor.scalar rp => Tensor.scalar (rp.slope * ci + rp.bias))
    { A := A', c := c' }

end ReLU

/-- Column-wise scaling of a matrix by a vector: scale each column `j` by `v[j]`. -/
def matColScaleSpec
  {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar)))
  (v : Tensor α (.dim n .scalar)) : Tensor α (.dim m (.dim n .scalar)) :=
  match A, v with
  | Tensor.dim rows, Tensor.dim vec =>
    Tensor.dim (fun i =>
      match rows i with
      | Tensor.dim cols =>
        Tensor.dim (fun j =>
          match cols j, vec j with
          | Tensor.scalar aij, Tensor.scalar vj => Tensor.scalar (aij * vj)))

/-- Elementwise positive part of a matrix: replace negative entries by `0`. -/
def matPosSpec {m n : Nat}
  (A : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n .scalar)) :=
  match A with
  | Tensor.dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | Tensor.dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | Tensor.scalar aij =>
            Tensor.scalar (if aij > Numbers.zero then aij else Numbers.zero)))

/-- Elementwise negative part of a matrix: replace positive entries by `0`. -/
def matNegSpec {m n : Nat}
  (A : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n .scalar)) :=
  match A with
  | Tensor.dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | Tensor.dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | Tensor.scalar aij =>
            Tensor.scalar (if aij > Numbers.zero then Numbers.zero else aij)))

/-- Extract the slope vector from a tensor of ReLU relaxations. -/
def reluRelaxSlopeVec {n : Nat}
  (relax : Tensor (ReLURelax α) (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match relax with
  | Tensor.dim r =>
    Tensor.dim (fun i => match r i with | Tensor.scalar rp => Tensor.scalar rp.slope)

/-- Extract the bias vector from a tensor of ReLU relaxations. -/
def reluRelaxBiasVec {n : Nat}
  (relax : Tensor (ReLURelax α) (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match relax with
  | Tensor.dim r =>
    Tensor.dim (fun i => match r i with | Tensor.scalar rp => Tensor.scalar rp.bias)

/- Interval forward (IBP) for MLP layer + ReLU -/
namespace IBP

/-!
Interval Bound Propagation (IBP) utilities for vector-shaped activations.

These are correctness-first helpers: they are intended to be sound for any `Context α` backend that
implements the scalar operations used by `Activation.Math.*_spec`.

The linear-layer bound helper `IBP.linear` lives in `NN.MLTheory.CROWN.Core`.
-/

/--
Interval bounds for ReLU on a vector.

This is the standard elementwise interval evaluation:
`relu([l,u]) = [relu(l), relu(u)]`.
-/
def relu {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l =>
        let l' := if l > 0 then l else 0
        Tensor.scalar l')
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u =>
        let u' := if u > 0 then u else 0
        Tensor.scalar u')
    { lo := outLo, hi := outHi }

/--
Re-export of the runtime-only monotone-activation IBP helper.

We keep this file compact and Mathlib-friendly for proofs, but we do not want to
maintain two copies of the same computational rule. Canonical implementation lives in:
`NN.MLTheory.CROWN.Runtime.Ops.IBP.map_minmax`.

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

/--
Interval bounds for leaky-ReLU.

Sound when the negative slope `αₗ` is nonnegative.
-/
def leakyRelu {n : Nat} (αₗ : α) (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  mapMinmax (fun x => Activation.Math.leakyReluSpec x αₗ) xB

/--
Interval bounds for ELU.

Sound for `alpha > 0`.
-/
def elu {n : Nat} (alpha : α) (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  mapMinmax (fun x => Activation.Math.eluSpec x alpha) xB

end IBP

/--
2-layer MLP payload used by this file.

Semantics: `y = W2 * relu(W1 * x + b1) + b2`.

PyTorch analogue: `torch.nn.Sequential(Linear(inDim,hidDim), ReLU(), Linear(hidDim,outDim))`.
-/
structure MLP2 (α : Type) (inDim hidDim outDim : Nat) where
  /-- First layer weight matrix. -/
  W1 : Tensor α (.dim hidDim (.dim inDim .scalar))
  /-- First layer bias vector. -/
  b1 : Tensor α (.dim hidDim .scalar)
  /-- Second layer weight matrix. -/
  W2 : Tensor α (.dim outDim (.dim hidDim .scalar))
  /-- Second layer bias vector. -/
  b2 : Tensor α (.dim outDim .scalar)

/-- Forward semantics for `MLP2` (used to state soundness theorems). -/
def forward {inDim hidDim outDim : Nat}
  (net : MLP2 α inDim hidDim outDim)
  (x : Tensor α (.dim inDim .scalar)) : Tensor α (.dim outDim .scalar) :=
  let l1 : Spec.LinearSpec α inDim hidDim := { weights := net.W1, bias := net.b1 }
  let l2 : Spec.LinearSpec α hidDim outDim := { weights := net.W2, bias := net.b2 }
  let z1 := Spec.linearSpec (α:=α) l1 x
  let a1 := Activation.reluSpec (α:=α) z1
  Spec.linearSpec (α:=α) l2 a1

/-- Build an `MLP2` from two `LinearSpec` records. -/
def ofLinearSpecs {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim) (l2 : Spec.LinearSpec α hidDim outDim) :
  MLP2 α inDim hidDim outDim :=
  { W1 := l1.weights, b1 := l1.bias, W2 := l2.weights, b2 := l2.bias }

/--
Compute an output interval box via pure IBP.

This is fast and always sound, but typically looser than CROWN/DeepPoly affine bounds.
-/
def boundIbp {inDim hidDim outDim : Nat}
  (net : MLP2 α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  -- z1 = W1 x + b1
  let b1B : Box α (.dim hidDim .scalar) := { lo := net.b1, hi := net.b1 }
  let z1B := IBP.linear net.W1 xB b1B
  let a1B := IBP.relu (n:=hidDim) z1B
  let b2B : Box α (.dim outDim .scalar) := { lo := net.b2, hi := net.b2 }
  IBP.linear net.W2 a1B b2B

/--
Single-pass affine (CROWN/DeepPoly-style) bounds for the 2-layer ReLU MLP.

This constructs upper and lower affine forms by combining:
- pre-activation intervals from IBP,
- per-neuron ReLU relaxations, and
- sign-splitting of the second-layer weights.

Note: this is the direct MLP-specialized implementation; the graph-level engine in
`NN.MLTheory.CROWN.Graph` is the long-term home for fully-general CROWN.
-/
def boundAffineCrown {inDim hidDim outDim : Nat}
  (net : MLP2 α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  -- Basic CROWN for 2-layer MLP:
  -- 1) Use IBP to get pre-activation bounds [l,u].
  -- 2) Build ReLU upper and lower linear relaxations per hidden unit.
  -- 3) Combine them with sign-splitting of W2 to obtain both an affine lower and upper bound,
  --    then evaluate those on the input box.
  let b1B : Box α (.dim hidDim .scalar) := { lo := net.b1, hi := net.b1 }
  let z1B := IBP.linear (α:=α) net.W1 xB b1B
  let relaxU := ReLU.relaxVector (α:=α) (n:=hidDim) z1B.lo z1B.hi
  let relaxL := ReLU.relaxVectorLower (α:=α) (n:=hidDim) z1B.lo z1B.hi
  let slopeU := reluRelaxSlopeVec (α:=α) (n:=hidDim) relaxU
  let biasU  := reluRelaxBiasVec  (α:=α) (n:=hidDim) relaxU
  let slopeL := reluRelaxSlopeVec (α:=α) (n:=hidDim) relaxL
  let biasL  := reluRelaxBiasVec  (α:=α) (n:=hidDim) relaxL

  let W2pos := matPosSpec (α:=α) (m:=outDim) (n:=hidDim) net.W2
  let W2neg := matNegSpec (α:=α) (m:=outDim) (n:=hidDim) net.W2

  -- Upper affine: W2pos uses ReLU upper, W2neg uses ReLU lower.
  let W2posU := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2pos slopeU
  let W2negL := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2neg slopeL
  let AU := Spec.matMulSpec (α:=α) (Tensor.addSpec W2posU W2negL) net.W1
  let innerU_pos := Tensor.addSpec (Tensor.mulSpec slopeU net.b1) biasU
  let innerL_neg := Tensor.addSpec (Tensor.mulSpec slopeL net.b1) biasL
  let cU :=
    Tensor.addSpec
      (Tensor.addSpec
        (Spec.matVecMulSpec (α:=α) W2pos innerU_pos)
        (Spec.matVecMulSpec (α:=α) W2neg innerL_neg))
      net.b2

  -- Lower affine: W2pos uses ReLU lower, W2neg uses ReLU upper.
  let W2posL := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2pos slopeL
  let W2negU := matColScaleSpec (α:=α) (m:=outDim) (n:=hidDim) W2neg slopeU
  let AL := Spec.matMulSpec (α:=α) (Tensor.addSpec W2posL W2negU) net.W1
  let innerL_pos := Tensor.addSpec (Tensor.mulSpec slopeL net.b1) biasL
  let innerU_neg := Tensor.addSpec (Tensor.mulSpec slopeU net.b1) biasU
  let cL :=
    Tensor.addSpec
      (Tensor.addSpec
        (Spec.matVecMulSpec (α:=α) W2pos innerL_pos)
        (Spec.matVecMulSpec (α:=α) W2neg innerU_neg))
      net.b2

  let affU : AffineVec α inDim outDim := AffineVec.ofLinear (α:=α) AU cU
  let affL : AffineVec α inDim outDim := AffineVec.ofLinear (α:=α) AL cL
  let BU := AffineVec.evalOnBox (α:=α) affU xB
  let BL := AffineVec.evalOnBox (α:=α) affL xB
  { lo := BL.lo, hi := BU.hi }

/--
End-to-end bound API exposed by this file.

This API returns the IBP bound (sound for all backends); `bound_affine_crown` is kept as a
reference implementation for the 2-layer ReLU case.
-/
def boundAffine {inDim hidDim outDim : Nat}
  (net : MLP2 α inDim hidDim outDim)
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

If `x ∈ [l, u]` and `rp := ReLU.relax_scalar l u`, then:
`relu(x) <= rp.slope * x + rp.bias`.

This is the standard CROWN/DeepPoly upper chord construction (arXiv:1811.00866).
-/
theorem relu_relax_scalar_upper_real
  (l u x : ℝ)
  (hlx : l ≤ x) (hxu : x ≤ u) :
  let rp := ReLU.relaxScalar (α:=ℝ) l u
  Activation.Math.reluSpec (α:=ℝ) x ≤ rp.slope * x + rp.bias := by
  -- Work by cases on signs of l,u (standard CROWN cases)
  unfold ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · -- both positive: rp.slope = 1, rp.bias = 0, relu(x)=x
      have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec, max_eq_left hxnonneg]
    · -- crossing: l ≤ 0 < u, rp.slope = u/(u-l), rp.bias = -(u/(u-l)*l)
      have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      have hne : (u - l) ≠ 0 := ne_of_gt hden
      simp only [hu, hlpos, if_true, if_false]
      -- two subcases depending on x sign
      by_cases hxpos : 0 < x
      · -- 0 < x ≤ u: relu x = x. Show x ≤ (u/(u-l))*x - (u/(u-l))*l
        have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.Math.reluSpec, max_eq_left hxnonneg]
        -- It suffices to prove: x ≤ (u/(u-l)) * (x - l)
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          -- Show (u - l) * x ≤ u * (x - l), then cancel (u - l) > 0
          have hrewrite : (u - l) * x - u * (x - l) = l * (u - x) := by
            ring
          have hxux : 0 ≤ u - x := sub_nonneg.mpr hxu
          have hxmul_le : l * (u - x) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hle0 hxux
          have hmul_goal : (u - l) * x ≤ u * (x - l) := by
            have : (u - l) * x - u * (x - l) ≤ 0 := by
              simpa [hrewrite] using hxmul_le
            exact sub_nonpos.mp this
          -- Divide both sides by (u - l) > 0 using le_div_iff₀ (group-with-zero variant)
          have hx_to_goal' : x ≤ (u * (x - l)) / (u - l) := by
            -- turn (u - l) * x ≤ u * (x - l) into x * (u - l) ≤ u * (x - l)
            have : x * (u - l) ≤ u * (x - l) := by simpa [mul_comm] using hmul_goal
            exact (le_div_iff₀ (G₀ := ℝ) hden).mpr this
          simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
            using hx_to_goal'
        -- Turn the RHS back into the original affine form
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by
          ring
        simpa [h2]
          using hx_to_goal
      · -- x ≤ 0: relu x = 0 and RHS = u/(u-l) * x + (-(u/(u-l) * l))
        have hxle : x ≤ 0 := le_of_not_gt hxpos
        -- ReLU x = 0 in this branch
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by
          ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · exact div_nonneg (le_of_lt hu) (le_of_lt hden)
          · linarith
        simpa [Activation.Math.reluSpec, max_eq_right hxle, h1]
          using this
  · -- u ≤ 0: relu x = 0 and rp.slope = 0, rp.bias = 0
    have hule : u ≤ 0 := le_of_not_gt hu
    have hxle0 : x ≤ 0 := le_trans hxu hule
    simp [hu, Activation.Math.reluSpec, hxle0]

/--
Vectorized ReLU relaxation (pointwise upper bound) over `ℝ`.

If `x ∈ [lo, hi]` and `rp := ReLU.relax_vector lo hi`, then for every component `i` we have
`relu(xᵢ) ≤ rpᵢ.slope * xᵢ + rpᵢ.bias`.
-/
theorem relu_relax_vector_pointwise_upper_real {n : Nat}
  (lo hi x : Tensor ℝ (.dim n .scalar))
  (hIn : Box.contains (α:=ℝ) { lo := lo, hi := hi } x) :
  ∀ i : Fin n,
    let li := match lo with | .dim flo => match flo i with | .scalar v => v
    let ui := match hi with | .dim fhi => match fhi i with | .scalar v => v
    let xi := match x with | .dim fx => match fx i with | .scalar v => v
    let rp := ReLU.relaxScalar (α:=ℝ) li ui
    Activation.Math.reluSpec (α:=ℝ) xi ≤ rp.slope * xi + rp.bias :=
  by
  classical
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      cases x with
      | dim fx =>
        intro i
        -- Record equalities for components at index i, to guide simp reductions
        have hIn_i := hIn i
        cases hli : flo i with
        | scalar li_ =>
          cases hui : fhi i with
          | scalar ui_ =>
            cases hxi : fx i with
            | scalar xi_ =>
              -- Extract scalar bounds at index i from the box containment hypothesis
              have hpair : li_ ≤ xi_ ∧ xi_ ≤ ui_ := by
                simpa [Box.contains, hli, hui, hxi] using hIn_i
              have hlx : li_ ≤ xi_ := hpair.1
              have hxu : xi_ ≤ ui_ := hpair.2
              -- Simplify the goal to eliminate let/match binders using recorded equalities
              simp [hli, hui, hxi]
              -- Finish with the scalar relaxation lemma
              exact
                relu_relax_scalar_upper_real (l:=li_) (u:=ui_) (x:=xi_) hlx hxu

/- Pure IBP soundness for the 2-layer MLP. -/
/--
Soundness of `IBP.linear` over `ℝ`.

If `x ∈ xB` and `b ∈ bB`, then `W*x + b` lies in the interval box computed by
`IBP.linear W xB bB`.
-/
theorem ibp_linear_sound_real {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar))
  (x : Tensor ℝ (.dim n .scalar)) (b : Tensor ℝ (.dim m .scalar))
  (hx : Box.contains (α:=ℝ) xB x) (hb : Box.contains (α:=ℝ) bB b) :
  Box.contains (α:=ℝ) (IBP.linear (α:=ℝ) W xB bB)
    (Spec.linearSpec (α:=ℝ) { weights := W, bias := b } x) := by
  classical
  -- Make the `BoundOps` instance explicit (and named) so `simp [instBO]` can unfold primitives.
  letI instBO : NN.MLTheory.CROWN.BoundOps ℝ :=
    { addDown := (· + ·)
      addUp := (· + ·)
      subDown := (· - ·)
      subUp := (· - ·)
      mulDown := (· * ·)
      mulUp := (· * ·) }
  -- Unpack structures
  cases W with
  | dim rows =>
    cases xB with
    | mk xBlo xBhi =>
      cases xBlo with
      | dim xlo =>
        cases xBhi with
        | dim xhi =>
          cases bB with
          | mk bBlo bBhi =>
            cases bBlo with
            | dim blo =>
              cases bBhi with
              | dim bhi =>
                cases x with
                | dim xv =>
                  cases b with
                  | dim bv =>
                    -- Destructure the output box and value to reach a pointwise goal
                    -- Reduce big Box.contains to pointwise scalar inequalities immediately
                    simp (config := { iota := true }) [IBP.linear, Spec.linearSpec,
                      Spec.matVecMulSpec]
                    intro i
                    -- Destructure bias components at i into scalars with equalities
                    cases hblo : blo i with
                    | scalar bi_lo =>
                      cases hbhi : bhi i with
                      | scalar bi_hi =>
                        cases hbv : bv i with
                        | scalar bi =>
                          -- Bias scalar bounds at i
                          have hb_i : bi_lo ≤ bi ∧ bi ≤ bi_hi := by
                            have hb_i0 := hb i
                            -- Unfold Box.contains for scalar bias at index i
                            simpa [Box.contains, hblo, hbhi, hbv] using hb_i0
                          -- Row i bounds from input intervals
                          cases hrow : rows i with
                    | dim cols =>
                      -- Local definitions for per-index contributions
                      let lower := fun (j : Fin n) =>
                        let aij := (match cols j with | Tensor.scalar a => a)
                        let xlo_j := (match xlo j with | Tensor.scalar v => v)
                        let xhi_j := (match xhi j with | Tensor.scalar v => v)
                        BoundOps.min2 (BoundOps.mulDown aij xlo_j) (BoundOps.mulDown aij xhi_j)
                      let upper := fun (j : Fin n) =>
                        let aij := (match cols j with | Tensor.scalar a => a)
                        let xlo_j := (match xlo j with | Tensor.scalar v => v)
                        let xhi_j := (match xhi j with | Tensor.scalar v => v)
                        BoundOps.max2 (BoundOps.mulUp aij xlo_j) (BoundOps.mulUp aij xhi_j)
                      let mid := fun (j : Fin n) =>
                        let aij := (match cols j with | Tensor.scalar a => a)
                        let xj := (match xv j with | Tensor.scalar v => v)
                        aij * xj
                      -- Per-j bounds
                      have per_j : ∀ j : Fin n, lower j ≤ mid j ∧ mid j ≤ upper j := by
                        intro j; cases hcol : cols j with
                        | scalar aij =>
                          cases hxlo : xlo j with
                          | scalar xlo_j =>
                            cases hxhi : xhi j with
                            | scalar xhi_j =>
                              cases hxv : xv j with
                              | scalar xj =>
                                have min2_le_left (a b : ℝ) : BoundOps.min2 a b ≤ a := by
                                  by_cases hab : a > b
                                  · have : b ≤ a := le_of_lt hab
                                    simp [BoundOps.min2, hab, this]
                                  · simp [BoundOps.min2, hab]
                                have min2_le_right (a b : ℝ) : BoundOps.min2 a b ≤ b := by
                                  by_cases hab : a > b
                                  · simp [BoundOps.min2, hab]
                                  · have : a ≤ b := le_of_not_gt hab
                                    simp [BoundOps.min2, hab, this]
                                have le_max2_left (a b : ℝ) : a ≤ BoundOps.max2 a b := by
                                  by_cases hab : a > b
                                  · simp [BoundOps.max2, hab]
                                  · have : a ≤ b := le_of_not_gt hab
                                    simp [BoundOps.max2, hab, this]
                                have le_max2_right (a b : ℝ) : b ≤ BoundOps.max2 a b := by
                                  by_cases hab : a > b
                                  · have : b ≤ a := le_of_lt hab
                                    simp [BoundOps.max2, hab, this]
                                  · simp [BoundOps.max2, hab]
                                -- input scalar bounds from hx at index j
                                have hxj : xlo_j ≤ xj ∧ xj ≤ xhi_j := by
                                  have h := hx j
                                  simpa [Box.contains, hxlo, hxhi, hxv] using h
                                have hlx := hxj.1; have hxu := hxj.2
                                let p1 := aij * xlo_j
                                let p2 := aij * xhi_j
                                by_cases hsign : 0 ≤ aij
                                ·
                                  have h1 : p1 ≤ aij * xj := by
                                    simpa [p1] using mul_le_mul_of_nonneg_left hlx hsign
                                  have h2 : aij * xj ≤ p2 := by
                                    simpa [p2] using mul_le_mul_of_nonneg_left hxu hsign
                                  constructor
                                  ·
                                    have : BoundOps.min2 p1 p2 ≤ aij * xj :=
                                      le_trans (min2_le_left p1 p2) h1
                                    simpa
                                      [lower, mid, hcol, hxlo, hxhi, hxv, p1, p2, instBO]
                                      using this
                                  ·
                                    have : aij * xj ≤ BoundOps.max2 p1 p2 :=
                                      le_trans h2 (le_max2_right p1 p2)
                                    simpa
                                      [upper, mid, hcol, hxlo, hxhi, hxv, p1, p2, instBO]
                                      using this
                                ·
                                  have hsign' : aij ≤ 0 := le_of_not_ge hsign
                                  have h1 : p2 ≤ aij * xj := by
                                    simpa [p2] using mul_le_mul_of_nonpos_left hxu hsign'
                                  have h2 : aij * xj ≤ p1 := by
                                    simpa [p1] using mul_le_mul_of_nonpos_left hlx hsign'
                                  constructor
                                  ·
                                    have : BoundOps.min2 p1 p2 ≤ aij * xj :=
                                      le_trans (min2_le_right p1 p2) h1
                                    simpa
                                      [lower, mid, hcol, hxlo, hxhi, hxv, p1, p2, instBO]
                                      using this
                                  ·
                                    have : aij * xj ≤ BoundOps.max2 p1 p2 :=
                                      le_trans h2 (le_max2_left p1 p2)
                                    simpa
                                      [upper, mid, hcol, hxlo, hxhi, hxv, p1, p2, instBO]
                                      using this
                      -- Fold monotonicity lemmas
                      have fold_lower_mid : ∀ (l : List (Fin n)) (acc1 acc2 : ℝ), acc1 ≤ acc2 →
                        l.foldl (fun acc j => acc + lower j) acc1 ≤
                        l.foldl (fun acc j => acc + mid j) acc2 := by
                        intro l; induction l with
                        | nil => intro acc1 acc2 h; simpa
                        | cons j l ih =>
                          intro acc1 acc2 h
                          have hj := (per_j j).1
                          have h' : acc1 + lower j ≤ acc2 + mid j := add_le_add h hj
                          simpa [List.foldl] using ih (acc1 + lower j) (acc2 + mid j) h'
                      have fold_mid_upper : ∀ (l : List (Fin n)) (acc1 acc2 : ℝ), acc1 ≤ acc2 →
                        l.foldl (fun acc j => acc + mid j) acc1 ≤
                        l.foldl (fun acc j => acc + upper j) acc2 := by
                        intro l; induction l with
                        | nil => intro acc1 acc2 h; simpa
                        | cons j l ih =>
                          intro acc1 acc2 h
                          have hj := (per_j j).2
                          have h' : acc1 + mid j ≤ acc2 + upper j := add_le_add h hj
                          simpa [List.foldl] using ih (acc1 + mid j) (acc2 + upper j) h'
                      -- Apply with initial 0 on the full index list
                      have hLsum : (List.finRange n).foldl (fun acc j => acc + lower j) 0 ≤
                                   (List.finRange n).foldl (fun acc j => acc + mid j) 0 :=
                        fold_lower_mid (List.finRange n) 0 0 (le_of_eq rfl)
                      have hUsum : (List.finRange n).foldl (fun acc j => acc + mid j) 0 ≤
                                   (List.finRange n).foldl (fun acc j => acc + upper j) 0 :=
                        fold_mid_upper (List.finRange n) 0 0 (le_of_eq rfl)
                      -- Finish by adding bias bounds
                      rcases hb_i with ⟨hbiL, hbiU⟩
                      -- Unfold containment at scalar shape to a pair of inequalities
                      -- Reduce contains at scalar shape and the map2_spec on RHS
                      -- Expose scalar fold results for lo/hi and the RHS mid-sum at index i
                      -- Lower sum over j
                      cases hFoldL :
                        (List.foldl
                          (fun acc j =>
                            AffineVec.evalOnBox.match_1 (α:=ℝ)
                              (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) => Tensor ℝ
                                Shape.scalar)
                              acc (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                Tensor.scalar
                                  (BoundOps.addDown accv
                                    (BoundOps.min2 (BoundOps.mulDown aij xlo) (BoundOps.mulDown aij
                                      xhi)))))
                          (Tensor.scalar 0) (List.finRange n))
                      with
                      | scalar sumL =>
                        -- Upper sum over j
                        cases hFoldU :
                          (List.foldl
                            (fun acc j =>
                              AffineVec.evalOnBox.match_1 (α:=ℝ)
                                (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) => Tensor ℝ
                                  Shape.scalar)
                                acc (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                  Tensor.scalar
                                    (BoundOps.addUp accv
                                      (BoundOps.max2 (BoundOps.mulUp aij xlo) (BoundOps.mulUp aij
                                        xhi)))))
                            (Tensor.scalar 0) (List.finRange n))
                        with
                        | scalar sumU =>
                          -- Mid (mat-vec) sum on RHS
                          cases hFoldM :
                            (match rows i with
                            | Tensor.dim colsA =>
                              List.foldl
                                (fun acc k =>
                                  Spec.matVecMulSpec.match_1 (α:=ℝ)
                                    (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                      Shape.scalar)
                                    acc (colsA k) (xv k) (fun s ak vk => Tensor.scalar (s + ak *
                                      vk)))
                                (Tensor.scalar 0) (List.finRange n))
                          with
                          | scalar sumM =>
                            -- Build the pair of scalar inequalities we need
                            -- Helper: interpret the Tensor-fold as a numeric fold.
                            have toScalar_fold_lower :
                                ∀ (l : List (Fin n)) (acc : ℝ),
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addDown accv
                                                  (BoundOps.min2 (BoundOps.mulDown aij xlo)
                                                    (BoundOps.mulDown aij xhi)))))
                                        (Tensor.scalar acc) l) =
                                    l.foldl (fun acc j => acc + lower j) acc := by
                              intro l
                              induction l with
                              | nil =>
                                intro acc
                                rfl
                              | cons j l ih =>
                                intro acc
                                cases hcol : cols j with
                                | scalar aij =>
                                  cases hxlo : xlo j with
                                  | scalar xlo_j =>
                                    cases hxhi : xhi j with
                                    | scalar xhi_j =>
                                      simp [Tensor.toScalar, List.foldl, lower, hcol, hxlo, hxhi]
                                      simpa [lower, hcol, hxlo, hxhi, instBO] using
                                        ih (acc + BoundOps.min2 (aij * xlo_j) (aij * xhi_j))
                            have toScalar_fold_upper :
                                ∀ (l : List (Fin n)) (acc : ℝ),
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addUp accv
                                                  (BoundOps.max2 (BoundOps.mulUp aij xlo)
                                                    (BoundOps.mulUp aij xhi)))))
                                        (Tensor.scalar acc) l) =
                                    l.foldl (fun acc j => acc + upper j) acc := by
                              intro l
                              induction l with
                              | nil =>
                                intro acc
                                rfl
                              | cons j l ih =>
                                intro acc
                                cases hcol : cols j with
                                | scalar aij =>
                                  cases hxlo : xlo j with
                                  | scalar xlo_j =>
                                    cases hxhi : xhi j with
                                    | scalar xhi_j =>
                                      simp [Tensor.toScalar, List.foldl, upper, hcol, hxlo, hxhi]
                                      simpa [upper, hcol, hxlo, hxhi, instBO] using
                                        ih (acc + BoundOps.max2 (aij * xlo_j) (aij * xhi_j))
                            have toScalar_fold_mid :
                                ∀ (l : List (Fin n)) (acc : ℝ),
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT k =>
                                          Spec.matVecMulSpec.match_1 (α:=ℝ)
                                            (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                              Shape.scalar)
                                            accT (cols k) (xv k) (fun s ak vk => Tensor.scalar (s +
                                              ak * vk)))
                                        (Tensor.scalar acc) l) =
                                    l.foldl (fun acc j => acc + mid j) acc := by
                              intro l
                              induction l with
                              | nil =>
                                intro acc
                                rfl
                              | cons j l ih =>
                                intro acc
                                cases hcol : cols j with
                                | scalar aij =>
                                  cases hxv : xv j with
                                  | scalar xj =>
                                    simp [Tensor.toScalar, List.foldl, mid, hcol, hxv]
                                    simpa using ih (acc + aij * xj)

                            -- Identify sumL/sumU/sumM with the corresponding numeric folds.
                            have sumL_def :
                                (List.finRange n).foldl (fun acc j => acc + lower j) 0 = sumL := by
                              have hto :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addDown accv
                                                  (BoundOps.min2 (BoundOps.mulDown aij xlo)
                                                    (BoundOps.mulDown aij xhi)))))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    (List.finRange n).foldl (fun acc j => acc + lower j) 0 := by
                                simpa using toScalar_fold_lower (List.finRange n) 0
                              have hscalar :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addDown accv
                                                  (BoundOps.min2 (BoundOps.mulDown aij xlo)
                                                    (BoundOps.mulDown aij xhi)))))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    sumL := by
                                simpa using (congrArg Tensor.toScalar hFoldL)
                              exact hto.symm.trans hscalar
                            have sumU_def :
                                (List.finRange n).foldl (fun acc j => acc + upper j) 0 = sumU := by
                              have hto :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addUp accv
                                                  (BoundOps.max2 (BoundOps.mulUp aij xlo)
                                                    (BoundOps.mulUp aij xhi)))))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    (List.finRange n).foldl (fun acc j => acc + upper j) 0 := by
                                simpa using toScalar_fold_upper (List.finRange n) 0
                              have hscalar :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT j =>
                                          AffineVec.evalOnBox.match_1 (α:=ℝ)
                                            (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                              Tensor ℝ Shape.scalar)
                                            accT (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                              Tensor.scalar
                                                (BoundOps.addUp accv
                                                  (BoundOps.max2 (BoundOps.mulUp aij xlo)
                                                    (BoundOps.mulUp aij xhi)))))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    sumU := by
                                simpa using (congrArg Tensor.toScalar hFoldU)
                              exact hto.symm.trans hscalar
                            have hFoldM' :
                                (List.foldl
                                  (fun acc k =>
                                    Spec.matVecMulSpec.match_1 (α:=ℝ)
                                      (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                        Shape.scalar)
                                      acc (cols k) (xv k) (fun s ak vk => Tensor.scalar (s + ak *
                                        vk)))
                                  (Tensor.scalar 0) (List.finRange n)) = Tensor.scalar sumM := by
                              simpa [hrow] using hFoldM
                            have sumM_def :
                                (List.finRange n).foldl (fun acc j => acc + mid j) 0 = sumM := by
                              have hto :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT k =>
                                          Spec.matVecMulSpec.match_1 (α:=ℝ)
                                            (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                              Shape.scalar)
                                            accT (cols k) (xv k) (fun s ak vk => Tensor.scalar (s +
                                              ak * vk)))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    (List.finRange n).foldl (fun acc j => acc + mid j) 0 := by
                                simpa using toScalar_fold_mid (List.finRange n) 0
                              have hscalar :
                                  Tensor.toScalar
                                      (List.foldl
                                        (fun accT k =>
                                          Spec.matVecMulSpec.match_1 (α:=ℝ)
                                            (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                              Shape.scalar)
                                            accT (cols k) (xv k) (fun s ak vk => Tensor.scalar (s +
                                              ak * vk)))
                                        (Tensor.scalar 0) (List.finRange n)) =
                                    sumM := by
                                simpa using (congrArg Tensor.toScalar hFoldM')
                              exact hto.symm.trans hscalar

                            have hsumL : sumL ≤ sumM := by
                              have := hLsum
                              simpa [sumL_def, sumM_def] using this
                            have hsumU : sumM ≤ sumU := by
                              have := hUsum
                              simpa [sumM_def, sumU_def] using this
                            have h1 : BoundOps.addDown sumL bi_lo ≤ sumM + bi := by
                              simpa [instBO] using (add_le_add hsumL hbiL)
                            have h2 : sumM + bi ≤ BoundOps.addUp sumU bi_hi := by
                              simpa [instBO] using (add_le_add hsumU hbiU)
                            -- Rewrite the goal into the scalar conjunction and finish with
                            -- `h1`/`h2`.
                            -- First, simplify the row/bias matches.
                            simp (config := { iota := true }) [hrow, hblo, hbhi, hbv]
                            -- Now `Box.contains` at scalar shape is just a conjunction.
                            have hLo :
                                IBP.linear.match_1 (α:=ℝ)
                                    (fun (s : Tensor ℝ Shape.scalar) => Tensor ℝ Shape.scalar)
                                    (List.foldl
                                      (fun acc j =>
                                        AffineVec.evalOnBox.match_1 (α:=ℝ)
                                          (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                            Tensor ℝ Shape.scalar)
                                          acc (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                            Tensor.scalar
                                              (BoundOps.addDown accv
                                                (BoundOps.min2 (BoundOps.mulDown aij xlo)
                                                  (BoundOps.mulDown aij xhi)))))
                                      (Tensor.scalar 0) (List.finRange n))
                                    (fun sv => Tensor.scalar (BoundOps.addDown sv bi_lo)) =
                                  Tensor.scalar (BoundOps.addDown sumL bi_lo) := by
                              simp (config := { iota := true }) [hFoldL]
                            have hHi :
                                IBP.linear.match_1 (α:=ℝ)
                                    (fun (s : Tensor ℝ Shape.scalar) => Tensor ℝ Shape.scalar)
                                    (List.foldl
                                      (fun acc j =>
                                        AffineVec.evalOnBox.match_1 (α:=ℝ)
                                          (fun (_acc _col _xlo _xhi : Tensor ℝ Shape.scalar) =>
                                            Tensor ℝ Shape.scalar)
                                          acc (cols j) (xlo j) (xhi j) (fun accv aij xlo xhi =>
                                            Tensor.scalar
                                              (BoundOps.addUp accv
                                                (BoundOps.max2 (BoundOps.mulUp aij xlo)
                                                  (BoundOps.mulUp aij xhi)))))
                                      (Tensor.scalar 0) (List.finRange n))
                                    (fun sv => Tensor.scalar (BoundOps.addUp sv bi_hi)) =
                                  Tensor.scalar (BoundOps.addUp sumU bi_hi) := by
                              simp (config := { iota := true }) [hFoldU]
                            have hVal :
                                Tensor.map2Spec (fun x1 x2 : ℝ => x1 + x2)
                                    (List.foldl
                                      (fun acc k =>
                                        Spec.matVecMulSpec.match_1 (α:=ℝ)
                                          (fun (_acc _a _v : Tensor ℝ Shape.scalar) => Tensor ℝ
                                            Shape.scalar)
                                          acc (cols k) (xv k) (fun s ak vk => Tensor.scalar (s + ak
                                            * vk)))
                                      (Tensor.scalar 0) (List.finRange n))
                                    (Tensor.scalar bi) =
                                  Tensor.scalar (sumM + bi) := by
                              simp (config := { iota := true }) [Tensor.map2Spec, hFoldM']
                            have hLoS := congrArg Tensor.toScalar hLo
                            have hHiS := congrArg Tensor.toScalar hHi
                            have hValS := congrArg Tensor.toScalar hVal
                            simp [Tensor.toScalar] at hLoS hHiS hValS
                            -- Finish the scalar containment goal via `Tensor.toScalar`.
                            have hcontains_scalar_iff (loT hiT yT : Tensor ℝ Shape.scalar) :
                                Box.contains (α:=ℝ) { lo := loT, hi := hiT } yT ↔
                                  (Tensor.toScalar loT ≤ Tensor.toScalar yT ∧
                                    Tensor.toScalar yT ≤ Tensor.toScalar hiT) := by
                              cases loT; cases hiT; cases yT; simp [Box.contains, Tensor.toScalar]
                            apply (hcontains_scalar_iff _ _ _).2
                            constructor
                            · -- lower bound
                              -- Rewrite `h1` into the exact scalar goal.
                              have h1' := h1
                              rw [← hLoS, ← hValS] at h1'
                              simpa [instBO] using h1'
                            · -- upper bound
                              have h2' := h2
                              rw [← hValS, ← hHiS] at h2'
                              simpa [instBO] using h2'

/- Helper: soundness of IBP.relu over ℝ -/
private theorem ibp_relu_sound_real {n : Nat}
  (zB : Box ℝ (.dim n .scalar))
  (z : Tensor ℝ (.dim n .scalar))
  (hz : Box.contains (α:=ℝ) zB z) :
  Box.contains (α:=ℝ) (IBP.relu (α:=ℝ) zB) (Activation.reluSpec (α:=ℝ) z) := by
  classical
  -- Reduce big Box.contains to pointwise scalar inequalities
  cases zB with
  | mk zBlo zBhi =>
    cases zBlo with
    | dim zlo =>
      cases zBhi with
      | dim zhi =>
        cases z with
        | dim zv =>
          simp [IBP.relu, Activation.reluSpec]
          intro i
          cases hzl : zlo i with
          | scalar l =>
            cases hzh : zhi i with
            | scalar u =>
              cases hzv : zv i with
              | scalar x =>
                have hx : l ≤ x ∧ x ≤ u := by
                  have := hz i
                  simpa [Box.contains, hzl, hzh, hzv] using this
                have hxlo : l ≤ x := hx.1
                have hxhi : x ≤ u := hx.2
                -- simplify the goal at index i to a pair of inequalities
                simp [hzl, hzh, hzv]
                -- `IBP.relu` uses `if a > 0 then a else 0`, which is `max a 0` over `ℝ`.
                have ite_gt_zero_eq_max (a : ℝ) : (if a > 0 then a else 0) = max a 0 := by
                  by_cases ha : a > 0
                  · have ha0 : 0 ≤ a := le_of_lt ha
                    simp [ha, max_eq_left ha0]
                  · have ha0 : a ≤ 0 := le_of_not_gt ha
                    simp [ha, max_eq_right ha0]
                constructor
                · -- lower bound: max l 0 ≤ max x 0 from l ≤ x
                  have : max l 0 ≤ max x 0 := max_le_max hxlo (le_rfl)
                  simpa [ite_gt_zero_eq_max, Activation.Math.reluSpec] using this
                · -- upper bound: max x 0 ≤ max u 0 from x ≤ u
                  have : max x 0 ≤ max u 0 := max_le_max hxhi (le_rfl)
                  simpa [ite_gt_zero_eq_max, Activation.Math.reluSpec] using this

/-- Soundness of pure IBP bounds for a 2-layer MLP over `ℝ`. -/
theorem bound_ibp_sound {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar))
  (x : Tensor ℝ (.dim inDim .scalar))
  (hx : Box.contains (α:=ℝ) xB x) :
  Box.contains (α:=ℝ) (boundIbp (α:=ℝ) net xB) (forward (α:=ℝ) net x) := by
  classical
  -- Unfold bound_ibp and forward
  -- Step 1: z1 ∈ IBP.linear(W1, xB, b1)
  -- Bias box is dirac at b1
  -- pointwise containment is trivial when lo=hi=b
  have hb1 : Box.contains (α:=ℝ) { lo := net.b1, hi := net.b1 } net.b1 := by
    cases net.b1 with
    | dim b1f =>
      intro i
      cases b1f i with
      | scalar _ =>
        simp [Box.contains]
  -- z1 containment
  have hz1 : Box.contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.W1 xB { lo := net.b1, hi := net.b1 })
      (Spec.linearSpec (α:=ℝ) { weights := net.W1, bias := net.b1 } x) := by
    exact ibp_linear_sound_real net.W1 xB { lo := net.b1, hi := net.b1 } x net.b1 hx hb1
  -- Step 2: a1 ∈ IBP.relu(z1B)
  have ha1 : Box.contains (α:=ℝ)
      (IBP.relu (α:=ℝ) (IBP.linear (α:=ℝ) net.W1 xB { lo := net.b1, hi := net.b1 }))
      (Activation.reluSpec (α:=ℝ)
        (Spec.linearSpec (α:=ℝ) { weights := net.W1, bias := net.b1 } x)) := by
    exact ibp_relu_sound_real _ _ hz1
  -- Step 3: y ∈ IBP.linear(W2, a1B, b2)
  -- Build a1B and b2B as in bound_ibp
  have hb2 : Box.contains (α:=ℝ) { lo := net.b2, hi := net.b2 } net.b2 := by
    cases net.b2 with
    | dim b2f =>
      intro i
      cases b2f i with
      | scalar _ =>
        simp [Box.contains]
  have hy : Box.contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.W2
        (IBP.relu (α:=ℝ) (IBP.linear (α:=ℝ) net.W1 xB { lo := net.b1, hi := net.b1 }))
        { lo := net.b2, hi := net.b2 })
      (Spec.linearSpec (α:=ℝ) { weights := net.W2, bias := net.b2 }
        (Activation.reluSpec (α:=ℝ)
          (Spec.linearSpec (α:=ℝ) { weights := net.W1, bias := net.b1 } x))) := by
    exact ibp_linear_sound_real net.W2
      (IBP.relu (α:=ℝ) (IBP.linear (α:=ℝ) net.W1 xB { lo := net.b1, hi := net.b1 }))
      { lo := net.b2, hi := net.b2 }
      (Activation.reluSpec (α:=ℝ)
        (Spec.linearSpec (α:=ℝ) { weights := net.W1, bias := net.b1 } x))
      net.b2
      ha1 hb2
  -- Combine: bound_ibp is exactly the composition of the above boxes
  -- Unfold bound_ibp and forward to match hy
  simpa [boundIbp, forward]

/--
Soundness of the affine-bound entrypoint for a 2-layer MLP over `ℝ`.

In this module `bound_affine` delegates to the IBP implementation, so this theorem is a direct
corollary of `bound_ibp_sound`.
-/
theorem bound_affine_sound {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar))
  (x : Tensor ℝ (.dim inDim .scalar))
  (hx : Box.contains (α:=ℝ) xB x) :
  Box.contains (α:=ℝ) (boundAffine (α:=ℝ) net xB) (forward (α:=ℝ) net x) := by
  -- `boundAffine` delegates to pure IBP bounds in this module.
  simpa [boundAffine] using bound_ibp_sound (net := net) (xB := xB) (x := x) hx

end Theorems

/- Public API -/
namespace Examples

/--
Compute both IBP bounds and affine-CROWN bounds for a 2-layer MLP around an `ε`-box.

The input set is the axis-aligned box centered at `x_center` with radius `eps` in each coordinate.
-/
def crownMlp2Bounds {inDim hidDim outDim : Nat}
  (l1 : Spec.LinearSpec α inDim hidDim)
  (l2 : Spec.LinearSpec α hidDim outDim)
  (x_center : Tensor α (.dim inDim .scalar)) (eps : α) :
  Box α (.dim outDim .scalar) × Box α (.dim outDim .scalar) :=
  let net := ofLinearSpecs (α:=α) l1 l2
  let xB : Box α (.dim inDim .scalar) :=
    let rad := Tensor.scaleSpec (Spec.fill (α:=α) eps (.dim inDim .scalar)) 1
    { lo := Tensor.subSpec x_center rad, hi := Tensor.addSpec x_center rad }
  (boundIbp (α:=α) net xB, boundAffineCrown (α:=α) net xB)

end Examples

/- Classification helpers based on logit bounds -/
namespace Classify

open NN.MLTheory.CROWN

/-- Extract the scalar component `t[i]` from a vector-shaped tensor. -/
def getVec {n : Nat} (t : Tensor α (.dim n .scalar)) (i : Fin n) : α :=
  match t with
  | Tensor.dim f =>
    match f i with
    | Tensor.scalar v => v

/-- Lower endpoint at index `i` from a vector box. -/
def lowerAt {n : Nat} (B : Box α (.dim n .scalar)) (i : Fin n) : α :=
  match B.lo with
  | Tensor.dim f => match f i with | Tensor.scalar v => v

/-- Upper endpoint at index `i` from a vector box. -/
def upperAt {n : Nat} (B : Box α (.dim n .scalar)) (i : Fin n) : α :=
  match B.hi with
  | Tensor.dim f => match f i with | Tensor.scalar v => v

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

/--
Return the argmax index of a concrete output vector (when `n > 0`), otherwise `none`.

This is a utility for pairing certified bounds with a predicted class.
-/
def argmax? {n : Nat} (y : Tensor α (.dim n .scalar)) : Option (Fin n) :=
  match y with
  | Tensor.dim f =>
    if h0 : 0 < n then
      let init : Fin n := ⟨0, h0⟩
      let (bestIdx, _) := (List.finRange n).foldl (fun (acc : Fin n × α) k =>
        let (_, bestVal) := acc
        let vk := match f k with | Tensor.scalar v => v
        if vk > bestVal then (k, vk) else acc
      ) (init, match f init with | Tensor.scalar v => v)
      some bestIdx
    else none

end Classify

end NN.MLTheory.CROWN
