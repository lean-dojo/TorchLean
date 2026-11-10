/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN
public import TorchLean.Verify.CROWN.Activation.ReLU
public import TorchLean.Verify.CROWN.Op.ExecCROWN
public import TorchLean.NN.IR.Builder
public import TorchLean.Tensor.Layers.Linear
import TorchLean.Tensor.Algebra
public import TorchLean.Verify.CROWN.BoundOps.Lawful
public import TorchLean.Verify.CROWN.FlatBox.Real

/-!
# Two-layer MLP

The two-layer ReLU network `TwoLayerMLP`, its forward semantics `forward` and its interval bound
propagation `TwoLayerMLP.boundIBP`, built from `IBP.linear` (`TorchLean.Verify.CROWN`) and
`Graph.boxReLU`. `TwoLayerMLP.boundIBP_sound` proves over `ℝ` that `TwoLayerMLP.boundIBP net xB`
contains `forward net x` for every `x` in `xB`. The passes on graphs of registered operators and
their soundness theorems, including α-CROWN with and without ReLU phases, are in
`TorchLean.CROWN.Op`.

`boundIBP` is the interval bound propagation of Gowal et al. (2018), Sec. 3, coded by hand.
Their Eq. (6) carries an affine layer in center-radius form, `μ' = W μ + b` and `r' = |W| r`.
`IBP.linear` uses the endpoint form, the smaller of `w * lo` and `w * hi` for each weight on the
lower side, which gives the same box over `ℝ` and lets each product round in one known direction.
The ReLU step is their Eq. (7), `[max lo 0, max hi 0]`, and both biases enter as point boxes.

Main declarations: `TwoLayerMLP`, `forward`, `TwoLayerMLP.boundIBP`, `TwoLayerMLP.boundIBP_sound`.

References:
- CROWN: Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions",
  NeurIPS 2018, arXiv:1811.00866, Sec. 3.2 (the ReLU line of `ReLU.relaxScalar_sound`).
- auto_LiRPA: Xu et al.,
  "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920.
- IBP: Gowal et al., "On the Effectiveness of Interval Bound Propagation for Training Verifiably
  Robust Models", 2018, arXiv:1810.12715, Sec. 3, Eqs. (6) and (7).
-/

@[expose] public section


namespace TorchLean.CROWN

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/--
Weights and biases of a two-layer ReLU network, which computes
`y = outputWeight * relu(hiddenWeight * x + hiddenBias) + outputBias`.
-/
structure TwoLayerMLP (α : Type) [Storage α] (inDim hidDim outDim : Nat) where
  /-- First layer weight matrix. -/
  hiddenWeight : Tensor α [hidDim, inDim]
  /-- First layer bias vector. -/
  hiddenBias : Tensor α [hidDim]
  /-- Second layer weight matrix. -/
  outputWeight : Tensor α [outDim, hidDim]
  /-- Second layer bias vector. -/
  outputBias : Tensor α [outDim]

/-- The output of `net` at `x`, which the soundness theorems bound. -/
def TwoLayerMLP.forward {inDim hidDim outDim : Nat}
  (net : TwoLayerMLP α inDim hidDim outDim)
  (x : Tensor α [inDim]) : Tensor α [outDim] :=
  let hiddenLayer : Linear α inDim hidDim :=
    { weights := net.hiddenWeight, bias := net.hiddenBias }
  let outputLayer : Linear α hidDim outDim :=
    { weights := net.outputWeight, bias := net.outputBias }
  let hiddenPreactivation := Linear.forward (α:=α) hiddenLayer x
  let hiddenActivation := Tensor.relu (α:=α) hiddenPreactivation
  Linear.forward (α:=α) outputLayer hiddenActivation

/--
Output box of `net` over the input box `xB` by interval propagation alone, with each bias as a
point box.

It is cheaper than the CROWN box of a graph output (`CROWN.Op.outputBoxCROWN?`) and usually looser.
Over `ℝ`, `TwoLayerMLP.boundIBP_sound` proves that it contains `TwoLayerMLP.forward net x` for every
`x` in `xB`.
-/
def TwoLayerMLP.boundIBP {inDim hidDim outDim : Nat} [BoundOps α]
  (net : TwoLayerMLP α inDim hidDim outDim)
  (xB : Box α (.dim inDim .scalar)) : Box α (.dim outDim .scalar) :=
  -- z1 = hiddenWeight x + hiddenBias
  let b1B : Box α (.dim hidDim .scalar) := { lo := net.hiddenBias, hi := net.hiddenBias }
  let z1B := IBP.linear net.hiddenWeight xB b1B
  let a1B : Box α (.dim hidDim .scalar) :=
    Graph.ofFlatBox (Graph.boxReLU (Graph.toFlatBox hidDim z1B))
  let b2B : Box α (.dim outDim .scalar) := { lo := net.outputBias, hi := net.outputBias }
  IBP.linear net.outputWeight a1B b2B

/-!
Soundness of the interval bounds over `ℝ`, by monotonicity of ReLU and the interval matrix-vector
bound `IBP.linear_sound_real`.
-/

open TwoLayerMLP

/-- `Graph.boxReLU`, read as a vector box, encloses the ReLU of every point of the box over `ℝ`. -/
private theorem IBP.relu_sound_real {n : Nat}
  (zB : Box ℝ (.dim n .scalar))
  (z : Tensor ℝ [n])
  (hz : Box.Contains (α:=ℝ) zB z) :
  Box.Contains (α:=ℝ)
    (Graph.ofFlatBox (Graph.boxReLU (Graph.toFlatBox n zB)) : Box ℝ (.dim n .scalar))
    (Tensor.relu (α:=ℝ) z) := by
  intro (i : Fin n)
  have hcoord := hz i
  change
    getScalar (Graph.boxReLU (Graph.toFlatBox n zB)).lo i ≤
        getScalar (Tensor.relu (α := ℝ) z) i ∧
      getScalar (Tensor.relu (α := ℝ) z) i ≤
        getScalar (Graph.boxReLU (Graph.toFlatBox n zB)).hi i
  simp only [Graph.boxReLU, Graph.toFlatBox, Tensor.relu, getScalar_map,
    Activation.relu_eq_max]
  exact ⟨max_le_max hcoord.1 (le_refl 0), max_le_max hcoord.2 (le_refl 0)⟩

/-- Over `ℝ`, `TwoLayerMLP.boundIBP net xB` contains `TwoLayerMLP.forward net x` for every `x` in
`xB`. -/
theorem TwoLayerMLP.boundIBP_sound {inDim hidDim outDim : Nat}
  (net : TwoLayerMLP ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar))
  (x : Tensor ℝ [inDim])
  (hx : Box.Contains (α:=ℝ) xB x) :
  Box.Contains (α:=ℝ) (boundIBP (α:=ℝ) net xB) (forward (α:=ℝ) net x) := by
  classical
  -- Each bias box is the point box at the bias, which contains the bias.
  have hb1 : Box.Contains (α:=ℝ) { lo := net.hiddenBias, hi := net.hiddenBias } net.hiddenBias := by
    intro i
    exact ⟨le_rfl, le_rfl⟩
  -- Step 1: z1 ∈ IBP.linear(hiddenWeight, xB, hiddenBias)
  have hz1 : Box.Contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias })
      (Linear.forward (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x) := by
    exact IBP.linear_sound_real net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias }
      x net.hiddenBias hx hb1
  -- Step 2: a1 ∈ boxRelu(z1B)
  have ha1 : Box.Contains (α:=ℝ)
      (Graph.ofFlatBox (Graph.boxReLU (Graph.toFlatBox hidDim
        (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias })))
        : Box ℝ (.dim hidDim .scalar))
      (Tensor.relu (α:=ℝ)
        (Linear.forward (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x)) := by
    exact IBP.relu_sound_real _ _ hz1
  -- Step 3: y ∈ IBP.linear(outputWeight, a1B, outputBias)
  have hb2 : Box.Contains (α:=ℝ) { lo := net.outputBias, hi := net.outputBias } net.outputBias := by
    intro i
    exact ⟨le_rfl, le_rfl⟩
  have hy : Box.Contains (α:=ℝ)
      (IBP.linear (α:=ℝ) net.outputWeight
        (Graph.ofFlatBox (Graph.boxReLU (Graph.toFlatBox hidDim
          (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias })))
          : Box ℝ (.dim hidDim .scalar))
        { lo := net.outputBias, hi := net.outputBias })
      (Linear.forward (α:=ℝ) { weights := net.outputWeight, bias := net.outputBias }
        (Tensor.relu (α:=ℝ)
          (Linear.forward (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x))) := by
    exact IBP.linear_sound_real net.outputWeight
      (Graph.ofFlatBox (Graph.boxReLU (Graph.toFlatBox hidDim
        (IBP.linear (α:=ℝ) net.hiddenWeight xB { lo := net.hiddenBias, hi := net.hiddenBias })))
        : Box ℝ (.dim hidDim .scalar))
      { lo := net.outputBias, hi := net.outputBias }
      (Tensor.relu (α:=ℝ)
        (Linear.forward (α:=ℝ) { weights := net.hiddenWeight, bias := net.hiddenBias } x))
      net.outputBias
      ha1 hb2
  -- `boundIBP` and `forward` unfold to the compositions bounded above.
  simpa [boundIBP, forward]

end TorchLean.CROWN
