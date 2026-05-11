/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Models.Mlp
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Backward CROWN propagation

Backward affine-bound propagation for tighter neural-network certificates.

This module implements the core *backward* CROWN idea: rather than pushing interval/affine bounds
forward layer by layer, start from an output objective and propagate its affine dependence on the
input backwards through the network.

At a high level, for each hidden layer `l` we maintain affine coefficients

`A^(l) x + b^(l)`

that over-approximate the chosen output objective, and update them using diagonal activation
relaxations `Λ^(l)` together with the layer weights/biases.

References / citations:
- Huan Zhang et al., “Efficient Neural Network Robustness Certification with General Activation
  Functions”, NeurIPS 2018.
  https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html
- Singh et al., “An Abstract Domain for Certifying Neural Networks” (DeepPoly), POPL 2019.

We keep this as the compact layer-wise backward propagation surface. The graph-IR verifier lives in
`NN.MLTheory.CROWN.Graph`; both modules use the same mathematical idea, but they serve different
integration points.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Propagation.Backward

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Per-neuron activation relaxation parameters. -/
structure NeuronRelax (α : Type) where
  /-- Slope of the lower affine envelope. -/
  slope_lower : α
  /-- Bias of the lower affine envelope. -/
  bias_lower : α
  /-- Slope of the upper affine envelope. -/
  slope_upper : α
  /-- Bias of the upper affine envelope. -/
  bias_upper : α

/-- Layer-wise relaxation parameters. -/
structure LayerRelax (α : Type) [Context α] where
  /-- Number of neurons covered by this relaxation record. -/
  dim : Nat
  /-- Per-neuron affine envelope parameters. -/
  params : Tensor (NeuronRelax α) (.dim dim .scalar)

/-- Extract relaxation slopes as diagonal matrix (for lower bound). -/
def layerSlopesLower {n : Nat} (relax : LayerRelax α)
    (h : relax.dim = n) : Tensor α (.dim n (.dim n .scalar)) :=
  by
    cases h
    cases relax.params with
    | dim f =>
        exact
          Tensor.dim (fun i : Fin relax.dim =>
            Tensor.dim (fun j : Fin relax.dim =>
              if decide (i.val = j.val) then
                match f i with
                | .scalar r => Tensor.scalar r.slope_lower
              else
                Tensor.scalar Numbers.zero))

/-- Extract relaxation slopes as diagonal matrix (for upper bound). -/
def layerSlopesUpper {n : Nat} (relax : LayerRelax α)
    (h : relax.dim = n) : Tensor α (.dim n (.dim n .scalar)) :=
  by
    cases h
    cases relax.params with
    | dim f =>
        exact
          Tensor.dim (fun i : Fin relax.dim =>
            Tensor.dim (fun j : Fin relax.dim =>
              if decide (i.val = j.val) then
                match f i with
                | .scalar r => Tensor.scalar r.slope_upper
              else
                Tensor.scalar Numbers.zero))

/-- Extract bias vector (for lower bound). -/
def layerBiasLower {n : Nat} (relax : LayerRelax α)
    (h : relax.dim = n) : Tensor α (.dim n .scalar) :=
  by
    cases h
    cases relax.params with
    | dim f =>
        exact
          Tensor.dim (fun i : Fin relax.dim =>
            match f i with
            | .scalar r => Tensor.scalar r.bias_lower)

/-- Extract bias vector (for upper bound). -/
def layerBiasUpper {n : Nat} (relax : LayerRelax α)
    (h : relax.dim = n) : Tensor α (.dim n .scalar) :=
  by
    cases h
    cases relax.params with
    | dim f =>
        exact
          Tensor.dim (fun i : Fin relax.dim =>
            match f i with
            | .scalar r => Tensor.scalar r.bias_upper)

/-- Network structure for backward propagation.
    Stores weights, biases, and pre-computed activation relaxations. -/
structure BackwardNetwork (α : Type) [Context α] where
  /-- Number of layers (not counting input) -/
  numLayers : Nat
  /-- Input dimension -/
  inDim : Nat
  /-- Output dimension -/
  outDim : Nat
  /-- Layer dimensions: dims[i] is output dim of layer i -/
  dims : Array Nat
  /-- Weight matrices: W[i] has shape [dims[i], dims[i-1]] -/
  weights : Array (Σ m n : Nat, Tensor α (.dim m (.dim n .scalar)))
  /-- Bias vectors: b[i] has shape [dims[i]] -/
  biases : Array (Σ n : Nat, Tensor α (.dim n .scalar))
  /-- Per-layer activation relaxations (empty for output layer) -/
  relaxations : Array (LayerRelax α)

/-- Backward state during propagation. -/
structure BackwardState (α : Type) [Context α] where
  /-- Current lower-bound affine coefficient matrix `A` (`output_dim × input_dim`). -/
  A_lower : Σ m n : Nat, Tensor α (.dim m (.dim n .scalar))
  /-- Current upper-bound affine coefficient matrix `A` (`output_dim × input_dim`). -/
  A_upper : Σ m n : Nat, Tensor α (.dim m (.dim n .scalar))
  /-- Current lower-bound affine bias vector `b` (`output_dim`). -/
  b_lower : Σ n : Nat, Tensor α (.dim n .scalar)
  /-- Current upper-bound affine bias vector `b` (`output_dim`). -/
  b_upper : Σ n : Nat, Tensor α (.dim n .scalar)

/-- Helper: matrix multiplication for sigma-typed tensors. -/
def sigmaMatMul (A : Σ m n : Nat, Tensor α (.dim m (.dim n .scalar)))
    (B : Σ p q : Nat, Tensor α (.dim p (.dim q .scalar))) :
    Option (Σ m q : Nat, Tensor α (.dim m (.dim q .scalar))) :=
  let ⟨m, n, matA⟩ := A
  let ⟨p, q, matB⟩ := B
  if h : n = p then
    some ⟨m, q, Spec.matMulSpec (α:=α) matA (by cases h; exact matB)⟩
  else
    none

/-- Helper: matrix-vector multiplication for sigma-typed tensors. -/
def sigmaMatVecMul (A : Σ m n : Nat, Tensor α (.dim m (.dim n .scalar)))
    (v : Σ n : Nat, Tensor α (.dim n .scalar)) :
    Option (Σ m : Nat, Tensor α (.dim m .scalar)) :=
  let ⟨m, n, matA⟩ := A
  let ⟨p, vecV⟩ := v
  if h : n = p then
    some ⟨m, Spec.matVecMulSpec (α:=α) matA (by cases h; exact vecV)⟩
  else
    none

/-- Helper: vector addition for sigma-typed tensors. -/
def sigmaVecAdd (v1 v2 : Σ n : Nat, Tensor α (.dim n .scalar)) :
    Option (Σ n : Nat, Tensor α (.dim n .scalar)) :=
  let ⟨n1, vec1⟩ := v1
  let ⟨n2, vec2⟩ := v2
  if h : n1 = n2 then
    some ⟨n1, Tensor.addSpec vec1 (by cases h; exact vec2)⟩
  else
    none

/-- Initialize backward state with identity at output. -/
def initBackwardState (outDim : Nat) : BackwardState α :=
  let identity : Tensor α (.dim outDim (.dim outDim .scalar)) :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar (if decide (i.val = j.val) then Numbers.one else Numbers.zero)))
  let zero : Tensor α (.dim outDim .scalar) :=
    Spec.fill (α:=α) Numbers.zero (.dim outDim .scalar)
  { A_lower := ⟨outDim, outDim, identity⟩
  , A_upper := ⟨outDim, outDim, identity⟩
  , b_lower := ⟨outDim, zero⟩
  , b_upper := ⟨outDim, zero⟩ }

/-- Process one layer in backward propagation.

    For layer l with weight W, bias b, and relaxation Λ:
    A^(l-1) = A^(l) · Λ · W
    b^(l-1) = A^(l) · (Λ · b + offset) + b^(l)

    Here we handle lower and upper bounds separately.
-/
def backwardOneLayer (state : BackwardState α)
    (W : Σ m n : Nat, Tensor α (.dim m (.dim n .scalar)))
    (bias : Σ n : Nat, Tensor α (.dim n .scalar))
    (relax : LayerRelax α) : Option (BackwardState α) := do
  let ⟨wm, wn, matW⟩ := W
  let ⟨bn, vecB⟩ := bias

  -- Check dimension compatibility
  if h : wm = relax.dim ∧ bn = wm then
    -- Build diagonal slope matrices for the relaxation dimension
    let slopeLower := layerSlopesLower (n:=relax.dim) relax rfl
    let slopeUpper := layerSlopesUpper (n:=relax.dim) relax rfl
    let biasLower := layerBiasLower (n:=relax.dim) relax rfl
    let biasUpper := layerBiasUpper (n:=relax.dim) relax rfl

    -- A_new_lower = A_lower · diag(slopeLower) · W
    -- First: temp = diag(slopeLower) · W
    -- Cast matW to match dimensions using equality h.1
    let matW' : Tensor α (.dim relax.dim (.dim wn .scalar)) :=
      cast (by rw [h.1]) matW
    let tempLower := Spec.matMulSpec (α:=α) slopeLower matW'
    -- Then: A_new = A_lower · temp
    let A_new_lower ← sigmaMatMul state.A_lower ⟨relax.dim, wn, tempLower⟩

    -- Similar for upper
    let tempUpper := Spec.matMulSpec (α:=α) slopeUpper matW'
    let A_new_upper ← sigmaMatMul state.A_upper ⟨relax.dim, wn, tempUpper⟩

    -- b_new_lower = A_lower · (slopeLower · bias + biasLower) + b_lower
    -- First: scaled_bias = slopeLower · bias + biasLower (elementwise)
    let scaledBiasLower : Tensor α (.dim relax.dim .scalar) :=
      match biasLower, vecB with
      | .dim bl, .dim vb =>
        Tensor.dim (fun i =>
          match bl i, vb ⟨i.val, by rw [h.2, h.1]; exact i.isLt⟩ with
          | .scalar bli, .scalar vbi =>
            match slopeLower with
            | .dim slrows =>
              match slrows i with
              | .dim slcols =>
                match slcols i with
                | .scalar si => Tensor.scalar (si * vbi + bli))

    let scaledBiasUpper : Tensor α (.dim relax.dim .scalar) :=
      match biasUpper, vecB with
      | .dim bu, .dim vb =>
        Tensor.dim (fun i =>
          match bu i, vb ⟨i.val, by rw [h.2, h.1]; exact i.isLt⟩ with
          | .scalar bui, .scalar vbi =>
            match slopeUpper with
            | .dim surows =>
              match surows i with
              | .dim sucols =>
                match sucols i with
                | .scalar si => Tensor.scalar (si * vbi + bui))

    -- Multiply by current A and add to b
    let Ab_lower ← sigmaMatVecMul state.A_lower ⟨relax.dim, scaledBiasLower⟩
    let b_new_lower ← sigmaVecAdd Ab_lower state.b_lower

    let Ab_upper ← sigmaMatVecMul state.A_upper ⟨relax.dim, scaledBiasUpper⟩
    let b_new_upper ← sigmaVecAdd Ab_upper state.b_upper

    return {
      A_lower := A_new_lower
      A_upper := A_new_upper
      b_lower := b_new_lower
      b_upper := b_new_upper
    }
  else
    none

/-- Run full backward propagation through network. -/
def runBackward (net : BackwardNetwork α) : Option (BackwardState α) := do
  let mut state := initBackwardState (α:=α) net.outDim

  -- Process layers from output to input
  for i in [0:net.numLayers] do
    let layerIdx := net.numLayers - 1 - i
    if hlw : layerIdx < net.weights.size then
      if hlb : layerIdx < net.biases.size then
        if hlr : layerIdx < net.relaxations.size then
          let W := net.weights[layerIdx]
          let b := net.biases[layerIdx]
          let relax := net.relaxations[layerIdx]
          match backwardOneLayer (α:=α) state W b relax with
          | some newState => state := newState
          | none => return state -- Early exit on error

  return state

/-- Evaluate backward bounds on input box to get output bounds.
    Given A·x + b with x ∈ [lo, hi], compute output interval.
-/
def evalBackwardBounds (outDim inDim : Nat) (state : BackwardState α)
    (xB : Box α (.dim inDim .scalar)) :
    Option (Box α (.dim outDim .scalar)) :=
  let ⟨mL, nL, A_lo⟩ := state.A_lower
  let ⟨mU, nU, A_up⟩ := state.A_upper
  let ⟨bDimL, b_lo⟩ := state.b_lower
  let ⟨bDimU, b_up⟩ := state.b_upper

  if hmL : mL = outDim then
    if hnL : nL = inDim then
      if hmU : mU = outDim then
        if hnU : nU = inDim then
          if hbL : bDimL = outDim then
            if hbU : bDimU = outDim then
              by
                cases hmL; cases hnL; cases hmU; cases hnU; cases hbL; cases hbU
                let bBLower : Box α (.dim outDim .scalar) := { lo := b_lo, hi := b_lo }
                let bBUpper : Box α (.dim outDim .scalar) := { lo := b_up, hi := b_up }
                let yLower := NN.MLTheory.CROWN.IBP.linear (α:=α) A_lo xB bBLower
                let yUpper := NN.MLTheory.CROWN.IBP.linear (α:=α) A_up xB bBUpper
                exact some { lo := yLower.lo, hi := yUpper.hi }
            else
              none
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

/-- Compute ReLU relaxation parameters from pre-activation bounds. -/
def computeReLURelax (n : Nat) (preB : Box α (.dim n .scalar)) : LayerRelax α :=
  match preB.lo, preB.hi with
  | .dim lo, .dim hi =>
    let params := Tensor.dim (fun i : Fin n =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let relax : NeuronRelax α :=
          if u < Numbers.zero then
            -- Inactive: y = 0
            { slope_lower := Numbers.zero
            , bias_lower := Numbers.zero
            , slope_upper := Numbers.zero
            , bias_upper := Numbers.zero }
          else if l > Numbers.zero then
            -- Active: y = x
            { slope_lower := Numbers.one
            , bias_lower := Numbers.zero
            , slope_upper := Numbers.one
            , bias_upper := Numbers.zero }
          else
            -- Crossing: lower y ≥ 0, upper y ≤ αx - αl
            let α := u / (u - l)
            { slope_lower := Numbers.zero  -- Conservative lower
            , bias_lower := Numbers.zero
            , slope_upper := α
            , bias_upper := -(α * l) }
        Tensor.scalar relax)
    { dim := n, params := params }

/-- Compute sigmoid relaxation parameters from pre-activation bounds. -/
def computeSigmoidRelax (n : Nat) (preB : Box α (.dim n .scalar)) : LayerRelax α :=
  match preB.lo, preB.hi with
  | .dim lo, .dim hi =>
    let params := Tensor.dim (fun i : Fin n =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let σl := Activation.Math.sigmoidSpec (α:=α) l
        let σu := Activation.Math.sigmoidSpec (α:=α) u
        -- Secant line for upper bound, tangent at midpoint for lower
        let slope_sec := if u > l + Numbers.epsilon then (σu - σl) / (u - l) else σl * (Numbers.one
          - σl)
        let mid := (l + u) * Numbers.pointfive
        let σmid := Activation.Math.sigmoidSpec (α:=α) mid
        let slope_tan := σmid * (Numbers.one - σmid)
        let relax : NeuronRelax α :=
          { slope_lower := slope_tan
          , bias_lower := σmid - slope_tan * mid
          , slope_upper := slope_sec
          , bias_upper := σl - slope_sec * l }
        Tensor.scalar relax)
    { dim := n, params := params }

/-- Compute tanh relaxation parameters from pre-activation bounds. -/
def computeTanhRelax (n : Nat) (preB : Box α (.dim n .scalar)) : LayerRelax α :=
  match preB.lo, preB.hi with
  | .dim lo, .dim hi =>
    let params := Tensor.dim (fun i : Fin n =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        let tl := Activation.Math.tanhSpec (α:=α) l
        let tu := Activation.Math.tanhSpec (α:=α) u
        -- Secant for one bound, tangent for other
        let slope_sec := if u > l + Numbers.epsilon then (tu - tl) / (u - l) else Numbers.one - tl *
          tl
        let mid := (l + u) * Numbers.pointfive
        let tmid := Activation.Math.tanhSpec (α:=α) mid
        let slope_tan := Numbers.one - tmid * tmid
        let relax : NeuronRelax α :=
          { slope_lower := slope_tan
          , bias_lower := tmid - slope_tan * mid
          , slope_upper := slope_sec
          , bias_upper := tl - slope_sec * l }
        Tensor.scalar relax)
    { dim := n, params := params }

namespace Theorems

/-- ReLU relaxation preserves dimension. -/
theorem relu_relax_dim_preserved (n : Nat) (preB : Box α (.dim n .scalar)) :
    (computeReLURelax (α:=α) n preB).dim = n := by
  unfold computeReLURelax
  match preB.lo, preB.hi with
  | .dim _, .dim _ => rfl

/-- Sigmoid relaxation preserves dimension. -/
theorem sigmoid_relax_dim_preserved (n : Nat) (preB : Box α (.dim n .scalar)) :
    (computeSigmoidRelax (α:=α) n preB).dim = n := by
  unfold computeSigmoidRelax
  match preB.lo, preB.hi with
  | .dim _, .dim _ => rfl

/-- Tanh relaxation preserves dimension. -/
theorem tanh_relax_dim_preserved (n : Nat) (preB : Box α (.dim n .scalar)) :
    (computeTanhRelax (α:=α) n preB).dim = n := by
  unfold computeTanhRelax
  match preB.lo, preB.hi with
  | .dim _, .dim _ => rfl

/-- Initial backward state has identity matrices of the same dimension. -/
theorem init_state_identity (outDim : Nat) :
    let state := initBackwardState (α:=α) outDim
    state.A_lower.1 = outDim ∧ state.A_lower.2.1 = outDim := by
  unfold initBackwardState
  exact ⟨rfl, rfl⟩

end Theorems

end NN.MLTheory.CROWN.Propagation.Backward
/-!
Backward (reverse) bound propagation for the CROWN development.

This file contains the backward pass that pushes linear bounds from outputs back to inputs through
the graph, as used by the CROWN-style certification proofs/checkers.
-/
