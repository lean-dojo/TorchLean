/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the normalizations

Each lemma here states one arm of `Op.rule?` for `layerNorm`, `batchNorm` or `batchNormRunning`.
`Op.rule?_layerNorm_matrix` gives the rule of `layerNorm` on a matrix and
`Op.batchNormRunning_forward` the forward of `batchNormRunning`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.layerNorm`: the matrix rule on the `(seqLen, embedDim)` views. -/
theorem rule?_layerNorm [Op.Capabilities α] {seqLen embedDim : Nat} (hSeq : seqLen > 0)
    (hEmbed : embedDim > 0) {lead x : Shape} (hlead : seqLen = lead.size)
    (hx : x = lead ++ [embedDim]) :
    (Op.layerNorm hSeq hEmbed hlead hx).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.layerNorm (Tensor.reshape xs.1 [seqLen, embedDim]
              (Shape.size_eq_of_append hlead hx)) [⟨1, Nat.one_lt_two⟩] xs.2.1 xs.2.2.1
              xs.2.2.2.1.item) x (Shape.size_eq_of_append hlead hx).symm, ())
        backward := fun xs g =>
          let gradients := LayerNorm.backward (sequenceLengthPositive := hSeq)
            (embeddingWidthPositive := hEmbed)
            (input := Tensor.reshape xs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (scale := xs.2.1)
            (outputGradient := Tensor.reshape g.1 [seqLen, embedDim]
              (Shape.size_eq_of_append hlead hx)) (epsilon := xs.2.2.2.1.item)
          (Tensor.reshape gradients.inputGradient x (Shape.size_eq_of_append hlead hx).symm,
            gradients.scaleGradient, gradients.biasGradient, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (LayerNorm.jvp (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
            (x := Tensor.reshape xs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (tangent := Tensor.reshape dxs.1 [seqLen, embedDim] (Shape.size_eq_of_append hlead hx))
            (gamma := xs.2.1) (dgamma := dxs.2.1) (_beta := xs.2.2.1) (dbeta := dxs.2.2.1)
            (epsilon := xs.2.2.2.1.item)) x (Shape.size_eq_of_append hlead hx).symm, ()) } :=
  rfl

/-- The rule of `Op.layerNorm` on a matrix, its rows standing for themselves: layer normalization
over the last axis and `LayerNorm.backward`. -/
theorem rule?_layerNorm_matrix [Op.Capabilities α] {seqLen embedDim : Nat} (hSeq : seqLen > 0)
    (hEmbed : embedDim > 0) (hlead : seqLen = Shape.size [seqLen])
    (hx : [seqLen, embedDim] = [seqLen] ++ [embedDim]) :
    (Op.layerNorm hSeq hEmbed hlead hx).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.layerNorm xs.1 [⟨1, Nat.one_lt_two⟩] xs.2.1 xs.2.2.1 xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let gradients := LayerNorm.backward (sequenceLengthPositive := hSeq)
            (embeddingWidthPositive := hEmbed) (input := xs.1) (scale := xs.2.1)
            (outputGradient := g.1) (epsilon := xs.2.2.2.1.item)
          (gradients.inputGradient, gradients.scaleGradient, gradients.biasGradient, (), ())
        jvp? := some fun xs dxs =>
          (LayerNorm.jvp (h_seq_pos := hSeq) (h_embed_pos := hEmbed) (x := xs.1)
            (tangent := dxs.1) (gamma := xs.2.1) (dgamma := dxs.2.1) (_beta := xs.2.2.1)
            (dbeta := dxs.2.2.1) (epsilon := xs.2.2.2.1.item), ()) } :=
  rfl

/-- The rule of `Op.batchNorm`. -/
theorem rule?_batchNorm [Op.Capabilities α] {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).Positive) :
    (Op.batchNorm hWellFormed).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.batchNorm xs.1 ⟨0, Nat.succ_pos _⟩ xs.2.1 xs.2.2.1 .batch xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let gradients := BatchNorm.backward (x := xs.1) (gamma := xs.2.1) (gradOutput := g.1)
            (epsilon := xs.2.2.2.1.item)
          (gradients.inputGradient, gradients.scaleGradient, gradients.biasGradient, (), ())
        jvp? := some fun xs dxs =>
          (BatchNorm.jvp (channels := channels) (sSpatial := spatial) (x := xs.1)
            (tangent := dxs.1) (gamma := xs.2.1) (dgamma := dxs.2.1) (_beta := xs.2.2.1)
            (dbeta := dxs.2.2.1) (epsilon := xs.2.2.2.1.item), ()) } :=
  rfl

/-- The forward of `Op.batchNormRunning`: batch normalization along `dim` with the running
statistics, epsilon read from its scalar port. -/
theorem batchNormRunning_forward [Op.Capabilities α] {s : Shape} (dim : Fin s.rank)
    (x : Tensor α s) (gamma beta mean var : Tensor α [s.extent dim]) (epsilon : Tensor α []) :
    (((Op.batchNormRunning dim).rule? (α := α)).get rfl).forward
        (x, gamma, beta, mean, var, epsilon, ()) =
      (Tensor.batchNorm x dim gamma beta (.running mean var) epsilon.item, ()) :=
  rfl

/-- The rule of `Op.batchNormRunning`. Its forward-mode rule holds the running statistics and
epsilon fixed: the input tangent times the broadcast scale over the denominator, the normalized
input times the broadcast scale tangent, and the broadcast bias tangent. -/
theorem rule?_batchNormRunning [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.batchNormRunning dim).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.batchNorm xs.1 dim xs.2.1 xs.2.2.1 (.running xs.2.2.2.1 xs.2.2.2.2.1)
            xs.2.2.2.2.2.1.item, ())
        backward := fun xs g =>
          let gradients := BatchNorm.runningBackward xs.1 dim xs.2.1 xs.2.2.2.1 xs.2.2.2.2.1
            xs.2.2.2.2.2.1.item g.1
          (gradients.inputGradient, gradients.scaleGradient, gradients.biasGradient, (), (), (),
            ())
        jvp? := some fun xs dxs =>
          let spread := fun t : Tensor α [s.extent dim] =>
            Tensor.broadcast t s _ (s.broadcastDims_axis dim)
          let std := Tensor.sqrt (spread (Tensor.maximum xs.2.2.2.2.1 (Tensor.full _ 0)) +
            Tensor.full s xs.2.2.2.2.2.1.item) .clampNegative
          (((dxs.1 * spread xs.2.1) / std + ((xs.1 - spread xs.2.2.2.1) / std) * spread dxs.2.1) +
            spread dxs.2.2.1, ()) } :=
  rfl

end Op

end TorchLean
