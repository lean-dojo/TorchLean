/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# LayerNorm IR Evaluation

The IR `layernorm axis` node interprets `axis` as the start of the normalized suffix.  Evaluation
reshapes the tensor to `(seqLen, embedDim)`, runs the spec 2D LayerNorm with `gamma=1` and
`beta=0`, then reshapes back.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Affine-free IR LayerNorm is exactly spec LayerNorm with unit scale and zero bias. -/
theorem layerNormWithoutAffine_eq_spec
    {α : Type} [TorchLean.Storage α] [Context α]
    (seqLen embedDim : Nat)
    (x : Tensor α [seqLen, embedDim])
    (hSeq : seqLen > 0)
    (hEmb : embedDim > 0) :
    Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim) x
      =
      Except.ok
        (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := x)
          (gamma := Tensor.full (α := α) (.dim embedDim .scalar) 1)
          (beta := Tensor.full (α := α) (.dim embedDim .scalar) 0)
          (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
  simp [Graph.layerNormWithoutAffine, Graph.layerNormMatrix, Graph.layerNormMatrixValue, hSeq, hEmb]
  rfl

/--
Local IR semantics for `layernorm axis`.

The hypotheses are the same contracts checked by the evaluator: `axis` must produce a valid
`(seqLen, embedDim)` view, the reshape must preserve element count, and the affine-free LayerNorm
step must succeed.
-/
theorem evalAt_layernorm_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (axis seqLen embedDim : Nat)
    (x : Tensor α s)
    (hParams : OpContracts.layerNormMatrixDims axis s = .ok (seqLen, embedDim))
    (hNumel : Spec.Shape.size s = Spec.Shape.size (.dim seqLen (.dim embedDim .scalar)))
    (y2d : Tensor α [seqLen, embedDim])
    (hLayerNorm :
      Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (Tensor.reshapeSpec (α := α) (source := s)
          (target := .dim seqLen (.dim embedDim .scalar)) x hNumel)
        =
        Except.ok y2d) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.layernorm axis) s s)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s
          (Tensor.reshapeSpec (α := α)
            (source := .dim seqLen (.dim embedDim .scalar)) (target := s) y2d hNumel.symm)) := by
  have hLayerNormMatrix :
      Graph.layerNormMatrix seqLen embedDim
          (Tensor.reshapeSpec (α := α) (source := s)
            (target := .dim seqLen (.dim embedDim .scalar)) x hNumel)
          (Tensor.full (α := α) [embedDim] 1) (Tensor.full (α := α) [embedDim] 0)
          TorchLean.normalizationEpsilon = .ok y2d := by
    simpa [Graph.layerNormWithoutAffine, Graph.layerNormMatrix] using hLayerNorm
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode, Graph.getNode?, Graph.unaryParentId, unaryParent?, Graph.expectShape, hParams,
    hNumel, Graph.resolveLayerNormAffine, hLayerNormMatrix, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
