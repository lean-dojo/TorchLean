/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the losses

Each lemma here states one arm of `Op.rule?` for a regression loss (`mseLoss`, `maeLoss`,
`huberLoss`, `logCoshLoss`, `cosineLoss`), the margin loss `hingeLoss`, a likelihood loss
(`binaryCrossEntropy`, `crossEntropy`, `crossEntropyProbabilities`, `poissonLoss`) or `distill`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.mseLoss`. -/
theorem rule?_mseLoss [Op.Capabilities α] {s : Shape} :
    (Op.mseLoss (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.meanSquaredError xs.1 xs.2.1), ())
        backward := fun xs g =>
          let gradient := Tensor.scale (Tensor.meanSquaredErrorBackward xs.1 xs.2.1)
            (Tensor.item g.1)
          (gradient, Tensor.neg gradient, ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.sum
            (Tensor.meanSquaredErrorBackward xs.1 xs.2.1 * (dxs.1 - dxs.2.1))), ()) } :=
  rfl

/-- The rule of `Op.maeLoss`. -/
theorem rule?_maeLoss [Op.Capabilities α] {s : Shape} :
    (Op.maeLoss (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.l1Loss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.l1LossBackward xs.1 xs.2.1) g.1.item, (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.l1LossBackward xs.1 xs.2.1) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.huberLoss`. -/
theorem rule?_huberLoss [Op.Capabilities α] {s : Shape} (mean : Bool) :
    (Op.huberLoss (s := s) mean).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.huberLoss xs.1 xs.2.1 xs.2.2.1.item mean), ())
        backward := fun xs g =>
            (Tensor.scale (Tensor.huberLossBackward xs.1 xs.2.1 xs.2.2.1.item mean)
              g.1.item, (), (),
              ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.huberLossBackward xs.1 xs.2.1 xs.2.2.1.item mean)
            dxs.1), ()) } :=
  rfl

/-- The rule of `Op.hingeLoss`. -/
theorem rule?_hingeLoss [Op.Capabilities α] {s : Shape} :
    (Op.hingeLoss (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.hingeLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.hingeLossBackward xs.1 xs.2.1) g.1.item, (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.hingeLossBackward xs.1 xs.2.1) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.poissonLoss`. -/
theorem rule?_poissonLoss [Op.Capabilities α] {s : Shape} :
    (Op.poissonLoss (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.poissonNLLLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.poissonNLLLossBackward xs.1 xs.2.1) g.1.item,
            (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.poissonNLLLossBackward xs.1 xs.2.1) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.logCoshLoss`. -/
theorem rule?_logCoshLoss [Op.Capabilities α] {s : Shape} :
    (Op.logCoshLoss (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.logCoshLoss xs.1 xs.2.1), ())
        backward := fun xs g => (Tensor.scale (Tensor.logCoshLossBackward xs.1 xs.2.1) g.1.item, (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.logCoshLossBackward xs.1 xs.2.1) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.binaryCrossEntropy`. -/
theorem rule?_binaryCrossEntropy [Op.Capabilities α] {s : Shape} :
    (Op.binaryCrossEntropy (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.scalar (Tensor.binaryCrossEntropy xs.1 xs.2.1 xs.2.2.1.item), ())
        backward := fun xs g =>
            (Tensor.scale
            (Tensor.binaryCrossEntropyBackward xs.1 xs.2.1 xs.2.2.1.item) g.1.item, (), (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (Tensor.binaryCrossEntropyBackward xs.1 xs.2.1 xs.2.2.1.item) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.cosineLoss`. -/
theorem rule?_cosineLoss [Op.Capabilities α] {s : Shape} :
    (Op.cosineLoss (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.cosineLoss xs.1 xs.2.1 (List.finRange s.rank) false
            xs.2.2.1.item) Shape.scalar
            ((Shape.size_reduce_finRange s false).trans Shape.size_nil.symm), ())
        backward := fun xs g =>
          (Tensor.scale (CosineLoss.fiberLossDeriv xs.1 xs.2.1 xs.2.2.1.item) g.1.item, (),
            (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (CosineLoss.fiberLossDeriv xs.1 xs.2.1 xs.2.2.1.item) dxs.1), ()) } :=
  rfl

/-- The rule of `Op.crossEntropy`. -/
theorem rule?_crossEntropy [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (mean : Bool) :
    (Op.crossEntropy dim mean).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.crossEntropy .logits xs.1 xs.2.1 dim mean), ())
        backward := fun xs g =>
          (Tensor.scale (Tensor.crossEntropyBackward .logits xs.1 xs.2.1 dim mean) g.1.item,
            Tensor.crossEntropyTargetBackward xs.1 dim g.1.item mean, ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
              (Tensor.crossEntropyBackward .logits xs.1 xs.2.1 dim mean) dxs.1 +
            Tensor.crossEntropy .logits xs.1 dxs.2.1 dim mean), ()) } :=
  rfl

/-- The rule of `Op.crossEntropyProbabilities`. -/
theorem rule?_crossEntropyProbabilities [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.crossEntropyProbabilities dim).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.scalar (Tensor.crossEntropy (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim), ())
        backward := fun xs g =>
          (Tensor.scale
              (Tensor.crossEntropyBackward (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim) g.1.item,
            (), (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot
            (Tensor.crossEntropyBackward (.probabilities xs.2.2.1.item) xs.1 xs.2.1 dim) dxs.1),
            ()) } :=
  rfl

/-- The rule of `Op.distill`. -/
theorem rule?_distill [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.distill dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.scalar (Tensor.distillLoss dim xs.1 xs.2.1), ())
        backward := fun xs g =>
          (Tensor.scale (Tensor.distillLossBackward dim xs.1 xs.2.1) g.1.item, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.scalar (Tensor.dot (Tensor.distillLossBackward dim xs.1 xs.2.1) dxs.1), ()) } :=
  rfl

end Op

end TorchLean
