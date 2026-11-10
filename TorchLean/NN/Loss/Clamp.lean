/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Loss.Axis
public import TorchLean.Tensor.Context.Real
import TorchLean.NN.Gradients.Activation

/-!
# Clamped probability losses over the reals

Cross-entropy of probabilities and binary cross-entropy clamp a probability into
`[epsilon, 1 - epsilon]` and take the logarithm at the upper clamp as `log1p (-epsilon)`, so the
logarithm never sees a rounded `1 - epsilon` in a floating-point format. Over the reals that is
the logarithm of the clamped probability, and both losses are the textbook formulas at it:

- `crossEntropy_probabilities_eq`: minus the target-weighted sum of the logarithms of the clamped
  probabilities, divided by the number of fibers along `dim` when `mean`.
- `BinaryCrossEntropy.term_eq`: `-(y log q + (1 - y) log (1 - q))` at the clamped probability `q`.

Both hold for every `epsilon`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor

/-- Over the reals, `log1p (-x)` is `log (1 - x)`. -/
private theorem Context.log1p_neg (x : ℝ) : Context.log1p (-x) = Real.log (1 - x) := by
  rw [Context.log1p_eq_log_one_add, ← sub_eq_add_neg]

/-- Over the reals, the cross-entropy of probabilities is minus the target-weighted sum of the
logarithms of the probabilities clamped into `[epsilon, 1 - epsilon]`, divided by the number of
fibers along `dim` when `mean`. -/
theorem Tensor.crossEntropy_probabilities_eq {s : Shape} (x target : Tensor ℝ s) (dim : Fin s.rank)
    (epsilon : ℝ) (mean : Bool) :
    Tensor.crossEntropy (.probabilities epsilon) x target dim mean =
      let clamp := fun v : ℝ =>
        let v := if v > epsilon then v else epsilon
        if v < 1 - epsilon then v else 1 - epsilon
      let total := Tensor.sum (target * Tensor.log (Tensor.map clamp x))
      if mean then meanOverAxisSlices (s := s) dim (-total) else -total := by
  have hlog : (Tensor.map (fun v : ℝ =>
        let v := if v > epsilon then v else epsilon
        if v < 1 - epsilon then MathFunctions.log v else Context.log1p (-epsilon)) x) =
      Tensor.log (Tensor.map (fun v : ℝ =>
        let v := if v > epsilon then v else epsilon
        if v < 1 - epsilon then v else 1 - epsilon) x) := by
    apply Tensor.Rep.ext
    intro c
    simp only [Tensor.log, Tensor.map_apply]
    split_ifs <;> simp [Context.log1p_neg, MathFunctions.log]
  simp only [Tensor.crossEntropy, hlog]

/-- Over the reals, binary cross-entropy is `-(y log q + (1 - y) log (1 - q))` at the probability
`q` clamped into `[epsilon, 1 - epsilon]`. -/
theorem BinaryCrossEntropy.term_eq (predicted target epsilon : ℝ) :
    let p := if predicted > epsilon then predicted else epsilon
    let q := if p < 1 - epsilon then p else 1 - epsilon
    BinaryCrossEntropy.term predicted target epsilon =
      -(target * Real.log q + (1 - target) * Real.log (1 - q)) := by
  intro p q
  simp only [BinaryCrossEntropy.term, p, q]
  split_ifs <;> simp [Context.log1p_neg, MathFunctions.log]

end TorchLean
