/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Layers.Loss

/-!
# Cross-entropy and cosine loss along any axes

`crossEntropy input x target dim` is the cross-entropy with classes along `dim`, at any positive
rank, and `input` says what `x` holds. With `.probabilities epsilon` the probabilities are clamped
into `[epsilon, 1 - epsilon]` before the logarithm, and the logarithm at the upper clamp is
`log1p (-epsilon)`. With `.logits` the logits are normalized by log-softmax, and a weighted term
whose log-probability is not finite is rebuilt from the shifted normalizer and a rescaled logit
difference. Both sum the weighted log-probabilities in one flat row-major fold over the whole
tensor and divide by the number of fibers along `dim` when `mean` holds.

The `.logits` case is `torch.nn.CrossEntropyLoss` (PyTorch 2.8 docs) with class-probability
targets, `l_n = -Σ_c y_{n,c} log softmax(x_n)_c` averaged as `Σ_n l_n / N`, at torch's defaults
`weight=None` and `label_smoothing=0.0`, coded by hand. Targets are probabilities or other
weights, never class indices. The log-probabilities come from `LogSoftmax.vec`, which shifts the
largest logit to an exact zero and takes `log1p` of the other exponentials. Over ℝ that's eq. (5)
of Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy" (2026,
arXiv:2610.31642), who reach the same value through a softplus of the others' log-sum-exp.
`CrossEntropyLoss` expects unnormalized logits, so the `.probabilities` case has no counterpart
there.

`cosineLoss x y dims keepdim epsilon` is the loss `1 - cos` of the fibers of `x` and `y` over
`dims`, computed from scaled norms by `TorchLean.CosineLoss.fiberLoss`. `cosineLoss_apply` reads an
entry as the loss of its fibers, and `cosineLoss_finRange` reads the loss over every axis as the
loss of the two whole tensors.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- The fiber of `x` over every axis is `x` with its shape spelled as the selected shape. -/
theorem ofFn_reduceSource_finRange (x : Tensor α s) (keepdim : Bool)
    (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    (Rep.ofFn fun r => x (s.reduceSource (List.finRange s.rank) keepdim o r) :
      Tensor α (s.selectAxes (List.finRange s.rank))) =
      castShape x s.selectAxes_finRange.symm := by
  refine eq_castShape_of_index _ fun c c' hcc => ?_
  rw [Rep.get_ofFn]
  congr 1
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.index_reduceSource, dite_eq_left hk, ite_eq_left (List.mem_finRange _),
    Shape.sortedAxes_finRange, List.idxOf_finRange]
  exact (hcc k).symm

variable [Context α]

/-! ## Cross-entropy -/

/-- What `crossEntropy` receives along the class axis. -/
inductive _root_.TorchLean.CrossEntropy.Input (α : Type)
  /-- Probabilities, clamped into `[epsilon, 1 - epsilon]` before the logarithm, with a NaN read
  as `epsilon`. The logarithm at the upper clamp is `log1p (-epsilon)`, so `1 - epsilon` is not
  rounded first. -/
  | probabilities (epsilon : α)
  /-- Logits, normalized by log-softmax. -/
  | logits

/-- Cross-entropy of `x` against `target` with classes along `dim`: minus the flat row-major sum of
the target-weighted log-probabilities, divided by the number of fibers along `dim` when `mean`. -/
def crossEntropy (input : CrossEntropy.Input α) (x target : Tensor α s) (dim : Fin s.rank)
    (mean : Bool := true) : α :=
  match input with
  | .probabilities epsilon =>
      let logClamped := fun v : α =>
        let v := if v > epsilon then v else epsilon
        if v < (1 : α) - epsilon then MathFunctions.log v else Context.log1p (-epsilon)
      let total := Tensor.sum (target * map logClamped x)
      if mean then meanOverAxisSlices (s := s) dim (-total) else -total
  | .logits =>
      let total := Tensor.sum (CrossEntropy.terms dim x target)
      if mean then meanOverAxisSlices (s := s) dim (-total) else -total

/-- The gradient of `crossEntropy input x target dim mean` with respect to `x`, divided by the
number of fibers along `dim` when `mean`. For probabilities it is `-target / x` strictly inside
the clamp window `(epsilon, 1 - epsilon)` and zero elsewhere, including at the two clamp kinks.
For logits it is the log-softmax VJP of `-target`, so each class fiber contributes
`softmax x * sum target - target`, the gradient of the loss for weighted, unnormalized and zero
targets alike. -/
def crossEntropyBackward (input : CrossEntropy.Input α) (x target : Tensor α s) (dim : Fin s.rank)
    (mean : Bool := true) : Tensor α s :=
  let grad := match input with
    | .probabilities epsilon => zipWith (fun q p =>
        if q > epsilon then
          if q < (1 : α) - epsilon then -p / q else 0
        else
          0) x target
    | .logits => logSoftmaxBackward (logSoftmax x dim) (map (fun x => -x) target) dim
  if mean then map (Context.divNat · (axisMeanDenom s dim)) grad else grad

/-- The gradient of `crossEntropy .logits x target dim mean` with respect to `target`, multiplied
by the upstream scalar `seed`: minus the weighted log-probabilities of `x` along `dim` at weight
`seed`, divided by the number of fibers when `mean`. The terms are computed with weight `seed`
through the same reconstruction as the loss, so a zero or small seed never multiplies an
overflowed log-probability. -/
def crossEntropyTargetBackward (x : Tensor α s) (dim : Fin s.rank) (seed : α := 1)
    (mean : Bool := true) : Tensor α s :=
  let terms := CrossEntropy.terms dim x (full s seed)
  if mean then map (fun value => Context.divNat (-value) (axisMeanDenom s dim)) terms
  else neg terms

/-! ## Cosine loss -/

/-- Cosine loss of `x` and `y` over the axes `dims`: `CosineLoss.fiberLoss` of the fibers
through each output coordinate, read in row-major order of `x`. -/
def cosineLoss (x y : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (epsilon : α := Context.defaultEpsilon) : Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o =>
    CosineLoss.fiberLoss (Rep.ofFn fun r => x (s.reduceSource dims keepdim o r))
      (Rep.ofFn fun r => y (s.reduceSource dims keepdim o r)) epsilon

/-- An entry of `cosineLoss x y dims keepdim epsilon` is the cosine loss of the fibers through
it. -/
theorem cosineLoss_apply (x y : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (epsilon : α) (o : (s.reduce dims keepdim).Coord) :
    cosineLoss x y dims keepdim epsilon o =
      CosineLoss.fiberLoss (Rep.ofFn fun r => x (s.reduceSource dims keepdim o r))
        (Rep.ofFn fun r => y (s.reduceSource dims keepdim o r)) epsilon :=
  Rep.get_ofFn _ o

/-- The cosine loss of two tensors does not depend on how their equal shape is spelled. -/
theorem _root_.TorchLean.CosineLoss.fiberLoss_castShape {t u : Shape} (h : t = u) (x y : Tensor α t)
    (epsilon : α) :
    CosineLoss.fiberLoss (castShape x h) (castShape y h) epsilon =
      CosineLoss.fiberLoss x y epsilon := by
  subst h
  rfl

/-- The cosine loss over every axis, read at its one output coordinate, is the loss of the two
whole tensors. -/
theorem cosineLoss_finRange (x y : Tensor α s) (keepdim : Bool) (epsilon : α)
    (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    cosineLoss x y (List.finRange s.rank) keepdim epsilon o =
      CosineLoss.fiberLoss x y epsilon := by
  rw [cosineLoss_apply, ofFn_reduceSource_finRange, ofFn_reduceSource_finRange,
    CosineLoss.fiberLoss_castShape]

end TorchLean.Tensor
