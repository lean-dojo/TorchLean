/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Broadcast -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Composite -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Convolution -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Literal -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.OnlineSoftmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Ordering -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Power -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Products -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Reduction -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Softmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay -- shake: keep
public import TorchLean.Tensor.Layers.Normalization.LayerNorm
import Mathlib.Util.TermReduce -- shake: keep

/-!
# Normalized rows for `roundoff`

We bound the error of `Normalization.normalizedRows`, which the JVP and backward kernels of layer
and batch normalization read, as one function of its rows. From rows whose entries are within an
absolute error of a reference, `roundoff` infers the error of each normalized value and of each
inverse standard deviation, and from these, for rows with mixed errors, a mixed bound on each. We
register the mixed bounds, so mixed analysis reads the entries of `Normalization.normalizedRows`
through them instead of analyzing the statistics of a row at every entry it reads.

FloatLib keeps the producer of a registered bound folded in both modes, so absolute analysis reads
these entries through `toReal_normalizedRows_fst` and `toReal_normalizedRows_snd`, which state
them by the definition of `Normalization.normalizedRows`.

- `normalizedRows_true_fst_mixedError`, `normalizedRows_true_snd_mixedError`: a normalized value
  and an inverse standard deviation of recentered rows (layer normalization), in mixed analysis.
- `normalizedRows_false_fst_mixedError`, `normalizedRows_false_snd_mixedError`: the same for rows
  that are not recentered (batch normalization).
- `abs_toReal_normalizedRows_true_fst_sub_le` and the three others like it: the absolute bounds
  the mixed bounds are inferred through, rules of this module only.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics TorchLean
open ExecFloat.Binary (toReal)

-- `normalizedRows` broadcasts tensors of shape `[rows]` at the shape `[rows, cols].eraseIdx 1`.
-- Unfolding `List.eraseIdx` in implicit arguments lets the rules read them at either shape.
attribute [local implicit_reducible] List.eraseIdx

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

section Inferred

/-- In every IEEE binary format, if each entry of `x` is within `xError c` of `xReference c`, then
under the inferred conditions entry `c` of `(Normalization.normalizedRows x epsilon true).1` is
within the inferred error of the same computation in real arithmetic on `xReference`. -/
roundoff theorem abs_toReal_normalizedRows_true_fst_sub_le := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols]) (xReference xError : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, |toReal (x c) - xReference c| ≤ xError c) (epsilon : Value)
    (c : Shape.Coord [rows, cols]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon true).1 c)

/-- In every IEEE binary format, if each entry of `x` is within `xError c` of `xReference c`, then
under the inferred conditions entry `o` of `(Normalization.normalizedRows x epsilon true).2` is
within the inferred error of the same computation in real arithmetic on `xReference`. -/
roundoff theorem abs_toReal_normalizedRows_true_snd_sub_le := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols]) (xReference xError : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, |toReal (x c) - xReference c| ≤ xError c) (epsilon : Value)
    (o : Shape.Coord [rows]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon true).2 o)

/-- In every IEEE binary format, if each entry of `x` is within `xError c` of `xReference c`, then
under the inferred conditions entry `c` of `(Normalization.normalizedRows x epsilon false).1` is
within the inferred error of the same computation in real arithmetic on `xReference`. -/
roundoff theorem abs_toReal_normalizedRows_false_fst_sub_le := fun [Storage Value]
    {rows cols : Nat} (x : Tensor Value [rows, cols])
    (xReference xError : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, |toReal (x c) - xReference c| ≤ xError c) (epsilon : Value)
    (c : Shape.Coord [rows, cols]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon false).1 c)

/-- In every IEEE binary format, if each entry of `x` is within `xError c` of `xReference c`, then
under the inferred conditions entry `o` of `(Normalization.normalizedRows x epsilon false).2` is
within the inferred error of the same computation in real arithmetic on `xReference`. -/
roundoff theorem abs_toReal_normalizedRows_false_snd_sub_le := fun [Storage Value]
    {rows cols : Nat} (x : Tensor Value [rows, cols])
    (xReference xError : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, |toReal (x c) - xReference c| ≤ xError c) (epsilon : Value)
    (o : Shape.Coord [rows]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon false).2 o)

attribute [local roundoff_rule] abs_toReal_normalizedRows_true_fst_sub_le
  abs_toReal_normalizedRows_true_snd_sub_le abs_toReal_normalizedRows_false_fst_sub_le
  abs_toReal_normalizedRows_false_snd_sub_le

/-- In every IEEE binary format, if each entry of `x` has mixed error `xRelative c`, `xAbsolute c`
against `xReference c`, then under the inferred conditions entry `c` of
`(Normalization.normalizedRows x epsilon true).1` is finite and has the inferred mixed error against
the same computation in real arithmetic on `xReference`. -/
roundoff mixed theorem normalizedRows_true_fst_mixedError := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols])
    (xReference xRelative xAbsolute : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, MixedError (toReal (x c)) (xReference c) (xRelative c) (xAbsolute c))
    (epsilon : Value) (c : Shape.Coord [rows, cols]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon true).1 c)

/-- In every IEEE binary format, if each entry of `x` has mixed error `xRelative c`, `xAbsolute c`
against `xReference c`, then under the inferred conditions entry `o` of
`(Normalization.normalizedRows x epsilon true).2` is finite and has the inferred mixed error against
the same computation in real arithmetic on `xReference`. -/
roundoff mixed theorem normalizedRows_true_snd_mixedError := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols])
    (xReference xRelative xAbsolute : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, MixedError (toReal (x c)) (xReference c) (xRelative c) (xAbsolute c))
    (epsilon : Value) (o : Shape.Coord [rows]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon true).2 o)

/-- In every IEEE binary format, if each entry of `x` has mixed error `xRelative c`, `xAbsolute c`
against `xReference c`, then under the inferred conditions entry `c` of
`(Normalization.normalizedRows x epsilon false).1` is finite and has the inferred mixed error
against the same computation in real arithmetic on `xReference`. -/
roundoff mixed theorem normalizedRows_false_fst_mixedError := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols])
    (xReference xRelative xAbsolute : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, MixedError (toReal (x c)) (xReference c) (xRelative c) (xAbsolute c))
    (epsilon : Value) (c : Shape.Coord [rows, cols]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon false).1 c)

/-- In every IEEE binary format, if each entry of `x` has mixed error `xRelative c`, `xAbsolute c`
against `xReference c`, then under the inferred conditions entry `o` of
`(Normalization.normalizedRows x epsilon false).2` is finite and has the inferred mixed error
against the same computation in real arithmetic on `xReference`. -/
roundoff mixed theorem normalizedRows_false_snd_mixedError := fun [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols])
    (xReference xRelative xAbsolute : Shape.Coord [rows, cols] → ℝ)
    (hx : ∀ c, MixedError (toReal (x c)) (xReference c) (xRelative c) (xAbsolute c))
    (epsilon : Value) (o : Shape.Coord [rows]) (_hformat : format.isIEEE = true) =>
  toReal ((Normalization.normalizedRows x epsilon false).2 o)

end Inferred

attribute [roundoff_rule] normalizedRows_true_fst_mixedError normalizedRows_true_snd_mixedError
  normalizedRows_false_fst_mixedError normalizedRows_false_snd_mixedError

/-- Entry `c` of `(Normalization.normalizedRows x epsilon recenter).1` has the real value of entry
`c` of the first component of the definition of `Normalization.normalizedRows`. -/
@[roundoff_rule] theorem toReal_normalizedRows_fst [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols]) (epsilon : Value) (recenter : Bool)
    (c : Shape.Coord [rows, cols]) :
    toReal (Tensor.Rep.get (Normalization.normalizedRows x epsilon recenter).1 c) =
      toReal (Tensor.Rep.get (delta% Normalization.normalizedRows x epsilon recenter).1 c) :=
  rfl

/-- Entry `o` of `(Normalization.normalizedRows x epsilon recenter).2` has the real value of entry
`o` of the second component of the definition of `Normalization.normalizedRows`. -/
@[roundoff_rule] theorem toReal_normalizedRows_snd [Storage Value] {rows cols : Nat}
    (x : Tensor Value [rows, cols]) (epsilon : Value) (recenter : Bool)
    (o : Shape.Coord [rows]) :
    toReal (Tensor.Rep.get (Normalization.normalizedRows x epsilon recenter).2 o) =
      toReal (Tensor.Rep.get (delta% Normalization.normalizedRows x epsilon recenter).2 o) :=
  rfl

end TorchLean.Roundoff
