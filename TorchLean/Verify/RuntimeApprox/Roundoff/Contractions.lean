/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Constants
public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.NF.CROWN
public import TorchLean.Verify.NumericalCertificate
public import TorchLean.Verify.Floats.IEEE754.Expressions
public import TorchLean.Verify.RuntimeApprox.NF.Attention
public import TorchLean.Verify.RuntimeApprox.NF.Convolution
public import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
public import TorchLean.Verify.RuntimeApprox.NF.Normalization
public import TorchLean.Verify.RuntimeApprox.NF.Optimizers
public import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Autograd -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxShift
public import TorchLean.Verify.RuntimeApprox.TransformerBlock
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Attention
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff
public import FloatLib.Numerics.Automation.Attributes
public import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
public import Mathlib.Data.Nat.Factorial.DoubleFactorial
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init
public import TorchLean.Verify.RuntimeApprox.Roundoff.Fold
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Power
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Products
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor

/-!
# Rounding error of checked scaled contractions

`TorchLean.Attention.scaledProducts` computes a scaled contraction in checked arithmetic and replays
it exactly when an operation loses range. `roundoff` reads its ordinary loop beside its flags, and
its replay rounds the exact scaled sum once, for one group of products and for the two of a
product rule, on one matrix (`TorchLean.Attention.scaledProductsSlice`) and over any leading batch
axes.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- One group of scaled products at a coordinate, ordinary and replayed. -/
roundoff theorem scaled_products_error :=
  fun [Storage Value] {m n p : Nat} (a : Tensor Value [m, n]) (b : Tensor Value [n, p])
      (scale : Value) (c : Shape.Coord [m, p]) (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      ((Attention.scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) scale) c)

/-- The mixed form of `scaled_products_error`. -/
roundoff mixed theorem scaled_products_mixedError :=
  fun [Storage Value] {m n p : Nat} (a : Tensor Value [m, n]) (b : Tensor Value [n, p])
      (scale : Value) (c : Shape.Coord [m, p]) (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      ((Attention.scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) scale) c)

/-- The two groups of products of a product rule, summed in one contraction. -/
roundoff theorem joint_products_error :=
  fun [Storage Value] {m n p : Nat} (a da : Tensor Value [m, n]) (b db : Tensor Value [n, p])
      (scale : Value) (c : Shape.Coord [m, p]) (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal ((Attention.scaledProductsSlice
      (fun group : Fin 2 => if group.val == 0 then da else a)
      (fun group => if group.val == 0 then b else db) scale) c)

/-- One group of scaled products over any leading batch axes at a coordinate, ordinary and
replayed. -/
roundoff theorem batched_products_error :=
  fun [Storage Value] {batch : Shape} {m n p : Nat}
      (a : Tensor Value (batch ++ [m, n])) (b : Tensor Value (batch ++ [n, p]))
      (scale : Value) (c : Shape.Coord (batch ++ [m, p]))
      (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      ((Attention.scaledProducts (fun _ : Fin 1 => a) (fun _ => b) scale) c)

/-- The mixed form of `batched_products_error`. -/
roundoff mixed theorem batched_products_mixedError :=
  fun [Storage Value] {batch : Shape} {m n p : Nat}
      (a : Tensor Value (batch ++ [m, n])) (b : Tensor Value (batch ++ [n, p]))
      (scale : Value) (c : Shape.Coord (batch ++ [m, p]))
      (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal
      ((Attention.scaledProducts (fun _ : Fin 1 => a) (fun _ => b) scale) c)

/-- The two groups of a product rule over any leading batch axes, summed in one contraction. -/
roundoff theorem batched_joint_products_error :=
  fun [Storage Value] {batch : Shape} {m n p : Nat}
      (a da : Tensor Value (batch ++ [m, n])) (b db : Tensor Value (batch ++ [n, p]))
      (scale : Value) (c : Shape.Coord (batch ++ [m, p]))
      (hformat : format.isIEEE = true) =>
    ExecFloat.Binary.toReal ((Attention.scaledProducts
      (fun group : Fin 2 => if group.val == 0 then da else a)
      (fun group => if group.val == 0 then b else db) scale) c)

end TorchLean.Roundoff
