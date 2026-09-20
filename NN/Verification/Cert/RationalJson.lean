/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Splines.PiecewisePolyCert
public import NN.Spec.Layers.Linear
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-! # Shape-checked decoding of exact rational parameters -/

@[expose] public section

namespace NN.Verification.Cert.RationalJson

open Lean _root_.Spec TorchLean

/-- Decode integer or fraction strings without floating-point conversion. -/
def decodeVector (n : Nat) (j : Json) : Except String (Tensor ℚ [n]) := do
  let xs ← (← j.getArr?).mapM fun x => do
    NN.Verification.Splines.PiecewisePolyCert.parseRatString (← x.getStr?)
  if h : xs.size = n then
    return Tensor.ofFn fun i => xs[i.val]'(by simp [h])
  else
    throw s!"expected {n} entries, received {xs.size}"

/-- Decode a row-major matrix and bias, checking both dimensions. -/
def decodeLinear (n : Nat) (j : Json) : Except String (Σ m : Nat, LinearSpec ℚ n m) := do
  let rows ← (← (← j.getObjVal? "weights").getArr?).mapM (decodeVector n)
  let bias ← decodeVector rows.size (← j.getObjVal? "bias")
  return ⟨rows.size, ⟨Tensor.dim (fun i => rows[i]), bias⟩⟩

end NN.Verification.Cert.RationalJson
