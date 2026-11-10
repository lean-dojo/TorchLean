/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.JSON
public import TorchLean.Tensor.Layers.Linear

/-! # Shape-checked decoding of exact rational parameters -/

@[expose] public section

namespace TorchLean.Certificate.RationalJson

open Lean

/-- Decode a JSON array of exactly `n` integer or fraction strings, such as `"-3"` and `"5/2"`,
into rationals with no floating-point conversion. -/
def decodeVector (n : Nat) (j : Json) : Except String (Tensor ℚ [n]) := do
  let xs ← (← j.getArr?).mapM fun x => do
    Json.parseRatString (← x.getStr?)
  if h : xs.size = n then
    return Tensor.ofFn fun i => xs[i.val]'(by simp [h])
  else
    throw s!"expected {n} entries, received {xs.size}"

/-- Decode a linear layer from `"weights"`, an array of rows of `n` entries each, and `"bias"`, one
entry per row, so the output dimension is the number of rows. -/
def decodeLinear (n : Nat) (j : Json) : Except String (Σ m : Nat, Linear ℚ n m) := do
  let rows ← (← (← j.getObjVal? "weights").getArr?).mapM (decodeVector n)
  let bias ← decodeVector rows.size (← j.getObjVal? "bias")
  return ⟨rows.size, ⟨Tensor.dim (fun i => rows[i]), bias⟩⟩

/-- Decode a nonempty layer list into a chain: each layer's output dimension is the next layer's
input dimension, and `comp` joins the decoded layers. -/
def decodeChain {Chain : Nat → Nat → Type}
    (decodeLayer : (n : Nat) → Json → Except String (Σ m : Nat, Chain n m))
    (comp : {n m k : Nat} → Chain n m → Chain m k → Chain n k) (n : Nat) :
    List Json → Except String (Σ m : Nat, Chain n m)
  | [] => .error "expected at least one layer"
  | j :: rest => do
      let ⟨m, first⟩ ← decodeLayer n j
      match rest with
      | [] => return ⟨m, first⟩
      | _ :: _ =>
          let ⟨k, tail⟩ ← decodeChain decodeLayer comp m rest
          return ⟨k, comp first tail⟩

end TorchLean.Certificate.RationalJson
