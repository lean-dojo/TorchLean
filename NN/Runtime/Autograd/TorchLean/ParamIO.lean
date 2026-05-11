/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN
public import Lean

/-!
# Parameter IO (Float, exact bitwise)

TorchLean examples often want a small "train once, save weights, reload later" workflow.

This module provides a small, explicit format for saving and loading *Float* parameter packs
(`Torch.TList Float ss`) without relying on floating-point JSON parsing.

## Format

We encode each `Float` value by its IEEE-754 bit pattern (`Float.toBits : Float → UInt64`) and
store those bits as JSON natural numbers.

This is:
- exact (round-trips every NaN payload and subnormal),
- stable across locales, and
- easy to validate (length = `Shape.size`).

The file layout is:

```json
{
  "format": "torchlean_paramlist_bits_v1",
  "params": [
    { "shape": [d1, d2, ...], "values": [u64bits, u64bits, ...] },
    ...
  ]
}
```
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean
namespace ParamIO

open Spec

def formatTag : String := "torchlean_paramlist_bits_v1"

def jsonNat (n : Nat) : Lean.Json :=
  Lean.Json.num (Lean.JsonNumber.fromInt (Int.ofNat n))

def floatToJsonBits (x : Float) : Lean.Json :=
  jsonNat x.toBits.toNat

def jsonBitsToFloat (j : Lean.Json) : Except String Float := do
  let n ← Lean.Json.getNat? j
  let limit : Nat := (2 : Nat) ^ 64
  if n >= limit then
    throw s!"ParamIO: float bits out of range (expected < 2^64, got {n})"
  let bits : UInt64 := UInt64.ofNat n
  pure (Float.ofBits bits)

def tensorOfFlatListExact {α : Type} [Zero α] (tag : String) :
    (s : Shape) → (xs : List α) → Except String (Tensor α s)
  | .scalar, [x] => pure (Tensor.scalar x)
  | .scalar, xs =>
      throw s!"{tag}: expected 1 scalar, got {xs.length}"
  | .dim n rest, xs => do
      let chunk := Shape.size rest
      let expected := n * chunk
      if xs.length != expected then
        throw s!"{tag}: expected {expected} scalars for shape {Shape.toList (.dim n rest)}, got {xs.length}"
      -- Build the dependent function `Fin n → Tensor α rest` by slicing the input list.
      let f : Fin n → Tensor α rest := fun i =>
        -- `splitAt` makes the slice total; correctness comes from the length check above.
        let off := i.val * chunk
        let slice := (xs.drop off).take chunk
        match tensorOfFlatListExact (tag := tag) rest slice with
        | Except.ok t => t
        | Except.error _ =>
            -- Unreachable because `xs.length` matches `n * chunk`.
            Spec.zeros α rest
      pure <| Tensor.dim f

def tensorToJsonBits (s : Shape) (t : Tensor Float s) : Lean.Json :=
  let dims : Lean.Json := Lean.Json.arr (Shape.toList s |>.toArray |>.map jsonNat)
  let values : Lean.Json :=
    Lean.Json.arr ((Spec.toList t).toArray.map floatToJsonBits)
  Lean.Json.mkObj [("shape", dims), ("values", values)]

def tensorFromJsonBits (tag : String) (s : Shape) (j : Lean.Json) :
    Except String (Tensor Float s) := do
  let o ← Lean.Json.getObj? j
  let shapeJ := (o.get? "shape").getD Lean.Json.null
  let valuesJ := (o.get? "values").getD (Lean.Json.arr #[])
  let dimsArr ← Lean.Json.getArr? shapeJ
  let dims : List Nat ←
    dimsArr.toList.mapM (fun d => Lean.Json.getNat? d)
  if Shape.ofList dims != s then
    throw s!"{tag}: shape mismatch (file={dims}, expected={Shape.toList s})"
  let valsArr ← Lean.Json.getArr? valuesJ
  let vals : List Float ←
    valsArr.toList.mapM (fun v => jsonBitsToFloat v)
  tensorOfFlatListExact (tag := tag) s vals

def tListToJsonBits {ss : List Shape} : Torch.TList Float ss → Lean.Json
  | .nil => Lean.Json.arr #[]
  | .cons (s := s) t ts =>
      match tListToJsonBits (ss := _) ts with
      | Lean.Json.arr xs =>
          Lean.Json.arr (#[tensorToJsonBits s t] ++ xs)
      | _ => Lean.Json.arr #[]

def tListFromJsonBits (tag : String) :
    {ss : List Shape} → (j : Lean.Json) → Except String (Torch.TList Float ss)
  | [], _ => pure .nil
  | s :: ss, j => do
      let xs ← Lean.Json.getArr? j
      if xs.size = 0 then
        throw s!"{tag}: missing parameter 1/{(s :: ss).length}"
      let headJ := xs.getD 0 Lean.Json.null
      let tailJ := Lean.Json.arr (xs.extract 1 xs.size)
      let head ← tensorFromJsonBits (tag := tag) s headJ
      let tail ← tListFromJsonBits (tag := tag) (ss := ss) tailJ
      pure (.cons head tail)

def writeTListBits (path : System.FilePath) {ss : List Shape}
    (ps : Torch.TList Float ss) (pretty : Bool := true) : IO Unit := do
  let top : Lean.Json :=
    Lean.Json.mkObj [("format", Lean.Json.str formatTag), ("params", tListToJsonBits ps)]
  let s := if pretty then top.pretty else top.compress
  IO.FS.writeFile path s

def readTListBits (path : System.FilePath) {ss : List Shape} :
    IO (Except String (Torch.TList Float ss)) := do
  let s ← IO.FS.readFile path
  match Lean.Json.parse s with
  | Except.error e =>
      pure (Except.error s!"ParamIO: JSON parse error: {e}")
  | Except.ok j =>
      match Lean.Json.getObj? j with
      | Except.error e =>
          pure (Except.error s!"ParamIO: expected object: {e}")
      | Except.ok o =>
          let fmt := (o.get? "format").getD (Lean.Json.str "")
          match Lean.Json.getStr? fmt with
          | Except.error _ =>
              pure (Except.error "ParamIO: missing `format` string")
          | Except.ok t =>
              if t != formatTag then
                pure (Except.error s!"ParamIO: unsupported format: {t}")
              else
                let paramsJ := (o.get? "params").getD (Lean.Json.arr #[])
                pure (tListFromJsonBits (tag := "ParamIO") (ss := ss) paramsJ)

end ParamIO
end TorchLean
end Autograd
end Runtime
