/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.IR.Builder
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# IR axis operations

Evaluate axis operations with the operator IR.

The tutorial builds and evaluates three graphs of registered operators at a chosen axis:

- `Op.softmax dim` (PyTorch: `torch.softmax(x, dim=dim)`)
- `Op.layerNorm` over the axes from `axis` on, the input viewed as rows of one axis
  (PyTorch: `F.layer_norm(x, normalized_shape=x.shape[axis:])`)
- `Op.cat dim` (PyTorch: `torch.cat(xs, dim=dim)`)

Softmax and cat take their axis as a `Fin` of the input's rank, so an axis value carries the bound
`axis.val < s.rank`: a numeral is read modulo the rank (`softmax 3` is axis `0`), and
`(⟨3, by decide⟩ : Fin 3)` does not elaborate. `Op.layerNorm` normalizes the last axis of its
input, and a reshape turns the axes from `axis` on into that axis. `Graph.Builder.build` runs the
structural check `Graph.check` before the graph is evaluated, and `Graph.eval` computes the outputs
with the operators' rules in Lean tensor operations. Everything runs on the CPU, with no LibTorch
dispatch and no device choice, and `--arithmetic` chooses the scalar implementation.

Run:

`scripts/lake.sh exe torchlean ir_axis_ops --arithmetic ieee`
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.Examples.IRAxisOps

open IR

/-- Command-line help for the IR axis-ops tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean IR axis-ops tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean ir_axis_ops [options]"
    , ""
    , "Options:"
    , "  --arithmetic native|ieee|complex"
    , ""
    , "Evaluates the checked operator graphs on CPU; no device or execution-mode selection."
    ]

/-!
## Small Operator Graphs
-/

/-- Softmax of the example's `[2, 3, 4]` input along `axis`. -/
def softmax (axis : Fin 3 := 1) : Except String Graph :=
  let (b, x) := ({} : Graph.Builder).input 0 (.value [2, 3, 4])
  let (b, ys) := b.op (Op.softmax (s := [2, 3, 4]) axis) [x]
  b.build ys.toArray

/-- The axes of the `[2, 3, 4]` input before `axis`, and the number of entries of the rest. -/
def rows (axis : Fin 3) : Shape × Nat :=
  ((([2, 3, 4] : List Nat).take axis.val).foldr (fun n s => .dim n s) .scalar,
    (([2, 3, 4] : List Nat).drop axis.val).foldr (· * ·) 1)

/--
LayerNorm of the `[2, 3, 4]` input over its axes from `axis` on, with scale one, bias zero and the
default epsilon (constants `1` to `3` of `layerNormPayload`). The graph views the input as the rows
`lead` followed by one axis of the remaining entries, applies `Op.layerNorm` over that axis and
views the result at the input shape. With the default `axis = 1` the last two axes, of extents `3`
and `4`, are normalized together.
-/
def layerNorm (axis : Fin 3 := 1) : Except String Graph :=
  let (lead, width) := rows axis
  if h : 0 < lead.size ∧ 0 < width ∧ Shape.size [2, 3, 4] = (lead ++ [width]).size then
    let (b, x) := ({} : Graph.Builder).input 0 (.value [2, 3, 4])
    let (b, scale) := b.const (.value [width])
    let (b, bias) := b.const (.value [width])
    let (b, epsilon) := b.const (.frozen [])
    let (b, xs) := b.op (Op.reshape (t := lead ++ [width]) h.2.2) [x]
    let (b, ys) := b.op (Op.layerNorm (lead := lead) h.1 h.2.1 rfl rfl)
      (xs ++ [scale, bias, epsilon])
    let (b, zs) := b.op (Op.reshape (t := [2, 3, 4]) h.2.2.symm) ys
    b.build zs.toArray
  else .error s!"layer norm from axis {axis}: the rows and the normalized axis need entries"

/-- The constants of `layerNorm axis` at scalar `α`: scale one, bias zero, the default epsilon. -/
def layerNormPayload {α : Type} [Storage α] [Context α] (axis : Fin 3 := 1) : Payload α :=
  let width := (rows axis).2
  { const? := fun
      | 1 => some ⟨.value [width], Tensor.full [width] 1⟩
      | 2 => some ⟨.value [width], Tensor.full [width] 0⟩
      | 3 => some ⟨.frozen [], Tensor.full [] normalizationEpsilon⟩
      | _ => none }

/--
`[2, 3, 4]` and `[2, 5, 4]` uniform tensors joined along `axis` by `Op.cat`, which reads inputs of
the shape `[2, 0, 4]` with the extent at `axis` replaced by `3` and `5`. Along axis one these are
the two tensors, and along another axis `build` rejects the operator's first input, whose port
differs from the one it reads. The keys `Random.keyOf 0 1` and `Random.keyOf 1 2` (seed,
counter) make both draws deterministic and distinct. The graph has no input.
-/
def cat (axis : Fin 3 := 1) : Except String Graph :=
  let (b, xs) := ({} : Graph.Builder).op
    (Op.randUniform (s := [2, 3, 4]) (Random.keyOf 0 1)) []
  let (b, ys) := b.op (Op.randUniform (s := [2, 5, 4]) (Random.keyOf 1 2)) []
  let (b, zs) := b.op (Op.cat (s := [2, 0, 4]) axis [3, 5]) (xs ++ ys)
  b.build zs.toArray

/-!
## Runner Helpers
-/

/--
A string of the first eight row-major entries of a tensor, or all of them when there are fewer,
ending in `, ...]` when entries are left out.
-/
def preview {α : Type} [Storage α] [ToString α] {σ : Shape}
    (tensor : Tensor α σ) : String :=
  let entries : Tensor α [min 8 σ.size] :=
    Tensor.flattenThenTake [] (min 8 σ.size) (Nat.min_le_right 8 σ.size) tensor
  let text := Tensor.pretty entries
  if σ.size > 8 then (text.dropEnd 1).toString ++ ", ...]" else text

/--
Evaluate a built graph with `Graph.eval` and print the shape and first scalars of its single value
output. Build and evaluation errors, and output lists of any other form, raise an `IO` error.
-/
def print
    {α : Type} [Storage α] [Context α] [ToString α]
    (tag : String) (g : Except String Graph) (payload : Payload α)
    (inputs : List (Op.Value α)) : IO Unit := do
  let g ← CLI.orThrow tag g
  match ← CLI.orThrow tag (g.eval payload inputs) with
  | [⟨⟨.value, shape⟩, output⟩] =>
      IO.println s!"[{tag}] output shape: {repr shape}"
      IO.println s!"[{tag}] first scalars: {preview output}"
  | _ => throw <| IO.userError s!"{tag}: expected one value output"

/--
Run every axis-op example at scalar type `α`.

The graphs carry no scalar type, so `main` runs the same examples at native `Float32`, at the
bit-level IEEE binary32 model or at complex numbers over it, as `--arithmetic` selects.
-/
def run
    {α : Type} [Storage α] [Context α] [ToString α]
    [FromFloat α]
    : IO Unit := do
  -- Flat entry `i` is `i / 10 - 1`, so the 24 input values step by `0.1` from `-1` to `1.3`.
  let inputTensor : Tensor α [2, 3, 4] :=
    Tensor.generateFlat [2, 3, 4] (fun i =>
      ofFloat ((Float.ofNat i) / 10.0 - 1.0))
  let input : Op.Value α := ⟨.value [2, 3, 4], inputTensor⟩

  IO.println ""
  IO.println "== IR axis ops tutorial =="
  IO.println "Each operator graph passes Graph.check, then Graph.eval computes it on CPU."

  IO.println ""
  IO.println "-- softmax axis=1 on shape [2,3,4]"
  print (tag := "softmax_middle_axis") (softmax 1) {} [input]

  IO.println ""
  IO.println "-- layerNorm from axis 1 on shape [2,3,4]"
  IO.println "PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]"
  print (tag := "layer_norm_middle_axes") (layerNorm 1) (layerNormPayload (α := α) 1) [input]

  IO.println ""
  IO.println "-- cat dim=1: [2,3,4] ++ [2,5,4] -> [2,8,4]"
  print (α := α) (tag := "cat_middle_dim") (cat 1) {} []

/--
Entry point: print the help for `--help` or `-h`, or run the tutorial at the scalar type that
`--arithmetic native|ieee|complex` picks (default `native`). Any other argument is an error.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (arithmetic, rest) ← CLI.orThrow "ir_axis_ops" <|
    Arithmetic.parseAndStrip args
  CLI.requireNoArgs "ir_axis_ops" rest
  arithmetic.log
  Arithmetic.withRuntime arithmetic @run

end TorchLean.Examples.IRAxisOps
