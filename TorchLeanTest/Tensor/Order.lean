/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Artifact
public import TorchLean.NN.IR.Order
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.ConfiguredBinary.Order
public import TorchLean.Tensor.Reduce.Order
public meta import TorchLean.NN.IR.Artifact
public meta import TorchLean.NN.IR.Order
public meta import TorchLean.NN.PyTorch.Export.IR
public meta import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Reduce.Order

/-!
# Orders of binary32 sums and selections

The entries `[2²⁴, 1, 1, 1]` have the exact sum `2²⁴ + 3`. `sumIn` gives a different binary32 sum
in each order, each within the depth bound of `TorchLean.IEEE754.abs_fold_add_sub_le`:

- `leftFold` gives `2²⁴`, the value of `Tensor.sum`: each `2²⁴ + 1` is a tie and rounds to the
  even `2²⁴`. Its tree has depth 4.
- `blocked 2` adds the blocks `2²⁴ + 1`, which rounds to `2²⁴`, and `1 + 1`, and gives `2²⁴ + 2`.
  Its tree has depth 3.
- `pairwise` adds `1 + (1 + 1)` first and gives `2²⁴ + 4`, the tie `2²⁴ + 3` rounded to even. Its
  tree has depth 3, `⌈log₂ 5⌉`.
- `given`, the tree `(e₃ + (e₂ + e₁)) + (start + e₀)`, also gives `2²⁴ + 4`. It covers 4 entries
  and, restricted, every smaller count, but not 5. Its scan folds the prefix of `k` entries with
  the tree restricted to them: `2²⁴`, `2²⁴`, `(e₂ + e₁) + 2²⁴ = 2²⁴ + 2` and `2²⁴ + 4`, where
  `cumsum` gives `2²⁴` four times.

`batchMatmulIn` multiplies the row `entries`, broadcast over a batch of two, by columns of ones,
so each product entry is the sum from zero in the order, the same bits as `sumIn`.

`repeated`, the tree `start + (e₀ + e₀)`, covers no positive count. At 6 entries each order is the
given tree of its own tree (`ReduceOrder.tree_ofTree`), and `Op.checkOrder` accepts `given` at 4
entries and refuses `repeated` (`Op.checkOrder_eq_ok`).

In the operator table, the graph of `Op.reduce .sum` over the four entries gives, under the IR
evaluator (`IR.Graph.eval`), the words of `sumIn` in each order, and in `repeated` it fails
`Op.check`. In the left fold `sumGraph given` is `sumGraph .leftFold`, so over `ℝ` it evaluates
to the outputs of the left fold (`IR.Graph.eval_leftFold_real`). It evaluates on every real input
of shape `[4]`, so the two graphs evaluate equally there. The wire record of the sum ends
with its order and decodes back. The JSON field of `given` is
`{"order": {"tree": [[3, [2, 1]], [null, 0]]}}`, which reads back as `given`, and a tree node with
three entries is an error. The PyTorch exporter writes the sum in the left fold and refuses it in
`pairwise`, since PyTorch fixes no order.

`marks` has the rows `[1, -0, +0, 1]` and `[-0, +0, NaN, 1]`. Along them `Tensor.cummax` and
`Tensor.cummin` move to a later entry at a tie, `-0` and `+0` included, and to a NaN, and from a NaN
only to a NaN. In each of the four orders `cummaxIn` and `cumminIn` give the same words and indices.
In every order that covers the row they are `cummax` and `cummin` at binary32, bit for bit
(`ConfiguredBinary.cummaxIn_eq`, `ConfiguredBinary.cumminIn_eq`), and over `ℝ`
(`Tensor.cummaxIn_real`, `Tensor.cumminIn_real`). The graph of `Op.cummax` gives those words under
the IR evaluator in each order and fails `Op.check` in `repeated`. The wire records of `Op.cummax`
and `Op.cummin` end with their order and decode back, and the PyTorch exporter writes the running
maximum in the left fold and refuses it in `pairwise`, as it does the sum.

`Attention.scaledMatmulIn` multiplies `rows`, the rows `[2²⁴, 1, 1, 1]` and `[2²⁴, 3, 3, 0]`, by a
column of ones and scales by one. The first row's exact sum `2²⁴ + 3` rounds once to `2²⁴ + 4` in
every order: `leftFold` and `blocked 2` absorb a `1` into `2²⁴` and replay the entry, and
`pairwise` and `given` add `2²⁴` to `3` last. The second row's sum `2²⁴ + 6` is exact in
`pairwise` and `given`, and `leftFold` and `blocked 2` give `2²⁴ + 8`, rounding the ties `2²⁴ + 3`
and `2²⁴ + 7` to even. The left fold gives the words of `Attention.scaledMatmul`. `Op.check` accepts
`Op.scaledMatmul` in `given` over the 4 entries and refuses it in `repeated`, its wire record ends
with its order and decodes back, and over `ℝ` it is `Attention.scaledMatmul` in `given`
(`Attention.scaledMatmulIn_real`).
-/

@[expose] public section

namespace TorchLean.Test.Order

open FloatLib.Floats
open Tensor (ReduceOrder)

/-- The binary32 entries `[2²⁴, 1, 1, 1]`. -/
def entries : Tensor Binary32 [4] :=
  Tensor.Rep.ofArray
    (#[0x4b800000, 0x3f800000, 0x3f800000, 0x3f800000].map ExecFloat.Binary.ofNatBits)
    (by simp [Shape.size])

#guard [ReduceOrder.leftFold, .blocked 2, .pairwise].map
    (fun order => ExecFloat.Binary.toNatBits (Tensor.sumIn order entries)) ==
  [0x4b800000, 0x4b800001, 0x4b800002]
#guard ExecFloat.Binary.toNatBits (Tensor.sum entries) == 0x4b800000
#guard [ReduceOrder.leftFold, .blocked 2, .pairwise].map (fun order => (order.tree 4).depth) ==
  [4, 3, 3]

/-- The tree `(e₃ + (e₂ + e₁)) + (start + e₀)`. -/
def given : ReduceOrder :=
  .ofTree (.node (.node (.leaf (some 3)) (.node (.leaf (some 2)) (.leaf (some 1))))
    (.node (.leaf none) (.leaf (some 0))))

#guard ExecFloat.Binary.toNatBits (Tensor.sumIn given entries) == 0x4b800002
#guard (given.tree 4).depth == 3
#guard (List.range 5).all fun n => decide (given.Covers n)
#guard !decide (given.Covers 5)
#guard (Tensor.Rep.data (Tensor.scanIn given (· + ·) 0 entries ⟨0, by decide⟩)).map
    ExecFloat.Binary.toNatBits == #[0x4b800000, 0x4b800000, 0x4b800001, 0x4b800002]
#guard (Tensor.Rep.data (Tensor.cumsum entries ⟨0, by decide⟩)).map ExecFloat.Binary.toNatBits ==
  #[0x4b800000, 0x4b800000, 0x4b800000, 0x4b800000]

/-- The row `entries`, its batch of one broadcast to two, times two columns of ones. -/
def batched (order : ReduceOrder) : Tensor Binary32 ([2] ++ [1, 1]) :=
  Tensor.batchMatmulIn (batchA := [1]) (batchB := [2]) (m := 1) (n := 4) (p := 1) order
    (by decide) (by decide) (Tensor.reshape entries ([1] ++ [1, 4]))
    (Tensor.Rep.ofFn fun _ => ExecFloat.Binary.ofNatBits 0x3f800000)

#guard [ReduceOrder.leftFold, .blocked 2, .pairwise, given].map
    (fun order => (Tensor.Rep.data (batched order)).map ExecFloat.Binary.toNatBits) ==
  [#[0x4b800000, 0x4b800000], #[0x4b800001, 0x4b800001], #[0x4b800002, 0x4b800002],
    #[0x4b800002, 0x4b800002]]

/-- The tree `start + (e₀ + e₀)`. -/
def repeated : ReduceOrder :=
  .ofTree (.node (.leaf none) (.node (.leaf (some 0)) (.leaf (some 0))))

#guard (List.range 4).all fun n => !decide (repeated.Covers (n + 1))

/-- At 6 entries each of four orders is the given tree of its own tree. -/
theorem ofTree_tree : ∀ order ∈ [ReduceOrder.leftFold, .pairwise, .blocked 3, given],
    (ReduceOrder.ofTree ((order.tree 6).map (Option.map Fin.val))).tree 6 = order.tree 6 :=
  fun order _ => ReduceOrder.tree_ofTree order 6

/-- `Op.checkOrder` accepts `given` at its 4 entries and refuses `repeated` there. -/
theorem checkOrder_given_repeated :
    Op.checkOrder "reduce" given 4 = .ok () ∧ Op.checkOrder "reduce" repeated 4 ≠ .ok () :=
  ⟨(Op.checkOrder_eq_ok _ _ _).mpr (by decide), mt (Op.checkOrder_eq_ok _ _ _).mp (by decide)⟩

/-- The graph of `Op.reduce .sum` over the input `[4]` in the order `order`. -/
def sumGraph (order : ReduceOrder) : IR.Graph :=
  ⟨#[.input 0 (.value [4]),
    .op ⟨_, _, Op.reduce (s := [4]) .sum [⟨0, by decide⟩] false (by decide) order⟩ #[⟨0, 0⟩]],
    #[⟨1, 0⟩]⟩

/-- The words of the output of `graph` on the input `x` under the IR evaluator, or its error. -/
def graphWords {s : Shape} (graph : IR.Graph) (x : Tensor Binary32 s) :
    Except String (Array Nat) := do
  match graph.eval (α := Binary32) {} [⟨.value s, x⟩] with
  | .ok [⟨⟨.value, _⟩, y⟩] => pure ((Tensor.Rep.data y).map ExecFloat.Binary.toNatBits)
  | .ok _ => throw "IR: expected one value output"
  | .error e => throw e

#guard [ReduceOrder.leftFold, .blocked 2, .pairwise, given].map
    (fun order => (graphWords (sumGraph order) entries).toOption) ==
  [some #[0x4b800000], some #[0x4b800001], some #[0x4b800002], some #[0x4b800002]]
#guard match graphWords (sumGraph repeated) entries with
  | .ok _ => false
  | .error message => (message.splitOn "the order does not cover the 4 reduced entries").length > 1

/-- `sumGraph given` in the left fold is `sumGraph .leftFold`. -/
theorem sumGraph_leftFold : (sumGraph given).leftFold = sumGraph .leftFold := by
  simp only [IR.Graph.leftFold, sumGraph, List.map_toArray, List.map_cons, List.map_nil,
    IR.Node.leftFold, Op.leftFold]

/-- Over `ℝ`, `sumGraph given` evaluates to the outputs of `sumGraph .leftFold`. -/
theorem eval_sumGraph_given {payload : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {outputs : List (Op.Value ℝ)} (h : (sumGraph given).eval payload inputs = .ok outputs) :
    (sumGraph .leftFold).eval payload inputs = .ok outputs :=
  sumGraph_leftFold ▸ IR.Graph.eval_leftFold_real h

/-- Over `ℝ`, `sumGraph given` evaluates on every input of shape `[4]`. -/
theorem eval_sumGraph_given_ok (x : Tensor ℝ [4]) :
    ∃ outputs, (sumGraph given).eval {} [⟨.value [4], x⟩] = .ok outputs :=
  ⟨_, rfl⟩

/-- Over `ℝ`, `sumGraph given` and `sumGraph .leftFold` evaluate equally on every input of shape
`[4]`. -/
theorem eval_sumGraph_eq (x : Tensor ℝ [4]) :
    (sumGraph given).eval {} [⟨.value [4], x⟩] =
      (sumGraph .leftFold).eval {} [⟨.value [4], x⟩] := by
  obtain ⟨outputs, h⟩ := eval_sumGraph_given_ok x
  rw [h, eval_sumGraph_given h]

/-- The binary32 rows `[1, -0, +0, 1]` and `[-0, +0, NaN, 1]`. -/
def marks : Tensor Binary32 [2, 4] :=
  Tensor.Rep.ofArray
    (#[0x3f800000, 0x80000000, 0x00000000, 0x3f800000,
      0x80000000, 0x00000000, 0x7fc00000, 0x3f800000].map ExecFloat.Binary.ofNatBits)
    (by simp [Shape.size])

/-- The words of the running values of a selection along the rows, and its indices. -/
def words (y : Tensor Binary32 [2, 4] × Tensor (Fin 4) [2, 4]) : Array Nat × Array Nat :=
  ((Tensor.Rep.data y.1).map ExecFloat.Binary.toNatBits, (Tensor.Rep.data y.2).map Fin.val)

/-- The words of `Tensor.cummax` along the rows of `marks`, and its indices. -/
def maxima : Array Nat × Array Nat :=
  (#[0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
    0x80000000, 0x00000000, 0x7fc00000, 0x7fc00000], #[0, 0, 0, 3, 0, 1, 2, 2])

/-- The words of `Tensor.cummin` along the rows of `marks`, and its indices. -/
def minima : Array Nat × Array Nat :=
  (#[0x3f800000, 0x80000000, 0x00000000, 0x00000000,
    0x80000000, 0x00000000, 0x7fc00000, 0x7fc00000], #[0, 1, 2, 2, 0, 1, 2, 2])

#guard words (Tensor.cummax marks ⟨1, by decide⟩) == maxima &&
  words (Tensor.cummin marks ⟨1, by decide⟩) == minima
#guard [ReduceOrder.leftFold, .pairwise, .blocked 2, given].all fun order =>
  words (Tensor.cummaxIn order marks ⟨1, by decide⟩) == maxima &&
    words (Tensor.cumminIn order marks ⟨1, by decide⟩) == minima

/-- At binary32, running maxima and minima along the rows in the order `given` are those of the
left fold, bit for bit. -/
theorem cummaxIn_cumminIn_given_binary32 (x : Tensor Binary32 [2, 4]) :
    Tensor.cummaxIn given x ⟨1, by decide⟩ = Tensor.cummax x ⟨1, by decide⟩ ∧
      Tensor.cumminIn given x ⟨1, by decide⟩ = Tensor.cummin x ⟨1, by decide⟩ :=
  ⟨ConfiguredBinary.cummaxIn_eq x _ (by decide), ConfiguredBinary.cumminIn_eq x _ (by decide)⟩

/-- Over `ℝ`, running maxima and minima along the rows in the order `given` are those of the left
fold. -/
theorem cummaxIn_cumminIn_given (x : Tensor ℝ [2, 4]) :
    Tensor.cummaxIn given x ⟨1, by decide⟩ = Tensor.cummax x ⟨1, by decide⟩ ∧
      Tensor.cumminIn given x ⟨1, by decide⟩ = Tensor.cummin x ⟨1, by decide⟩ :=
  ⟨Tensor.cummaxIn_real x _ (by decide), Tensor.cumminIn_real x _ (by decide)⟩

/-- The binary32 rows `[2²⁴, 1, 1, 1]` and `[2²⁴, 3, 3, 0]`. -/
def rows : Tensor Binary32 [2, 4] :=
  Tensor.Rep.ofArray
    (#[0x4b800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x4b800000, 0x40400000, 0x40400000, 0x00000000].map ExecFloat.Binary.ofNatBits)
    (by simp [Shape.size])

/-- The binary32 one. -/
def one : Binary32 := ExecFloat.Binary.ofNatBits 0x3f800000

/-- The words of `1 * (rows * ones)` in the order `order`. -/
def scaledWords (order : ReduceOrder) : Array Nat :=
  (Tensor.Rep.data (Attention.scaledMatmulIn (batchA := []) (batchB := []) (batch := []) (m := 2)
    (n := 4) (p := 1) order (by decide) (by decide) rows (Tensor.Rep.ofFn fun _ => one)
    one)).map ExecFloat.Binary.toNatBits

#guard (Tensor.Rep.data (Attention.scaledMatmul (batchA := []) (batchB := []) (batch := [])
    (m := 2) (n := 4) (p := 1) (by decide) (by decide) rows (Tensor.Rep.ofFn fun _ => one)
    one)).map ExecFloat.Binary.toNatBits == #[0x4b800002, 0x4b800004]
#guard [ReduceOrder.leftFold, .blocked 2, .pairwise, given].map scaledWords ==
  [#[0x4b800002, 0x4b800004], #[0x4b800002, 0x4b800004], #[0x4b800002, 0x4b800003],
    #[0x4b800002, 0x4b800003]]

/-- `Op.scaledMatmul` of a `[2, 4]` and a `[4, 1]` matrix in the order `order`. -/
def scaledOp (order : ReduceOrder) : Σ ins outs, Op ins outs :=
  ⟨_, _, Op.scaledMatmul (batchA := []) (batchB := []) (batch := []) (m := 2) (n := 4) (p := 1)
    false false (by decide) (by decide) (by decide) (by decide) order⟩

#guard (Op.check (scaledOp given).2.2).toOption.isSome &&
  match Op.check (scaledOp repeated).2.2 with
  | .ok _ => false
  | .error message => (message.splitOn "the order does not cover the 4 reduced entries").length > 1

/-- Over `ℝ`, the scaled product in the order `given` is `Attention.scaledMatmul`. -/
theorem scaledMatmulIn_given (left : Tensor ℝ [2, 4]) (right : Tensor ℝ [4, 1]) (scale : ℝ) :
    Attention.scaledMatmulIn (batchA := []) (batchB := []) (batch := []) (m := 2) (n := 4)
        (p := 1) given (by decide) (by decide) left right scale =
      Attention.scaledMatmul (batchA := []) (batchB := []) (batch := []) (m := 2) (n := 4)
        (p := 1) (by decide) (by decide) left right scale :=
  Attention.scaledMatmulIn_real (batchA := []) (batchB := []) (batch := []) (m := 2) (n := 4)
    (p := 1) _ _ left right scale (by decide)

/-- The graph of `Op.cummax` along the rows of the input `[2, 4]` in the order `order`, with the
running values as output. -/
def maxGraph (order : ReduceOrder) : IR.Graph :=
  ⟨#[.input 0 (.value [2, 4]),
    .op ⟨_, _, Op.cummax (s := [2, 4]) ⟨1, by decide⟩ order⟩ #[⟨0, 0⟩]],
    #[⟨1, 0⟩]⟩

#guard [ReduceOrder.leftFold, .pairwise, .blocked 2, given].all fun order =>
  (graphWords (maxGraph order) marks).toOption == some maxima.1
#guard match graphWords (maxGraph repeated) marks with
  | .ok _ => false
  | .error message => (message.splitOn "the order does not cover the 4 reduced entries").length > 1

#guard [ReduceOrder.leftFold, .pairwise, .blocked 2, given].all fun order =>
  [(⟨_, _, Op.reduce (s := [4]) .sum [⟨0, by decide⟩] false (by decide) order⟩ :
      Σ ins outs, Op ins outs),
    ⟨_, _, Op.cummax (s := [2, 4]) ⟨1, by decide⟩ order⟩,
    ⟨_, _, Op.cummin (s := [2, 4]) ⟨1, by decide⟩ order⟩, scaledOp order].all fun o =>
  let wire := Op.encode o
  wire.fields.getLast? == some (.order order) && (Op.decode wire).map Op.encode == some wire

/-- The operator field the IR importer reads from the JSON text `text`. -/
def field (text : String) : Except String Op.Field :=
  Lean.Json.parse text >>= Json.opField

#guard (Json.ofOpField (.order given)).compress == "{\"order\":{\"tree\":[[3,[2,1]],[null,0]]}}"
#guard (field "{\"order\": {\"tree\": [[3, [2, 1]], [null, 0]]}}").toOption == some (.order given)
#guard match field "{\"order\": {\"tree\": [1, 2, 3]}}" with
  | .ok _ => false
  | .error message => (message.splitOn "a reduction tree node has two entries").length > 1

/-- The module `PyTorch.emit` writes for `graph`, or its refusal. -/
def emitted (graph : IR.Graph) : Except String String := do
  PyTorch.emit ⟨graph, {}, #[]⟩ (options := { includeTrainingSkeleton := false })

#guard [sumGraph, maxGraph].all fun graph =>
  (emitted (graph .leftFold)).toOption.isSome &&
    match emitted (graph .pairwise) with
    | .ok _ => false
    | .error message =>
        (message.splitOn "in an order other than the left fold has no PyTorch translation").length >
          1

end TorchLean.Test.Order
