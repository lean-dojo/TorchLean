/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.BoundOps
public import TorchLean.Verify.JSON
public import TorchLean.Verify.Tensor

/-!
# VNNLIB-style output specifications

The benchmark-independent part of the VNN-COMP artifact boundary. An instance pairs an input box
with an unsafe output region, a disjunction of conjunctions of linear rows
$\mathrm{mat}\,y\leq\mathrm{rhs}$, and `loadSuite` reads instances from the compact
`vnnlib_suite_v0_1` JSON export. Model-specific checkers such as MNIST-FC build the graph and its
output bounds. The arithmetic that interprets the rows belongs to this module: a term is refuted
when the lower bound of one of its rows over the output box, rounded downward, exceeds the row's
right-hand side.
-/

@[expose] public section

namespace TorchLean.Examples.VNNComp

open Lean
open Lean.Json
open CROWN

/-- One conjunction term $\mathrm{mat}\,y\leq\mathrm{rhs}$ in a VNNLIB disjunction. -/
structure Term where
  /-- Number of inequalities in this conjunction. -/
  rows : Nat
  /-- Network output dimension. -/
  cols : Nat
  /-- Left-hand-side coefficients. -/
  mat : Tensor Float [rows, cols]
  /-- Right-hand-side thresholds. -/
  rhs : Tensor Float [rows]

/-- A VNNLIB-style unsafe-region spec: a disjunction of conjunction terms. -/
abbrev Property := Array Term

/--
One exported VNN-COMP instance. Its `spec` is a disjunction whose terms are conjunctions
$\mathrm{mat}\,y\leq\mathrm{rhs}$ over the network output vector `y`.
-/
structure Instance where
  /-- Instance id copied from the exported suite JSON. -/
  id : Nat
  /-- Ordered input endpoints, decoded at the artifact boundary. -/
  input : FlatBox Float
  /-- Unsafe output-region specification. -/
  property : Property

/--
Load a suite in the compact JSON format `vnnlib_suite_v0_1`.

Each entry of `instances` has an `id`, finite input endpoints `input_lo` and `input_hi` of equal
length with `input_lo ≤ input_hi`, and a `spec` array of terms with a rectangular `mat` and one
`rhs` entry per row.
-/
def loadSuite (path : String) : IO (Array Instance) := do
  let top ← IO.ofExcept <| Json.object "top-level" (← Json.readFile path)
  let format ← IO.ofExcept <|
    Json.field "top-level" "format" top >>= Json.string "top-level.format"
  IO.ofExcept <| Json.expectFormat format "vnnlib_suite_v0_1"
  let instArr ← IO.ofExcept <|
    Json.field "top-level" "instances" top >>= Json.array "top-level.instances"
  let mut out : Array Instance := #[]
  for ex in instArr do
    let exo ← IO.ofExcept <| Json.object "instance" ex
    let id ← IO.ofExcept <|
      Json.field "instance" "id" exo >>= Json.expectNat "instance.id"
    let lo ← IO.ofExcept <|
      Json.field "instance" "input_lo" exo >>=
        Json.parseFiniteFloatArray "instance.input_lo"
    let hi ← IO.ofExcept <|
      Json.field "instance" "input_hi" exo >>=
        Json.parseFiniteFloatArray "instance.input_hi"
    let inputDim := lo.size
    let lo ← Tensor.requireArray "input_lo" [inputDim] lo
    let hi ← Tensor.requireArray "input_hi" [inputDim] hi
    if !Tensor.boundsOrdered lo hi then
      throw <| IO.userError s!"instance {id}: input box endpoints are mismatched or reversed"
    let specArr ← IO.ofExcept <|
      Json.field "instance" "spec" exo >>= Json.array "instance.spec"
    let mut specOut : Property := #[]
    for t in specArr do
      let termObj ← IO.ofExcept <| Json.object "spec term" t
      let matJ ← IO.ofExcept <| Json.field "spec term" "mat" termObj
      let mat ← IO.ofExcept <| Json.parseFiniteFloatMatrix "spec term.mat" matJ
      let rhs ← IO.ofExcept <|
        Json.field "spec term" "rhs" termObj >>=
          Json.parseFiniteFloatArray "spec term.rhs"
      let rows := mat.size
      let cols := (mat[0]?).map Array.size |>.getD 0
      let some mat := Tensor.ofRows? rows cols mat
        | throw <| IO.userError "spec term.mat: ragged coefficient matrix"
      let rhs ← Tensor.requireArray "spec term.rhs" [rows] rhs
      specOut := specOut.push { rows, cols, mat, rhs }
    out := out.push { id := id, input := { dim := inputDim, lo, hi }, property := specOut }
  pure out

/--
Lower-bound one linear row over an output interval box.

For each coefficient $a_j$, the minimum of $a_jy_j$ over
$y_j\in[\mathrm{lo}_j,\mathrm{hi}_j]$ is the smaller of the endpoint products. Each product and
partial sum is rounded to nearest and then stepped down to the adjacent binary64 value
(`BoundOps.mulDown`, `BoundOps.addDown`), so the result is at most the real minimum of the row.
-/
def rowLowerBoundOnBox {n : Nat} (row yLo yHi : Tensor Float [n]) : Float :=
  let lowerProducts := Tensor.zipWith BoundOps.mulDown row yLo
  let upperProducts := Tensor.zipWith BoundOps.mulDown row yHi
  Tensor.foldl BoundOps.addDown 0.0 (Tensor.zipWith min lowerProducts upperProducts)

/-- `Term.refutedByOutputBox box term` is `true` when `box` has the term's width `cols`, its bounds
are ordered, and some row's lower bound over it exceeds that row's right-hand side. -/
def Term.refutedByOutputBox (box : FlatBox Float) (term : Term) : Bool :=
  if h : box.dim = term.cols then
    let lo := Tensor.castShape box.lo (congrArg (fun n => Shape.dim n .scalar) h)
    let hi := Tensor.castShape box.hi (congrArg (fun n => Shape.dim n .scalar) h)
    Tensor.boundsOrdered lo hi &&
      (List.finRange term.rows).any (fun i =>
        rowLowerBoundOnBox (term.mat.unstack i) lo hi > term.rhs.getScalar i)
  else false

/--
Check whether an unsafe VNNLIB spec is refuted by an output interval box.

The spec is a disjunction of conjunctions, so no output in the box lies in the unsafe region when
every disjunct is refuted, and one row lower bound above its right-hand side refutes a conjunction.
The check runs on host `Float` through the `BoundOps` rounding boundary, and no Lean theorem links
its result to the real-valued semantics of the graph.
-/
def refutedByOutputBox (box : FlatBox Float) (spec : Property) : Bool :=
  spec.all (Term.refutedByOutputBox box)

end TorchLean.Examples.VNNComp
