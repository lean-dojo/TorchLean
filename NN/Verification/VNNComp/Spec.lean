/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Verification.Util.Json
public import NN.Verification.Util.Tensor

/-!
# VNNLIB-style output specifications

This module contains the benchmark-independent part of the VNN-COMP artifact boundary:

- an input box plus a disjunction-of-conjunctions output spec,
- a JSON loader for the compact `vnnlib_suite_v0_1` export format,
- the outward-rounded interval-box refutation check used by the executable VNN-COMP runner.

Model-specific checkers, such as MNIST-FC, build a graph and output bounds. The arithmetic for
interpreting the VNNLIB rows lives here.
-/

@[expose] public section

namespace NN.Verification.VNNComp.VNNLib

open Lean
open Json
open NN.Verification.Json
open NN.MLTheory.CROWN
open TorchLean

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
abbrev Spec := Array Term

/--
One exported VNN-COMP instance.

`spec` is a disjunction-of-conjunctions: each term is a conjunction
$\mathrm{mat}\,y\leq\mathrm{rhs}$ over the
network output vector `y`.
-/
structure Instance where
  /-- Instance id copied from the exported suite JSON. -/
  id : Nat
  /-- Ordered input endpoints, decoded at the artifact boundary. -/
  input : FlatBox Float
  /-- Unsafe output-region specification. -/
  spec : Spec

/--
Load the compact VNNLIB suite JSON format used by TorchLean checkers.

Expected top-level format: `vnnlib_suite_v0_1`.
-/
def loadSuite (path : String) : IO (Array Instance) := do
  let top ← readJsonObjectFile path
  expectFormat top "vnnlib_suite_v0_1"
  let instArr ← expectFieldArray top "instances" "top-level"
  let mut out : Array Instance := #[]
  for ex in instArr do
    let exo ← expectObject ex "instance"
    let id ← expectFieldNat exo "id" "instance"
    let lo ← expectFieldFiniteFloatArray exo "input_lo" "instance"
    let hi ← expectFieldFiniteFloatArray exo "input_hi" "instance"
    let inputDim := lo.size
    let lo ← NN.Verification.Util.Tensor.requireVecOfArray "input_lo" inputDim lo
    let hi ← NN.Verification.Util.Tensor.requireVecOfArray "input_hi" inputDim hi
    if !NN.Verification.Util.Tensor.boundsOrdered lo hi then
      throw <| IO.userError s!"instance {id}: input box endpoints are mismatched or reversed"
    let specArr ← expectFieldArray exo "spec" "instance"
    let mut specOut : Spec := #[]
    for t in specArr do
      let termObj ← expectObject t "spec term"
      let matJ ← expectField termObj "mat" "spec term"
      let mat ← expectFiniteFloatMatrix matJ "spec term.mat"
      let rhs ← expectFieldFiniteFloatArray termObj "rhs" "spec term"
      let rows := mat.size
      let cols := (mat[0]?).map Array.size |>.getD 0
      let some mat := NN.Verification.Util.Tensor.matOfArray rows cols mat
        | throw <| IO.userError "spec term.mat: ragged coefficient matrix"
      let rhs ← NN.Verification.Util.Tensor.requireVecOfArray "spec term.rhs" rows rhs
      specOut := specOut.push { rows, cols, mat, rhs }
    out := out.push { id := id, input := { dim := inputDim, lo, hi }, spec := specOut }
  pure out

/--
Lower-bound one linear row over an output interval box.

For each coefficient $a_j$, the minimum of $a_jy_j$ over
$y_j\in[\mathrm{lo}_j,\mathrm{hi}_j]$ is the smaller of the endpoint products.
-/
def rowLowerBoundOnBox {n : Nat} (row yLo yHi : Tensor Float [n]) : Float :=
  let lowerProducts := Tensor.map2Spec BoundOps.mulDown row yLo
  let upperProducts := Tensor.map2Spec BoundOps.mulDown row yHi
  Tensor.foldl BoundOps.addDown 0.0 (Tensor.map2Spec min lowerProducts upperProducts)

/-- Check whether a conjunction term is refuted by an equally dimensioned output box. -/
def termRefutedByOutputBox (box : FlatBox Float) (term : Term) : Bool :=
  if h : box.dim = term.cols then
    let lo : Tensor Float [term.cols] := h ▸ box.lo
    let hi : Tensor Float [term.cols] := h ▸ box.hi
    NN.Verification.Util.Tensor.boundsOrdered lo hi &&
      (List.finRange term.rows).any (fun i =>
        rowLowerBoundOnBox (term.mat.unstack i) lo hi > term.rhs.getScalar i)
  else false

/--
Check whether an unsafe VNNLIB spec is refuted by an output interval box.

The spec is a disjunction of conjunctions. To prove the unsafe region is empty, every disjunct must
be refuted. For a conjunction, it is enough for one row lower bound to exceed its right-hand side.
This executable predicate uses the explicit host-`Float` `BoundOps` boundary; its Boolean result is
not itself a Lean theorem about real-valued graph semantics.
-/
def refutedByOutputBox (box : FlatBox Float) (spec : Spec) : Bool :=
  spec.all (termRefutedByOutputBox box)

end NN.Verification.VNNComp.VNNLib
