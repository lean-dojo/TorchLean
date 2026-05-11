/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.IR.Semantics
public import NN.MLTheory.CROWN.Graph

/-!
# One semantic universe

End-to-end “one semantic universe” tutorial (single graph, many semantics, one checker).

This tutorial is intentionally **not** part of the beginner training tutorials; it lives under
`NN/Examples/Advanced`.

We build one medium IR graph:

`x ↦ tanh(sum(Linear2(ReLU(Linear1(x)))))`

and then:
1) evaluate it under multiple scalar semantics (`ℝ`, `FP32`, `IEEE32Exec`);
2) run IBP under multiple interval semantics (endpoints in `ℝ`, `FP32`, `IEEE32Exec` with directed
  rounding);
3) empirically sanity-check that `evalIEEE(G,x)` lies in the IBP output box for random `x ∈ B`;
4) point to the Lean theorem that the Boolean checker is sound (`Box.containsDecBool_sound`).

Notes:
- `ℝ` and `FP32` instantiations are proof-oriented and noncomputable (they typecheck, but do not run
  as an executable).
- `IEEE32Exec` is fully executable inside Lean, so we use it for the runnable sanity check.

Run:
  `lake exe torchlean one_semantic_universe --samples 50`
-/

@[expose] public section


namespace NN.Examples.Advanced.OneSemanticUniverse

open Spec
open Tensor
open NN.Tensor
open NN.API

open NN.IR
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

open TorchLean.Floats

def inDim : Nat := 4
def hidDim : Nat := 5
def outDim : Nat := 3

def xShape : Spec.Shape := NN.Tensor.Shape.Vec inDim
def hShape : Spec.Shape := NN.Tensor.Shape.Vec hidDim
def yShape : Spec.Shape := NN.Tensor.Shape.Vec outDim

def W1Shape : Spec.Shape := NN.Tensor.Shape.Mat hidDim inDim
def b1Shape : Spec.Shape := NN.Tensor.Shape.Vec hidDim
def W2Shape : Spec.Shape := NN.Tensor.Shape.Mat outDim hidDim
def b2Shape : Spec.Shape := NN.Tensor.Shape.Vec outDim

structure Params (α : Type) where
  /-- Weight matrix for layer 1. -/
  W1 : Spec.Tensor α W1Shape
  /-- Bias for layer 1. -/
  b1 : Spec.Tensor α b1Shape
  /-- Weight matrix for layer 2. -/
  W2 : Spec.Tensor α W2Shape
  /-- Bias for layer 2. -/
  b2 : Spec.Tensor α b2Shape

def Params.map {α β : Type} (f : α → β) (p : Params α) : Params β :=
  { W1 := Spec.mapTensor f p.W1
    b1 := Spec.mapTensor f p.b1
    W2 := Spec.mapTensor f p.W2
    b2 := Spec.mapTensor f p.b2 }

def paramsFloat : Params Float :=
  { W1 := tensorND! (ty := Float) [hidDim, inDim]
      [ 0.15, -0.12, 0.08, 0.05
      , 0.02, 0.11, -0.09, 0.07
      , -0.04, 0.06, 0.10, -0.03
      , 0.09, 0.01, 0.04, 0.13
      , -0.07, 0.03, 0.12, -0.02 ]
    b1 := tensorND! (ty := Float) [hidDim] [0.01, -0.02, 0.03, 0.0, 0.02]
    W2 := tensorND! (ty := Float) [outDim, hidDim]
      [ 0.05, 0.08, -0.06, 0.03, 0.07
      , -0.04, 0.02, 0.09, -0.01, 0.06
      , 0.10, -0.03, 0.04, 0.05, -0.08 ]
    b2 := tensorND! (ty := Float) [outDim] [0.02, -0.01, 0.00] }

def g : NN.IR.Graph :=
  let n0 : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := xShape }
  let n1 : NN.IR.Node := { id := 1, parents := [0], kind := .linear, outShape := hShape }
  let n2 : NN.IR.Node := { id := 2, parents := [1], kind := .relu, outShape := hShape }
  let n3 : NN.IR.Node := { id := 3, parents := [2], kind := .linear, outShape := yShape }
  let n4 : NN.IR.Node := { id := 4, parents := [3], kind := .sum, outShape := Spec.Shape.scalar }
  let n5 : NN.IR.Node := { id := 5, parents := [4], kind := .tanh, outShape := Spec.Shape.scalar }
  { nodes := #[n0, n1, n2, n3, n4, n5] }

def mkPayload {α : Type} [Context α] (p : Params α) : NN.IR.Payload α :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := hidDim, inDim := inDim, W := p.W1, b := p.b1 }
      else if id = 3 then
        some { outDim := outDim, inDim := hidDim, W := p.W2, b := p.b2 }
      else
        none }

def mkParamStore {α : Type} [Context α] (p : Params α) (xB : FlatBox α) : ParamStore α :=
  { inputBoxes := (Std.HashMap.emptyWithCapacity).insert 0 xB
    linearWB :=
      (Std.HashMap.emptyWithCapacity)
        |>.insert 1 { m := hidDim, n := inDim, w := p.W1, b := p.b1 }
        |>.insert 3 { m := outDim, n := hidDim, w := p.W2, b := p.b2 } }

def evalOut
    {α : Type} [Context α] [Inhabited α] [DecidableEq Spec.Shape]
    (p : Params α) (x : Spec.Tensor α xShape) : Except String (Spec.Tensor α Spec.Shape.scalar) :=
      do
  let payload := mkPayload (α := α) p
  let input : DVal α := DVal.mk (α := α) xShape x
  let v ← NN.IR.Graph.denote (α := α) (g := g) (payload := payload) (input := input) (outputId := 5)
  NN.IR.Graph.expectShape (α := α) (expected := Spec.Shape.scalar) v

/-!
### Proof-only instantiations (typechecks)

These `have` lines ensure we can interpret **the same graph** under:
- `ℝ` (reference semantics),
- `FP32` (proof-oriented rounding-on-ℝ model),
- plus IBP over those endpoint types.

They live in a propositional `example`, so they do not affect the executable tutorial.
-/
section ProofOnly

noncomputable example : True := by
  have _ :
      ∀ (p : Params ℝ) (x : Spec.Tensor ℝ xShape),
        Except String (Spec.Tensor ℝ Spec.Shape.scalar) :=
    fun p x => evalOut (α := ℝ) p x
  have _ :
      ∀ (p : Params TorchLean.Floats.FP32) (x : Spec.Tensor TorchLean.Floats.FP32 xShape),
        Except String (Spec.Tensor TorchLean.Floats.FP32 Spec.Shape.scalar) :=
    fun p x => evalOut (α := TorchLean.Floats.FP32) p x
  have _ :
      ∀ (ps : ParamStore ℝ), Array (Option (FlatBox ℝ)) :=
    fun ps => runIBP (α := ℝ) g ps
  have _ :
      ∀ (ps : ParamStore TorchLean.Floats.FP32),
        Array (Option (FlatBox TorchLean.Floats.FP32)) :=
    fun ps => runIBP (α := TorchLean.Floats.FP32) g ps
  trivial

end ProofOnly

def x0Float : Spec.Tensor Float xShape :=
  tensorND! (ty := Float) [inDim] [0.3, -0.2, 0.1, 0.4]

def xBoxOf (α : Type) [Semantics.Scalar α] [Runtime.Scalar α] (eps : Float) : Box α xShape :=
  let x0 : Spec.Tensor α xShape := Common.castTensor Runtime.ofFloat x0Float
  let r : α := Runtime.ofFloat eps
  let rad : Spec.Tensor α xShape := Spec.fill (α := α) r xShape
  { lo := Spec.Tensor.subSpec (α := α) x0 rad
    hi := Spec.Tensor.addSpec (α := α) x0 rad }

def toFlatXBox {α : Type} [Context α] (B : Box α xShape) : FlatBox α :=
  { dim := inDim, lo := B.lo, hi := B.hi }

def scalarBoxOfFlat (B : FlatBox IEEE32Exec) : Except String (Box IEEE32Exec Spec.Shape.scalar) :=
  do
  if h : B.dim = 1 then
    let loT : Spec.Tensor IEEE32Exec (Shape.Vec 1) :=
      Spec.Tensor.castVecDim (α := IEEE32Exec) (n := B.dim) (m := 1) h B.lo
    let hiT : Spec.Tensor IEEE32Exec (Shape.Vec 1) :=
      Spec.Tensor.castVecDim (α := IEEE32Exec) (n := B.dim) (m := 1) h B.hi
    let l : IEEE32Exec := Spec.Tensor.vecGet (α := IEEE32Exec) loT fin0!
    let u : IEEE32Exec := Spec.Tensor.vecGet (α := IEEE32Exec) hiT fin0!
    pure { lo := Spec.Tensor.scalar l, hi := Spec.Tensor.scalar u }
  else
    throw s!"expected a scalar FlatBox (dim=1), got dim={B.dim}"

def sampleInBoxIEEE (seed idx : Nat) (B : Box IEEE32Exec xShape) : Spec.Tensor IEEE32Exec xShape :=
  let key := rand.keyOf seed idx
  let u : Spec.Tensor IEEE32Exec xShape := rand.uniform (α := IEEE32Exec) key (s := xShape)
  let w := Spec.Tensor.subSpec (α := IEEE32Exec) B.hi B.lo
  let xRaw := Spec.Tensor.addSpec (α := IEEE32Exec) B.lo (Spec.Tensor.mulSpec (α := IEEE32Exec) u
    w)
  -- Clamp to be sure we land inside `[lo,hi]` despite rounding.
  let xLo := Spec.Tensor.maxSpec (α := IEEE32Exec) xRaw B.lo
  Spec.Tensor.minSpec (α := IEEE32Exec) xLo B.hi

def showIEEECheck (samples : Nat) : IO Unit := do
  IO.println "== One semantic universe tutorial =="
  IO.println s!"graph nodes = {g.nodes.size}"

  let pIEEE : Params IEEE32Exec := paramsFloat.map Runtime.ofFloat
  let BIEEE : Box IEEE32Exec xShape := xBoxOf (α := IEEE32Exec) (eps := 0.05)
  let xBIEEE : FlatBox IEEE32Exec := toFlatXBox (α := IEEE32Exec) BIEEE

  -- Evaluate at the center point.
  let x0IEEE : Spec.Tensor IEEE32Exec xShape := Common.castTensor Runtime.ofFloat x0Float
  match evalOut (α := IEEE32Exec) pIEEE x0IEEE with
  | .error msg => throw <| IO.userError msg
  | .ok y0 =>
      IO.println s!"[eval IEEE32Exec] y(x0) = {Spec.pretty y0}"

  -- Compute IBP box (IEEE endpoints with directed rounding via `BoundOps IEEE32Exec`).
  let ps := mkParamStore (α := IEEE32Exec) pIEEE xBIEEE
  let ibp := runIBP (α := IEEE32Exec) g ps
  let some outFlat := ibp[5]! | throw <| IO.userError "IBP produced no output box at node 5"
  let outBox ←
    match scalarBoxOfFlat outFlat with
    | .error msg => throw <| IO.userError msg
    | .ok b => pure b
  IO.println s!"[IBP IEEE endpoints] lo = {Spec.pretty outBox.lo}"
  IO.println s!"[IBP IEEE endpoints] hi = {Spec.pretty outBox.hi}"

  -- Empirical sanity: random x ∈ B, check eval(x) ∈ IBP(B).
  let mut okCount : Nat := 0
  for k in [0:samples] do
    let x := sampleInBoxIEEE (seed := 12345) (idx := k) BIEEE
    let inOk := Box.containsDecBool (α := IEEE32Exec) (s := xShape) BIEEE x
    if inOk != true then
      throw <| IO.userError s!"internal error: sampled x not in box (k={k})"
    match evalOut (α := IEEE32Exec) pIEEE x with
    | .error msg => throw <| IO.userError msg
    | .ok y =>
        let outOk := Box.containsDecBool (α := IEEE32Exec) (s := Spec.Shape.scalar) outBox y
        if outOk then
          okCount := okCount + 1
        else
          IO.println s!"[counterexample?] k={k}"
          IO.println s!"x = {Spec.pretty x}"
          IO.println s!"y = {Spec.pretty y}"
  IO.println s!"sanity: {okCount}/{samples} samples satisfied evalIEEE(x) ∈ IBP(B)"
  IO.println "checker theorem: `NN.MLTheory.CROWN.Box.containsDecBool_sound`"

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let (samples?, rest) ← Common.orThrow "OneSemanticUniverse" <| CLI.takeNatFlagOnce args
    "samples"
  Common.orThrow "OneSemanticUniverse" <| CLI.requireNoArgs rest
  showIEEECheck (samples := samples?.getD 50)

end NN.Examples.Advanced.OneSemanticUniverse
