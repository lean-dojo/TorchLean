/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.IR.Check
public import NN.Runtime.Autograd.Compiled.IRExec

/-!
# IR axis operations

IR axis-ops smoke tutorial.

This file is a small “regression guard” for three ops where TorchLean’s IR uses an explicit `axis`:

- `softmax axis` (PyTorch: `torch.softmax(x, dim=axis)`)
- `concat axis` (PyTorch: `torch.cat(xs, dim=axis)`)
- `layernorm axis`
  PyTorch: `F.layer_norm(x, normalized_shape=x.shape[axis:])`

Why this tutorial exists:

* These three ops are easy to accidentally restrict to “last axis only” (because the spec primitives
  we reuse are last-axis).
* The denotational IR semantics intentionally supports the PyTorch meaning on *any* valid axis:
  it implements non-last axes by reshaping/permuting into a form the spec primitive already
  supports.
* The compiled IRExec backend is more conservative today. This tutorial runs compiled execution only for
  supported cases and prints an explicit skip for known backend gaps, instead of treating the
  backend covers more than it does.

Run:

`lake exe torchlean ir_axis_ops --dtype float --backend eager`
-/

@[expose] public section

namespace NN.Examples.Advanced.IRAxisOps

open NN.IR

open Runtime.Autograd.Compiled

/-!
## Test Shapes

We keep shapes small so this tutorial runs instantly, but still exercises the “axis is not last / not 0”
code paths.
-/

abbrev s234 : Spec.Shape := NN.Tensor.shapeOfDims [2, 3, 4]
abbrev s254 : Spec.Shape := NN.Tensor.shapeOfDims [2, 5, 4]
abbrev s284 : Spec.Shape := NN.Tensor.shapeOfDims [2, 8, 4]

/-!
## Small IR Graphs
-/

def gSoftmaxAxis1 : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := s234 }
    , { id := 1, parents := [0], kind := .softmax (axis := 1), outShape := s234 }
    ] }

def gLayerNormRank3Axis1 : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := s234 }
    , { id := 1, parents := [0], kind := .layernorm (axis := 1), outShape := s234 }
    ] }

def gConcatAxis1 : NN.IR.Graph :=
  -- The input node is intentionally unused: the concat example uses `rand_uniform` sources so we
  -- don’t have to carry an explicit constant payload in this tutorial.
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := Spec.Shape.scalar }
    , { id := 1, parents := [], kind := .randUniform (seed := 0), outShape := s234 }
    , { id := 2, parents := [], kind := .randUniform (seed := 1), outShape := s254 }
    , { id := 3, parents := [1, 2], kind := .concat (axis := 1), outShape := s284 }
    ] }

/-!
## Runner Helpers
-/

def checkIR (tag : String) (g : NN.IR.Graph) : IO Unit := do
  API.Common.orThrow s!"{tag}:checkWellFormed" <| g.checkWellFormed
  API.Common.orThrow s!"{tag}:checkShapes" <| NN.IR.Graph.checkShapes g

def firstScalars {α : Type} [ToString α] : List α → String
  | [] => "[]"
  | xs =>
      let ys := xs.take 8
      "[" ++ ", ".intercalate (ys.map toString) ++ (if xs.length > ys.length then ", ..." else "")
        ++ "]"

def runOne
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (tag : String)
    (g : NN.IR.Graph)
    (payload : NN.IR.Payload α)
    (inputShape : Spec.Shape)
    (x : Spec.Tensor α inputShape)
    (outputId : Fin g.nodes.size)
    (runCompiled : Bool := true) : IO Unit := do
  checkIR tag g

  -- Spec semantics (denotational model).
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inputShape x
  let outSpec ← API.Common.orThrow s!"{tag}:spec" <|
    NN.IR.Graph.denote (α := α) (g := g) (payload := payload) (input := input) (outputId :=
      outputId)
  let ⟨sSpec, tSpec⟩ := outSpec
  IO.println s!"[{tag}] spec outShape: {repr sSpec}"
  IO.println s!"[{tag}] spec first scalars: {firstScalars (Spec.toList tSpec)}"

  if !runCompiled then
    IO.println s!"[{tag}] compiled skipped: current IRExec backend supports fewer axis cases than the spec semantics."
    return ()

  -- Compiled bridge (IR → executable SSA) + execution.
  let eg ← API.Common.orThrow s!"{tag}:compiled" <|
    Runtime.Autograd.Compiled.execGraphOfIR (α := α) (g := g) (payload := payload)
  let xExec : Spec.Tensor α eg.inShape ←
    if hIn : inputShape = eg.inShape then
      pure <| Spec.Tensor.castShape (t := x) hIn
    else
      -- This should never happen for well-formed tutorials, but keep the error readable if someone
      -- edits the graph and forgets to update the `inputShape` argument.
      throw <| IO.userError
        s!"{tag}: inputShape mismatch: arg={repr inputShape}, graph={repr eg.inShape}"
  let valsExec := Runtime.Autograd.Compiled.ExecGraphData.denoteAll (α := α) eg xExec
  let outExec ←
    if hOut : outputId.1 < valsExec.size then
      pure <| valsExec[outputId.1]'hOut
    else
      throw <| IO.userError
        s!"{tag}: compiled output index out of bounds: index={outputId.1}, size={valsExec.size}"
  let ⟨sExec, tExec⟩ := outExec
  IO.println s!"[{tag}] compiled outShape: {repr sExec}"
  IO.println s!"[{tag}] compiled first scalars: {firstScalars (Spec.toList tExec)}"

  if sExec = sSpec then
    pure ()
  else
    throw <| IO.userError
      s!"{tag}: spec/compiled outShape mismatch: spec={repr sSpec}, compiled={repr sExec}"

def runOnce
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (cast : Float → α) : IO Unit := do
  let payload : NN.IR.Payload α := {}

  -- A small but nontrivial 2×3×4 input tensor.
  let x234 : Spec.Tensor α s234 ←
    API.Common.orThrow "ir_axis_ops:input234" <|
      API.Common.tensorFGen (α := α) cast [2, 3, 4] (fun i =>
        (Float.ofNat i) / 10.0 - 1.0)

  IO.println ""
  IO.println "== IR axis ops tutorial =="
  IO.println "This checks shape validation + runs both the spec semantics and compiled IRExec path."

  IO.println ""
  IO.println "-- softmax axis=1 on shape [2,3,4]"
  runOne (α := α) (tag := "softmax_axis1") (g := gSoftmaxAxis1) (payload := payload)
    (inputShape := s234) (x := x234) (outputId := ⟨1, by decide⟩) (runCompiled := false)

  IO.println ""
  IO.println "-- layernorm axis=1 on rank-3 shape [2,3,4]"
  IO.println "PyTorch meaning: normalized_shape = x.shape[axis:] = [3,4]"
  runOne (α := α) (tag := "layernorm_rank3_axis1") (g := gLayerNormRank3Axis1) (payload := payload)
    (inputShape := s234) (x := x234) (outputId := ⟨1, by decide⟩) (runCompiled := false)

  IO.println ""
  IO.println "-- concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]"
  let x0 : Spec.Tensor α Spec.Shape.scalar := Spec.Tensor.scalar (cast 0.0)
  runOne (α := α) (tag := "concat_axis1") (g := gConcatAxis1) (payload := payload)
    (inputShape := Spec.Shape.scalar) (x := x0) (outputId := ⟨3, by decide⟩)

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args
  _root_.NN.API.TorchLean.Module.withRuntime args (fun {α} _ _ _ _ cast _opts rest => do
    API.Common.orThrow "ir_axis_ops" <| API.CLI.requireNoArgs rest
    runOnce (α := α) cast)

end NN.Examples.Advanced.IRAxisOps
