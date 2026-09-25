/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core
public import NN.Tests.Runtime.Floats.Utils

/-!
# Eager gradient flags

Exercise every CPU eager constructor with frozen and trainable inputs. Mixed graphs also check
the numerical gradients and the absence of cotangents on frozen intermediate nodes.
-/

@[expose] public section

namespace Tests.Floats.RequiresGrad

open Spec TorchLean Runtime.Autograd

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"requiresGrad: {label}"

/-- Check all input-flag combinations, including each parameter being trainable on its own. -/
def checkOperation (label : String) (inputs : Array (Spec.SomeTensor Float))
    (record : Tape Float → Array Nat → Result (Tape Float × Nat)) : IO Unit := do
  let mut referenceGrads : Array (Option (Spec.SomeTensor Float)) := #[]
  for mask in (List.range (2 ^ inputs.size)).reverse do
    let mut tape : Tape Float := Tape.empty
    let mut ids : Array Nat := #[]
    let mut flags : Array Bool := #[]
    for input in inputs do
      let trainable := mask / 2 ^ ids.size % 2 == 1
      let (next, id) := tape.leaf input.tensor (requiresGrad := trainable)
      tape := next
      ids := ids.push id
      flags := flags.push trainable
    let (result, outId) ← okOrThrow (record tape ids)
    let some output := result.getNode? outId
      | throw <| IO.userError s!"{label}: missing output"
    expect s!"{label}/{mask}/output flag" (output.requiresGrad == (mask != 0))
    expect s!"{label}/{mask}/appended one node"
      (result.size == tape.size + 1 && outId == tape.size)
    for i in [:ids.size] do
      expect s!"{label}/{mask}/leaf {i}"
        ((result.getNode? ids[i]!).any (fun node => node.requiresGrad == flags[i]!))
    if mask != 0 then
      let seed := Spec.SomeTensor.ofTensor (Tensor.full output.value.shape (1 : Float))
      let grads ← okOrThrow (result.backwardDense outId seed)
      if mask == 2 ^ inputs.size - 1 then
        referenceGrads := grads
      for i in [:ids.size] do
        expect s!"{label}/{mask}/gradient presence {i}"
          ((grads[ids[i]!]?).join.isSome == flags[i]!)
        if flags[i]! then
          let some actual := (grads[ids[i]!]?).join
            | throw <| IO.userError s!"{label}: missing trainable gradient"
          let some expected := (referenceGrads[ids[i]!]?).join
            | throw <| IO.userError s!"{label}: missing reference gradient"
          Utils.assertArrayApprox s!"{label}/{mask}/trainable gradient {i}"
            (actual.tensor.to (Array Float)) (expected.tensor.to (Array Float))

def checkElementwise : IO Unit := do
  let vector := Spec.SomeTensor.ofTensor ([1.0, 2.0] : Tensor Float [2])
  let unaryOps : Array (String × (Tape Float → Nat → Result (Tape Float × Nat))) :=
    #[("unary", fun t x => t.unary (σ := [2]) (τ := [2]) "identity" x
        (fun value => value) (fun _ grad => grad)),
      ("scale", fun t x => t.scale (s := [2]) x 2),
      ("abs", Tape.abs (s := [2])),
      ("sqrt", Tape.sqrt (s := [2])),
      ("clamp", fun t x => t.clamp (s := [2]) x 0 3),
      ("sin", Tape.sin (s := [2])),
      ("cos", Tape.cos (s := [2])),
      ("relu", Tape.relu (s := [2])),
      ("sigmoid", Tape.sigmoid (s := [2])),
      ("tanh", Tape.tanh (s := [2])),
      ("gelu", Tape.gelu (s := [2])),
      ("softmaxLast", Tape.softmaxLast (s := [2])),
      ("logSoftmaxLast", Tape.logSoftmaxLast (s := [2])),
      ("softplus", Tape.softplus (s := [2])),
      ("exp", Tape.exp (s := [2])),
      ("log", Tape.log (s := [2])),
      ("inv", Tape.inv (s := [2])),
      ("safeLog", fun t x => t.safeLog (s := [2]) x),
      ("sum", Tape.sum (s := [2]))]
  for (name, operation) in unaryOps do
    checkOperation name #[vector] fun t ids => operation t ids[0]!
  let otherVector := Spec.SomeTensor.ofTensor ([3.0, 1.0] : Tensor Float [2])
  let binaryOps : Array (String × (Tape Float → Nat → Nat → Result (Tape Float × Nat))) :=
    #[("add", Tape.add (s := [2])),
      ("sub", Tape.sub (s := [2])),
      ("mul", Tape.mul (s := [2])),
      ("div", Tape.div (s := [2])),
      ("max", Tape.max (s := [2])),
      ("min", Tape.min (s := [2])),
      ("mseLoss", Tape.mseLoss (s := [2]))]
  for (name, operation) in binaryOps do
    checkOperation name #[vector, otherVector] fun t ids => operation t ids[0]! ids[1]!

def checkShapeAndIndexing : IO Unit := do
  let vector := Spec.SomeTensor.ofTensor ([1.0, 2.0] : Tensor Float [2])
  let matrix := Spec.SomeTensor.ofTensor ([[1.0, 2.0], [3.0, 4.0]] : Tensor Float [2, 2])
  let indices : Tensor (Fin 2) [2] := Tensor.ofFn (fun _ => ⟨1, by decide⟩)
  checkOperation "flatten" #[matrix] fun t ids => t.flatten (s := [2, 2]) ids[0]!
  checkOperation "reshape" #[matrix] fun t ids =>
    t.reshape (s₁ := [2, 2]) (s₂ := [4]) ids[0]! (by decide)
  checkOperation "swapAdjacentAtDepth" #[matrix] fun t ids =>
    t.swapAdjacentAtDepth (s := [2, 2]) 0 ids[0]!
  checkOperation "broadcastTo" #[Spec.SomeTensor.ofTensor (Tensor.scalar (2 : Float))]
    fun t ids => t.broadcastTo (Shape.CanBroadcastTo.scalarTo [2, 2]) ids[0]!
  checkOperation "reduceSum" #[matrix] fun t ids =>
    t.reduceSum (s := [2, 2]) 1 ids[0]!
  checkOperation "reduceMean" #[matrix] fun t ids =>
    t.reduceMean (s := [2, 2]) 1 ids[0]!
  checkOperation "concatLeadingAxis" #[vector, vector] fun t ids =>
    t.concatLeadingAxis (n := 2) (m := 2) (s := .scalar) ids[0]! ids[1]!
  checkOperation "sliceLeadingAxisRange" #[vector] fun t ids =>
    t.sliceLeadingAxisRange (n := 2) (s := .scalar) ids[0]! 1 1 (by decide)
  checkOperation "select" #[matrix] fun t ids =>
    t.select (s := [2, 2]) ids[0]! 1 ⟨1, by decide⟩
  checkOperation "indexSelect" #[matrix] fun t ids =>
    t.indexSelect (s := [2, 2]) ids[0]! 1 2 indices
  checkOperation "scatterAdd" #[matrix, matrix] fun t ids =>
    t.scatterAdd (s := [2, 2]) ids[0]! ids[1]! 1 2 indices
  let empty := Spec.SomeTensor.ofTensor (Tensor.full [0] (0 : Float))
  checkOperation "empty sum" #[empty] fun t ids => t.sum (s := [0]) ids[0]!
  checkOperation "empty concat" #[empty, vector] fun t ids =>
    t.concatLeadingAxis (n := 0) (m := 2) (s := .scalar) ids[0]! ids[1]!

def checkLayers : IO Unit := do
  let vector := Spec.SomeTensor.ofTensor ([1.0, 2.0] : Tensor Float [2])
  let matrix := Spec.SomeTensor.ofTensor ([[1.0, 2.0], [3.0, 4.0]] : Tensor Float [2, 2])
  checkOperation "linear" #[matrix, vector, vector] fun t ids =>
    t.linear (inDim := 2) (outDim := 2) ids[0]! ids[1]! ids[2]!
  checkOperation "matmul" #[matrix, matrix] fun t ids =>
    t.matmul (m := 2) (n := 2) (p := 2) ids[0]! ids[1]!
  let batched := Spec.SomeTensor.ofTensor (Tensor.full [3, 2, 2] (1 : Float))
  checkOperation "broadcast matmul" #[matrix, batched] fun t ids =>
    t.matmul (m := 2) (n := 2) (p := 2) ids[0]! ids[1]!
      (batchB := [3]) (batch := [3])
  checkOperation "layerNorm" #[matrix, vector, vector] fun t ids =>
    t.layerNorm (seqLen := 2) (embedDim := 2) (by decide) (by decide)
      ids[0]! ids[1]! ids[2]!
  checkOperation "batchNorm" #[matrix, vector, vector] fun t ids =>
    t.batchNorm (channels := 2) (sSpatial := [2]) (by simp [Shape.wellFormed])
      ids[0]! ids[1]! ids[2]!
  checkOperation "multiHeadAttention" #[matrix, matrix, matrix, matrix, matrix] fun t ids =>
    t.multiHeadAttention (n := 2) (numHeads := 1) (dModel := 2) (headDim := 2) (by decide)
      ids[0]! ids[1]! ids[2]! ids[3]! ids[4]!

def checkConvolutionAndPooling : IO Unit := do
  let spatial : Tensor Nat [1] := [2]
  let kernel : Tensor Nat [1] := [1]
  let stride : Tensor Nat [1] := [1]
  let padding : Tensor Nat [1] := [0]
  let weights := Spec.SomeTensor.ofTensor ([[[2.0]]] : Tensor Float [1, 1, 1])
  let bias := Spec.SomeTensor.ofTensor ([1.0] : Tensor Float [1])
  let input := Spec.SomeTensor.ofTensor ([[1.0, 2.0]] : Tensor Float [1, 2])
  checkOperation "conv" #[weights, bias, input] fun t ids =>
    t.conv (d := 1) (inC := 1) (outC := 1) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := spatial) ids[0]! ids[1]! ids[2]!
  checkOperation "convTranspose" #[weights, bias, input] fun t ids =>
    t.convTranspose (d := 1) (inC := 1) (outC := 1) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := spatial) ids[0]! ids[1]! ids[2]!
  checkOperation "maxPool" #[input] fun t ids =>
    t.maxPool (d := 1) (C := 1) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := spatial) ids[0]!
  checkOperation "avgPool" #[input] fun t ids =>
    t.avgPool (d := 1) (C := 1) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := spatial) ids[0]!
  checkOperation "smoothMaxPool" #[input] fun t ids =>
    t.smoothMaxPool (d := 1) (C := 1) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := spatial) ids[0]! 1

/-- The same trainable value contributes twice, while a frozen unary/binary chain is skipped. -/
def checkMixedGraph : IO Unit := do
  let (t1, frozen) := (Tape.empty : Tape Float).leaf (Tensor.scalar (2 : Float))
    (requiresGrad := false)
  let (t2, scaled) ← okOrThrow (t1.scale (s := .scalar) frozen 3)
  let (t3, squareRoot) ← okOrThrow (t2.sqrt (s := .scalar) frozen)
  let (t4, constant) ← okOrThrow (t3.mul (s := .scalar) scaled squareRoot)
  let (t5, trainable) := t4.leaf (Tensor.scalar (5 : Float))
  let (t6, product) ← okOrThrow (t5.mul (s := .scalar) trainable constant)
  let (t7, total) ← okOrThrow (t6.add (s := .scalar) product product)
  let grads ← okOrThrow (t7.backwardDense total
    (Spec.SomeTensor.ofTensor (Tensor.scalar (1 : Float))))
  let sparse ← okOrThrow (t7.backwardScalar total)
  let dense ← okOrThrow (t7.backwardDenseAll total
    (Spec.SomeTensor.ofTensor (Tensor.scalar (1 : Float))))
  for id in #[frozen, scaled, squareRoot, constant] do
    expect s!"mixed/frozen flag {id}"
      ((t7.getNode? id).any (fun node => !node.requiresGrad))
    expect s!"mixed/no optional cotangent {id}" ((grads[id]?).join.isNone)
    expect s!"mixed/no sparse cotangent {id}" ((sparse.get? id).isNone)
    let some zero := dense[id]?
      | throw <| IO.userError "mixed: missing totalized gradient"
    let zero ← okOrThrow (Tape.requireGrad (τ := .scalar) zero)
    expect s!"mixed/totalized zero {id}" (zero.item == 0)
  let some gradient := (grads[trainable]?).join
    | throw <| IO.userError "mixed: missing trainable gradient"
  let gradient ← okOrThrow (Tape.requireGrad (τ := .scalar) gradient)
  Tests.Utils.assertApprox "mixed/trainable gradient" gradient.item (12 * Float.sqrt 2)
  let value ← okOrThrow (t7.requireValue (s := .scalar) total)
  Tests.Utils.assertApprox "mixed/forward" value.item (60 * Float.sqrt 2)

/-- A frozen matmul branch still contributes its value to a trainable parameter's gradient. -/
def checkMixedMatmul : IO Unit := do
  for trainLeft in #[false, true] do
    let (t1, left) := (Tape.empty : Tape Float).leaf
      ([[1.0, 2.0], [3.0, 4.0]] : Tensor Float [2, 2]) (requiresGrad := trainLeft)
    let (t2, right) := t1.leaf ([[5.0], [6.0]] : Tensor Float [2, 1])
      (requiresGrad := !trainLeft)
    let (t3, leftScaled) ← okOrThrow (t2.scale (s := [2, 2]) left 2)
    let (t4, rightScaled) ← okOrThrow (t3.scale (s := [2, 1]) right 3)
    let (t5, product) ← okOrThrow
      (t4.matmul (m := 2) (n := 2) (p := 1) leftScaled rightScaled)
    let (t6, total) ← okOrThrow (t5.sum (s := [2, 1]) product)
    let grads ← okOrThrow (t6.backwardDense total
      (Spec.SomeTensor.ofTensor (Tensor.scalar (1 : Float))))
    let (frozen, frozenIntermediate, trainable) :=
      if trainLeft then (right, rightScaled, left) else (left, leftScaled, right)
    expect "matmul/frozen leaf" ((grads[frozen]?).join.isNone)
    expect "matmul/frozen intermediate" ((grads[frozenIntermediate]?).join.isNone)
    let some gradient := (grads[trainable]?).join
      | throw <| IO.userError "matmul: missing trainable gradient"
    if trainLeft then
      let gradient ← okOrThrow (Tape.requireGrad (τ := [2, 2]) gradient)
      Utils.assertArrayApprox "matmul/left gradient" (gradient.to (Array Float)) #[30, 36, 30, 36]
    else
      let gradient ← okOrThrow (Tape.requireGrad (τ := [2, 1]) gradient)
      Utils.assertArrayApprox "matmul/right gradient" (gradient.to (Array Float)) #[24, 36]

/-- Run the CPU constructor and mixed-graph regressions. -/
def run : IO Unit := do
  checkElementwise
  checkShapeAndIndexing
  checkLayers
  checkConvolutionAndPooling
  checkMixedGraph
  checkMixedMatmul
  IO.println "CPU eager requiresGrad propagation: OK"

end Tests.Floats.RequiresGrad
