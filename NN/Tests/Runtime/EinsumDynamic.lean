/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.EinsumDynamic
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Runtime.Autograd.Torch.Core.Trainer.GraphOps

/-!
Dynamic einsum regressions: independent coordinate sums and seeded pullbacks, normalized matrix
dispatch, and graph-size and retained CUDA allocation bounds for the large ellipsis contraction.
-/

public section

namespace NN.Tests.Runtime.EinsumDynamic

open Spec TorchLean Runtime.Autograd

private def check (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"dynamic einsum: {label}"

private def close (label : String) (actual expected : Array Float) (tolerance : Float) :
    IO Unit := do
  check s!"{label}: length" (actual.size == expected.size)
  for i in [:actual.size] do
    check s!"{label}[{i}]: {actual[i]!} vs {expected[i]!}"
      (actual[i]!.isFinite &&
        Float.abs (actual[i]! - expected[i]!) ≤ tolerance * (1 + Float.abs expected[i]!))

/-- Enumerate label coordinates directly, including diagonal and singleton-broadcast indices. -/
private def reference {sA sB sOut : Shape} (explicit : String)
    (a : Tensor Float sA) (b : Tensor Float sB) (seed : Tensor Float sOut) :
    IO (Array Float × Array Float × Array Float) := do
  let .ok parsed := Model.F.Einsum.parseEquation explicit
    | throw <| IO.userError "invalid reference equation"
  let [subA, subB] := parsed.inputs
    | throw <| IO.userError "reference requires two inputs"
  let some subOut := parsed.output?
    | throw <| IO.userError "reference requires explicit output"
  check "reference has no ellipsis" (!(subA.hasEll || subB.hasEll || subOut.hasEll))
  let labelsA := subA.pre
  let labelsB := subB.pre
  let labelsOut := subOut.pre
  let axes := (labelsA ++ labelsB).eraseDups
  let occurrences := labelsA.zip sA.toList ++ labelsB.zip sB.toList
  let dimensions := axes.map fun label =>
    occurrences.foldl (fun dimension pair =>
      if pair.1 == label && dimension == 1 then pair.2 else dimension) 1
  let av := a.to (Array Float)
  let bv := b.to (Array Float)
  let sv := seed.to (Array Float)
  let mut output := Array.replicate sOut.size (0.0 : Float)
  let mut gradA := Array.replicate sA.size (0.0 : Float)
  let mut gradB := Array.replicate sB.size (0.0 : Float)
  for flat in [:dimensions.prod] do
    let coordinate (label : Char) : Nat :=
      let index := axes.findIdx (· == label)
      flat / (dimensions.drop (index + 1)).prod % dimensions[index]!
    let offset (labels : List Char) (shape : Shape) : Nat :=
      (labels.zip shape.toList).foldl (fun index pair =>
        index * pair.2 + if pair.2 == 1 then 0 else coordinate pair.1) 0
    let ia := offset labelsA sA
    let ib := offset labelsB sB
    let io := offset labelsOut sOut
    output := output.modify io (· + av[ia]! * bv[ib]!)
    gradA := gradA.modify ia (· + sv[io]! * bv[ib]!)
    gradB := gradB.modify ib (· + sv[io]! * av[ia]!)
  pure (output, gradA, gradB)

private def evaluate {sA sB sOut : Shape} (device : NN.Backend.Device)
    (equation : String) (a : Tensor Float sA) (b : Tensor Float sB)
    (seed : Tensor Float sOut) (expectFast : Bool) (maxElements : Nat) :
    IO (Array Float × Array Float × Array Float) := do
  let session ← Torch.Internal.EagerSession.new (α := Float) { device, execution := .eager }
  try
    let ar ← session.input a (requiresGrad := true)
    let br ← session.input b (requiresGrad := true)
    let some ⟨shape, output⟩ ←
      Model.F.einsum? (α := Float) (m := Torch.Internal.EagerM Float)
        equation [⟨sA, ar⟩, ⟨sB, br⟩] session
      | throw <| IO.userError s!"rejected valid equation: {equation}"
    if h : shape = sOut then
      let output : Torch.TensorRef Float sOut := h ▸ output
      let nodes : Array (Option String × Shape) ← if device == NN.Backend.Device.cuda then
          pure <| (← session.cudaTape.get).nodes.map fun node => (node.name, node.value.s)
        else
          pure <| (← session.tape.get).nodes.map fun node => (node.name, node.value.shape)
      let matmuls := nodes.filter fun node =>
        node.1 == some "matmul" || node.1 == some "bmm"
      check s!"{equation}: matrix dispatch" (matmuls.size == if expectFast then 1 else 0)
      if expectFast then
        check s!"{equation}: no elementwise contraction product"
          (nodes.all fun node => node.1 != some "mul")
      if maxElements > 0 then
        check s!"{equation}: largest forward tensor"
          (nodes.all fun node => node.2.size ≤ maxElements)
      let value ← session.getValue output
      let gradients ← session.backwardDenseAll output seed
      let ga ← Torch.Internal.EagerSession.grad gradients ar
      let gb ← Torch.Internal.EagerSession.grad gradients br
      pure (value.to (Array Float), ga.to (Array Float), gb.to (Array Float))
    else
      throw <| IO.userError s!"{equation}: expected {sOut}, got {shape}"
  finally session.resetTape

private def checkCase (device : NN.Backend.Device) (equation explicit : String)
    (sA sB sOut : Shape) (expectFast : Bool := true) (maxElements : Nat := 0) : IO Unit := do
  let a : Tensor Float sA := Tensor.generateFlat _ fun i => (i % 11).toFloat / 7 - 0.6
  let b : Tensor Float sB := Tensor.generateFlat _ fun i => (i % 7).toFloat / 5 - 0.4
  let seed : Tensor Float sOut := Tensor.generateFlat _ fun i => (i % 5).toFloat / 3 - 0.5
  let expected ← reference explicit a b seed
  let tolerance := if device == .cuda then 2e-5 else 1e-10
  for expression in [equation, explicit] do
    let actual ← evaluate device expression a b seed expectFast maxElements
    close s!"{expression}: values" actual.1 expected.1 tolerance
    close s!"{expression}: left VJP" actual.2.1 expected.2.1 tolerance
    close s!"{expression}: right VJP" actual.2.2 expected.2.2 tolerance

/-- Record the reported large shape without allocating either inputs or intermediate tensors. -/
private def checkLargeGraph : IO Unit := do
  let aShape : Shape := [8, 16, 128, 64]
  let bShape : Shape := [8, 16, 64, 128]
  let outShape : Shape := [8, 16, 128, 128]
  for equation in ["...ij,...jk->...ik", "abij,abjk->abik"] do
    let program : TypedGraph.GraphM.M Float [aShape, bShape]
        (Option (Σ s : Shape, TypedGraph.GraphM.Var s)) := do
      let a ← TypedGraph.GraphM.arg 0 aShape
      let b ← TypedGraph.GraphM.arg 1 bShape
      Model.F.einsum? (α := Float) (m := TypedGraph.GraphM.M Float [aShape, bShape])
        equation [⟨aShape, a⟩, ⟨bShape, b⟩]
    let .ok (some output, graph) := TypedGraph.GraphM.run program
      | throw <| IO.userError s!"{equation}: failed to record large contraction"
    check "large contraction output shape" (output.fst == outShape)
    -- The sole node is the matrix result: 8 MiB in binary32, versus a 512 MiB product.
    check "large contraction has only its output node" (graph.nodeShapes == [outShape])
    check "large contraction forward tensor bound"
      (graph.nodeShapes.all fun shape => shape.size * 4 ≤ 8 * 1024 * 1024)

/-- Measure the large contraction while the eager tape retains all forward intermediates. -/
private def checkLargeCuda : IO Unit := do
  Runtime.Autograd.Cuda.Buffer.requireNativeRuntime
  let aShape : Shape := [8, 16, 128, 64]
  let bShape : Shape := [8, 16, 64, 128]
  let outShape : Shape := [8, 16, 128, 128]
  let session ← Torch.Internal.EagerSession.new (α := Float)
    { device := .cuda, execution := .eager }
  try
    let a ← session.input (Tensor.ones (α := Float) aShape) (requiresGrad := true)
    let b ← session.input (Tensor.ones (α := Float) bShape) (requiresGrad := true)
    Runtime.Autograd.Cuda.LibTorch.synchronize
    let before ← Runtime.Autograd.Cuda.Buffer.allocatorStats
    let some output ← Model.F.einsum (α := Float) (m := Torch.Internal.EagerM Float)
      (sOut := outShape) "...ij,...jk->...ik" [⟨aShape, a⟩, ⟨bShape, b⟩] session
      | throw <| IO.userError "large CUDA contraction was rejected"
    Runtime.Autograd.Cuda.LibTorch.synchronize
    let after ← Runtime.Autograd.Cuda.Buffer.allocatorStats
    let growth := after.allocatedBytes.toNat - before.allocatedBytes.toNat
    IO.println s!"einsum CUDA large before: allocated={before.allocatedBytes} \
      reserved={before.reservedBytes} peak_allocated={before.peakAllocatedBytes} \
      peak_reserved={before.peakReservedBytes}"
    IO.println s!"einsum CUDA large forward: allocated={after.allocatedBytes} \
      reserved={after.reservedBytes} peak_allocated={after.peakAllocatedBytes} \
      peak_reserved={after.peakReservedBytes} allocated_growth={growth} \
      logical_output_bytes={outShape.size * 4}"
    check "large CUDA retained allocation increase below 128 MiB" (growth < 128 * 1024 * 1024)
    let value ← session.getValue output
    let values := value.to (Array Float)
    check "large CUDA output length" (values.size == outShape.size)
    check "large CUDA all-one contraction values" (values.all fun x => x == 64.0)
    check "large CUDA tape remains live" (!(← session.cudaTape.get).nodes.isEmpty)
  finally session.resetTape

private def checkRejected (device : NN.Backend.Device) : IO Unit := do
  for (equation, shapes) in
      [("...ij,...jk->...ik->ik", [[2, 3], [3, 4]]),
       ("...ij...,...jk->...ik", [[2, 3], [3, 4]]),
       ("ij,jk->ik", [[2, 3]]),
       ("ij,jk->ik", [[2, 3], [3, 4], [4]]),
       ("ij,jk->ik", [[1, 2, 3], [3, 4]]),
       ("...ij,...jk->...ik", [[3], [3, 4]]),
       ("...ij,...jk->...ik", [[2, 2, 3], [4, 3, 4]]),
       ("...ij,...jk->...ik", [[2, 3], [5, 4]]),
       ("...ij,...jk->...ix", [[2, 3], [3, 4]]),
       ("ii,ij->j", [[1, 2], [2, 3]]),
       ("ii->i", [[0, 0]]),
       ("i,i->", [[0], [0]])] do
    let session ← Torch.Internal.EagerSession.new (α := Float) { device, execution := .eager }
    try
      let mut inputs : List (Σ s : Shape, Torch.TensorRef Float s) := []
      for shape in shapes do
        let input ← session.input (Tensor.ones (α := Float) shape)
        inputs := inputs ++ [⟨shape, input⟩]
      let result ← Model.F.einsum? (α := Float) (m := Torch.Internal.EagerM Float)
        equation inputs session
      check s!"rejected equation {equation} / {shapes}" result.isNone
    finally session.resetTape

/-- Three operands must still use the general contraction and accumulate every leaf's VJP. -/
private def checkThreeInputs (device : NN.Backend.Device) : IO Unit := do
  let session ← Torch.Internal.EagerSession.new (α := Float) { device, execution := .eager }
  try
    let a ← session.input ([2.0, 3.0] : Tensor Float [2]) (requiresGrad := true)
    let b ← session.input ([5.0, 7.0] : Tensor Float [2]) (requiresGrad := true)
    let c ← session.input ([11.0, 13.0] : Tensor Float [2]) (requiresGrad := true)
    let some output ← Model.F.einsum (α := Float) (m := Torch.Internal.EagerM Float) (sOut := [])
      "i,i,i->" [⟨[2], a⟩, ⟨[2], b⟩, ⟨[2], c⟩] session
      | throw <| IO.userError "three-input contraction was rejected"
    close "three-input values" ((← session.getValue output).to (Array Float)) #[383] 1e-6
    let gradients ← session.backwardDenseAll output (Tensor.scalar 2.0)
    close "three-input first VJP"
      ((← Torch.Internal.EagerSession.grad gradients a).to (Array Float)) #[110, 182] 1e-6
    close "three-input second VJP"
      ((← Torch.Internal.EagerSession.grad gradients b).to (Array Float)) #[44, 78] 1e-6
    close "three-input third VJP"
      ((← Torch.Internal.EagerSession.grad gradients c).to (Array Float)) #[20, 42] 1e-6
  finally session.resetTape

/-- Run on CPU by default; `.cuda` exercises the same values, VJPs, and dispatch on LibTorch. -/
def run (device : NN.Backend.Device := .cpu) : IO Unit := do
  checkLargeGraph
  if device == .cuda then checkLargeCuda
  for (batch, subscript) in
      [([], ""), ([2], "a"), ([2, 3], "ab"), ([2, 1, 2], "abc"),
       ([1, 2, 1, 2, 1], "abcde")] do
    let explicit := s!"{subscript}ij,{subscript}jk->{subscript}ik"
    checkCase device "...ij,...jk->...ik" explicit
      (batch ++ [2, 3]) (batch ++ [3, 4]) (batch ++ [2, 4])
  checkCase device "...ij,...jk" "abij,abjk->abik"
    [2, 3, 2, 3] [2, 3, 3, 4] [2, 3, 2, 4]
  checkCase device "...zy,...yx" "azy,ayx->axz" [2, 3, 4] [2, 4, 5] [2, 5, 3]
  checkCase device " ...ij , ...jk -> ...ik " "abij,abjk->abik"
    [2, 3, 5, 7] [2, 3, 7, 4] [2, 3, 5, 4] (maxElements := 210)
  -- Right-aligned missing axes, singleton axes on both sides, and independent batch labels.
  checkCase device "...ij,...jk->...ik" "abij,bjk->abik"
    [2, 3, 2, 3] [3, 3, 4] [2, 3, 2, 4]
  checkCase device "...ij,...jk->...ik" "bij,abjk->abik"
    [3, 2, 3] [2, 3, 3, 4] [2, 3, 2, 4]
  checkCase device "...ij,jk->...ik" "abij,jk->abik"
    [2, 3, 2, 3] [3, 4] [2, 3, 2, 4]
  checkCase device "ij,...jk->...ik" "ij,abjk->abik"
    [2, 3] [2, 3, 3, 4] [2, 3, 2, 4]
  checkCase device "...ij,...jk->...ik" "abij,abjk->abik"
    [2, 1, 2, 3] [1, 3, 3, 4] [2, 3, 2, 4]
  checkCase device "aij,bjk->abik" "aij,bjk->abik"
    [2, 2, 3] [3, 3, 4] [2, 3, 2, 4]
  -- Transposes on either operand, arbitrary batch order, and output permutation.
  checkCase device "...ji,...jk->...ik" "abji,abjk->abik"
    [2, 3, 3, 2] [2, 3, 3, 4] [2, 3, 2, 4]
  checkCase device "...ij,...kj->...ik" "abij,abkj->abik"
    [2, 3, 2, 3] [2, 3, 4, 3] [2, 3, 2, 4]
  checkCase device "...ji,...kj->...ki" "abji,abkj->abki"
    [2, 3, 3, 2] [2, 3, 4, 3] [2, 3, 4, 2]
  checkCase device "i...j,j...k->k...i" "iabj,jabk->kabi"
    [2, 2, 3, 3] [3, 2, 3, 4] [4, 2, 3, 2]
  checkCase device "abij,bajk->baik" "abij,bajk->baik"
    [2, 3, 2, 3] [3, 2, 3, 4] [3, 2, 2, 4]
  -- Empty batches, rows, columns, and contraction axes use matmul's empty-sum semantics.
  for (sA, sB, sOut) in
      [([0, 2, 3], [1, 3, 4], [0, 2, 4]),
       ([2, 0, 3], [2, 3, 4], [2, 0, 4]),
       ([2, 2, 3], [2, 3, 0], [2, 2, 0]),
       ([2, 2, 0], [2, 0, 4], [2, 2, 4])] do
    checkCase device "...ij,...jk->...ik" "aij,ajk->aik" sA sB sOut
  -- Singleton contraction broadcasting, implicit batch reduction, and repeated labels.
  checkCase device "...ij,...jk->...ik" "aij,ajk->aik"
    [2, 2, 1] [2, 3, 4] [2, 2, 4] (expectFast := false)
  checkCase device "aij,ajk" "aij,ajk->ik"
    [2, 2, 3] [2, 3, 4] [2, 4] (expectFast := false)
  checkCase device "...ij,...jk->ik" "aij,ajk->ik"
    [2, 2, 3] [2, 3, 4] [2, 4] (expectFast := false)
  checkCase device "ii,ij->j" "ii,ij->j" [2, 2] [2, 3] [3] (expectFast := false)
  checkCase device "ii,j" "ii,j->j" [2, 2] [3] [3] (expectFast := false)
  checkCase device "ij,jk->iik" "ij,jk->iik" [2, 3] [3, 4] [2, 2, 4]
    (expectFast := false)
  checkCase device "ij,ij->ij" "ij,ij->ij" [2, 3] [2, 3] [2, 3] (expectFast := false)
  checkCase device "i,i->" "i,i->" [3] [3] [] (expectFast := false)
  checkThreeInputs device
  checkRejected device
  IO.println s!"dynamic einsum values, VJPs, and allocation bounds ({reprStr device}): passed"

end NN.Tests.Runtime.EinsumDynamic
