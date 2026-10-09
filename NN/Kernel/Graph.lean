/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.IR.Infer
public import NN.IR.Payload
public import NN.Runtime.Autograd.Engine.LibTorch.Tape
import NN.Kernel.Runtime
import NN.Runtime.Autograd.Engine.LibTorch.Convert
import NN.Runtime.Autograd.Engine.LibTorch.ConvPool
import NN.Runtime.Autograd.Engine.LibTorch.Kernels
import NN.Runtime.Autograd.Engine.LibTorch.Shape
import NN.Runtime.Autograd.Torch.Core.TensorTransfer

/-!
# Resident execution of custom-operation graphs

The canonical graph supplies shapes and dependencies. Custom nodes use NVRTC; established
operations use the existing LibTorch buffer interface. Preparation checks the whole graph before
launching a node. This is a forward runner, not an automatic custom-operation VJP or a proof about
native execution. Buffer aliases remain reference counted; the runner never explicitly releases
borrowed inputs or view aliases.
-/

public section

namespace NN.IR

open Spec TorchLean Runtime.Autograd.LibTorch

namespace Internal

private def parent (values : Array AnyBuffer) (index : Nat) : IO AnyBuffer :=
  match values[index]? with
  | some value => pure value
  | none => throw (IO.userError s!"IR native: missing parent {index}")

private def upload {s : Shape} (tensor : Tensor Float32 s) : IO Buffer := do
  let host ← Runtime.Autograd.Torch.TensorTransfer.toFloatTensor tensor
  Buffer.ofFloatArrayIO (Convert.flattenFloat host)

private def broadcast (source target : Shape) : Except String (Buffer → Buffer) := do
  let _ ← AnyBuffer.numelU32 target
  if h : Shape.CanBroadcastTo source target then
    if source = target then pure id
    else pure fun value =>
      Buffer.broadcastTo value source.toArray target.toArray (Broadcast.axisMap h)
  else throw s!"IR native: cannot broadcast {repr source} to {repr target}"

private def prepare (graph : Graph) (payload : Payload Float32) (node : Node) :
    Except String (Array AnyBuffer → IO Buffer) := do
  let _ ← AnyBuffer.numelU32 node.outShape
  let parents ← node.parents.mapM fun index => graph.getNode index
  let unary (op : Buffer → Buffer) : Except String (Array AnyBuffer → IO Buffer) := do
    let first ← Infer.expectUnaryParent node.kind.tag node.parents
    pure fun values => do pure (op (← parent values first).buf)
  let binary (op : Buffer → Buffer → Buffer) :
      Except String (Array AnyBuffer → IO Buffer) := do
    let (first, second) ← Infer.expectBinaryParents node.kind.tag node.parents
    pure fun values => do pure (op (← parent values first).buf (← parent values second).buf)
  match node.kind with
  | .input => throw "IR native: inputs are supplied separately"
  | .const _ =>
      let some constant := payload.const? node.id |
        throw s!"IR native: node {node.id}: missing constant"
      if constant.n != node.outShape.size then
        throw s!"IR native: node {node.id}: constant size mismatch"
      pure fun _ => upload constant.v
  | .custom name _ _ =>
      let some program := payload.custom? node.id |
        throw s!"IR native: custom {name}: missing checked body"
      let .native .binary32 := program.scalar.precision |
        throw s!"IR native: custom {name}: resident buffers require native binary32"
      let bits := fun x => (program.scalar.encode x).toUInt64
      let _ ← NN.Kernel.Cuda.source .binary32 bits node.parents.size program.expression
      pure fun values => do
        let inputs ← node.parents.mapM fun index => do pure (← parent values index).buf
        NN.Kernel.Internal.runBuffers .binary32 bits program.expression inputs
          node.outShape.size.toUInt64
  | .detach | .reshape _ _ | .flatten _ => unary id
  | .add => binary Buffer.add
  | .sub => binary Buffer.sub
  | .mulElem => binary Buffer.mul
  | .abs => unary Buffer.abs
  | .sqrt => unary Buffer.sqrt
  | .relu => unary Buffer.relu
  | .sigmoid => unary Buffer.sigmoid
  | .tanh => unary Buffer.tanh
  | .exp => unary Buffer.exp
  | .log =>
      throw s!"IR native: node {node.id}: log requires a checked positive-input domain"
  | .inv => unary Buffer.inv
  | .sin => unary Buffer.sin
  | .cos => unary Buffer.cos
  | .softmax axis =>
      let first ← Infer.expectUnaryParent "softmax" node.parents
      let axis ← AnyBuffer.natToU32Checked axis
      pure fun values => do
        Buffer.softmax (← parent values first).buf node.outShape.toArray axis
  | .layernorm axis =>
      let first ← Infer.expectUnaryParent "layernorm" node.parents
      let (rows, cols) ← OpContracts.layerNormMatrixDims axis node.outShape
      let rows ← AnyBuffer.natToU32Checked rows
      let cols ← AnyBuffer.natToU32Checked cols
      let affine ← match payload.layerNorm? node.id with
        | none => pure (do
            let gamma ← Buffer.fullIO cols 1
            let beta ← Buffer.fullIO cols 0
            pure (gamma, beta, (TorchLean.normalizationEpsilon : Float32).toFloat))
        | some params => do
            if params.normalizedShape != Shape.ofList (node.outShape.toList.drop axis) then
              throw s!"IR native: layernorm {node.id}: payload suffix mismatch"
            pure (do pure (← upload params.gamma, ← upload params.beta, params.eps.toFloat))
      pure fun values => do
        let input ← parent values first
        if rows == 0 then return input.buf
        let (gamma, beta, epsilon) ← affine
        pure (Buffer.layerNormFwd input.buf gamma beta rows cols
          (1 / Float.ofNat cols.toNat) epsilon).1
  | .conv config =>
      if config.spatialRank == 0 || config.spatialRank > 3 then
        throw s!"IR native: conv {node.id}: LibTorch supports one to three spatial dimensions"
      let first ← Infer.expectUnaryParent "conv" parents
      let some params := payload.conv? node.id |
        throw s!"IR native: conv {node.id}: missing payload"
      if !params.matchesConfig config then
        throw s!"IR native: conv {node.id}: payload configuration mismatch"
      let leading := Shape.ofList (first.outShape.toList.take config.channelAxis)
      if params.input leading != first.outShape then
        throw s!"IR native: conv {node.id}: payload input shape mismatch"
      let _ ← AnyBuffer.natToU32Checked leading.size
      let groups ← AnyBuffer.natToU32Checked config.groups
      let kernelShape := Shape.ofList
        (config.outChannels :: config.inChannels :: Tensor.to config.kernel (List Nat))
      let _ ← AnyBuffer.numelU32 kernelShape
      for geometry in [config.kernel, config.stride, config.padding,
          config.paddingAfter, config.dilation] do
        for extent in Tensor.to geometry (List Nat) do
          let _ ← AnyBuffer.natToU32Checked extent
      let input ← Infer.expectUnaryParent "conv" node.parents
      pure fun values => do
        Buffer.conv (← parent values input).buf (← upload params.spec.kernel)
          (← upload params.spec.bias) first.outShape.toArray kernelShape.toArray
          (Tensor.to config.stride (List Nat)).toArray
          (Tensor.to config.padding (List Nat)).toArray
          (Tensor.to config.paddingAfter (List Nat)).toArray
          (Tensor.to config.dilation (List Nat)).toArray groups
  | .broadcastTo source target => unary (← broadcast source target)
  | .maxPool config | .avgPool config =>
      if config.spatialRank > 3 then
        throw s!"IR native: {node.kind.tag}: LibTorch supports one to three spatial dimensions"
      let first ← Infer.expectUnaryParent node.kind.tag parents
      let plan ← OpContracts.planPool node.kind.tag config first.outShape
      -- Flatten only the preserved prefix; each spatial window remains independent.
      let channels ← AnyBuffer.natToU32Checked plan.leading.size
      let spatial := (Tensor.to plan.spatial (List Nat)).toArray
      let kernel := (Tensor.to config.kernel (List Nat)).toArray
      let stride := (Tensor.to config.stride (List Nat)).toArray
      let padding := (Tensor.to config.padding (List Nat)).toArray
      for dimensions in [spatial, kernel, stride, padding] do
        for extent in dimensions do
          let _ ← AnyBuffer.natToU32Checked extent
      let _ ← AnyBuffer.numelU32 (Shape.ofList kernel.toList)
      if node.kind matches .avgPool _ then
        unary fun input => torchleanAvgPoolFwdCuda input spatial kernel stride padding channels
      else
        unary fun input => torchleanMaxPoolFwdCuda input spatial kernel stride padding channels
  | .matmul =>
      let (left, right) ← Infer.expectBinaryParents "matmul" parents
      let dims ← OpContracts.matmulDims left.outShape right.outShape
      let batch ← AnyBuffer.natToU32Checked dims.leading.size
      let rows ← AnyBuffer.natToU32Checked dims.rows
      let inner ← AnyBuffer.natToU32Checked dims.inner
      let cols ← AnyBuffer.natToU32Checked dims.cols
      let leftShape := dims.leftLeading.concat (Shape.ofList [dims.rows, dims.inner])
      let rightShape := dims.rightLeading.concat (Shape.ofList [dims.inner, dims.cols])
      let leftBroadcast ← broadcast leftShape
        (dims.leading.concat (Shape.ofList [dims.rows, dims.inner]))
      let rightBroadcast ← broadcast rightShape
        (dims.leading.concat (Shape.ofList [dims.inner, dims.cols]))
      binary fun a b => Buffer.bmm (leftBroadcast a) (rightBroadcast b) batch rows inner cols
  | _ => throw s!"IR native: node {node.id}: unsupported operation {node.kind.tag}"

end Internal

/-- Execute the canonical graph on resident binary32 buffers, returning every node's value.

Inputs are keyed by node id and validated before execution. All shape, payload and operation
coverage checks run before any graph node is launched. Established operations retain the existing
LibTorch FFI trust and failure policy; custom compilation and checked-read failures are IO errors.
This runner neither records a differentiation tape nor invents gradients for custom bodies.
-/
def Graph.runBuffers (graph : Graph) (payload : Payload Float32)
    (inputs : Nat → Option AnyBuffer) : IO (Array AnyBuffer) := do
  IO.ofExcept graph.checkShapes
  Buffer.requireNativeRuntime
  let mut actions : Array (Array AnyBuffer → IO Buffer) := #[]
  for node in graph.nodes do
    if node.kind matches .input then
      let some value := inputs node.id |
        throw (IO.userError s!"IR native: missing input at node {node.id}")
      if value.s != node.outShape then
        throw (IO.userError s!"IR native: input shape mismatch at node {node.id}")
      let value ← IO.ofExcept (AnyBuffer.validate value)
      unless Buffer.dtype value.buf == .float32 do
        throw (IO.userError s!"IR native: input dtype must be binary32 at node {node.id}")
      actions := actions.push (fun _ => pure value.buf)
    else
      actions := actions.push (← IO.ofExcept (Internal.prepare graph payload node))
  let mut values : Array AnyBuffer := #[]
  for node in graph.nodes do
    let some action := actions[node.id]? |
      throw (IO.userError s!"IR native: missing prepared node {node.id}")
    let result : AnyBuffer := { s := node.outShape, buf := ← action values }
    values := values.push (← IO.ofExcept (AnyBuffer.validate result))
  pure values

end NN.IR
