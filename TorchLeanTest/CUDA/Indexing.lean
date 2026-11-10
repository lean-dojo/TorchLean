/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanTest.Harness.CUDA
public import TorchLean.CUDA.LibTorch.Controls
public import TorchLean.NN.Tape.Apply

/-!
# Gather and scatter on CUDA against the CPU tape

The CUDA entries of `Op.gather`, `Op.indexAt` and `Op.scatter .sum` against the CPU tape in
`Float32`, values and gradients bit for bit, with deterministic reductions. The fixtures select
along each axis of a rank-3 tensor with repeated indices and take one index along an axis (the eager
`indexAt`), also with negative zeros in the cotangent, on a one-entry input and on an empty one.
They gather along one axis without batch pairs, along two axes with a batch pair and along unsorted
axes, and they run a sum scatter through index tuples over both axes of a matrix and one along a
single axis, both onto a nonzero base with an index repeated three times. The remaining fixtures are
an empty selection, negative zeros, a cancellation that only an ordered fold keeps, an aliased base
and updates, and a replacing scatter, which has no entry. The CUDA entries are tested on these
fixtures only.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace Indexing

open TorchLean.Tensor

/-- A `Float32` tensor of shape `s` with distinct entries whose sums round. -/
def ramp (s : Shape) (offset : Float) : Tensor Float32 s :=
  Tensor.Rep.ofFlatFn fun i => (Float.ofNat i.val * 0.37 + offset).toFloat32

/-- The bits of the entries of a `Float32` tensor, row-major. -/
def bits {s : Shape} (x : Tensor Float32 s) : Array UInt32 :=
  (x.to (Array Float32)).map Float32.toBits

/-- A device buffer holding a `Float32` tensor. -/
def buffer {s : Shape} (x : Tensor Float32 s) : LibTorch.SomeBuffer :=
  { s, buf := LibTorch.Buffer.ofFloatArray ⟨(x.to (Array Float32)).map Float32.toFloat⟩ }

/-- The bits of a device buffer of `Float32` entries. -/
def deviceBits (b : LibTorch.Buffer) : Array UInt32 :=
  (LibTorch.Buffer.toFloatArray b).data.map fun v => v.toFloat32.toBits

/-- The gradient of node `id` on a CPU tape. -/
def cpuGrad {s : Shape} (grads : Array (SomeTensor Float32)) (id : Nat) :
    IO (Tensor Float32 s) := do
  let some g := grads[id]? | throw <| IO.userError s!"gather test: no gradient at {id}"
  if h : g.shape = s then pure (g.cast h)
  else throw <| IO.userError s!"gather test: gradient shape at {id}"

/-- Fail unless the CPU and CUDA bits agree. -/
def same (label : String) (host cuda : Array UInt32) : IO Unit :=
  unless host == cuda do
    throw <| IO.userError s!"{label}: CUDA bits {cuda} differ from the CPU tape's {host}"

/-- A vector of `count` indices below `n` from `values`. -/
def indexVector (n count : Nat) (h : 0 < n) (values : List Nat) : Tensor (Fin n) [count] :=
  Tensor.Rep.ofFlatFn fun i => ⟨values.getD i.val 0 % n, Nat.mod_lt _ h⟩

/-- A gather-shaped operator `op` (an operand and an index input) on the CPU and CUDA tapes: the
value and the operand's pullback of `seed` must have the same bits on both. -/
def checkGather {s t out : Shape} {bounds : List Nat} (label : String)
    (op : Op [.value s, .index bounds t] [.value out])
    (index : Tensor (Port.IndexTuple bounds) t) (seed : Tensor Float32 out := ramp out 0.3) :
    IO Unit := do
  let x := ramp s (-1.1)
  let (host, xId) := Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float32))) (.ofTensor x)
  let (host, yId, ()) ← IO.ofExcept (Tape.apply host label op (xId, index, ()))
  let y ← IO.ofExcept (Tape.requireValue (α := Float32) (t := host) (s := out) yId)
  let grads ← IO.ofExcept (Tape.backwardDenseAll host yId (SomeTensor.ofTensor seed))
  let dx ← cpuGrad (s := s) grads xId
  let (device, xIdc) := Tape.leaf (t := Tape.empty) (buffer x)
  let (device, yIdc) ← Harness.record device label op (xIdc, index, ())
  let yc ← IO.ofExcept (LibTorch.Tape.requireValue (t := device) yIdc out)
  let gradsc ← IO.ofExcept (Tape.backwardDenseAll (t := device) yIdc (buffer seed))
  let some dxc := gradsc[xIdc]? | throw <| IO.userError s!"{label}: no CUDA gradient"
  same s!"{label} forward" (bits y) (deviceBits yc)
  same s!"{label} pullback" (bits dx) (deviceBits dxc.buf)

/-- A sum scatter `op` (a base, an index input and the updates) on the CPU and CUDA tapes: the value
and the pullbacks to the base and the updates must have the same bits on both. -/
def checkScatter {s t u : Shape} {bounds : List Nat} (label : String)
    (op : Op [.value s, .index bounds t, .value u] [.value s])
    (index : Tensor (Port.IndexTuple bounds) t) (base : Tensor Float32 s := ramp s (-2.3))
    (updates : Tensor Float32 u := ramp u 0.7) : IO Unit := do
  let seed := ramp s 0.1
  let (host, baseId) := Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float32))) (.ofTensor base)
  let (host, updatesId) := Tape.leaf (t := host) (.ofTensor updates)
  let (host, yId, ()) ← IO.ofExcept (Tape.apply host label op (baseId, index, updatesId, ()))
  let y ← IO.ofExcept (Tape.requireValue (α := Float32) (t := host) (s := s) yId)
  let grads ← IO.ofExcept (Tape.backwardDenseAll host yId (SomeTensor.ofTensor seed))
  let dBase ← cpuGrad (s := s) grads baseId
  let dUpdates ← cpuGrad (s := u) grads updatesId
  let (device, baseIdc) := Tape.leaf (t := Tape.empty) (buffer base)
  let (device, updatesIdc) := Tape.leaf (t := device) (buffer updates)
  let (device, yIdc) ← Harness.record device label op (baseIdc, index, updatesIdc, ())
  let yc ← IO.ofExcept (LibTorch.Tape.requireValue (t := device) yIdc s)
  let gradsc ← IO.ofExcept (Tape.backwardDenseAll (t := device) yIdc (buffer seed))
  let some dBasec := gradsc[baseIdc]? | throw <| IO.userError s!"{label}: no CUDA base gradient"
  let some dUpdatesc := gradsc[updatesIdc]? |
    throw <| IO.userError s!"{label}: no CUDA update gradient"
  same s!"{label} forward" (bits y) (deviceBits yc)
  same s!"{label} base pullback" (bits dBase) (deviceBits dBasec.buf)
  same s!"{label} update pullback" (bits dUpdates) (deviceBits dUpdatesc.buf)

/-- Negative zeros at the even positions of a `Float32` tensor, distinct values at the odd ones. -/
def signedZeros (s : Shape) : Tensor Float32 s :=
  Tensor.Rep.ofFlatFn fun i =>
    if i.val % 2 = 0 then (-0.0 : Float).toFloat32 else (Float.ofNat i.val * 0.37).toFloat32

/-- One index along `dim`, as the eager `indexAt` records it on each device (`Op.indexAt`). -/
def checkIndexAt (s : Shape) (dim : Fin s.rank) (index : Fin (s.extent dim))
    (seed : Tensor Float32 (s.eraseIdx dim) := ramp (s.eraseIdx dim) 0.3)
    (label : String := s!"select(axis={dim.val}, index={index.val})") : IO Unit := do
  let op := Op.indexAt dim index
  let x := ramp s (-1.1)
  let (host, xId) := Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float32))) (.ofTensor x)
  let (host, yId, ()) ← IO.ofExcept (Tape.apply host label op (xId, ()))
  let y ← IO.ofExcept (Tape.requireValue (α := Float32) (t := host) (s := s.eraseIdx dim) yId)
  let grads ← IO.ofExcept (Tape.backwardDenseAll host yId (SomeTensor.ofTensor seed))
  let dx ← cpuGrad (s := s) grads xId
  let (device, xIdc) := Tape.leaf (t := Tape.empty) (buffer x)
  let (device, yIdc) ← Harness.record device label op (xIdc, ())
  let yc ← IO.ofExcept (LibTorch.Tape.requireValue (t := device) yIdc (s.eraseIdx dim))
  let gradsc ← IO.ofExcept (Tape.backwardDenseAll (t := device) yIdc (buffer seed))
  let some dxc := gradsc[xIdc]? | throw <| IO.userError s!"{label}: no CUDA gradient"
  same s!"{label} forward" (bits y) (deviceBits yc)
  same s!"{label} pullback" (bits dx) (deviceBits dxc.buf)

/-- A sum scatter whose updates are its base, one node in both inputs. -/
def checkScatterAliased {s t : Shape} {bounds : List Nat} (label : String)
    (op : Op [.value s, .index bounds t, .value s] [.value s])
    (index : Tensor (Port.IndexTuple bounds) t) : IO Unit := do
  let x := ramp s (-2.3)
  let seed := ramp s 0.1
  let (host, xId) := Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float32))) (.ofTensor x)
  let (host, yId, ()) ← IO.ofExcept (Tape.apply host label op (xId, index, xId, ()))
  let y ← IO.ofExcept (Tape.requireValue (α := Float32) (t := host) (s := s) yId)
  let grads ← IO.ofExcept (Tape.backwardDenseAll host yId (SomeTensor.ofTensor seed))
  let dx ← cpuGrad (s := s) grads xId
  let (device, xIdc) := Tape.leaf (t := Tape.empty) (buffer x)
  let (device, yIdc) ← Harness.record device label op (xIdc, index, xIdc, ())
  let yc ← IO.ofExcept (LibTorch.Tape.requireValue (t := device) yIdc s)
  let gradsc ← IO.ofExcept (Tape.backwardDenseAll (t := device) yIdc (buffer seed))
  let some dxc := gradsc[xIdc]? | throw <| IO.userError s!"{label}: no CUDA gradient"
  same s!"{label} forward" (bits y) (deviceBits yc)
  same s!"{label} pullback" (bits dx) (deviceBits dxc.buf)

/-- Index selection along `dim`, as a gather through the indices of `Tensor.selectIndices`. -/
def checkSelect (s : Shape) (dim : Fin s.rank) (count : Nat)
    (indices : Tensor (Fin (s.axisSize dim)) [count]) : IO Unit :=
  checkGather s!"index_select(axis={dim.val})"
    (Op.gather (s.selectDims dim) (s.selectBatch dim count)
      (Shape.GatherValid.select s dim count))
    (Tensor.map (fun tuple => tuple ⟨0, Nat.one_pos⟩) (Tensor.selectIndices dim indices))

/-- The index tuples of a gather along axes `2` then `0`: `(2 * i + 1) % 4` and `i % 3`. -/
def reversedIndices : Tensor (Port.IndexTuple [4, 3]) [5] :=
  Tensor.Rep.ofFlatFn fun i => fun k =>
    ⟨(if k.val = 0 then 2 * i.val + 1 else i.val) % ([4, 3] : List Nat).get k,
      Nat.mod_lt _ (by revert k; decide)⟩

/-- The index tuples of the two-axis gather: rows `i % 3` and columns `(i / 2) % 4`. -/
def pairIndices : Tensor (Port.IndexTuple [3, 4]) [2, 3] :=
  Tensor.Rep.ofFlatFn fun i => fun k =>
    ⟨(if k.val = 0 then i.val % 3 else (i.val / 2) % 4) % ([3, 4] : List Nat).get k,
      Nat.mod_lt _ (by revert k; decide)⟩

/-- Run the select, gather and scatter checks with deterministic algorithms on and cuDNN
benchmarking off, then restore both settings. -/
def run : IO Unit := do
  let previous ← LibTorch.getDeterministic
  let benchmark ← LibTorch.getCuDNNBenchmark
  LibTorch.setCuDNNBenchmark false
  LibTorch.setDeterministic true
  try
    let s : Shape := [3, 4, 2]
    checkSelect s ⟨0, by decide⟩ 5 (indexVector 3 5 (by decide) [2, 0, 2, 1, 2])
    checkSelect s ⟨1, by decide⟩ 5 (indexVector 4 5 (by decide) [3, 3, 0, 3, 1])
    checkSelect s ⟨2, by decide⟩ 5 (indexVector 2 5 (by decide) [1, 1, 0, 1, 1])
    checkIndexAt s ⟨1, by decide⟩ ⟨2, by decide⟩
    checkIndexAt s ⟨2, by decide⟩ ⟨0, by decide⟩
    -- The selected row keeps the cotangent's negative zeros and the other rows are `+0`, with
    -- zero rows after the selected row, on both sides of it and before it. A one-entry input has
    -- no zero rows and no swaps, and the last input has no entries.
    checkIndexAt s ⟨1, by decide⟩ ⟨0, by decide⟩ (seed := signedZeros _)
      (label := "select(axis=1, index=0) with negative-zero cotangents")
    checkIndexAt s ⟨1, by decide⟩ ⟨2, by decide⟩ (seed := signedZeros _)
      (label := "select(axis=1, index=2) with negative-zero cotangents")
    checkIndexAt s ⟨1, by decide⟩ ⟨3, by decide⟩ (seed := signedZeros _)
      (label := "select(axis=1, index=3) with negative-zero cotangents")
    checkIndexAt [1] ⟨0, by decide⟩ ⟨0, by decide⟩ (seed := signedZeros _)
      (label := "select of [1] with a negative-zero cotangent")
    checkIndexAt [0, 3, 2] ⟨1, by decide⟩ ⟨1, by decide⟩
    checkSelect s ⟨1, by decide⟩ 0 (indexVector 4 0 (by decide) [])
    checkGather "gather(dims=[1])"
      (Op.gather (s := [3, 4, 2]) (t := [5]) [⟨1, by decide⟩] []
        ⟨by decide, by decide, by decide⟩)
      (indexVector 4 5 (by decide) [3, 3, 0, 3, 1])
    checkGather "gather(dims=[0, 2], batch=[(1, 0)])"
      (Op.gather (s := [3, 2, 4]) (t := [2, 3]) [⟨0, by decide⟩, ⟨2, by decide⟩]
        [(⟨1, by decide⟩, ⟨0, by decide⟩)] ⟨by decide, by decide, by decide⟩)
      pairIndices
    checkGather "gather(dims=[2, 0])"
      (Op.gather (s := [3, 2, 4]) (t := [5]) [⟨2, by decide⟩, ⟨0, by decide⟩] []
        ⟨by decide, by decide, by decide⟩)
      reversedIndices
    -- Negative-zero cotangents: the pullback starts from positive zeros, on both devices.
    checkGather "gather(dims=[1]) with negative-zero cotangents"
      (Op.gather (s := [3, 4, 2]) (t := [5]) [⟨1, by decide⟩] []
        ⟨by decide, by decide, by decide⟩)
      (indexVector 4 5 (by decide) [3, 3, 0, 3, 1])
      (seed := Tensor.Rep.ofFlatFn fun _ => (-0.0 : Float).toFloat32)
    let base : Shape := [3, 4]
    let selected := indexVector 4 5 (by decide) [3, 3, 0, 3, 1]
    checkScatter "scatter_add(axis=1)"
      (Op.scatter (s := base) (t := base.set 1 5) .sum (List.finRange 2)
        (List.nodup_finRange _))
      (Tensor.map (Op.indexOfAxisTuples base.extent _)
        (Tensor.selectSources (s := base) ⟨1, by decide⟩ selected))
    checkScatter "scatter(dims=[0])"
      (Op.scatter (s := [3, 2]) (t := [4]) .sum [⟨0, by decide⟩] (by decide))
      (indexVector 3 4 (by decide) [2, 0, 2, 2])
    -- An ordered fold keeps `(2^24 - 2^24) + 1 = 1`, and adding the updates in reverse gives `0`.
    let big : Float32 := (16777216.0 : Float).toFloat32
    checkScatter "scatter(dims=[0]) cancellation"
      (Op.scatter (s := [1]) (t := [2]) .sum [⟨0, by decide⟩] (by decide))
      (indexVector 1 2 (by decide) [0, 0])
      (base := Tensor.Rep.ofFlatFn fun _ => big)
      (updates := Tensor.Rep.ofFlatFn fun i =>
        if i.val = 0 then -big else (1.0 : Float).toFloat32)
    -- Negative zeros in the base, touched by negative-zero updates and untouched.
    checkScatter "scatter(dims=[0]) negative zeros"
      (Op.scatter (s := [3]) (t := [2]) .sum [⟨0, by decide⟩] (by decide))
      (indexVector 3 2 (by decide) [0, 0])
      (base := Tensor.Rep.ofFlatFn fun i =>
        if i.val < 2 then (-0.0 : Float).toFloat32 else (1.0 : Float).toFloat32)
      (updates := Tensor.Rep.ofFlatFn fun _ => (-0.0 : Float).toFloat32)
    checkScatterAliased "scatter(dims=[0]) aliased"
      (Op.scatter (s := [2, 2]) (t := [2]) .sum [⟨0, by decide⟩] (by decide))
      (indexVector 2 2 (by decide) [1, 1])
    -- A replacing scatter has no entry: the tape comes back with no new node and the error says so.
    let (device, baseIdc) := Tape.leaf (t := Tape.empty) (buffer (ramp [3] 0.0))
    let (device, updatesIdc) := Tape.leaf (t := device) (buffer (ramp [2] 1.0))
    let (after, result) ← LibTorch.Tape.apply device "scatter_replace"
      (Op.scatter (s := [3]) (t := [2]) .replace [⟨0, by decide⟩] (by decide))
      (baseIdc, indexVector 3 2 (by decide) [0, 1], updatesIdc, ()) (pure .float32)
    let unsupported := match result with
      | .error message => (message.splitOn "is unsupported by LibTorch").length > 1
      | .ok _ => false
    unless unsupported && after.nodes.size == device.nodes.size do
      throw <| IO.userError "scatter_replace: expected the missing-entry error and no new node"
    IO.println "  gather and scatter: CUDA entries match the CPU tape bit for bit"
  finally
    LibTorch.setDeterministic previous
    LibTorch.setCuDNNBenchmark benchmark

end Indexing
end CUDA
end TorchLean.Test
