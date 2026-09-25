/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.API
public import NN.Tests.Runtime.IRExecScalingRegression

/-!
# IRExec Scaling Regression

Dedicated native driver for measuring lowering, execution, and disposal independently. The lowered
graph is owned by an `IO.Ref`; clearing that reference after the inspection function returns times
its destruction without relying on compiler decisions about the lifetime of a local variable.

Run one size per process to compare peak RSS without carrying allocator state between sizes.
-/

public section

open Spec TorchLean
open Runtime.Autograd.IRExec

namespace Tests.IRExecScaling

private def reluChain (n : Nat) : NN.IR.Graph :=
  { nodes := (Array.range (n + 1)).map fun i =>
      if i = 0 then { id := i, parents := #[], kind := .input, outShape := [4] }
      else { id := i, parents := #[i - 1], kind := .relu, outShape := [4] } }

@[noinline] private def lowerInto (source : IO.Ref NN.IR.Graph)
    (result : IO.Ref (Option (ForwardGraph Float))) : IO Unit := do
  let graph ← source.get
  match lowerToForwardGraph (α := Float) graph {} with
  | .error error => throw <| IO.userError error
  | .ok graph => result.set (some graph)

@[noinline] private def inspect (result : IO.Ref (Option (ForwardGraph Float)))
    (n : Nat) (execute : Bool) : IO (Nat × UInt64) := do
  let some graph ← result.get | throw <| IO.userError "missing lowered graph"
  unless graph.ss.length == n do
    throw <| IO.userError "lowering lost nodes"
  if !execute then
    return (0, 0)
  if h : [4] = graph.inShape then
    let x : Tensor Float [4] :=
      Tensor.Internal.Rep.ofFlatFn fun i => #[-3.0, 2.0, -0.0, 7.0][i.val]!
    let table ← IO.mkRef (#[] : Array (Spec.SomeTensor Float))
    let start ← IO.monoNanosNow
    table.set (graph.denoteAll (Tensor.castShape x h))
    let stop ← IO.monoNanosNow
    let values ← table.get
    unless values.size == n + 1 do
      throw <| IO.userError "execution lost nodes"
    let mut hash : UInt64 := 14695981039346656037
    for value in values do
      unless value.shape == [4] do
        throw <| IO.userError "execution changed a shape"
      for item in value.tensor.data do
        hash := (hash ^^^ item.toBits) * 1099511628211
    return (stop - start, hash)
  else
    throw <| IO.userError "lowering changed the input shape"

private def memory : IO String := do
  let status ← IO.FS.readFile "/proc/self/status"
  return String.intercalate " " <|
    (status.splitOn "\n").filter fun line =>
      line.startsWith "VmRSS:" || line.startsWith "VmHWM:"

/-- Measure the complete lowered artifact, including releasing all retained shape prefixes. -/
def benchmark (n : Nat) (execute : Bool := true) : IO Unit := do
  let source ← IO.mkRef (reluChain n)
  let result ← IO.mkRef (none : Option (ForwardGraph Float))
  let start ← IO.monoNanosNow
  lowerInto source result
  let lowered ← IO.monoNanosNow
  let live ← memory
  let (execution, hash) ← inspect result n execute
  let beforeDrop ← IO.monoNanosNow
  result.set none
  let afterDrop ← IO.monoNanosNow
  IO.println s!"n={n} lower_ns={lowered - start} eval_ns={execution} \
    drop_ns={afterDrop - beforeDrop} hash={hash} {live}"

end Tests.IRExecScaling

/-- Run the dedicated scaling driver at the requested node count. -/
def main (args : List String) : IO Unit := do
  match args with
  | ["--check"] => Tests.IRExecScalingRegression.check
  | ["--lower-only", size] =>
      let some n := size.toNat? | throw <| IO.userError "expected a natural node count"
      Tests.IRExecScaling.benchmark n false
  | [size] =>
      let some n := size.toNat? | throw <| IO.userError "expected a natural node count"
      Tests.IRExecScaling.benchmark n
  | _ => throw <| IO.userError "usage: irexec_scaling [--lower-only] NODE_COUNT | --check"
