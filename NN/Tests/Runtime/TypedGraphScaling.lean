/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import NN.Runtime.Autograd.TypedGraph
import NN.Runtime.Autograd.Torch.Core.TypedGraph

/-!
Dedicated increasing-size TypedGraph benchmark. Run each size in a separate process so the
process high-water memory includes only that graph. Timings separate construction, checked
lowering, retained dense backward, and the final backward that releases the saved graph.
The `public` mode exercises the ordinary checked VJP API. Output hashes fold the `Float.toBits`
observations of every tensor element. This fixture contains no NaNs.
-/

open Spec TorchLean Runtime.Autograd
open Runtime.Autograd.TypedGraph

namespace TypedGraphScaling

private def chain (count : Nat) :
    GraphM.M Float [[4]] (GraphM.Var [4]) := do
  let mut value ← GraphM.arg 0 [4]
  for _ in [0:count] do
    value ← GraphM.relu value
  pure value

private def hashTensor (value : Spec.SomeTensor Float) : UInt64 :=
  (Storage.toArray value.tensor.buffer).foldl
    (fun hash scalar => (hash ^^^ scalar.toBits) * 1099511628211) 14695981039346656037

private def hashValues (values : Array (Spec.SomeTensor Float)) : UInt64 :=
  values.foldl (fun hash value => (hash ^^^ hashTensor value) * 1099511628211)
    14695981039346656037

private def memory : IO String := do
  let status ← IO.FS.readFile "/proc/self/status"
  pure <| String.intercalate " " <|
    (status.splitOn "\n").filter fun line =>
      line.startsWith "VmRSS:" || line.startsWith "VmHWM:"

/-- Keep the pure reverse pass inside the timed IO action, before the ending clock read. -/
@[noinline] private def executeBackward {ss : List Shape} (compiled : Compiled Float [[4]] ss)
    (index : Proofs.Idx ([[4]] ++ ss) [4]) (seed : Tensor Float [4]) :
    IO (Array (Spec.SomeTensor Float)) :=
  pure (compiled.backwardDenseAllFrom index seed)

@[noinline] private def executePublic
    (graph : Torch.TypedGraph Float [[4]] [4]) (inputs : TensorPack Float [[4]])
    (seed : Tensor Float [4]) : IO (TensorPack Float [[4]] × Tensor Float [4]) :=
  okOrThrow (graph.vjpChecked inputs () seed)

def run (count : Nat) (mode : String := "compiled") : IO Unit := do
  let start ← IO.monoNanosNow
  let (output, state) ← okOrThrow (GraphM.run (chain count))
  let built ← IO.monoNanosNow
  IO.println s!"phase=construct n={count} ns={built - start} {← memory}"
  let inputs : TensorPack Float [[4]] :=
    .cons (Tensor.Internal.Rep.ofArray #[-2.0, -0.0, 0.5, 3.0] (by decide)) .nil
  let index ← okOrThrow (GraphM.mkIdx (_α := Float) (Γ := [[4]]) state.nodeShapes output)
  let seed : Tensor Float [4] := Tensor.full [4] 1.0
  if mode == "legacy" then
    let beforeLower ← IO.monoNanosNow
    let (tape, context) ← okOrThrow (lowerToTapeChecked state.data inputs ())
    let lowered ← IO.monoNanosNow
    IO.println s!"phase=lower n={count} ns={lowered - beforeLower} {← memory}"
    let valuesHash := hashValues context.toShapeErasedArray
    let beforeBackward ← IO.monoNanosNow
    let gradients ← okOrThrow
      (backwardDenseAllFrom (Γ := [[4]]) (ss := state.nodeShapes) tape index seed)
    let backward ← IO.monoNanosNow
    IO.println s!"phase=backward n={count} ns={backward - beforeBackward} {← memory}"
    IO.println s!"n={count} values={valuesHash} gradients={hashValues gradients} \
      tensors={gradients.size}"
  else if mode == "public" then
    let graph : Torch.TypedGraph Float [[4]] [4] :=
      { nodeShapes := state.nodeShapes, data := state.data, output := index }
    let beforeVjp ← IO.monoNanosNow
    let result ← executePublic graph inputs seed
    let afterVjp ← IO.monoNanosNow
    IO.println s!"phase=checked_vjp n={count} ns={afterVjp - beforeVjp} {← memory}"
    let gradientsHash := hashValues result.1.toShapeErasedArray
    let beforeRelease ← IO.monoNanosNow
    let repeated ← executePublic graph inputs seed
    let afterRelease ← IO.monoNanosNow
    IO.println s!"phase=checked_vjp_release n={count} ns={afterRelease - beforeRelease} {← memory}"
    unless gradientsHash == hashValues repeated.1.toShapeErasedArray do
      throw <| IO.userError "repeated public VJP changed the gradient Float.toBits hash"
    IO.println s!"n={count} input_gradients={gradientsHash} \
      output={hashTensor (Spec.SomeTensor.ofTensor repeated.2)}"
  else
    let beforeLower ← IO.monoNanosNow
    let compiled ← okOrThrow (compileChecked state.data inputs ())
    let lowered ← IO.monoNanosNow
    IO.println s!"phase=lower n={count} ns={lowered - beforeLower} {← memory}"
    let valuesHash := hashValues compiled.context.values
    let beforeBackward ← IO.monoNanosNow
    let gradients ← executeBackward compiled index seed
    let backward ← IO.monoNanosNow
    IO.println s!"phase=backward n={count} ns={backward - beforeBackward} {← memory}"
    IO.println s!"n={count} values={valuesHash} gradients={hashValues gradients} \
      tensors={gradients.size}"
    -- This second pass keeps the compiled handle live during the first timing. Its timing
    -- includes releasing the final owner of the graph's retained shape-prefix metadata.
    let beforeRelease ← IO.monoNanosNow
    let repeated ← executeBackward compiled index seed
    let released ← IO.monoNanosNow
    IO.println s!"phase=backward_release n={count} ns={released - beforeRelease} {← memory}"
    IO.println s!"n={count} repeated_gradients={hashValues repeated} tensors={repeated.size}"

end TypedGraphScaling

def main (args : List String) : IO Unit := do
  match args with
  | [size] => TypedGraphScaling.run size.toNat!
  | [size, mode] =>
      if mode == "legacy" || mode == "public" then TypedGraphScaling.run size.toNat! mode
      else throw <| IO.userError "mode must be legacy or public"
  | _ => throw <| IO.userError "usage: typedgraph-scaling NODE_COUNT [legacy|public]"
