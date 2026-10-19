/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.MoE
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Session
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Tests.Runtime.Cuda.Utils

/-!
# Mixture-of-experts runtime checks

The gating and mixing of `nn.moe` against `Spec.MoE.combineSpec` and `Spec.MoE.gatesVecSpec`, on
tokens whose experts are decided by ties, including a tie between `+0` and `-0`. On the CPU tape the
result, the gradient with respect to the expert outputs, and the gradient with respect to the
logits match the spec bit for bit or by finite differences; on CUDA the routing mask matches the
spec bit for bit on rows full of ties and NaNs, and the rest matches the CPU tape.

The last check is the binary32 example of `NN.Proofs.Models.MoE`: the logits `0` and `2⁻³⁰` round to
the same probability, so routing on probabilities keeps the first expert where routing on logits
keeps the second, on the host and on the GPU.
-/

@[expose] public section

namespace NN.Tests.Runtime.MoE

open Spec TorchLean TorchLean.Tensor
open Runtime.Autograd.Torch.Internal

/-- Router logits for three tokens and four experts. The second token's first, second and fourth
logits tie; the third token's first, second and fourth are `0`, `-0` and `-0`. -/
def logits : Tensor Float [3, 4] :=
  (Tensor.from (#[0.5, -1.0, 2.0, 0.25, 1.0, 1.0, 0.0, 1.0, 0.0, -0.0, -3.0, -0.0] :
    Array Float)).reshape [3, 4] (by decide)

/-- Outputs of the four experts for the three tokens, of width two. -/
def outputs : Tensor Float [4, 3, 2] :=
  (Tensor.from (Array.ofFn (n := 24) fun i => ((i.val * 7 % 11 : Nat).toFloat - 5) / 4)).reshape
    [4, 3, 2] (by decide)

/-- The spec of the layer on every token. -/
def expected (k : Nat) (scores : Tensor Float [3, 4]) : Tensor Float [3, 2] :=
  Tensor.dim fun t =>
    Spec.MoE.combineSpec (Spec.MoE.gatesVecSpec k (Tensor.unstack scores t)) fun e =>
      Tensor.unstack (Tensor.unstack outputs e) t

/-- Run the gating and mixing on `device`, returning the result and the gradients of its sum with
respect to the logits and the expert outputs. -/
def mix (device : NN.Backend.Device) (k : Nat) :
    IO (Tensor Float [3, 2] × Tensor Float [3, 4] × Tensor Float [4, 3, 2]) := do
  let session ← EagerSession.new (α := Float) { device }
  let l ← session.input logits (requiresGrad := true)
  let o ← session.input outputs (requiresGrad := true)
  let y ← (nn.Internal.moeMix (m := EagerM Float) (α := Float) k l o).run session
  let value ← session.getValue y
  let grads ← session.backwardDenseAll y (Tensor.full [3, 2] 1.0)
  pure (value, ← EagerSession.grad grads l, ← EagerSession.grad grads o)

/-- Equal as bit patterns, entry by entry. -/
def sameBits {s : Shape} (x y : Tensor Float s) : Bool :=
  let ax := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) x
  let ay := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) y
  ax.size == ay.size && (List.range ax.size).all fun i => (ax.get! i).toBits == (ay.get! i).toBits

/-- The gate of expert `e` for token `t` under the spec. -/
def gate (k : Nat) (t : Fin 3) (e : Fin 4) : Float :=
  (Spec.MoE.gatesVecSpec k (Tensor.unstack logits t)).getScalar e

/-- On the CPU tape: the spec's bits, the gates as the output gradient, and the spec's
derivative with respect to the logits of the token without ties. -/
def checkCpu : IO Unit := do
  for k in [1, 2, 4] do
    let (value, dLogits, dOutputs) ← mix .cpu k
    unless sameBits value (expected k logits) do
      throw <| IO.userError s!"MoE k={k}: CPU mixing differs from Spec.MoE"
    for e in List.finRange 4 do
      for t in List.finRange 3 do
        let row := Tensor.unstack (Tensor.unstack dOutputs e) t
        for w in List.finRange 2 do
          unless (row.getScalar w).toBits == (gate k t e).toBits do
            throw <| IO.userError s!"MoE k={k}: output gradient of expert {e} is not its gate"
    let step := 1e-6
    for e in List.finRange 4 do
      let shifted (h : Float) : Tensor Float [3, 4] :=
        Tensor.dim fun t => Tensor.dim fun j =>
          Tensor.scalar ((Tensor.unstack logits t).getScalar j +
            if t.val == 0 && j == e then h else 0)
      let total (scores : Tensor Float [3, 4]) : Float :=
        (List.finRange 3).foldl (fun acc t =>
          acc + (Tensor.unstack (expected k scores) t).getScalar 0 +
            (Tensor.unstack (expected k scores) t).getScalar 1) 0
      let numeric := (total (shifted step) - total (shifted (-step))) / (2 * step)
      Tests.Utils.assertApprox s!"MoE k={k}: logit gradient of expert {e}"
        ((Tensor.unstack dLogits 0).getScalar e) numeric 1e-6

/-- Rows of six entries drawn from `-1, 0, -0, 1, 2` and NaN, with between two and six different
values in a row, so that most rows tie. -/
def tiedRows : Tensor Float [64, 6] :=
  let pool : Array Float := #[-1.0, 0.0, -0.0, 1.0, 2.0, (0.0 : Float) / 0.0]
  Tensor.dim fun r => Tensor.dim fun j =>
    Tensor.scalar pool[(r.val * 7 + j.val * (r.val % 4 + 1)) % 6]!

/-- The CUDA mask has the spec's bits for every `k`, on rows of ties, signed zeros and NaNs. -/
def checkCudaMask : IO Unit := do
  for k in List.range 8 do
    let session ← EagerSession.new (α := Float) { device := .cuda }
    let x ← session.input tiedRows
    let mask ← session.getValue (← session.topKMask k x)
    unless sameBits mask (Spec.MoE.topKMaskSpec k tiedRows) do
      throw <| IO.userError s!"MoE: CUDA top-{k} mask differs from Spec.MoE.topKMaskSpec"

/-- On CUDA, the result and both gradients match the CPU tape. -/
def checkCudaMix : IO Unit := do
  for k in [1, 2, 4] do
    let (value, dLogits, dOutputs) ← mix .cuda k
    let (value', dLogits', dOutputs') ← mix .cpu k
    Tests.Cuda.Utils.assertTensorApprox s!"MoE k={k}: CUDA mixing" value value' (tol := 1e-12)
    Tests.Cuda.Utils.assertTensorApprox s!"MoE k={k}: CUDA logit gradient" dLogits dLogits'
      (tol := 1e-12)
    Tests.Cuda.Utils.assertTensorApprox s!"MoE k={k}: CUDA output gradient" dOutputs dOutputs'
      (tol := 1e-12)

/-- In binary32 the logits `0` and `2⁻³⁰` have the same softmax probability. -/
def checkRoundedTie (device : NN.Backend.Device) : IO Unit := do
  let close : Tensor Float32 [2] := Tensor.dim fun j =>
    Tensor.scalar (if j.val == 0 then 0 else (1 : Float32) / 1073741824)
  let probabilities := Activation.softmaxVecSpec close
  unless probabilities.getScalar 0 == probabilities.getScalar 1 do
    throw <| IO.userError "MoE: binary32 probabilities of 0 and 2^-30 differ"
  unless Spec.MoE.kept 1 (fun j => probabilities.getScalar j) 0 &&
      Spec.MoE.kept 1 (fun j => close.getScalar j) 1 do
    throw <| IO.userError "MoE: routing on binary32 probabilities should keep expert 0"
  if device == .cuda then
    let session ← EagerSession.new (α := Float32) { device := .cuda }
    let x ← session.input close
    let p ← session.softmaxLast x
    let mask ← session.getValue (← session.topKMask 1 p)
    unless mask.getScalar 0 == 1 && mask.getScalar 1 == 0 do
      throw <| IO.userError "MoE: CUDA routing on binary32 probabilities should keep expert 0"

/-- Four stacked experts of 4-5-4, two kept per token, on three tokens, with fixed seeds. -/
def stackedLayer : nn.Layer [3, 4] [3, 4] := nn.Layer.moeMlp (E := 3) 2 5 .relu 11 12 13

/-- A row vector as a one-row matrix, and back. -/
def asRow {α : Type} [Storage α] {n : Nat} (v : Tensor α [n]) : Tensor α [1, n] :=
  Tensor.dim fun _ => v
def ofRow {α : Type} [Storage α] {n : Nat} (A : Tensor α [1, n]) : Tensor α [n] :=
  Tensor.unstack A 0

/-- `nn.moeMlp` predicts what `Spec.MoE.forwardSpec` computes from the layer's own weights, with the
router and the experts written as the spec's matrix products. The trainer runs in binary32, so the
spec runs there too, on the weights and input rounded once as the trainer rounds them: bit for bit
on the CPU tape, and within rounding on CUDA. -/
def checkStacked (device : NN.Backend.Device) : IO Unit := do
  let weights : TensorPack Float [[4, 4], [4], [4, 4, 5], [4, 5], [4, 5, 4], [4, 4]] :=
    stackedLayer.initState
  let .cons wr (.cons br (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))))) := weights
  let r {s : Shape} (t : Tensor Float s) : Tensor Float32 s := Tensor.map Float.toFloat32 t
  let wrT : Tensor Float32 [4, 4] := Tensor.dim fun j => Tensor.dim fun e =>
    Tensor.scalar ((Tensor.unstack (r wr) e).getScalar j)
  let router (v : Tensor Float32 [4]) : Tensor Float32 [4] :=
    addSpec (ofRow (matMulSpec (asRow v) wrT)) (r br)
  let expert (e : Fin 4) (v : Tensor Float32 [4]) : Tensor Float32 [4] :=
    let h := Activation.reluSpec
      (addSpec (ofRow (matMulSpec (asRow v) (Tensor.unstack (r w1) e))) (Tensor.unstack (r b1) e))
    addSpec (ofRow (matMulSpec (asRow h) (Tensor.unstack (r w2) e))) (Tensor.unstack (r b2) e)
  let x : Tensor Float [3, 4] := Tensor.dim fun t => Tensor.dim fun j =>
    Tensor.scalar (((t.val * 5 + j.val * 3) % 7 : Nat).toFloat / 2 - 1.5)
  let expected : Tensor Float [3, 4] := Tensor.map Float32.toFloat <| Tensor.dim fun t =>
    Spec.MoE.forwardSpec 2 router expert (Tensor.unstack (r x) t)
  let model : nn.Builder (nn.Sequential [3, 4] [3, 4]) :=
    pure (nn.Sequential.fromLayer stackedLayer)
  let trainer := TorchLean.Trainer.new model { device }
  let session ← trainer.open
  let predicted ← session.predict x
  if device == .cuda then
    Tests.Cuda.Utils.assertTensorApprox "nn.moeMlp on CUDA" predicted expected (tol := 1e-5)
  else
    unless sameBits predicted expected do
      throw <| IO.userError "nn.moeMlp on the CPU tape differs from Spec.MoE.forwardSpec"

def run (device : NN.Backend.Device := .cpu) : IO Unit := do
  checkStacked device
  if device == .cuda then
    checkCudaMask
    checkCudaMix
  else
    checkCpu
  checkRoundedTie device
  IO.println s!"mixture of experts ({repr device}): spec bits, stacked experts, gradients, ties: ok"

end NN.Tests.Runtime.MoE
