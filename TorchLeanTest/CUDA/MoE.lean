/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.MoE
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Session
public import TorchLean.NN.ScalarTrainer.EagerOps
public import TorchLeanTest.Harness.CUDA

/-!
# Mixture-of-experts runtime checks

The gating and mixing of `nn.moe` on LibTorch's CUDA backend, on tokens whose experts are decided
by ties, including a tie between `+0` and `-0`. The routing mask has the bits of
`MoE.topKMaskInnermost` on rows full of ties, signed zeros and NaNs, and the result and both
gradients match the CPU tape. `nn.moeMLP` predicts what `MoE.forward` computes within
rounding, and the mask of a scalar keeps it when `0 < k` and passes no gradient.

The last check is the binary32 example of `TorchLean.MoE`: the logits `0` and `2⁻³⁰` round to
the same probability on the host, and routing on the GPU's probabilities keeps the first expert.
-/

@[expose] public section

namespace TorchLean.Test.MoE

open Backend.Device TorchLean.Tensor

/-- Router logits for three tokens and four experts. The second token's first, second and fourth
logits tie, and the third token's first, second and fourth are `0`, `-0` and `-0`. -/
def logits : Tensor Float [3, 4] :=
  (Tensor.from (#[0.5, -1.0, 2.0, 0.25, 1.0, 1.0, 0.0, 1.0, 0.0, -0.0, -3.0, -0.0] :
    Array Float)).reshape [3, 4] (by decide)

/-- Outputs of the four experts for the three tokens, of width two. -/
def outputs : Tensor Float [4, 3, 2] :=
  (Tensor.from (Array.ofFn (n := 24) fun i => ((i.val * 7 % 11 : Nat).toFloat - 5) / 4)).reshape
    [4, 3, 2] (by decide)

/-- Run the gating and mixing on `device`, returning the result and the gradients of its sum with
respect to the logits and the expert outputs. -/
def mix (device : Backend.Device) (k : Nat) :
    IO (Tensor Float [3, 2] × Tensor Float [3, 4] × Tensor Float [4, 3, 2]) := do
  let session ← Session.create (α := Float) { device }
  let l ← session.input logits (requiresGrad := true)
  let o ← session.input outputs (requiresGrad := true)
  let y ← (nn.moeMix (m := SessionM Float) (α := Float) k l o).run session
  unless (← session.backendSelections).any (fun selected => selected.op.tag == "topKMask") do
    throw <| IO.userError "MoE routing bypassed backend selection and audit"
  let value ← session.getValue y
  let grads ← session.backwardDenseAll y (Tensor.full [3, 2] 1.0)
  pure (value, ← Session.grad grads l, ← Session.grad grads o)

/-- Equal as bit patterns, entry by entry. -/
def sameBits {s : Shape} (x y : Tensor Float s) : Bool :=
  let ax := x.to FloatArray
  let ay := y.to FloatArray
  ax.size == ay.size && (List.range ax.size).all fun i => (ax.get! i).toBits == (ay.get! i).toBits

/-- On a row of one entry, the top-`k` mask on `device` is one when `0 < k` and zero otherwise, a
NaN included, and the recorded pullback of `mask * x` gives `x` the mask's value, NaN inputs too: no
gradient flows through the mask. -/
def checkSingleEntryMask (device : Backend.Device) : IO Unit := do
  for (k, kept, value) in [(0, (0.0 : Float), (2.5 : Float)), (1, 1.0, 2.5), (3, 1.0, 2.5),
      (0, 0.0, Float.ofBits 0x7ff8000000000000), (1, 1.0, Float.ofBits 0x7ff8000000000000),
      (3, 1.0, Float.ofBits 0x7ff8000000000000)] do
    let session ← Session.create (α := Float) { device }
    let x ← session.input (Tensor.full [1] value) (requiresGrad := true)
    let mask ← (Ops.apply (m := SessionM Float) (α := Float)
      (.topKMask (s := [1]) ⟨0, by decide⟩ k) (x, ())).run session
    unless ((← session.getValue mask).getScalar 0).toBits == kept.toBits do
      throw <| IO.userError s!"MoE: top-{k} mask of one entry on {repr device} is not {kept}"
    let y ← (Ops.apply (m := SessionM Float) (α := Float) .mul
      (mask, x, ())).run session
    let grads ← session.backwardDenseAll y (Tensor.full [1] 1.0)
    unless ((← Session.grad grads x).getScalar 0).toBits == kept.toBits do
      throw <| IO.userError s!"MoE: a gradient reached an entry through its top-{k} mask"

/-- Rows of six entries drawn from `-1, 0, -0, 1, 2` and NaN, with between two and six different
values in a row, so that most rows tie. -/
def tiedRows : Tensor Float [64, 6] :=
  let pool : Array Float := #[-1.0, 0.0, -0.0, 1.0, 2.0, (0.0 : Float) / 0.0]
  Tensor.dim fun r => Tensor.dim fun j =>
    Tensor.scalar pool[(r.val * 7 + j.val * (r.val % 4 + 1)) % 6]!

/-- The CUDA mask has the spec's bits for every `k`, on rows of ties, signed zeros and NaNs. -/
def checkCUDAMask : IO Unit := do
  for k in List.range 8 do
    let session ← Session.create (α := Float) { device := cuda }
    let x ← session.input tiedRows
    let mask ← session.getValue (← session.apply "top_k_mask"
      (.topKMask (s := [64, 6]) ⟨1, by decide⟩ k) (x, ()))
    unless sameBits mask (MoE.topKMaskInnermost k tiedRows) do
      throw <| IO.userError s!"MoE: CUDA top-{k} mask differs from TorchLean.MoE.topKMaskInnermost"

/-- On CUDA, the result and both gradients match the CPU tape. -/
def checkCUDAMix : IO Unit := do
  for k in [1, 2, 4] do
    let (value, dLogits, dOutputs) ← mix .cuda k
    let (value', dLogits', dOutputs') ← mix .cpu k
    Harness.assertTensorApprox s!"MoE k={k}: CUDA mixing" value value' (tol := 1e-12)
    Harness.assertTensorApprox s!"MoE k={k}: CUDA logit gradient" dLogits dLogits'
      (tol := 1e-12)
    Harness.assertTensorApprox s!"MoE k={k}: CUDA output gradient" dOutputs dOutputs'
      (tol := 1e-12)

/-- In binary32 the logits `0` and `2⁻³⁰` have the same softmax probability. -/
def checkRoundedTie (device : Backend.Device) : IO Unit := do
  let close : Tensor Float32 [2] := Tensor.dim fun j =>
    Tensor.scalar (if j.val == 0 then 0 else (1 : Float32) / 1073741824)
  let probabilities := Tensor.softmax close ⟨0, Nat.one_pos⟩
  unless probabilities.getScalar 0 == probabilities.getScalar 1 do
    throw <| IO.userError "MoE: binary32 probabilities of 0 and 2^-30 differ"
  if device == .cuda then
    let session ← Session.create (α := Float32) { device := cuda }
    let x ← session.input close
    let p ← session.apply "softmax" (.softmax (s := [2]) ⟨0, Nat.one_pos⟩) (x, ())
    let mask ← session.getValue (← session.apply "top_k_mask"
      (.topKMask (s := [2]) ⟨0, by decide⟩ 1) (p, ()))
    unless mask.getScalar 0 == 1 && mask.getScalar 1 == 0 do
      throw <| IO.userError "MoE: CUDA routing on binary32 probabilities should keep expert 0"

/-- Four stacked experts of 4-5-4, two kept per token, on three tokens, built from seed 11. -/
def stacked (activation : Activation.Kind) : nn.Sequential [3, 4] [3, 4] :=
  nn.build 11 (nn.moeMLP (tokens := 3) (width := 4) (experts := 4) 2 5 activation)

/-- The tensor specification of each expert activation. -/
def activate {s : Shape} : Activation.Kind → Tensor Float32 s → Tensor Float32 s
  | .relu => Tensor.relu
  | .geluTanh => Tensor.map Activation.geluTanh
  | .silu => Tensor.map Activation.silu
  | .tanh => Tensor.tanh
  | .sigmoid => Tensor.sigmoid

/-- A vector as a one-row matrix. -/
def asRow {α : Type} [Storage α] {n : Nat} (v : Tensor α [n]) : Tensor α [1, n] :=
  Tensor.dim fun _ => v

/-- The row of a one-row matrix. -/
def ofRow {α : Type} [Storage α] {n : Nat} (A : Tensor α [1, n]) : Tensor α [n] :=
  Tensor.unstack A 0

/-- `nn.moeMLP` predicts what `MoE.forward` computes from the built model's initial
weights, with the router and the experts written as the spec's matrix products. The trainer runs in
binary32, so the spec runs there too, on the weights and input rounded once as the trainer rounds
them, and the CUDA prediction agrees within rounding. -/
def checkStacked (device : Backend.Device) (activation : Activation.Kind) : IO Unit := do
  let model := stacked activation
  let weights : TensorPack Float [[4, 4], [4], [4, 4, 5], [4, 5], [4, 5, 4], [4, 4]] ←
    if h : model.stateShapes = [[4, 4], [4], [4, 4, 5], [4, 5], [4, 5, 4], [4, 4]] then
      pure (h ▸ model.initState)
    else
      throw <| IO.userError s!"nn.moeMlp has parameters {model.stateShapes}"
  let .cons wr (.cons br (.cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))))) := weights
  let r {s : Shape} (t : Tensor Float s) : Tensor Float32 s := Tensor.map Float.toFloat32 t
  let wrT : Tensor Float32 [4, 4] := Tensor.dim fun j => Tensor.dim fun e =>
    Tensor.scalar ((Tensor.unstack (r wr) e).getScalar j)
  let router (v : Tensor Float32 [4]) : Tensor Float32 [4] :=
    ofRow (Tensor.matmul (asRow v) wrT) + r br
  let expert (e : Fin 4) (v : Tensor Float32 [4]) : Tensor Float32 [4] :=
    let h := activate activation
      ((ofRow
          (Tensor.matmul (asRow v) (Tensor.unstack (r w1) e))) + Tensor.unstack (r b1) e)
    ofRow (Tensor.matmul (asRow h) (Tensor.unstack (r w2) e)) + Tensor.unstack (r b2) e
  let x : Tensor Float [3, 4] := Tensor.dim fun t => Tensor.dim fun j =>
    Tensor.scalar (((t.val * 5 + j.val * 3) % 7 : Nat).toFloat / 2 - 1.5)
  let expected : Tensor Float [3, 4] := Tensor.map Float32.toFloat <| Tensor.dim fun t =>
    MoE.forward 2 router expert (Tensor.unstack (r x) t)
  let trainer := Trainer.create model { device }
  let session ← trainer.open
  let predicted ← session.predict x
  if device == .cuda then
    Harness.assertTensorApprox s!"nn.moeMlp {reprStr activation} on CUDA" predicted
      expected (tol := 1e-5)

/-- Run `checkStacked` for five activations, `checkRoundedTie` and `checkSingleEntryMask` on
`device`, and `checkCUDAMask` and `checkCUDAMix` when `device` is CUDA. -/
def run (device : Backend.Device) : IO Unit := do
  for activation in [.relu, .geluTanh, .silu, .tanh, .sigmoid] do
    checkStacked device activation
  if device == .cuda then
    checkCUDAMask
    checkCUDAMix
  checkRoundedTie device
  checkSingleEntryMask device
  IO.println s!"mixture of experts ({repr device}): spec bits, stacked experts, gradients, ties: ok"

end TorchLean.Test.MoE
