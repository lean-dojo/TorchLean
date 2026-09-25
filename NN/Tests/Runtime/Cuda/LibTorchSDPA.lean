/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session
public import NN.Tests.Runtime.Cuda.Attention

/-!
# CUDA Kernel Coverage: LibTorch SDPA

Tests for the checked paired attention API.

The same ABI runs ATen kernels in LibTorch builds and portable C in CPU builds. An independent
two-key reference and finite differences check values and gradients, including a hard boolean mask
with a fully blocked row. TorchLean's global tape owns the attention node and its backward call.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LibTorchSDPA

open Spec TorchLean
open Tests.Cuda.Attention (checked)
open Runtime.Autograd.Cuda

abbrev batch : Nat := 2
abbrev n : Nat := 2
abbrev d : Nat := 2

abbrev s : Shape := [batch, n, d]
abbrev maskShape : Shape := [batch, n, n]

def q : Tensor Float s :=
  (Tensor.from #[
    0.10, -0.20,
    0.30,  0.05,
   -0.15,  0.25,
    0.40, -0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def k : Tensor Float s :=
  (Tensor.from #[
    0.05,  0.20,
   -0.10,  0.30,
    0.15, -0.25,
    0.35,  0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def v : Tensor Float s :=
  (Tensor.from #[
    0.20, -0.05,
    0.10,  0.30,
   -0.20,  0.15,
    0.05, -0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def dOut : Tensor Float s :=
  (Tensor.from #[
    1.00, 0.50,
   -0.25, 0.75,
    0.30, 1.20,
   -0.60, 0.40
  ]).reshape [batch, n, d] (by dsimp; decide)

/-- `1` marks an allowed key. The last query row is fully blocked. -/
def hardMask : Tensor Float maskShape :=
  (Tensor.from #[
    1.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,
    0.0, 0.0
  ]).reshape [batch, n, n] (by dsimp; decide)

/-- For this two-key case, the first probability is the logistic of the score difference. -/
@[no_expose] def referenceForward (qs ks vs : FloatArray) (masked : Bool) (scale : Float) :
    FloatArray := FloatArray.mk <| Id.run do
  let maskValues := Runtime.Autograd.Cuda.Convert.flattenFloat (s := maskShape) hardMask
  let mut values : Array Float := #[]
  for b in [:batch] do
    for i in [:n] do
      let firstAllowed := !masked || maskValues.get! ((b * n + i) * n) != 0.0
      let secondAllowed := !masked || maskValues.get! ((b * n + i) * n + 1) != 0.0
      let mut difference : Float := 0.0
      for c in [:d] do
        difference := difference + qs.get! ((b * n + i) * d + c) *
          (ks.get! (b * n * d + c) - ks.get! ((b * n + 1) * d + c))
      let (p, r) : Float × Float :=
        match firstAllowed, secondAllowed with
        | true, true =>
            let probability := 1.0 / (1.0 + Float.exp (-scale * difference))
            (probability, 1.0 - probability)
        | true, false => (1.0, 0.0)
        | false, true => (0.0, 1.0)
        | false, false => (0.0, 0.0)
      for c in [:d] do
        values := values.push
          (p * vs.get! (b * n * d + c) + r * vs.get! ((b * n + 1) * d + c))
  return values

@[no_expose] def perturb (xs : FloatArray) (index : Nat) (delta : Float) : FloatArray :=
  FloatArray.mk <| (Array.range xs.size).map fun j =>
    xs.get! j + if j = index then delta else 0.0

@[no_expose] def referenceLoss (qs ks vs : FloatArray) (masked : Bool) (scale : Float) : Float :=
  Id.run do
    let ys := referenceForward qs ks vs masked scale
    let seed := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) dOut
    let mut loss : Float := 0.0
    for i in [:ys.size] do
      loss := loss + ys.get! i * seed.get! i
    return loss

/-- Check all raw Q/K/V gradients against an independent reference. -/
@[no_expose] def checkReference (masked : Bool) (scale : Float)
    (output dq dk dv : Runtime.Autograd.Cuda.Buffer) : IO Unit := do
  let qs := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) q
  let ks := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) k
  let vs := Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) v
  Tests.Cuda.Utils.assertFloatArrayApprox "attention two-key reference"
    (Runtime.Autograd.Cuda.Buffer.toFloatArray output)
    (referenceForward qs ks vs masked scale) 1e-4
  let qGrad := Runtime.Autograd.Cuda.Buffer.toFloatArray dq
  let kGrad := Runtime.Autograd.Cuda.Buffer.toFloatArray dk
  let vGrad := Runtime.Autograd.Cuda.Buffer.toFloatArray dv
  let step : Float := 1e-4
  for i in [:qs.size] do
    let expectedQ :=
      (referenceLoss (perturb qs i step) ks vs masked scale -
       referenceLoss (perturb qs i (-step)) ks vs masked scale) / (2.0 * step)
    let expectedK :=
      (referenceLoss qs (perturb ks i step) vs masked scale -
       referenceLoss qs (perturb ks i (-step)) vs masked scale) / (2.0 * step)
    let expectedV :=
      (referenceLoss qs ks (perturb vs i step) masked scale -
       referenceLoss qs ks (perturb vs i (-step)) masked scale) / (2.0 * step)
    Tests.Utils.assertApprox s!"attention finite-difference dQ[{i}]" (qGrad.get! i) expectedQ 1e-4
    Tests.Utils.assertApprox s!"attention finite-difference dK[{i}]" (kGrad.get! i) expectedK 1e-4
    Tests.Utils.assertApprox s!"attention finite-difference dV[{i}]" (vGrad.get! i) expectedV 1e-4

@[no_expose] def expectFailure {α : Type} (message : String) (action : IO α) : IO Unit := do
  let rejected ← do
    try
      discard action
      pure false
    catch _ =>
      pure true
  unless rejected do throw <| IO.userError message

/-- Portable tests check planning; CUDA builds also exercise session selection. -/
@[no_expose] def selectAttention (profile : NN.Backend.BackendProfile) :
    IO NN.Backend.KernelCapsule := do
  let options := Runtime.Autograd.Torch.Config.withBackendProfile
    ({} : Runtime.Autograd.Torch.Config) profile
  match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
  | .cpuStub =>
      let planned ← Tests.Cuda.Utils.okOrThrow <|
        options.planBackendOp .scaledDotProductAttention
      pure planned.capsule
  | _ =>
      let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
      session.selectedCapsule .scaledDotProductAttention

/-- Aligned float32 heads also exercise a shape eligible for the efficient pair, when available. -/
@[no_expose] def checkAlignedPair : IO Unit := do
  let zeros ← Runtime.Autograd.Cuda.Buffer.zerosIO 256
  let values := Runtime.Autograd.Cuda.Buffer.ofFloatArray <|
    FloatArray.mk <| (Array.range 256).map fun i => Float.ofNat (i / 32)
  let maskValues := Runtime.Autograd.Cuda.Buffer.ofFloatArray <|
    FloatArray.mk <| (Array.range 64).map fun i => if i < 8 then 0.0 else 1.0
  let seed := Runtime.Autograd.Cuda.Buffer.full 256 1.0
  let output ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
    zeros zeros values maskValues 1 1 8 32 0.25
  let (dq, dk, dv) ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd output seed
  let ys := Runtime.Autograd.Cuda.Buffer.toFloatArray output
  let dqs := Runtime.Autograd.Cuda.Buffer.toFloatArray dq
  let dks := Runtime.Autograd.Cuda.Buffer.toFloatArray dk
  let dvs := Runtime.Autograd.Cuda.Buffer.toFloatArray dv
  for i in [:256] do
    Tests.Utils.assertApprox s!"aligned attention output[{i}]" (ys.get! i)
      (if i < 32 then 0.0 else 3.5) 1e-4
    Tests.Utils.assertApprox s!"aligned attention dQ[{i}]" (dqs.get! i) 0.0 1e-4
    Tests.Utils.assertApprox s!"aligned attention dK[{i}]" (dks.get! i) 0.0 1e-4
    Tests.Utils.assertApprox s!"aligned attention dV[{i}]" (dvs.get! i) 0.875 1e-4
  for buffer in #[zeros, values, maskValues, seed, output, dq, dk, dv] do
    discard <| Runtime.Autograd.Cuda.Buffer.releaseIO buffer

@[no_expose] def sdpBackends : Array (String × LibTorch.SDPBackend) :=
  #[("flash", .flash), ("efficient", .efficient), ("math", .math), ("cudnn", .cuDNN)]

@[no_expose] def configureProviders (selection : String) : IO Unit := do
  for (name, backend) in sdpBackends do
    LibTorch.setSDPEnabled backend (selection == "auto" || selection == name)

@[no_expose] def withProviderSettings (action : IO Unit) : IO Unit := do
  let flags ← sdpBackends.mapM fun (_, backend) => do
    let enabled ← LibTorch.getSDPEnabled backend
    pure (backend, enabled)
  let precision ← LibTorch.getMatmulPrecision
  try
    LibTorch.setMatmulPrecision .ieee
    action
  finally
    for (backend, enabled) in flags do
      LibTorch.setSDPEnabled backend enabled
    LibTorch.setMatmulPrecision precision

/-- Bounded, nonconstant inputs with aligned head width and nonzero Q/K derivatives. -/
@[no_expose] def providerValues (multiplier offset modulus : Nat) (divisor : Float) :
    FloatArray :=
  FloatArray.mk <| (Array.range 256).map fun i =>
    (Float.ofNat ((i * multiplier + offset) % modulus) -
      Float.ofNat (modulus / 2)) / divisor

@[no_expose] def providerMask (kind : String) : FloatArray :=
  FloatArray.mk <| (Array.range 64).map fun index =>
    let row := index / 8
    let col := index % 8
    if kind == "causal" then
      if col ≤ row then 1.0 else 0.0
    else if kind == "masked-zero" then
      if row != 0 && (row + col) % 3 != 0 then 1.0 else 0.0
    else 1.0

/-- Float64 scalar reference; the bounded fixture keeps all exponentials finite. -/
@[no_expose] def providerReference (qs ks vs seed maskValues : FloatArray) (scale : Float) :
    FloatArray × FloatArray × FloatArray × FloatArray := Id.run do
  let mut ys : Array Float := Array.replicate 256 0.0
  let mut dqs : Array Float := Array.replicate 256 0.0
  let mut dks : Array Float := Array.replicate 256 0.0
  let mut dvs : Array Float := Array.replicate 256 0.0
  for i in [:8] do
    let mut weights : Array Float := Array.replicate 8 0.0
    let mut total : Float := 0.0
    for j in [:8] do
      if maskValues.get! (i * 8 + j) != 0.0 then
        let mut score : Float := 0.0
        for c in [:32] do
          score := score + qs.get! (i * 32 + c) * ks.get! (j * 32 + c)
        let weight := Float.exp (score * scale)
        weights := weights.set! j weight
        total := total + weight
    let probabilities := weights.map fun weight => if total == 0.0 then 0.0 else weight / total
    let mut dp : Array Float := Array.replicate 8 0.0
    let mut rowDot : Float := 0.0
    for j in [:8] do
      let mut derivative : Float := 0.0
      for c in [:32] do
        derivative := derivative + seed.get! (i * 32 + c) * vs.get! (j * 32 + c)
      dp := dp.set! j derivative
      rowDot := rowDot + probabilities[j]! * derivative
    for j in [:8] do
      let probability := probabilities[j]!
      let ds := probability * (dp[j]! - rowDot) * scale
      for c in [:32] do
        let qi := i * 32 + c
        let kj := j * 32 + c
        ys := ys.set! qi (ys[qi]! + probability * vs.get! kj)
        dqs := dqs.set! qi (dqs[qi]! + ds * ks.get! kj)
        dks := dks.set! kj (dks[kj]! + ds * qs.get! qi)
        dvs := dvs.set! kj (dvs[kj]! + probability * seed.get! qi)
  return (FloatArray.mk ys, FloatArray.mk dqs, FloatArray.mk dks, FloatArray.mk dvs)

/-- Only SDK eligibility diagnostics may skip a forced provider's forward call. -/
@[no_expose] def unsupportedProvider (message : String) : Bool :=
  #["No available kernel", "No viable backend", "needs the math SDPA pair"].any
    fun text => (message.splitOn text).length > 1

@[no_expose] def checkProviderFixture (selection maskKind : String) (native : Bool) :
    IO Unit := do
  if native then configureProviders selection
  let qs := providerValues 7 3 29 32.0
  let ks := providerValues 11 5 31 32.0
  let vs := providerValues 13 7 23 16.0
  let seeds := providerValues 5 1 19 16.0
  let masks := providerMask maskKind
  let scale := 1.0 / Float.sqrt 32.0
  let (expectedY, expectedDQ, expectedDK, expectedDV) :=
    providerReference qs ks vs seeds masks scale
  unless (Array.range 256).any (fun i => (expectedDQ.get! i).abs > 1e-6) &&
      (Array.range 256).any (fun i => (expectedDK.get! i).abs > 1e-6) do
    throw <| IO.userError "attention provider fixture must have nonzero Q and K gradients"
  let q ← Buffer.ofFloatArrayIO qs
  let k ← Buffer.ofFloatArrayIO ks
  let v ← Buffer.ofFloatArrayIO vs
  let mask ← Buffer.ofFloatArrayIO masks
  let seed ← Buffer.ofFloatArrayIO seeds
  try
    let result ← IO.lazyPure fun _ =>
      Buffer.libTorchAttentionFwd q k v mask (if maskKind == "unmasked" then 0 else 1)
        1 8 32 scale
    match result with
    | .error message =>
        if native && selection != "auto" && selection != "math" &&
            unsupportedProvider message then
          IO.println s!"UNSUPPORTED attention {selection}/float32/{maskKind}: {message}"
        else throw <| IO.userError message
    | .ok output =>
        try
          -- Eligibility flags affect future forwards, not an already saved backward pair.
          if native then configureProviders "all-disabled"
          for input in #[q, k, v, mask] do
            discard <| Buffer.releaseIO input
          let (dq, dk, dv) ← checked fun _ => Buffer.libTorchAttentionBwd output seed
          try
            for (label, actual, expected) in
                #[("output", output, expectedY), ("dQ", dq, expectedDQ),
                  ("dK", dk, expectedDK), ("dV", dv, expectedDV)] do
              let values ← Buffer.toFloatArrayIO actual
              for i in [:values.size] do
                unless (values.get! i).isFinite do
                  throw <| IO.userError s!"attention {selection}/{maskKind}/{label}[{i}] nonfinite"
              Tests.Cuda.Utils.assertFloatArrayApprox
                s!"attention {selection}/{maskKind}/{label}" values expected 1e-4
            IO.println s!"PASS attention {selection}/float32/{maskKind}, saved-provider backward"
          finally
            for gradient in #[dq, dk, dv] do
              discard <| Buffer.releaseIO gradient
        finally
          discard <| Buffer.releaseIO output
  finally
    for input in #[q, k, v, mask, seed] do
      discard <| Buffer.releaseIO input

@[no_expose] def checkScaleRange : IO Unit := do
  let values ← Buffer.ofFloatArrayIO (FloatArray.mk #[1.0])
  try
    let result ← IO.lazyPure fun _ =>
      Buffer.libTorchAttentionFwd values values values values 0 1 1 1 1e40
    match result with
    | .error message =>
        unless (message.splitOn "scale exceeds the float32 range").length > 1 do
          throw <| IO.userError s!"unexpected attention scale error: {message}"
    | .ok output =>
        discard <| Buffer.releaseIO output
        throw <| IO.userError "attention accepted scale 1e40 outside the float32 contract"
  finally
    discard <| Buffer.releaseIO values

@[no_expose] def checkAllDisabled : IO Unit := do
  configureProviders "all-disabled"
  let values ← Buffer.ofFloatArrayIO (providerValues 7 3 29 32.0)
  try
    let result ← IO.lazyPure fun _ =>
      Buffer.libTorchAttentionFwd values values values values 0 1 8 32 0.25
    match result with
    | .error message =>
        unless unsupportedProvider message do
          throw <| IO.userError s!"unexpected all-disabled attention error: {message}"
        IO.println "PASS attention all-disabled returns Except.error"
    | .ok output =>
        discard <| Buffer.releaseIO output
        throw <| IO.userError "attention executed with all providers disabled"
  finally
    discard <| Buffer.releaseIO values

@[no_expose] def runProviderTests (selection : String) : IO Unit := do
  unless (#["all", "auto", "math", "flash", "efficient", "cudnn", "all-disabled"]).contains
      selection do
    throw <| IO.userError s!"unknown attention provider selector: {selection}"
  match Buffer.runtimeStatus with
  | .cpuStub =>
      unless selection == "all" do
        throw <| IO.userError "forced ATen provider tests require the native CUDA runtime"
      for kind in #["unmasked", "causal", "masked-zero"] do
        checkProviderFixture "portable" kind false
      checkScaleRange
  | .nativeUnavailable =>
      throw <| IO.userError "attention provider tests require an available CUDA device"
  | .nativeAvailable =>
      withProviderSettings do
        if selection == "all" || selection == "all-disabled" then checkAllDisabled
        let selections := if selection == "all" then
          #["math", "auto", "efficient", "flash", "cudnn"] else #[selection]
        for provider in selections do
          unless provider == "all-disabled" do
            for kind in #["unmasked", "causal", "masked-zero"] do
              checkProviderFixture provider kind true
        configureProviders "math"
        checkScaleRange

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: LibTorch SDPA ==="
  Tests.Cuda.Attention.checkPairedBuffers

  let qBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) q)
  let kBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) k)
  let vBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) v)
  let dOutBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) dOut)
  let emptyMask ← Runtime.Autograd.Cuda.Buffer.zerosIO 0

  let batch32 := UInt32.ofNat batch
  let n32 := UInt32.ofNat n
  let d32 := UInt32.ofNat d
  let scale : Float := 1.0 / Float.sqrt (Float.ofNat d)

  let libTorchY ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
    qBuf kBuf vBuf emptyMask 0 batch32 n32 d32 scale
  let (libTorchDQ, libTorchDK, libTorchDV) ← checked fun _ =>
    Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd libTorchY dOutBuf
  checkReference false scale libTorchY libTorchDQ libTorchDK libTorchDV

  let maskBuf := Runtime.Autograd.Cuda.Buffer.ofFloatArray
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := maskShape) hardMask)
  let libTorchMaskedY ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
    qBuf kBuf vBuf maskBuf 1 batch32 n32 d32 scale
  let (libTorchMaskedDQ, libTorchMaskedDK, libTorchMaskedDV) ← checked fun _ =>
    Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd libTorchMaskedY dOutBuf
  checkReference true scale libTorchMaskedY libTorchMaskedDQ libTorchMaskedDK libTorchMaskedDV
  checkAlignedPair

  for testScale in #[0.0, -0.7] do
    let output ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
      qBuf kBuf vBuf maskBuf 1 batch32 n32 d32 testScale
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd output dOutBuf
    checkReference true testScale output dq dk dv
    for buffer in #[output, dq, dk, dv] do
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO buffer

  -- Forward state owns its inputs and supports repeated VJPs until the output is released.
  let savedQ ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <|
    Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) q
  let savedK ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <|
    Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) k
  let savedV ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <|
    Runtime.Autograd.Cuda.Convert.flattenFloat (s := s) v
  let savedMask ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO <|
    Runtime.Autograd.Cuda.Convert.flattenFloat (s := maskShape) hardMask
  let retainedY ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
    savedQ savedK savedV savedMask 1 batch32 n32 d32 scale
  for input in #[savedQ, savedK, savedV, savedMask] do
    discard <| Runtime.Autograd.Cuda.Buffer.releaseIO input
  for _ in [:2] do
    let (dq, dk, dv) ← checked fun _ =>
      Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd retainedY dOutBuf
    checkReference true scale retainedY dq dk dv
    for gradient in #[dq, dk, dv] do
      discard <| Runtime.Autograd.Cuda.Buffer.releaseIO gradient
  discard <| Runtime.Autograd.Cuda.Buffer.releaseIO retainedY
  expectFailure "attention accepted a released forward buffer" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd retainedY dOutBuf
  expectFailure "attention accepted an unrelated buffer as forward state" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd qBuf dOutBuf

  -- Exercise capsule routing and the global TorchLean tape with the paired ATen backward.
  let selectedAttention ← selectAttention NN.Backend.BackendProfile.checkedCuda
  unless selectedAttention.sameIdentity NN.Backend.Attention.libTorchDirectAttention do
    throw <| IO.userError <|
      s!"default checkedCuda selected unexpected attention capsule `{selectedAttention.name}`"
  let compositionProfile : NN.Backend.BackendProfile :=
    { NN.Backend.BackendProfile.checkedCuda with
      policy :=
        { NN.Backend.BackendProfile.checkedCuda.policy with
          provider := .only .torchLean
          vjpMode := .torchLeanTape } }
  let comparisonAttention ← selectAttention compositionProfile
  unless comparisonAttention.sameIdentity NN.Backend.Attention.torchLeanComposed do
    throw <| IO.userError "explicit composition profile did not select composed attention"
  let base0 : Runtime.Autograd.Cuda.Tape := Runtime.Autograd.Cuda.Tape.empty
  let (base1, wqId) := Runtime.Autograd.Cuda.Tape.leaf (t := base0)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wq)
  let (base2, wkId) := Runtime.Autograd.Cuda.Tape.leaf (t := base1)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wk)
  let (base3, wvId) := Runtime.Autograd.Cuda.Tape.leaf (t := base2)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wv)
  let (base4, woId) := Runtime.Autograd.Cuda.Tape.leaf (t := base3)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wo)
  let (base5, xId) := Runtime.Autograd.Cuda.Tape.leaf (t := base4)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.x)
  if Runtime.Autograd.Cuda.Buffer.runtimeStatus == .nativeAvailable then
    withProviderSettings do
      configureProviders "all-disabled"
      let rejected ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := base5)
        (n := Tests.Cuda.Attention.n) (numHeads := Tests.Cuda.Attention.numHeads)
        (dModel := Tests.Cuda.Attention.dModel) (headDim := Tests.Cuda.Attention.headDim)
        (h1 := Tests.Cuda.Attention.n_ne_zero) wqId wkId wvId woId xId
        (mask := some Tests.Cuda.Attention.mask) (attentionCapsule := selectedAttention)
      match rejected with
      | .error message =>
          unless unsupportedProvider message do
            throw <| IO.userError s!"unexpected attention tape error: {message}"
      | .ok _ => throw <| IO.userError "attention tape ignored disabled providers"
  let libTorchTapeResult ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := base5)
    (n := Tests.Cuda.Attention.n) (numHeads := Tests.Cuda.Attention.numHeads)
    (dModel := Tests.Cuda.Attention.dModel) (headDim := Tests.Cuda.Attention.headDim)
    (h1 := Tests.Cuda.Attention.n_ne_zero) wqId wkId wvId woId xId
    (mask := some Tests.Cuda.Attention.mask)
    (attentionCapsule := selectedAttention)
  let (libTorchTape, libTorchOutId) ← Tests.Cuda.Utils.okOrThrow libTorchTapeResult
  let composedTapeResult ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := base5)
    (n := Tests.Cuda.Attention.n) (numHeads := Tests.Cuda.Attention.numHeads)
    (dModel := Tests.Cuda.Attention.dModel) (headDim := Tests.Cuda.Attention.headDim)
    (h1 := Tests.Cuda.Attention.n_ne_zero) wqId wkId wvId woId xId
    (mask := some Tests.Cuda.Attention.mask)
    (attentionCapsule := comparisonAttention)
  let (composedTape, composedOutId) ← Tests.Cuda.Utils.okOrThrow composedTapeResult
  let modelOutShape : Shape :=
    [Tests.Cuda.Attention.n, Tests.Cuda.Attention.dModel]
  let libTorchTapeOut ← Tests.Cuda.Utils.cudaValue
    (s := modelOutShape) libTorchTape libTorchOutId
  let composedTapeOut ← Tests.Cuda.Utils.cudaValue
    (s := modelOutShape) composedTape composedOutId
  Tests.Cuda.Utils.assertTensorApprox (s := modelOutShape)
    "LibTorch attention capsule output" libTorchTapeOut composedTapeOut (tol := 2e-2)
  let seed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := modelOutShape
      buf := Runtime.Autograd.Cuda.Buffer.full
        (UInt32.ofNat (Spec.Shape.size modelOutShape)) 1.0 }
  let libTorchGrads ← Tests.Cuda.Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll libTorchTape libTorchOutId seed
  let composedSeed : Runtime.Autograd.Cuda.AnyBuffer :=
    { s := modelOutShape
      buf := Runtime.Autograd.Cuda.Buffer.full
        (UInt32.ofNat (Spec.Shape.size modelOutShape)) 1.0 }
  let composedGrads ← Tests.Cuda.Utils.okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll composedTape composedOutId composedSeed
  let libTorchDx ← Tests.Cuda.Utils.cudaGrad (s := modelOutShape) libTorchGrads xId
  let composedDx ← Tests.Cuda.Utils.cudaGrad (s := modelOutShape) composedGrads xId
  Tests.Cuda.Utils.assertTensorApprox (s := modelOutShape)
    "LibTorch attention capsule TorchLean backward" libTorchDx composedDx (tol := 2e-2)

  let shortQ ← Runtime.Autograd.Cuda.Buffer.zerosIO 1
  let rejected ← do
    try
      let _ ← checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
        shortQ kBuf vBuf emptyMask 0 batch32 n32 d32 scale
      pure false
    catch _ =>
      pure true
  unless rejected do
    throw <| IO.userError "libtorch sdpa accepted a Q buffer with the wrong size"
  expectFailure "attention accepted a dOut buffer with the wrong size" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionBwd libTorchY shortQ
  expectFailure "attention accepted a mask buffer with the wrong size" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
      qBuf kBuf vBuf shortQ 1 batch32 n32 d32 scale
  expectFailure "attention accepted an invalid hasMask flag" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
      qBuf kBuf vBuf maskBuf 2 batch32 n32 d32 scale
  expectFailure "attention accepted a non-finite scale" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
      qBuf kBuf vBuf emptyMask 0 batch32 n32 d32 (1.0 / 0.0)
  expectFailure "attention accepted an overflowing shape" <|
    checked fun _ => Runtime.Autograd.Cuda.Buffer.libTorchAttentionFwd
      qBuf kBuf vBuf emptyMask 0 4294967295 4294967295 4294967295 scale
  runProviderTests "all"
  IO.println "== LibTorch SDPA bridge: OK =="

@[no_expose] def main (args : List String) : IO Unit :=
  match args with
  | [] => run
  | ["--provider", selection] => runProviderTests selection
  | _ => throw <| IO.userError
      "usage: libtorch_sdpa_test [--provider all|auto|math|flash|efficient|cudnn|all-disabled]"

end LibTorchSDPA
end Cuda
end Tests

@[no_expose] def main (args : List String) : IO Unit :=
  Tests.Cuda.LibTorchSDPA.main args
