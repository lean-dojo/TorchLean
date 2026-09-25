/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: Real FFT

Low-level coverage for the packed real FFT buffer primitives:

- `Buffer.rfft1dPacked`: `(batch, n)` real float32 rows to `(batch, n/2+1, 2)`,
- `Buffer.irfft1dPacked`: packed half-spectrum back to normalized real rows.

The CUDA backend calls LibTorch's `at::fft_rfft` and `at::fft_irfft`. These tests check the runtime
buffer contract; autograd-facing spectral layers are covered separately.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Fft

open Runtime.Autograd.Cuda
open Spec TorchLean

-- Buffer comparisons and the `floatArray` literal wrapper come from `Cuda.Utils`. The tolerances
-- here are loose by the standards of the other CUDA suites because a packed real FFT accumulates
-- float32 rounding across every butterfly stage.
open Tests.Cuda.Utils (floatArray assertFloatArrayApprox)

def dotFloatArray (a b : FloatArray) : Float := Id.run do
  let mut acc := 0.0
  let n := min a.size b.size
  for i in [:n] do
    acc := acc + a.get! i * b.get! i
  pure acc

def perturbArray (xs : Array Float) (i : Nat) (delta : Float) : Array Float :=
  match xs[i]? with
  | none => xs
  | some x => xs.setIfInBounds i (x + delta)

def spectralConvLoss
    (x wRe wIm dY : Array Float) (grid width modes : UInt32) : Float :=
  let y :=
    Buffer.spectralConv1dRfftFwd
      (Buffer.ofFloatArray (floatArray x))
      (Buffer.ofFloatArray (floatArray wRe))
      (Buffer.ofFloatArray (floatArray wIm))
      grid width modes
  dotFloatArray (Buffer.toFloatArray y) (floatArray dY)

def assertFiniteDiff
    (msg : String) (analytic : FloatArray) (idx : Nat) (fd : Float) (tol : Float) : IO Unit := do
  Utils.assertApprox s!"{msg}[{idx}]" (analytic.get! idx) fd tol

def runKnownSpectrum : IO Unit := do
  IO.println "== rfft1d packed known spectra =="

  -- Two rows, n=4. The second row catches the sign convention:
  -- DFT([1,2,3,4]) at k=1 is `-2 + 2i` for the `exp(-2*pi*i*k*t/n)` convention used by cuFFT.
  let x := Buffer.ofFloatArray (floatArray #[
    1.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0, 4.0
  ])
  let got := Buffer.toFloatArray (Buffer.rfft1dPacked x 2 4)
  let expected := floatArray #[
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -2.0, 2.0, -2.0, 0.0
  ]
  assertFloatArrayApprox "rfft1dPacked known" got expected (tol := 1e-4)

def runRoundtripEvenOdd : IO Unit := do
  IO.println "== rfft1d/irfft1d packed roundtrip =="

  -- Even and odd lengths exercise different Nyquist-bin handling. cuFFT's inverse is
  -- unnormalized, so the runtime wrapper scales by `1/n` before returning.
  let even := floatArray #[
    0.25, -0.50, 1.00, 0.75, -1.25, 0.50, 0.125, -0.875,
    -0.30, 0.20, 0.90, -0.10, 0.45, -0.65, 1.10, -0.95
  ]
  let evenBuf := Buffer.ofFloatArray even
  let evenBack := Buffer.toFloatArray (Buffer.irfft1dPacked (Buffer.rfft1dPacked evenBuf 2 8) 2 8)
  assertFloatArrayApprox "rfft/irfft roundtrip even" evenBack even (tol := 2e-4)

  let odd := floatArray #[
    0.10, 0.30, -0.20, 0.70, -0.40,
    -0.60, 0.80, 0.15, -0.25, 0.55
  ]
  let oddBuf := Buffer.ofFloatArray odd
  let oddBack := Buffer.toFloatArray (Buffer.irfft1dPacked (Buffer.rfft1dPacked oddBuf 2 5) 2 5)
  assertFloatArrayApprox "rfft/irfft roundtrip odd" oddBack odd (tol := 2e-4)

def runSpectralConvIdentity : IO Unit := do
  IO.println "== spectralConv1dRfft identity/full-spectrum check =="

  -- With all retained RFFT bins and identity channel weights, the fused spectral convolution is
  -- exactly `irfft(rfft(x))`, so it should return the input up to float32/cuFFT roundoff.
  let x := floatArray #[
    0.25, -0.50,
    1.00, 0.75,
    -1.25, 0.50,
    0.125, -0.875
  ]
  let wRe := floatArray #[
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0
  ]
  let wIm := floatArray (Array.replicate 12 0.0)
  let got :=
    Buffer.toFloatArray
      (Buffer.spectralConv1dRfftFwd
        (Buffer.ofFloatArray x)
        (Buffer.ofFloatArray wRe)
        (Buffer.ofFloatArray wIm)
        4 2 3)
  assertFloatArrayApprox "spectralConv1dRfft identity" got x (tol := 3e-4)

def checkSpectralConvFiniteDiff (grid width modes : UInt32)
    (x wRe wIm dY : Array Float) : IO Unit := do
  -- This validates the explicit VJP kernels against the scalar pairing
  --   L(x,w) = sum(spectralConv1dRfft(x,w) * dY).
  -- The half-spectrum adjoint has subtle `2/n` factors for interior frequencies, so this test
  -- checks the numeric VJP directly instead of relying only on shape-level tape coverage.
  let eps := 1e-2
  let tol := 2e-2

  let xBuf ← Buffer.ofFloatArrayIO (floatArray x)
  let wReBuf ← Buffer.ofFloatArrayIO (floatArray wRe)
  let wImBuf ← Buffer.ofFloatArrayIO (floatArray wIm)
  let dYBuf ← Buffer.ofFloatArrayIO (floatArray dY)
  let before ← Buffer.allocatorStats
  let (dXBuf, dWReBuf, dWImBuf) ← IO.lazyPure fun _ =>
    Buffer.spectralConv1dRfftBwd xBuf wReBuf wImBuf dYBuf grid width modes
  let after ← Buffer.allocatorStats
  let expectedAllocations : UInt64 := if modes == 0 then 1 else 3
  unless after.allocCount - before.allocCount == expectedAllocations do
    throw <| IO.userError "spectral backward allocated discarded gradient payloads"
  let expectedBytes := 4 * (x.size + wRe.size + wIm.size)
  unless after.liveBytes.toNat == before.liveBytes.toNat + expectedBytes do
    throw <| IO.userError "spectral backward retained temporary tensor payloads"
  let dX ← Buffer.toFloatArrayIO dXBuf
  let dWRe ← Buffer.toFloatArrayIO dWReBuf
  let dWIm ← Buffer.toFloatArrayIO dWImBuf
  unless dX.size == x.size && dWRe.size == wRe.size && dWIm.size == wIm.size do
    throw <| IO.userError "spectral backward returned incorrect gradient sizes"
  for buffer in #[dXBuf, dWReBuf, dWImBuf] do
    discard <| Buffer.releaseIO buffer
  let retired ← Buffer.allocatorStats
  unless retired.liveBytes == before.liveBytes do
    throw <| IO.userError "spectral backward did not retire its result payloads"
  for buffer in #[xBuf, wReBuf, wImBuf, dYBuf] do
    discard <| Buffer.releaseIO buffer

  for i in [:x.size] do
    let lp := spectralConvLoss (perturbArray x i eps) wRe wIm dY grid width modes
    let lm := spectralConvLoss (perturbArray x i (-eps)) wRe wIm dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dX" dX i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wRe.size] do
    let lp := spectralConvLoss x (perturbArray wRe i eps) wIm dY grid width modes
    let lm := spectralConvLoss x (perturbArray wRe i (-eps)) wIm dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dWRe" dWRe i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wIm.size] do
    let lp := spectralConvLoss x wRe (perturbArray wIm i eps) dY grid width modes
    let lm := spectralConvLoss x wRe (perturbArray wIm i (-eps)) dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dWIm" dWIm i ((lp - lm) / (2.0 * eps)) tol

def runSpectralConvFiniteDiff : IO Unit := do
  IO.println "== spectralConv1dRfft backward finite differences and payload lifetime =="
  checkSpectralConvFiniteDiff 4 1 3
    #[0.20, -0.40, 0.70, 1.10] #[0.75, -0.30, 0.20]
    #[0.00, 0.45, 0.00] #[1.00, -0.50, 0.25, 0.75]
  -- Width two detects channel transposition; odd grids distinguish the final bin from Nyquist.
  for (grid, modes) in #[(1, 1), (4, 0), (4, 2), (4, 3), (5, 0), (5, 2), (5, 3)] do
    let values := fun (count phase : Nat) =>
      (Array.range count).map fun i => Float.ofNat ((i * 7 + phase) % 17) / 10.0 - 0.8
    IO.println s!"  grid={grid}, width=2, modes={modes}"
    checkSpectralConvFiniteDiff (UInt32.ofNat grid) 2 (UInt32.ofNat modes)
      (values (grid * 2) 1) (values (modes * 4) 3) (values (modes * 4) 5)
      (values (grid * 2) 9)

def runSpectralConvValidation : IO Unit := do
  let before ← Buffer.allocatorStats
  for (grid, width) in #[(0, 2), (4, 0)] do
    match Tape.Internal.spectralConv1dRfft
        (grid := grid) (width := width) (modes := 0) (t := Tape.empty) 0 0 0 with
    | .error message =>
        unless message.contains "must be positive" do
          throw <| IO.userError s!"spectral dimension validation: {message}"
    | .ok _ => throw <| IO.userError "spectral convolution accepted an empty grid or width"
  let after ← Buffer.allocatorStats
  unless after.allocCount == before.allocCount do
    throw <| IO.userError "spectral dimension validation allocated tensor payloads"

def runSpectralConvTapeNode : IO Unit := do
  IO.println "== spectralConv1dRfft CUDA tape node =="

  -- This is the autograd-facing runtime check: the tape node should return the same forward value
  -- and parent cotangents as the direct low-level fused VJP primitives.
  let xShape : Shape := [4, 1]
  let wShape : Shape := [3, 1, 1]
  let xA := floatArray #[0.20, -0.40, 0.70, 1.10]
  let wReA := floatArray #[0.75, -0.30, 0.20]
  let wImA := floatArray #[0.00, 0.45, 0.00]
  let dYA := floatArray #[1.00, -0.50, 0.25, 0.75]
  let xB := Buffer.ofFloatArray xA
  let wReB := Buffer.ofFloatArray wReA
  let wImB := Buffer.ofFloatArray wImA
  let dYB := Buffer.ofFloatArray dYA

  let (t1, xId) := Tape.empty.leaf { s := xShape, buf := xB } (some "x")
  let (t2, wReId) := t1.leaf { s := wShape, buf := wReB } (some "wRe")
  let (t3, wImId) := t2.leaf { s := wShape, buf := wImB } (some "wIm")
  let (t4, yId) ← Utils.okOrThrow <|
    Tape.Internal.spectralConv1dRfft
      (grid := 4) (width := 1) (modes := 3) (t := t3) xId wReId wImId

  let y ← Utils.okOrThrow <| Tape.requireValue (t := t4) yId xShape
  let directY := Buffer.spectralConv1dRfftFwd xB wReB wImB 4 1 3
  assertFloatArrayApprox "spectralConv1dRfft tape forward"
    (Buffer.toFloatArray y) (Buffer.toFloatArray directY) (tol := 2e-4)

  let grads ← Utils.okOrThrow <|
    Tape.backwardDenseAll (t := t4) yId { s := xShape, buf := dYB }
  let dX ← Utils.cudaGrad (s := xShape) grads xId
  let dWRe ← Utils.cudaGrad (s := wShape) grads wReId
  let dWIm ← Utils.cudaGrad (s := wShape) grads wImId
  let (directDX, directDWRe, directDWIm) := Buffer.spectralConv1dRfftBwd xB wReB wImB dYB 4 1 3
  assertFloatArrayApprox "spectralConv1dRfft tape dX"
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := xShape) dX)
    (Buffer.toFloatArray directDX)
    (tol := 2e-4)
  assertFloatArrayApprox "spectralConv1dRfft tape dWRe"
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := wShape) dWRe)
    (Buffer.toFloatArray directDWRe)
    (tol := 2e-4)
  assertFloatArrayApprox "spectralConv1dRfft tape dWIm"
    (Runtime.Autograd.Cuda.Convert.flattenFloat (s := wShape) dWIm)
    (Buffer.toFloatArray directDWIm)
    (tol := 2e-4)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: real FFT ==="
  runKnownSpectrum
  runRoundtripEvenOdd
  runSpectralConvIdentity
  runSpectralConvFiniteDiff
  runSpectralConvValidation
  runSpectralConvTapeNode

end Fft
end Cuda
end Tests
