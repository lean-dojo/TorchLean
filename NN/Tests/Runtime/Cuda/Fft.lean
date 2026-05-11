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

The CUDA backend uses cuFFT. The non-CUDA build uses a direct CPU DFT stub. These tests are
intentionally about the runtime buffer contract, not yet about an autograd-facing spectral layer.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Fft

open Runtime.Autograd.Cuda
open Spec

def floatArrayOfList (xs : List Float) : FloatArray :=
  FloatArray.mk xs.toArray

def assertFloatArrayApprox (msg : String) (a b : FloatArray) (tol : Float := 1e-4) : IO Unit := do
  if a.size != b.size then
    throw <| IO.userError s!"{msg}: size mismatch ({a.size} vs {b.size})"
  for i in [:a.size] do
    Utils.assertApprox s!"{msg}[{i}]" (a.get! i) (b.get! i) tol

def dotFloatArray (a b : FloatArray) : Float := Id.run do
  let mut acc := 0.0
  let n := min a.size b.size
  for i in [:n] do
    acc := acc + a.get! i * b.get! i
  pure acc

def perturbList : List Float → Nat → Float → List Float
  | [], _, _ => []
  | x :: xs, 0, delta => (x + delta) :: xs
  | x :: xs, i + 1, delta => x :: perturbList xs i delta

def spectralConvLoss
    (x wRe wIm dY : List Float) (grid width modes : UInt32) : Float :=
  let y :=
    Buffer.spectralConv1dRfftFwd
      (Buffer.ofFloatArray (floatArrayOfList x))
      (Buffer.ofFloatArray (floatArrayOfList wRe))
      (Buffer.ofFloatArray (floatArrayOfList wIm))
      grid width modes
  dotFloatArray (Buffer.toFloatArray y) (floatArrayOfList dY)

def assertFiniteDiff
    (msg : String) (analytic : FloatArray) (idx : Nat) (fd : Float) (tol : Float) : IO Unit := do
  Utils.assertApprox s!"{msg}[{idx}]" (analytic.get! idx) fd tol

def runKnownSpectrum : IO Unit := do
  IO.println "== rfft1d packed known spectra =="

  -- Two rows, n=4. The second row catches the sign convention:
  -- DFT([1,2,3,4]) at k=1 is `-2 + 2i` for the `exp(-2*pi*i*k*t/n)` convention used by cuFFT.
  let x := Buffer.ofFloatArray (floatArrayOfList [
    1.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0, 4.0
  ])
  let got := Buffer.toFloatArray (Buffer.rfft1dPacked x 2 4)
  let expected := floatArrayOfList [
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -2.0, 2.0, -2.0, 0.0
  ]
  assertFloatArrayApprox "rfft1dPacked known" got expected (tol := 1e-4)

def runRoundtripEvenOdd : IO Unit := do
  IO.println "== rfft1d/irfft1d packed roundtrip =="

  -- Even and odd lengths exercise different Nyquist-bin handling. cuFFT's inverse is
  -- unnormalized, so the runtime wrapper scales by `1/n` before returning.
  let even := floatArrayOfList [
    0.25, -0.50, 1.00, 0.75, -1.25, 0.50, 0.125, -0.875,
    -0.30, 0.20, 0.90, -0.10, 0.45, -0.65, 1.10, -0.95
  ]
  let evenBuf := Buffer.ofFloatArray even
  let evenBack := Buffer.toFloatArray (Buffer.irfft1dPacked (Buffer.rfft1dPacked evenBuf 2 8) 2 8)
  assertFloatArrayApprox "rfft/irfft roundtrip even" evenBack even (tol := 2e-4)

  let odd := floatArrayOfList [
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
  let x := floatArrayOfList [
    0.25, -0.50,
    1.00, 0.75,
    -1.25, 0.50,
    0.125, -0.875
  ]
  let wRe := floatArrayOfList [
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0
  ]
  let wIm := floatArrayOfList (List.replicate 12 0.0)
  let got :=
    Buffer.toFloatArray
      (Buffer.spectralConv1dRfftFwd
        (Buffer.ofFloatArray x)
        (Buffer.ofFloatArray wRe)
        (Buffer.ofFloatArray wIm)
        4 2 3)
  assertFloatArrayApprox "spectralConv1dRfft identity" got x (tol := 3e-4)

def runSpectralConvFiniteDiff : IO Unit := do
  IO.println "== spectralConv1dRfft backward finite differences =="

  -- This validates the explicit VJP kernels against the scalar pairing
  --   L(x,w) = sum(spectralConv1dRfft(x,w) * dY).
  -- The half-spectrum adjoint has subtle `2/n` factors for interior frequencies, so this test is
  -- intentionally small and direct rather than relying on only shape-level tape coverage.
  let grid : UInt32 := 4
  let width : UInt32 := 1
  let modes : UInt32 := 3
  let x : List Float := [0.20, -0.40, 0.70, 1.10]
  let wRe : List Float := [0.75, -0.30, 0.20]
  let wIm : List Float := [0.00, 0.45, 0.00]
  let dY : List Float := [1.00, -0.50, 0.25, 0.75]
  let eps := 1e-2
  let tol := 2e-2

  let xBuf := Buffer.ofFloatArray (floatArrayOfList x)
  let wReBuf := Buffer.ofFloatArray (floatArrayOfList wRe)
  let wImBuf := Buffer.ofFloatArray (floatArrayOfList wIm)
  let dYBuf := Buffer.ofFloatArray (floatArrayOfList dY)
  let dX :=
    Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdX xBuf wReBuf wImBuf dYBuf grid width modes)
  let dWRe :=
    Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdWRe xBuf wReBuf wImBuf dYBuf grid width modes)
  let dWIm :=
    Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdWIm xBuf wReBuf wImBuf dYBuf grid width modes)

  for i in [:x.length] do
    let lp := spectralConvLoss (perturbList x i eps) wRe wIm dY grid width modes
    let lm := spectralConvLoss (perturbList x i (-eps)) wRe wIm dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dX" dX i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wRe.length] do
    let lp := spectralConvLoss x (perturbList wRe i eps) wIm dY grid width modes
    let lm := spectralConvLoss x (perturbList wRe i (-eps)) wIm dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dWRe" dWRe i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wIm.length] do
    let lp := spectralConvLoss x wRe (perturbList wIm i eps) dY grid width modes
    let lm := spectralConvLoss x wRe (perturbList wIm i (-eps)) dY grid width modes
    assertFiniteDiff "spectralConv1dRfft dWIm" dWIm i ((lp - lm) / (2.0 * eps)) tol

def runSpectralConvTapeNode : IO Unit := do
  IO.println "== spectralConv1dRfft CUDA tape node =="

  -- This is the autograd-facing smoke test: the tape node should return the same forward value and
  -- parent cotangents as the direct low-level fused VJP primitives.
  let xShape : Shape := .dim 4 (.dim 1 .scalar)
  let wShape : Shape := .dim 3 (.dim 1 (.dim 1 .scalar))
  let xA := floatArrayOfList [0.20, -0.40, 0.70, 1.10]
  let wReA := floatArrayOfList [0.75, -0.30, 0.20]
  let wImA := floatArrayOfList [0.00, 0.45, 0.00]
  let dYA := floatArrayOfList [1.00, -0.50, 0.25, 0.75]
  let xB := Buffer.ofFloatArray xA
  let wReB := Buffer.ofFloatArray wReA
  let wImB := Buffer.ofFloatArray wImA
  let dYB := Buffer.ofFloatArray dYA

  let (t1, xId) := Tape.empty.leaf { s := xShape, buf := xB } (some "x")
  let (t2, wReId) := t1.leaf { s := wShape, buf := wReB } (some "wRe")
  let (t3, wImId) := t2.leaf { s := wShape, buf := wImB } (some "wIm")
  let (t4, yId) ← Utils.okOrThrow <|
    Tape.spectralConv1dRfft (grid := 4) (width := 1) (modes := 3) (t := t3) xId wReId wImId

  let y ← Utils.okOrThrow <| Tape.requireValue (t := t4) yId xShape
  let directY := Buffer.spectralConv1dRfftFwd xB wReB wImB 4 1 3
  assertFloatArrayApprox "spectralConv1dRfft tape forward"
    (Buffer.toFloatArray y) (Buffer.toFloatArray directY) (tol := 2e-4)

  let grads ← Utils.okOrThrow <|
    Tape.backwardDenseAll (t := t4) yId { s := xShape, buf := dYB }
  let dX ← Utils.cudaGrad (s := xShape) grads xId
  let dWRe ← Utils.cudaGrad (s := wShape) grads wReId
  let dWIm ← Utils.cudaGrad (s := wShape) grads wImId
  assertFloatArrayApprox "spectralConv1dRfft tape dX"
    (_root_.Runtime.Autograd.Cuda.Convert.flattenFloat (s := xShape) dX)
    (Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdX xB wReB wImB dYB 4 1 3))
    (tol := 2e-4)
  assertFloatArrayApprox "spectralConv1dRfft tape dWRe"
    (_root_.Runtime.Autograd.Cuda.Convert.flattenFloat (s := wShape) dWRe)
    (Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdWRe xB wReB wImB dYB 4 1 3))
    (tol := 2e-4)
  assertFloatArrayApprox "spectralConv1dRfft tape dWIm"
    (_root_.Runtime.Autograd.Cuda.Convert.flattenFloat (s := wShape) dWIm)
    (Buffer.toFloatArray (Buffer.spectralConv1dRfftBwdWIm xB wReB wImB dYB 4 1 3))
    (tol := 2e-4)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: real FFT ==="
  runKnownSpectrum
  runRoundtripEvenOdd
  runSpectralConvIdentity
  runSpectralConvFiniteDiff
  runSpectralConvTapeNode

end Fft
end Cuda
end Tests
