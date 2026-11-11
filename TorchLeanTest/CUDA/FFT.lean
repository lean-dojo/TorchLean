/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Functional.Spectral
public import TorchLean.NN.Session.Optimizer
public import TorchLean.NN.ScalarTrainer.EagerOps
public import TorchLeanTest.Harness.CUDA

/-!
# CUDA Kernel Coverage: Real FFT

Low-level coverage of the packed real FFT buffer primitives:

- `Buffer.rfftPacked`: `(batch, n)` real float32 rows to `(batch, n/2+1, 2)`,
- `Buffer.irfftPacked`: packed half-spectrum back to normalized real rows,
- `Buffer.rfftAdjoint`: the adjoint of the real transform, at both ends of the binary32 range.

The CUDA backend calls ATen's `aten::fft_rfft` and, for the inverse, `aten::_fft_c2r`, the
operator inside `fft_irfft`. The spectral convolution checks run the public Lean composition,
compare its values and pullbacks with the dense reference on both devices, and check the pullbacks
against finite differences on odd and even grids, with two channels and with no retained modes.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace FFT

open LibTorch Buffer

open Harness (assertFloatArrayApprox)

/-- The dot product of `a` and `b`, failing unless they have one size. -/
def dotFloatArray (a b : FloatArray) : IO Float := do
  unless a.size == b.size do
    throw <| IO.userError "spectral finite-difference loss: output/seed length mismatch"
  let mut acc := 0.0
  for i in [:a.size] do
    acc := acc + a.get! i * b.get! i
  pure acc

/-- `xs` with `delta` added to entry `i`, or `xs` itself when `i` is out of range. -/
def perturbArray (xs : Array Float) (i : Nat) (delta : Float) : Array Float :=
  match xs[i]? with
  | none => xs
  | some x => xs.setIfInBounds i (x + delta)

/-- The output of the public spectral composition on `device` along `path`, with the pullbacks of
the seed `dY` to the input and both weight parts when `backward` is set. On CUDA the tape must hold
one `rfft` and one `irfft` node on the automatic path and none on another. -/
def evaluateSpectralConv (grid width modes : Nat) (x wRe wIm dY : Array Float)
    (device : Backend.Device := .cuda) (path : nn.functional.SpectralPath := .automatic)
    (backward : Bool := true) : IO (FloatArray × FloatArray × FloatArray × FloatArray) := do
  if valid : 0 < grid ∧ 0 < width ∧ modes ≤ grid / 2 + 1 then
    unless x.size == grid * width && dY.size == grid * width &&
        wRe.size == modes * width * width && wIm.size == modes * width * width do
      throw <| IO.userError "spectral test input shape mismatch"
    let session ← Session.create (α := Float) { device }
    try
      let xRef ← session.input
        (Tensor.generateFlat [grid, width] fun i => x[i]!) (requiresGrad := backward)
      let realRef ← session.input
        (Tensor.generateFlat [modes, width, width] fun i => wRe[i]!) (requiresGrad := backward)
      let imagRef ← session.input
        (Tensor.generateFlat [modes, width, width] fun i => wIm[i]!) (requiresGrad := backward)
      let outputRef ← (nn.functional.spectralConv (α := Float) (m := SessionM Float)
        valid.1 valid.2.1 valid.2.2 xRef realRef imagRef path) session
      let output := FloatArray.mk ((← session.getValue outputRef).to (Array Float))
      if device == .cuda then
        let tape ← session.cudaTape
        let forwardCount := (tape.nodes.filter fun node => node.name == some "rfft").size
        let inverseCount := (tape.nodes.filter fun node => node.name == some "irfft").size
        let expected := if path == .automatic then 1 else 0
        unless forwardCount == expected && inverseCount == expected do
          throw <| IO.userError "spectral composition selected incorrect FFT primitives"
      if !backward then return (output, FloatArray.empty, FloatArray.empty, FloatArray.empty)
      let gradients ← session.backwardDenseAll outputRef
        (Tensor.generateFlat [grid, width] fun i => dY[i]!)
      let dx ← Session.grad gradients xRef
      let dReal ← Session.grad gradients realRef
      let dImag ← Session.grad gradients imagRef
      return (output, FloatArray.mk (dx.to (Array Float)),
        FloatArray.mk (dReal.to (Array Float)), FloatArray.mk (dImag.to (Array Float)))
    finally session.resetTape
  else
    throw <| IO.userError "spectral test dimensions must be positive and modes within the spectrum"

/-- The dot product of `dY` with the output of the spectral convolution on CUDA at `x`, `wRe`
and `wIm`, the scalar whose central differences `checkSpectralConvFiniteDiff` takes. -/
def spectralConvLoss
    (x wRe wIm dY : Array Float) (grid width modes : Nat) : IO Float := do
  let (y, _, _, _) ← evaluateSpectralConv grid width modes x wRe wIm dY (backward := false)
  dotFloatArray y (FloatArray.mk dY)

/-- Fail unless entry `idx` of `analytic` and the finite difference `fd` are finite and within
`tol` of each other. -/
def assertFiniteDiff
    (msg : String) (analytic : FloatArray) (idx : Nat) (fd : Float) (tol : Float) : IO Unit := do
  Harness.assertApprox s!"{msg}[{idx}]" (analytic.get! idx) fd tol

/-- `Buffer.rfftPacked` of the rows `[1, 0, 0, 0]` and `[1, 2, 3, 4]` against their packed half
spectra to `1e-4`. The second row fixes the sign convention `exp(-2πikt/n)`. -/
def runKnownSpectrum : IO Unit := do
  IO.println "== rfft1d packed known spectra =="

  -- Two rows, n=4. The second row catches the sign convention:
  -- DFT([1,2,3,4]) at k=1 is `-2 + 2i` for the `exp(-2*pi*i*k*t/n)` convention used by cuFFT.
  let x ← Buffer.ofFloatArrayIO (FloatArray.mk #[
    1.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 3.0, 4.0
  ])
  let got := Buffer.toFloatArray (Buffer.rfftPacked x 2 4)
  let expected := FloatArray.mk #[
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    10.0, 0.0, -2.0, 2.0, -2.0, 0.0
  ]
  assertFloatArrayApprox "rfft1dPacked known" got expected (tol := 1e-4)

/-- `Buffer.irfftPacked` after `Buffer.rfftPacked` returns two rows of length 8 and two rows of
length 5 to `2e-4`. -/
def runRoundtripEvenOdd : IO Unit := do
  IO.println "== rfft1d/irfft1d packed roundtrip =="

  -- Even and odd lengths exercise different Nyquist-bin handling. The adapter requests
  -- backward normalization from ATen's inverse, which includes the `1/n` factor.
  let even := FloatArray.mk #[
    0.25, -0.50, 1.00, 0.75, -1.25, 0.50, 0.125, -0.875,
    -0.30, 0.20, 0.90, -0.10, 0.45, -0.65, 1.10, -0.95
  ]
  let evenBuf ← Buffer.ofFloatArrayIO even
  let evenBack := Buffer.toFloatArray (Buffer.irfftPacked (Buffer.rfftPacked evenBuf 2 8) 2 8)
  assertFloatArrayApprox "rfft/irfft roundtrip even" evenBack even (tol := 2e-4)

  let odd := FloatArray.mk #[
    0.10, 0.30, -0.20, 0.70, -0.40,
    -0.60, 0.80, 0.15, -0.25, 0.55
  ]
  let oddBuf ← Buffer.ofFloatArrayIO odd
  let oddBack := Buffer.toFloatArray (Buffer.irfftPacked (Buffer.rfftPacked oddBuf 2 5) 2 5)
  assertFloatArrayApprox "rfft/irfft roundtrip odd" oddBack odd (tol := 2e-4)

/-- With all three bins of a grid of four retained and identity channel weights, the spectral
convolution of two channels returns its input to `3e-4`. -/
def runSpectralConvIdentity : IO Unit := do
  IO.println "== spectral convolution identity/full-spectrum check =="

  -- With every RFFT bin retained and identity channel weights, the spectral convolution is
  -- `irfft(rfft(x))`, which returns the input up to float32 and cuFFT roundoff.
  let x := #[
    0.25, -0.50,
    1.00, 0.75,
    -1.25, 0.50,
    0.125, -0.875
  ]
  let wRe := #[
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0
  ]
  let wIm := Array.replicate 12 0.0
  let (got, _, _, _) ← evaluateSpectralConv 4 2 3 x wRe wIm x (backward := false)
  assertFloatArrayApprox "spectral convolution identity" got (FloatArray.mk x) (tol := 3e-4)

/-- `Buffer.rfftAdjoint` sends a DC cotangent of `2^126` or of `2^-149` to that value at each of
four entries, so it neither amplifies its spectrum into overflow nor normalizes tiny results to
zero. The large cotangent passes through the public spectral composition as well, and no payload
outlives the checks. Exact powers of two keep both range boundaries free of decimal rounding. -/
def runAdjointRange : IO Unit := do
  IO.println "== real FFT adjoint range =="
  let before ← Buffer.Memory.read
  for dc in #[Float.ofNat (2 ^ 126), 1.0 / Float.ofNat (2 ^ 149)] do
    let gradient ← Buffer.ofFloatArrayIO <| FloatArray.mk #[dc, 0.0, 0.0, 0.0, 0.0, 0.0]
    let result ← IO.lazyPure fun _ => Buffer.rfftAdjoint gradient 1 4
    assertFloatArrayApprox "rfft adjoint preserves finite DC"
      (← Buffer.toFloatArrayIO result) (FloatArray.mk #[dc, dc, dc, dc]) (tol := 0.0)
    for buffer in #[gradient, result] do
      discard <| Buffer.releaseIO buffer
  -- Exercise the same large cotangent through the public spectral composition and Lean tape.
  let large := Float.ofNat (2 ^ 126)
  let (output, dx, _, _) ← evaluateSpectralConv 4 1 1
    #[0.0, 0.0, 0.0, 0.0] #[large] #[0.0] #[1.0, 1.0, 1.0, 1.0]
  assertFloatArrayApprox "spectral zero forward" output (FloatArray.mk #[0.0, 0.0, 0.0, 0.0]) 0.0
  assertFloatArrayApprox "spectral finite large dX" dx
    (FloatArray.mk #[large, large, large, large]) 0.0
  let after ← Buffer.Memory.read
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "FFT range checks retained tensor payloads"

/-- The spectral convolution of `grid` points, `width` channels and `modes` modes at `x`, `wRe`
and `wIm` with cotangent `dY`: its value and pullbacks match the dense reference on the CPU and on
CUDA to `3e-4`, and the pullbacks match central differences of `sum (spectralConv x w * dY)`
(step `1e-2`) to `2e-2`. The native evaluation leaves the live bytes unchanged. -/
def checkSpectralConvFiniteDiff (grid width modes : Nat)
    (x wRe wIm dY : Array Float) : IO Unit := do
  -- The composed pullback is checked against central differences of the scalar pairing
  --   L(x,w) = sum(spectralConv(x,w) * dY).
  -- The half-spectrum adjoint has factors `2/n` at interior frequencies, so these differences are
  -- checked as well as the agreement of the native and dense-reference results on both devices.
  let eps := 1e-2
  let tol := 2e-2

  let before ← Buffer.Memory.read
  let (actual, dX, dWRe, dWIm) ← evaluateSpectralConv grid width modes x wRe wIm dY
  let after ← Buffer.Memory.read
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "spectral composition retained tensor payloads after session reset"
  unless dX.size == x.size && dWRe.size == wRe.size && dWIm.size == wIm.size do
    throw <| IO.userError "spectral backward returned incorrect gradient sizes"
  for device in [Backend.Device.cpu, Backend.Device.cuda] do
    let (expected, refDX, refDWRe, refDWIm) ←
      evaluateSpectralConv grid width modes x wRe wIm dY device .denseReference
    assertFloatArrayApprox "spectral forward dense parity" actual expected (tol := 3e-4)
    assertFloatArrayApprox "spectral dX dense parity" dX refDX (tol := 3e-4)
    assertFloatArrayApprox "spectral dWRe dense parity" dWRe refDWRe (tol := 3e-4)
    assertFloatArrayApprox "spectral dWIm dense parity" dWIm refDWIm (tol := 3e-4)

  for i in [:x.size] do
    let lp ← spectralConvLoss (perturbArray x i eps) wRe wIm dY grid width modes
    let lm ← spectralConvLoss (perturbArray x i (-eps)) wRe wIm dY grid width modes
    assertFiniteDiff "spectral convolution dX" dX i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wRe.size] do
    let lp ← spectralConvLoss x (perturbArray wRe i eps) wIm dY grid width modes
    let lm ← spectralConvLoss x (perturbArray wRe i (-eps)) wIm dY grid width modes
    assertFiniteDiff "spectral convolution dWRe" dWRe i ((lp - lm) / (2.0 * eps)) tol

  for i in [:wIm.size] do
    let lp ← spectralConvLoss x wRe (perturbArray wIm i eps) dY grid width modes
    let lm ← spectralConvLoss x wRe (perturbArray wIm i (-eps)) dY grid width modes
    assertFiniteDiff "spectral convolution dWIm" dWIm i ((lp - lm) / (2.0 * eps)) tol

/-- `checkSpectralConvFiniteDiff` with one channel on a grid of four with three modes, and with two
channels at the grid and mode counts `(1, 1)`, `(4, 0)`, `(4, 2)`, `(4, 3)`, `(5, 0)`, `(5, 2)`
and `(5, 3)`. -/
def runSpectralConvFiniteDiff : IO Unit := do
  IO.println "== spectral convolution backward finite differences and payload lifetime =="
  checkSpectralConvFiniteDiff 4 1 3
    #[0.20, -0.40, 0.70, 1.10] #[0.75, -0.30, 0.20]
    #[0.00, 0.45, 0.00] #[1.00, -0.50, 0.25, 0.75]
  -- Width two detects a transposition of the channels, and odd grids tell the last bin apart
  -- from a Nyquist bin.
  for (grid, modes) in #[(1, 1), (4, 0), (4, 2), (4, 3), (5, 0), (5, 2), (5, 3)] do
    let values := fun (count phase : Nat) =>
      (Array.range count).map fun i => Float.ofNat ((i * 7 + phase) % 17) / 10.0 - 0.8
    IO.println s!"  grid={grid}, width=2, modes={modes}"
    checkSpectralConvFiniteDiff grid 2 modes
      (values (grid * 2) 1) (values (modes * 4) 3) (values (modes * 4) 5)
      (values (grid * 2) 9)

/-- Run every real FFT check of this file. -/
def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: real FFT ==="
  runKnownSpectrum
  runRoundtripEvenOdd
  runSpectralConvIdentity
  runAdjointRange
  runSpectralConvFiniteDiff

end FFT
end CUDA
end TorchLean.Test
