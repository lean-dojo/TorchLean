/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLeanTest.Harness.CUDA
public import TorchLean.NN.Tape.Apply
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# CUDA Kernel Coverage: Multi-Head Attention

Compares CPU eager tape vs CUDA eager tape for `multi_head_attention` (forward + backward).
A NaN beside finite entries of a native score cotangent must request the host replay.

The case stays small so float64/float32 roundoff differences stay limited.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.Test
namespace CUDA
namespace Attention

open TorchLean.Tensor

/-- Number of positions in the test sequence. -/
abbrev n : Nat := 2
/-- Number of attention heads. -/
abbrev numHeads : Nat := 2
/-- Feature width of each input and output position. -/
abbrev dModel : Nat := 4
/-- Feature width of each head. -/
abbrev headDim : Nat := 2

/-- The sequence is nonempty, as `Op.multiheadAttention` requires. -/
theorem n_ne_zero : n ≠ 0 := by decide

/-- Width of the concatenated heads, `numHeads * headDim`. -/
abbrev projDim : Nat := numHeads * headDim

/-- Query projection weight. -/
def wq : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.01, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08,
    0.09, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16
  ]).reshape [dModel, projDim] (by dsimp; decide)

/-- Key projection weight. -/
def wk : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.02, 0.01, 0.04, 0.03,
    0.06, 0.05, 0.08, 0.07,
    0.10, 0.09, 0.12, 0.11,
    0.14, 0.13, 0.16, 0.15
  ]).reshape [dModel, projDim] (by dsimp; decide)

/-- Value projection weight. -/
def wv : Tensor Float [dModel, projDim] :=
  (Tensor.from #[
    0.03, 0.00, 0.01, 0.02,
    0.00, 0.03, 0.02, 0.01,
    0.01, 0.02, 0.03, 0.00,
    0.02, 0.01, 0.00, 0.03
  ]).reshape [dModel, projDim] (by dsimp; decide)

/-- Output projection weight. -/
def wo : Tensor Float [projDim, dModel] :=
  (Tensor.from #[
    0.05, 0.00, 0.01, 0.02,
    0.00, 0.05, 0.02, 0.01,
    0.01, 0.02, 0.05, 0.00,
    0.02, 0.01, 0.00, 0.05
  ]).reshape [projDim, dModel] (by dsimp; decide)

/-- Input sequence of the tape comparison. -/
def x : Tensor Float [n, dModel] :=
  (Tensor.from #[
    0.10, -0.20, 0.05, 0.30,
    -0.05, 0.25, -0.10, 0.15
  ]).reshape [n, dModel] (by dsimp; decide)

/-- Attention mask, `true` where a query may attend to a key: the first position attends to
both positions and the second only to itself. -/
def mask : Tensor Bool [n, n] :=
  (Tensor.from #[
    true,  true,
    false, true
  ]).reshape [n, n] (by dsimp; decide)

/-- Evaluate a checked buffer operation at this point in a test's ownership sequence. -/
@[no_expose] def checked {α : Type} (action : Unit → Except String α) : IO α := do
  let result ← IO.lazyPure action
  IO.ofExcept result

/-- Throw unless `got` matches `want` entrywise: NaN matches NaN, infinities match by sign, and
finite entries agree within `tol`. -/
def assertMatches (label : String) (got want : FloatArray) (tol : Float) : IO Unit :=
  unless got.size == want.size && (List.range want.size).all (fun i =>
      let (a, b) := (got.get! i, want.get! i)
      (a.isNaN && b.isNaN) || a == b || Float.abs (a - b) ≤ tol) do
    throw <| IO.userError s!"{label}: got {got.toList}, expected {want.toList}"

/-- The two-pass reference on one masked sequence of `n` positions of width one, from flat
row-major arrays: the output of `Attention.scaledDotProduct` and the `(dQ, dK, dV)` of
`Attention.scaledDotProductBackward` for the output cotangent `g`, at their scale `1 / √1 = 1`.
A nonzero mask entry allows its key. -/
def reference (n : Nat) (h : n ≠ 0) (q k v g mask : FloatArray) :
    FloatArray × FloatArray × FloatArray × FloatArray :=
  let column (values : FloatArray) : Tensor Float [n, 1] :=
    Tensor.Rep.ofFn fun c => values.get! (Shape.Coord.index c 0)
  let allowed : Tensor Bool [n, n] := Tensor.Rep.ofFn fun c =>
    mask.get! (Shape.Coord.index c 0 * n + Shape.Coord.index c 1) != 0
  let ctx : Attention.Inputs Float n n 1 h h := ⟨column q, column k, column v, some allowed⟩
  let (dq, dk, dv) := Attention.scaledDotProductBackward ctx (column g)
  ((Attention.scaledDotProduct ctx).to FloatArray, dq.to FloatArray, dk.to FloatArray,
    dv.to FloatArray)

/-- Compare the native `(dQ, dK, dV)` with the `reference` gradients, then release them. -/
def assertGradients (label : String) (expected : FloatArray × FloatArray × FloatArray × FloatArray)
    (dq dk dv : LibTorch.Buffer) (tol : Float) : IO Unit := do
  let (_, rdq, rdk, rdv) := expected
  for (name, gradient, want) in [("dQ", dq, rdq), ("dK", dk, rdk), ("dV", dv, rdv)] do
    assertMatches s!"{label} {name}" (← LibTorch.Buffer.toFloatArrayIO gradient) want tol
    discard <| LibTorch.Buffer.releaseIO gradient

/-- Saved probabilities and borrowed Q/K/V support repeated VJPs after output release. A fully
blocked row follows the two-pass reference: zero weights times a nonfinite query, key, value or
cotangent give NaN, as in `Attention.scaledDotProduct` and `Attention.scaledDotProductBackward`. -/
def checkSavedBuffers : IO Unit := do
  let before ← LibTorch.Buffer.Memory.read
  let qs := FloatArray.mk #[0.0, 0.0]
  let ks := FloatArray.mk #[1.0, -1.0]
  let vs := FloatArray.mk #[2.0, 4.0]
  let rowMask := FloatArray.mk #[1.0, 1.0, 0.0, 0.0]
  let q ← LibTorch.Buffer.ofFloatArrayIO qs
  let k ← LibTorch.Buffer.ofFloatArrayIO ks
  let v ← LibTorch.Buffer.ofFloatArrayIO vs
  let allowed ← LibTorch.Buffer.ofFloatArrayIO rowMask
  let (output, probabilities) ← checked fun _ =>
    LibTorch.Buffer.attentionForward q k v (some allowed) 1 2 1 1.0
  discard <| LibTorch.Buffer.releaseIO allowed
  Harness.assertFloatArrayApprox "attention forward"
    (← LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[3.0, 0.0]) 1e-5
  Harness.assertFloatArrayApprox "attention saved probabilities"
    (← LibTorch.Buffer.toFloatArrayIO probabilities)
    (FloatArray.mk #[0.5, 0.5, 0.0, 0.0]) 1e-5
  discard <| LibTorch.Buffer.releaseIO output
  for blockedSeed in #[7.0, 0.0 / 0.0] do
    let gs := FloatArray.mk #[1.0, blockedSeed]
    let seed ← LibTorch.Buffer.ofFloatArrayIO gs
    let (dq, dk, dv) ← checked fun _ =>
      LibTorch.Buffer.attentionBackward q k v probabilities seed 1 2 1 1.0
    assertGradients s!"attention blocked cotangent {blockedSeed}"
      (reference 2 (by decide) qs ks vs gs rowMask) dq dk dv 1e-5
    discard <| LibTorch.Buffer.releaseIO seed
  -- A nonfinite query of the fully blocked row meets a zero score cotangent in dSᵀ Q.
  let allowed ← LibTorch.Buffer.ofFloatArrayIO rowMask
  let gs := FloatArray.mk #[1.0, 7.0]
  let seed ← LibTorch.Buffer.ofFloatArrayIO gs
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let blockedQs := FloatArray.mk #[0.0, value]
    let expected := reference 2 (by decide) blockedQs ks vs gs rowMask
    let blockedQ ← LibTorch.Buffer.ofFloatArrayIO blockedQs
    let (output, saved) ← checked fun _ =>
      LibTorch.Buffer.attentionForward blockedQ k v (some allowed) 1 2 1 1.0
    assertMatches s!"attention blocked query {value} output"
      (← LibTorch.Buffer.toFloatArrayIO output) expected.1 0.0
    discard <| LibTorch.Buffer.releaseIO output
    let (dq, dk, dv) ← checked fun _ =>
      LibTorch.Buffer.attentionBackward blockedQ k v saved seed 1 2 1 1.0
    assertGradients s!"attention blocked query {value}" expected dq dk dv 0.0
    for buffer in #[blockedQ, saved] do
      discard <| LibTorch.Buffer.releaseIO buffer
  for buffer in #[allowed, seed] do
    discard <| LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v, probabilities] do
    discard <| LibTorch.Buffer.releaseIO buffer
  -- A fully blocked row multiplies its zero weights by a nonfinite value or key.
  let zero ← LibTorch.Buffer.zerosIO 1
  let one ← LibTorch.Buffer.fullIO 1 1.0
  let (zeros, ones) := (FloatArray.mk #[0.0], FloatArray.mk #[1.0])
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let nonfinite ← LibTorch.Buffer.fullIO 1 value
    let (blocked, saved) ← checked fun _ =>
      LibTorch.Buffer.attentionForward zero zero nonfinite (some zero) 1 1 1 1.0
    assertMatches s!"attention blocked value {value} output"
      (← LibTorch.Buffer.toFloatArrayIO blocked)
      (reference 1 (by decide) zeros zeros (FloatArray.mk #[value]) ones zeros).1 0.0
    for buffer in #[blocked, saved] do
      discard <| LibTorch.Buffer.releaseIO buffer
    -- At `Q = 0` both scales multiply only zeros and NaNs, so the reference at its scale `1` gives
    -- the expected values of both scaling branches.
    let expected := reference 1 (by decide) zeros (FloatArray.mk #[value]) ones ones zeros
    for scale in #[1.0, 0.5] do
      let (blocked, saved) ← checked fun _ =>
        LibTorch.Buffer.attentionForward zero nonfinite one (some zero) 1 1 1 scale
      assertMatches s!"attention blocked key {value} output, scale {scale}"
        (← LibTorch.Buffer.toFloatArrayIO blocked) expected.1 0.0
      discard <| LibTorch.Buffer.releaseIO blocked
      let (dq, dk, dv) ← checked fun _ =>
        LibTorch.Buffer.attentionBackward zero nonfinite one saved one 1 1 1 scale
      assertGradients s!"attention blocked key {value}, scale {scale}" expected dq dk dv 0.0
      discard <| LibTorch.Buffer.releaseIO saved
    discard <| LibTorch.Buffer.releaseIO nonfinite
  -- Allowed NaN and either infinity are not fully blocked rows, even at zero scale.
  for value in #[0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0] do
    let nonfinite ← LibTorch.Buffer.fullIO 1 value
    for (label, query, key) in #[("K", one, nonfinite), ("Q", nonfinite, one)] do
      for scale in #[1.0, 0.0] do
        let (invalid, invalidSaved) ← checked fun _ =>
          LibTorch.Buffer.attentionForward query key one (some one) 1 1 1 scale
        unless ((← LibTorch.Buffer.toFloatArrayIO invalid).get! 0).isNaN do
          throw <| IO.userError
            s!"attention hid allowed nonfinite arithmetic ({label}={value}, scale={scale})"
        for buffer in #[invalid, invalidSaved] do
          discard <| LibTorch.Buffer.releaseIO buffer
    discard <| LibTorch.Buffer.releaseIO nonfinite
  for buffer in #[zero, one] do
    discard <| LibTorch.Buffer.releaseIO buffer
  let empty ← LibTorch.Buffer.zerosIO 0
  for shape in (#[(0, 2, 1), (1, 0, 1), (1, 2, 0)] :
      Array (UInt32 × UInt32 × UInt32)) do
    let (batch, rows, cols) := shape
    let (emptyOutput, emptyProbabilities) ← checked fun _ =>
      LibTorch.Buffer.attentionForward empty empty empty none batch rows cols 1.0
    unless (← LibTorch.Buffer.sizeIO emptyProbabilities) == batch * rows * rows do
      throw <| IO.userError "empty attention returned probabilities with the wrong size"
    let (dq, dk, dv) ← checked fun _ =>
      LibTorch.Buffer.attentionBackward
        empty empty empty emptyProbabilities empty batch rows cols 1.0
    for result in #[emptyOutput, dq, dk, dv] do
      unless (← LibTorch.Buffer.sizeIO result) == 0 do
        throw <| IO.userError "attention returned a nonempty output/gradient for an empty shape"
      discard <| LibTorch.Buffer.releaseIO result
    discard <| LibTorch.Buffer.releaseIO emptyProbabilities
  discard <| LibTorch.Buffer.releaseIO empty
  let after ← LibTorch.Buffer.Memory.read
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "attention saved-buffer checks retained tensor payloads"

/-- Discard masked overflow and avoid intermediate overflow for shrinking and growing scales. -/
def checkFiniteInputRegressions : IO Unit := do
  let before ← LibTorch.Buffer.Memory.read
  let q ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[1e20, 1e20]
  let k ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[0.0, 1e20]
  let v ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 3.0]
  let mask ← LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1.0, 0.0, 1.0, 0.0]
  let (output, probabilities) ← checked fun _ =>
    LibTorch.Buffer.attentionForward q k v (some mask) 1 2 1 1.0
  Harness.assertFloatArrayApprox "attention discards masked overflow"
    (← LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[2.0, 2.0]) 0.0
  for buffer in #[q, k, v, mask, output, probabilities] do
    discard <| LibTorch.Buffer.releaseIO buffer
  let q ← LibTorch.Buffer.zerosIO 1
  let k ← LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[Float.ofNat (2 ^ 100)]
  let v ← LibTorch.Buffer.fullIO 1 1.0
  let (output, probabilities) ← checked fun _ =>
    LibTorch.Buffer.attentionForward q k v none 1 1 1 (Float.ofNat (2 ^ 60))
  Harness.assertFloatArrayApprox "attention scales the dot product"
    (← LibTorch.Buffer.toFloatArrayIO output) (FloatArray.mk #[1.0]) 0.0
  for buffer in #[q, k, v, output, probabilities] do
    discard <| LibTorch.Buffer.releaseIO buffer
  -- The raw dot product overflows, but the default head-dimension scale makes it finite.
  let q ← LibTorch.Buffer.fullIO 4 1.4e19
  let k ← LibTorch.Buffer.fullIO 4 1.4e19
  let v ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 2.0, 4.0, 4.0]
  let seed ← LibTorch.Buffer.fullIO 4 1.0
  for scale in #[1.0 / Float.sqrt 2.0, -1.0 / Float.sqrt 2.0] do
    let (output, probabilities) ← checked fun _ =>
      LibTorch.Buffer.attentionForward q k v none 1 2 2 scale
    Harness.assertFloatArrayApprox "attention avoids raw dot-product overflow"
      (← LibTorch.Buffer.toFloatArrayIO output)
      (FloatArray.mk #[3.0, 3.0, 3.0, 3.0]) 0.0
    let (dq, dk, dv) ← checked fun _ =>
      LibTorch.Buffer.attentionBackward q k v probabilities seed 1 2 2 scale
    Harness.assertFloatArrayApprox "attention large-input dQ"
      (← LibTorch.Buffer.toFloatArrayIO dq)
      (FloatArray.mk #[0.0, 0.0, 0.0, 0.0]) 0.0
    -- Compare after rescaling to keep the tolerance relative to these large finite derivatives.
    let normalizedDK := (← LibTorch.Buffer.toFloatArrayIO dk).data.map (· / 1.4e19)
    Harness.assertFloatArrayApprox "attention large-input dK" (FloatArray.mk normalizedDK)
      (FloatArray.mk #[-2.0 * scale, -2.0 * scale, 2.0 * scale, 2.0 * scale]) 1e-5
    Harness.assertFloatArrayApprox "attention large-input dV"
      (← LibTorch.Buffer.toFloatArrayIO dv)
      (FloatArray.mk #[1.0, 1.0, 1.0, 1.0]) 0.0
    for buffer in #[output, probabilities, dq, dk, dv] do
      discard <| LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v, seed] do
    discard <| LibTorch.Buffer.releaseIO buffer
  -- Tiny Float64 scales through `scaledProduct`: the products of these operands overflow binary32,
  -- so the guard asks for the host replay.
  let q ← LibTorch.Buffer.fullIO 2 1e30
  let k ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[0.0, 1e30]
  let v ← LibTorch.Buffer.ofFloatArrayIO <| FloatArray.mk #[2.0, 4.0]
  for (scale, expected) in #[(1e-46, 4.0), (-1e-46, 2.0), (0.0, 3.0)] do
    let (output, probabilities) ← checked fun _ =>
      LibTorch.Buffer.attentionForward q k v none 1 2 1 scale
    Harness.assertFloatArrayApprox "attention retains tiny Float64 scale"
      (← LibTorch.Buffer.toFloatArrayIO output)
      (FloatArray.mk #[expected, expected]) 0.0
    for buffer in #[output, probabilities] do
      discard <| LibTorch.Buffer.releaseIO buffer
  for buffer in #[q, k, v] do
    discard <| LibTorch.Buffer.releaseIO buffer
  -- `2^60 * 2^60` is finite and the scale `2^-150` is below binary32's range: the split keeps the
  -- product `2^-30` on the device, with no replay.
  let a ← LibTorch.Buffer.fullIO 1 (Float.ofNat (2 ^ 60))
  let (product, replay) ← checked fun _ =>
    .ok (LibTorch.Buffer.scaledBmm a a 1 1 1 1 0 0 (1 / Float.ofNat (2 ^ 150)))
  if replay then
    throw <| IO.userError "scaled product of a tiny scale asked for the host replay"
  Harness.assertFloatArrayEq "scaled product of a tiny scale"
    (← LibTorch.Buffer.toFloatArrayIO product)
    (FloatArray.mk #[1 / Float.ofNat (2 ^ 30)])
  for buffer in #[a, product] do
    discard <| LibTorch.Buffer.releaseIO buffer
  let after ← LibTorch.Buffer.Memory.read
  unless after.liveBytes == before.liveBytes do
    throw <| IO.userError "attention scaling checks retained tensor payloads"

/-- Masked attention from scores over a batch of two sequences on the device against the CPU tape:
the output and the gradients of the scores, the value and the multiplier. -/
def checkMaskedFromScores : IO Unit := do
  let scoreShape : Shape := [2, n, n]
  -- Rows and keys are both `n`, so the value and the output have one shape.
  let outShape : Shape := [2, n, headDim]
  let scores : Tensor Float [2, n, n] :=
    (Tensor.from #[0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0]).reshape [2, n, n]
      (by dsimp; decide)
  let value : Tensor Float [2, n, headDim] :=
    (Tensor.from #[1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 4.0, -0.5]).reshape [2, n, headDim]
      (by dsimp; decide)
  -- An inverted-dropout multiplier: one dropped entry per sequence, the kept ones scaled by 2.
  let multiplier : Tensor Float [2, n, n] :=
    (Tensor.from #[2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0]).reshape [2, n, n] (by dsimp; decide)
  let seed : Tensor Float [2, n, headDim] :=
    (Tensor.from #[1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 1.5, -2.0]).reshape [2, n, headDim]
      (by dsimp; decide)
  let op := Op.attentionFromScores (batch := [2]) (rows := n) (keys := n)
    (width := headDim) (some mask)
  let (c1, cs) := Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float))) (.ofTensor scores)
  let (c2, cv) := Tape.leaf (t := c1) (.ofTensor value)
  let (c3, cm) := Tape.leaf (t := c2) (.ofTensor multiplier)
  let (c4, cy, ()) ← IO.ofExcept
    (Tape.apply (α := Float) c3 "attention_from_scores" op (cs, cv, cm, ()))
  let yCpu ← Harness.cpuValue (s := outShape) c4 cy
  let gradsCpu ← IO.ofExcept <|
    Tape.backwardDenseAll (V := SomeTensor Float) (t := c4) cy (SomeTensor.ofTensor seed)
  let (g1, gs) := Tape.leaf (t := Tape.empty)
    (Harness.tensorToSomeBuffer scores)
  let (g2, gv) := Tape.leaf (t := g1) (Harness.tensorToSomeBuffer value)
  let (g3, gm) :=
    Tape.leaf (t := g2) (Harness.tensorToSomeBuffer multiplier)
  let (g4, recorded) ← LibTorch.Tape.apply g3 "attention_from_scores" op
    (gs, gv, gm, ()) (pure .float32)
  let (gy, ()) ← IO.ofExcept recorded
  let yCuda ← Harness.cudaValue (s := outShape) g4 gy
  let gradsCuda ← IO.ofExcept
    (Tape.backwardDenseAll (t := g4) gy (Harness.tensorToSomeBuffer seed))
  Harness.assertTensorApprox "masked attention from scores, output" yCuda yCpu (tol := 1e-4)
  Harness.assertTensorApprox "masked attention from scores, scores gradient"
    (← Harness.cudaGrad (s := scoreShape) gradsCuda gs)
    (← Harness.cpuGrad (s := scoreShape) gradsCpu cs) (tol := 1e-4)
  Harness.assertTensorApprox "masked attention from scores, value gradient"
    (← Harness.cudaGrad (s := outShape) gradsCuda gv)
    (← Harness.cpuGrad (s := outShape) gradsCpu cv) (tol := 1e-4)
  Harness.assertTensorApprox "masked attention from scores, multiplier gradient"
    (← Harness.cudaGrad (s := scoreShape) gradsCuda gm)
    (← Harness.cpuGrad (s := scoreShape) gradsCpu cm) (tol := 1e-4)

/-- A NaN beside finite entries of the native score cotangent requests the host replay: binary32
weights `0.5`, values `[1e30, 1e30]`, multiplier `[[1e30, 1e30], [1, 1]]` and output cotangent
`[1, 1e-10]` overflow the first row's weighted cotangent, and the score, value and multiplier
cotangents must have the bits of `Attention.valuesBackward`, NaNs matching NaNs. -/
def checkValuesBackwardReplay : IO Unit := do
  let weights : Tensor Float32 [1, 2, 2] := [[[0.5, 0.5], [0.5, 0.5]]]
  let value : Tensor Float32 [1, 2, 1] := [[[1e30], [1e30]]]
  let multiplier : Tensor Float32 [1, 2, 2] := [[[1e30, 1e30], [1, 1]]]
  let gradient : Tensor Float32 [1, 2, 1] := [[[1], [1e-10]]]
  let buffer {s : Shape} (x : Tensor Float32 s) : IO LibTorch.Buffer :=
    LibTorch.Buffer.ofFloatArrayIO
      (FloatArray.mk ((x.to (Array Float32)).map Float32.toFloat)) .float32
  let w ← buffer weights
  let v ← buffer value
  let m ← buffer multiplier
  let g ← buffer gradient
  let (ds, dv, dm) ← checked fun _ =>
    .ok (LibTorch.Buffer.attentionValuesBackward w v m g 1 2 2 1)
  let (sds, sdv, sdm) :=
    Attention.valuesBackward (batch := [1]) weights value multiplier gradient
  for (label, computed, expected) in [("score", ds, sds.to (Array Float32)),
      ("value", dv, sdv.to (Array Float32)), ("multiplier", dm, sdm.to (Array Float32))] do
    let got ← LibTorch.Buffer.toFloatArrayIO computed
    let want := expected.map Float32.toFloat
    unless got.size == want.size && (List.range want.size).all (fun i =>
        let (a, b) := (got.get! i, want[i]!)
        a.toBits == b.toBits || (a.isNaN && b.isNaN)) do
      throw <| IO.userError
        s!"attention {label} cotangent replay: got {got.toList}, expected {want.toList}"
  for buffer in #[w, v, m, g, ds, dv, dm] do
    discard <| LibTorch.Buffer.releaseIO buffer

/-- The number of entries of two tensors of one shape whose bits differ, and the largest absolute
difference. -/
def differences {s : Shape} (x y : Tensor Float s) : Nat × Float :=
  let ax := x.to FloatArray
  let ay := y.to FloatArray
  (List.range ax.size).foldl (init := (0, 0)) fun (count, largest) i =>
    let (a, b) := (ax.get! i, ay.get! i)
    (if a.toBits == b.toBits then count else count + 1, max largest (Float.abs (a - b)))

/-- Run the buffer and mask checks, then compare the CUDA and CPU tapes on `x` (value and gradients
to `2e-2`) and a batch of two sequences with one tape per sequence. -/
def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: multi_head_attention ==="
  checkSavedBuffers
  checkFiniteInputRegressions
  checkValuesBackwardReplay

  -- A blocked extreme score must not influence stabilization. The second row checks the explicit
  -- all-blocked convention used by hard-masked attention.
  let extremeScores ← LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[1000.0, -1000.0, 3.0, 4.0]
  let extremeMask ← LibTorch.Buffer.ofFloatArrayIO <|
    FloatArray.mk #[0.0, 1.0, 0.0, 0.0]
  let extremeOut := LibTorch.Buffer.hardMaskedSoftmaxByRow
    extremeScores extremeMask 2 2
  let extremeHost := LibTorch.Buffer.toFloatArray extremeOut
  Harness.assertApprox "hard mask ignores blocked row maximum[0]" (extremeHost.get! 0) 0.0
    (tol := 1e-3)
  Harness.assertApprox "hard mask preserves allowed probability[1]" (extremeHost.get! 1) 1.0
    (tol := 1e-3)
  Harness.assertApprox "all-blocked hard mask row[0]" (extremeHost.get! 2) 0.0 (tol := 1e-3)
  Harness.assertApprox "all-blocked hard mask row[1]" (extremeHost.get! 3) 0.0 (tol := 1e-3)
  discard <| LibTorch.Buffer.releaseIO extremeScores
  discard <| LibTorch.Buffer.releaseIO extremeMask
  discard <| LibTorch.Buffer.releaseIO extremeOut

  let outShape : Shape := [n, dModel]

  -- CPU tape
  let t0 : Tape (SomeTensor Float) := Tape.empty
  let (t1, wqId) := Tape.leaf (t := t0) (.ofTensor wq) (name := some "wq")
  let (t2, wkId) := Tape.leaf (t := t1) (.ofTensor wk) (name := some "wk")
  let (t3, wvId) := Tape.leaf (t := t2) (.ofTensor wv) (name := some "wv")
  let (t4, woId) := Tape.leaf (t := t3) (.ofTensor wo) (name := some "wo")
  let (t5, xId) := Tape.leaf (t := t4) (.ofTensor x) (name := some "x")
  let (t6, yId, ()) ← IO.ofExcept
    (Tape.apply (α := Float) t5 "multi_head_attention"
      (Op.multiheadAttention (batch := []) (n := n) (numHeads := numHeads)
        (dModel := dModel) (headDim := headDim) n_ne_zero (some mask))
      (wqId, wkId, wvId, woId, xId, ()))
  let yCpu ← Harness.cpuValue (s := outShape) t6 yId
  let seedCpu : SomeTensor Float :=
    SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float))
  let gradsCpu ← IO.ofExcept (Tape.backwardDenseAll (V := SomeTensor Float) (t
      := t6) yId seedCpu)
  let dxCpu ← Harness.cpuGrad (s := outShape) gradsCpu xId
  let dWqCpu ← Harness.cpuGrad (s := [dModel, projDim]) gradsCpu wqId
  let dWkCpu ← Harness.cpuGrad (s := [dModel, projDim]) gradsCpu wkId
  let dWvCpu ← Harness.cpuGrad (s := [dModel, projDim]) gradsCpu wvId
  let dWoCpu ← Harness.cpuGrad (s := [projDim, dModel]) gradsCpu woId

  -- CUDA tape
  let t0c : Tape LibTorch.SomeBuffer :=
    Tape.empty
  let (t1c, wqIdc) := Tape.leaf (t := t0c) (Harness.tensorToSomeBuffer wq)
    (name := some "wq")
  let (t2c, wkIdc) := Tape.leaf (t := t1c) (Harness.tensorToSomeBuffer wk)
    (name := some "wk")
  let (t3c, wvIdc) := Tape.leaf (t := t2c) (Harness.tensorToSomeBuffer wv)
    (name := some "wv")
  let (t4c, woIdc) := Tape.leaf (t := t3c) (Harness.tensorToSomeBuffer wo)
    (name := some "wo")
  let (t5c, xIdc) := Tape.leaf (t := t4c) (Harness.tensorToSomeBuffer x)
    (name := some "x")
  let (t6c, directResult) ← LibTorch.Tape.apply t5c "multi_head_attention"
      (Op.multiheadAttention (batch := []) (n := n) (numHeads := numHeads)
        (dModel := dModel) (headDim := headDim) n_ne_zero (some mask))
      (wqIdc, wkIdc, wvIdc, woIdc, xIdc, ()) (pure .float32)
  let (yIdc, ()) ← IO.ofExcept directResult
  let yCuda ← Harness.cudaValue (s := outShape) t6c yIdc
  let seedCuda : LibTorch.SomeBuffer :=
    { s := outShape,
      buf := ← LibTorch.Buffer.fullIO
        (UInt32.ofNat (Shape.size outShape)) 1.0 }
  let gradsCuda ← IO.ofExcept
    (Tape.backwardDenseAll (t := t6c) yIdc seedCuda)
  let dxCuda ← Harness.cudaGrad (s := outShape) gradsCuda xIdc
  let dWqCuda ← Harness.cudaGrad (s := [dModel, projDim]) gradsCuda wqIdc
  let dWkCuda ← Harness.cudaGrad (s := [dModel, projDim]) gradsCuda wkIdc
  let dWvCuda ← Harness.cudaGrad (s := [dModel, projDim]) gradsCuda wvIdc
  let dWoCuda ← Harness.cudaGrad (s := [projDim, dModel]) gradsCuda woIdc

  -- Distinct samples are essential here: duplicated samples cannot expose a permutation that
  -- accidentally exchanges the batch and head axes.
  let xSecond : Tensor Float [n, dModel] :=
    (Tensor.from #[
      1.0, 2.0, 3.0, 1.0,
      -2.0, 1.0, 1.0, 3.0
    ]).reshape [n, dModel] (by dsimp; decide)
  let batchIdentity : Tensor Float [dModel, dModel] :=
    (Tensor.from #[
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ]).reshape [dModel, dModel] (by dsimp; decide)
  let xFirst : Tensor Float [n, dModel] :=
    (Tensor.from #[
      4.0, 0.0, 1.0, 0.0,
      0.0, 4.0, 0.0, 1.0
    ]).reshape [n, dModel] (by dsimp; decide)
  let xBatch : Tensor Float [2, n, dModel] :=
    Tensor.stack 0 fun i => if i.val = 0 then xFirst else xSecond
  let batchShape : Shape := [2, n, dModel]
  let tb0 : Tape LibTorch.SomeBuffer :=
    Tape.empty
  let (tb1, bwq) :=
    Tape.leaf (t := tb0) (Harness.tensorToSomeBuffer batchIdentity)
  let (tb2, bwk) :=
    Tape.leaf (t := tb1) (Harness.tensorToSomeBuffer batchIdentity)
  let (tb3, bwv) :=
    Tape.leaf (t := tb2) (Harness.tensorToSomeBuffer batchIdentity)
  let (tb4, bwo) :=
    Tape.leaf (t := tb3) (Harness.tensorToSomeBuffer batchIdentity)
  let (tb5, bx) := Tape.leaf (t := tb4) (Harness.tensorToSomeBuffer xBatch)
  let (tb6, batchResult) ← LibTorch.Tape.apply tb5 "multi_head_attention"
    (Op.multiheadAttention (batch := [2]) (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) n_ne_zero (some mask))
    (bwq, bwk, bwv, bwo, bx, ()) (pure .float32)
  let (byId, ()) ← IO.ofExcept batchResult
  let yBatch ← Harness.cudaValue (s := batchShape) tb6 byId
  let batchSeed : LibTorch.SomeBuffer :=
    { s := batchShape,
      buf := ← LibTorch.Buffer.fullIO
        (UInt32.ofNat (Shape.size batchShape)) 1.0 }
  let batchGrads ← IO.ofExcept
    (Tape.backwardDenseAll (t := tb6) byId batchSeed)
  let dxBatch ← Harness.cudaGrad (s := batchShape) batchGrads bx
  let dWqBatch ← Harness.cudaGrad (s := [dModel, projDim]) batchGrads bwq
  let dWkBatch ← Harness.cudaGrad (s := [dModel, projDim]) batchGrads bwk
  let dWvBatch ← Harness.cudaGrad (s := [dModel, projDim]) batchGrads bwv
  let dWoBatch ← Harness.cudaGrad (s := [projDim, dModel]) batchGrads bwo

  -- Compare the batch with two independent CPU tapes, summing shared weight gradients.
  let cpuSample := fun (input : Tensor Float [n, dModel]) => do
    let (tc1, cq) :=
      Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float))) (.ofTensor batchIdentity)
    let (tc2, ck) := Tape.leaf (t := tc1) (.ofTensor batchIdentity)
    let (tc3, cv) := Tape.leaf (t := tc2) (.ofTensor batchIdentity)
    let (tc4, co) := Tape.leaf (t := tc3) (.ofTensor batchIdentity)
    let (tc5, cx) := Tape.leaf (t := tc4) (.ofTensor input)
    let (tc6, cy, ()) ← IO.ofExcept <|
      Tape.apply tc5 "multi_head_attention"
        (Op.multiheadAttention (batch := []) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) n_ne_zero (some mask)) (cq, ck, cv, co, cx, ())
    let output ← Harness.cpuValue (s := outShape) tc6 cy
    let gradients ← IO.ofExcept <|
      Tape.backwardDenseAll tc6 cy (SomeTensor.ofTensor (Tensor.full outShape (1.0 : Float)))
    let dx ← Harness.cpuGrad (s := outShape) gradients cx
    let dq ← Harness.cpuGrad (s := [dModel, projDim]) gradients cq
    let dk ← Harness.cpuGrad (s := [dModel, projDim]) gradients ck
    let dv ← Harness.cpuGrad (s := [dModel, projDim]) gradients cv
    let dw ← Harness.cpuGrad (s := [projDim, dModel]) gradients co
    pure (output, dx, dq, dk, dv, dw)
  let (y0, dx0, dq0, dk0, dv0, dw0) ← cpuSample xFirst
  let (y1, dx1, dq1, dk1, dv1, dw1) ← cpuSample xSecond
  let expectedY : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then y0 else y1
  let expectedDx : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then dx0 else dx1
  Harness.assertTensorApprox "batched mha forward" yBatch expectedY (tol := 2e-2)
  Harness.assertTensorApprox "batched mha dx" dxBatch expectedDx (tol := 2e-2)
  Harness.assertTensorApprox "batched mha dWq" dWqBatch (Tensor.add dq0 dq1) (tol := 2e-2)
  Harness.assertTensorApprox "batched mha dWk" dWkBatch (Tensor.add dk0 dk1) (tol := 2e-2)
  Harness.assertTensorApprox "batched mha dWv" dWvBatch (Tensor.add dv0 dv1) (tol := 2e-2)
  Harness.assertTensorApprox "batched mha dWo" dWoBatch (Tensor.add dw0 dw1) (tol := 2e-2)

  -- The batched entry against the entry of each sequence on the device.
  let cudaSample := fun (input : Tensor Float [n, dModel]) => do
    let (tc1, cq) := Tape.leaf (t := Tape.empty)
      (Harness.tensorToSomeBuffer batchIdentity)
    let (tc2, ck) := Tape.leaf (t := tc1)
      (Harness.tensorToSomeBuffer batchIdentity)
    let (tc3, cv) := Tape.leaf (t := tc2)
      (Harness.tensorToSomeBuffer batchIdentity)
    let (tc4, co) := Tape.leaf (t := tc3)
      (Harness.tensorToSomeBuffer batchIdentity)
    let (tc5, cx) := Tape.leaf (t := tc4) (Harness.tensorToSomeBuffer input)
    let (tc6, recorded) ← LibTorch.Tape.apply tc5 "multi_head_attention"
      (Op.multiheadAttention (batch := []) (n := n) (numHeads := numHeads)
        (dModel := dModel) (headDim := headDim) n_ne_zero (some mask)) (cq, ck, cv, co, cx, ())
      (pure .float32)
    let (cy, ()) ← IO.ofExcept recorded
    let output ← Harness.cudaValue (s := outShape) tc6 cy
    let seed : LibTorch.SomeBuffer :=
      { s := outShape,
        buf := ← LibTorch.Buffer.fullIO
          (UInt32.ofNat (Shape.size outShape)) 1.0 }
    let gradients ← IO.ofExcept (Tape.backwardDenseAll (t := tc6) cy seed)
    let dx ← Harness.cudaGrad (s := outShape) gradients cx
    let dq ← Harness.cudaGrad (s := [dModel, projDim]) gradients cq
    let dk ← Harness.cudaGrad (s := [dModel, projDim]) gradients ck
    let dv ← Harness.cudaGrad (s := [dModel, projDim]) gradients cv
    let dw ← Harness.cudaGrad (s := [projDim, dModel]) gradients co
    pure (output, dx, dq, dk, dv, dw)
  let (cy0, cdx0, cdq0, cdk0, cdv0, cdw0) ← cudaSample xFirst
  let (cy1, cdx1, cdq1, cdk1, cdv1, cdw1) ← cudaSample xSecond
  let perSequenceY : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then cy0 else cy1
  let perSequenceDx : Tensor Float batchShape :=
    Tensor.stack 0 fun i => if i.val = 0 then cdx0 else cdx1
  let compared := #[
    ("forward", differences yBatch perSequenceY),
    ("dx", differences dxBatch perSequenceDx),
    ("dWq", differences dWqBatch (Tensor.add cdq0 cdq1)),
    ("dWk", differences dWkBatch (Tensor.add cdk0 cdk1)),
    ("dWv", differences dWvBatch (Tensor.add cdv0 cdv1)),
    ("dWo", differences dWoBatch (Tensor.add cdw0 cdw1))]
  IO.println <| "batched mha entry vs each sequence's entry " ++
    s!"(entries whose bits differ, largest |difference|): {compared}"
  for (label, _, largest) in compared do
    unless largest ≤ 2e-2 do
      throw <| IO.userError s!"batched mha entry vs each sequence's entry, {label}: {largest}"
  -- The values and the input's gradient of a sequence do not depend on the others and agree bit
  -- for bit on this fixture. The projection gradients are summed over all rows at once, so their
  -- last bits may differ.
  for (label, differing, _) in compared.extract 0 2 do
    unless differing == 0 do
      throw <| IO.userError <|
        s!"batched mha entry vs each sequence's entry, {label}: {differing} entries differ"

  Harness.assertTensorApprox (s := outShape) "mha forward" yCuda yCpu (tol := 2e-2)
  Harness.assertTensorApprox (s := outShape) "mha dx" dxCuda dxCpu (tol := 2e-2)
  Harness.assertTensorApprox (s := [dModel, projDim]) "mha dWq" dWqCuda dWqCpu (tol := 2e-2)
  Harness.assertTensorApprox (s := [dModel, projDim]) "mha dWk" dWkCuda dWkCpu (tol := 2e-2)
  Harness.assertTensorApprox (s := [dModel, projDim]) "mha dWv" dWvCuda dWvCpu (tol := 2e-2)
  Harness.assertTensorApprox (s := [projDim, dModel]) "mha dWo" dWoCuda dWoCpu (tol := 2e-2)
  checkMaskedFromScores

end Attention
end CUDA
end TorchLean.Test
