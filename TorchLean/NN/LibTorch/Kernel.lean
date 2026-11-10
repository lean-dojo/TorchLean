/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Activation
public import TorchLean.NN.LibTorch.Indexing
public import TorchLean.CUDA.LibTorch.ConvPool
public import TorchLean.NN.LibTorch.Attention
public import TorchLean.NN.LibTorch.Fourier
public import TorchLean.CUDA.LibTorch.Broadcast
public import TorchLean.Tensor.Axes
public import TorchLean.NN.Backend.LibTorch
public import TorchLean.Tensor.Op.Table.Rule

/-!
# The CUDA implementation table

`LibTorch.Kernel.ofOp op` is the CUDA implementation of the registered operator `op`, attributes
included, or `none` where LibTorch has none. `LibTorch.Tape.apply` records an operator on the CUDA
tape `Tape SomeBuffer` through its entry. An entry holds what the operator needs on the device: the
LibTorch kernel calls, the checks of shapes against the `UInt32` ABI, the node with its parents and
gradient flag, the workspace the node keeps for its backward (released with the tape) and the
release of every temporary buffer. The entries are private to this module.

The backward closures return fresh gradient buffers, which the tape's gradient accumulator owns,
and a derivative that needs intermediate buffers releases them before it returns.
-/

public section

open TorchLean.Shape (size)

namespace TorchLean
namespace LibTorch

/-- A parent or an output of an entry: a tape node for a value or detached port, a host tensor of
`Float` for a frozen port, the host tensor for an index, Boolean or integer port. -/
abbrev Arg : Port → Type :=
  Port.Slot (fun _ => Nat) (fun s => Tensor Float s) (fun _ => Nat)

/-- The parents or the outputs of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Args : List Port → Type := List.TProd Arg

/-- The CUDA implementation of one registered operator with its attributes. -/
structure Kernel (ins outs : List Port) where
  /-- Record the operator on the tape from its parents under the node label given, reading the
  session's buffer format only where it creates a buffer without inputs. It returns the
  extended tape with the outputs, or the tape it was given with the error, and throws no
  exception. A pure entry takes the tape without keeping a second reference to it, so the
  append updates the node array in place. -/
  record : Tape SomeBuffer → String → IO LibTorch.Dtype → Args ins →
    IO (Tape SomeBuffer × Result (Args outs))
  /-- Whether the entry runs on buffers of a configured scalar format (`Buffer.format?`). -/
  configuredFormats : Bool := false
  /-- The entry's contract, built with the builders of `Backend.LibTorch` or, for
  multi-head attention, `Backend.LibTorch.multiheadAttention`: name, trust level, forward and
  VJP support, layout, notes on its shape, layout, value and VJP claims, and numerical policy.
  `detach`'s entry, which no planner selects, has none. -/
  contract? : Option Backend.KernelContract

/-- `.ok ()` when the native `Buffer.divCount` boundary accepts a division of a `dtype` buffer by
`count`, and otherwise the error of the kernel `op`. The boundary aborts the process on a binary32
count of `2 ^ 29` or more, a binary64 count of `2 ^ 53` or more and an encoded buffer, so the
means, losses and normalizations below run this check first. -/
def checkDivCount (op : String) (dtype : Dtype) (count : Nat) : Result Unit :=
  match dtype with
  | .float32 =>
    if count < 2 ^ 29 then pure ()
    else throw s!"autograd: cuda: {op}: a binary32 count must be below 2^29, got {count}"
  | .float64 =>
    if count < 2 ^ 53 then pure ()
    else throw s!"autograd: cuda: {op}: a binary64 count must be below 2^53, got {count}"
  | .encoded => throw s!"autograd: cuda: {op}: a binary32 or binary64 buffer is required"

namespace Kernel

open Backend.LibTorch Coverage
open TorchLean.Tape (Node)

/-- An entry whose recording is pure and returns one value node. A failed recording returns the
tape it was given, whatever nodes the recording appended before it failed. -/
@[inline] private def ofResult {ins : List Port} {s : Shape}
    (contract? : Option Backend.KernelContract)
    (record : Tape SomeBuffer → String → Args ins → Result (Tape SomeBuffer × Nat))
    (configuredFormats : Bool := false) : Kernel ins [.value s] where
  record t name _ args := pure <| match record t name args with
    | .ok (t, id) => (t, .ok (id, ()))
    | .error message => (t, .error message)
  configuredFormats
  contract?

/-! ## Shared helpers -/

/-- Fold the leading axes into a row count and keep the last axis as the column count, for the row
kernels. A scalar, an empty last axis or an empty leading fold is an error. -/
private def foldRowsColsLastAxis (s : Shape) : Result (UInt32 × UInt32) := do
  match s.reverse with
  | [] =>
      throw "autograd: softmax: scalar input is not supported"
  | cols :: restRev =>
      let rowsFold : Nat := restRev.foldl (init := 1) (fun acc d => acc * d)
      if cols = 0 then
        throw "autograd: softmax: last dimension is 0"
      else if rowsFold = 0 then
        throw "autograd: softmax: folded leading dimension is 0"
      else
        let some rows32 := Bytes.toUInt32? rowsFold | throw SomeBuffer.tooLarge
        let some cols32 := Bytes.toUInt32? cols | throw SomeBuffer.tooLarge
        pure (rows32, cols32)

/-- The same buffer at another shape of equal size, as a view the node does not own. The pullback
copies the cotangent. -/
@[inline] private def reshapeNode (s t : Shape) (tape : Tape SomeBuffer) (name : String) (x : Nat) :
    Result (Tape SomeBuffer × Nat) :=
  Tape.unary tape name x s t (fun x => x) (fun _x dLdy => Buffer.copy dLdy) (ownsValue := false)

/-- A scalar buffer broadcast to `outShape`, the pullback of a reduction to a scalar. -/
private def broadcastScalarToShape (g : Buffer) (outShape : Shape) : Buffer :=
  let outDims := outShape.toArray
  let axisMap := Array.replicate outDims.size 0
  Buffer.broadcastTo g #[] outDims axisMap

/-! ## Elementwise entries -/

private def add {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "add" elementwise .torchLeanTape) (configuredFormats := true)
      fun t name (a, b, ()) =>
    Tape.binary t name a b s s s Buffer.add (fun _a _b dLdy => Buffer.duplicate dLdy)

private def sub {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "sub" elementwise .torchLeanTape) (configuredFormats := true)
      fun t name (a, b, ()) =>
    Tape.binary t name a b s s s Buffer.sub fun _a _b dLdy => (Buffer.copy dLdy, Buffer.neg dLdy)

private def mul {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "mul" elementwise .torchLeanTape) (configuredFormats := true)
      fun t name (a, b, ()) =>
    Tape.binary t name a b s s s Buffer.mul fun a b dLdy => (Buffer.mul dLdy b, Buffer.mul dLdy a)

/-- Division with the quotient-rule pullback in the eager schedule `(a / b) / b`: squaring `b`
first could overflow or underflow. -/
private def div {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "div" empty .torchLeanTape) (configuredFormats := true)
      fun t name (a, b, ()) =>
    Tape.binary t name a b s s s Buffer.div fun a b dLdy =>
      let da := Buffer.div dLdy b
      let aOverB := Buffer.div a b
      let aOverB2 := Buffer.div aOverB b
      let dLdyA := Buffer.mul dLdy aOverB2
      let db := Buffer.releaseThen aOverB <| Buffer.releaseThen aOverB2 <|
        Buffer.releaseThen dLdyA (Buffer.neg dLdyA)
      (da, db)

/-- Direct negation, preserving signed zero, with the sign-reversing pullback. -/
private def neg {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (pointwiseContract "neg" empty .torchLeanTape) (configuredFormats := true)
      fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.neg fun _ seed => Buffer.neg seed

private def scale {s : Shape} : Kernel [.value s, .frozen []] [.value s] :=
  ofResult (pointwiseContract "scale" elementwise .torchLeanTape) fun t name (x, c, ()) =>
    Tape.unary t name x s s (fun x => Buffer.scale x c.item) fun _x dLdy => Buffer.scale dLdy c.item

private def abs {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (pointwiseContract "abs" elementwise .backendVJP) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.abs Buffer.absBackward

private def sqrtClampNegative {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (pointwiseContract "sqrt" elementwise .backendVJP) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.sqrtClampNegative Buffer.sqrtClampNegativeBackward

private def sqrt {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (sqrtClampNegative (s := s)).contract? fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.sqrt Buffer.sqrtBackward

private def clamp {s : Shape} : Kernel [.value s, .frozen [], .frozen []] [.value s] :=
  ofResult (pointwiseContract "clamp" elementwise .backendVJP) fun t name (x, lo, hi, ()) =>
    Tape.unary t name x s s (fun x => Buffer.clamp x lo.item hi.item)
      fun x dLdy => Buffer.clampBackward x dLdy lo.item hi.item

/-- The maximum of two parents. Distinct parents split ties as `Buffer.maxBackward` does, and one
node at both positions receives the upstream gradient once. -/
private def max {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "max" elementwise .backendVJP) fun t name (a, b, ()) =>
    if a == b then
      Tape.unary t name a s s (fun x => Buffer.max x x) fun _ seed => Buffer.scale seed 1.0
    else
      Tape.binary t name a b s s s Buffer.max Buffer.maxBackward

/-- The minimum of two parents, with the tie and repeated-parent conventions of `max`. -/
private def min {s : Shape} : Kernel [.value s, .value s] [.value s] :=
  ofResult (pointwiseContract "min" elementwise .backendVJP) fun t name (a, b, ()) =>
    if a == b then
      Tape.unary t name a s s (fun x => Buffer.min x x) fun _ seed => Buffer.scale seed 1.0
    else
      Tape.binary t name a b s s s Buffer.min Buffer.minBackward

private def relu {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (primitiveContract "relu" elementwise
      "ReLU forward follows the pointwise activation contract."
      "ReLU VJP is checked through runtime autograd tests."
      .backendVJP .notApplicable) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.relu Buffer.reluBackward

/-- Exponential from the vendor math library. The pullback recomputes `exp x` as a temporary. -/
private def exp {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "exp" elementwise .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.exp fun x dLdy =>
      let ex := Buffer.exp x
      Buffer.releaseThen ex <| Buffer.mul dLdy ex

/-- Sine, with angles measured in radians, from the vendor math library. -/
private def sin {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "sin" empty .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.sin fun x dLdy =>
      let derivative := Buffer.cos x
      Buffer.releaseThen derivative <| Buffer.mul derivative dLdy

/-- Cosine, with angles measured in radians, from the vendor math library. -/
private def cos {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "cos" empty .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.cos fun x dLdy =>
      let sine := Buffer.sin x
      let derivative := Buffer.scale sine (-1.0)
      Buffer.releaseThen sine <| Buffer.releaseThen derivative <| Buffer.mul derivative dLdy

/-- The natural logarithm from the vendor math library, with IEEE 754 values outside the positive
reals (`-inf` at zero, NaN below it). -/
private def log {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "log" elementwise .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.log fun x dLdy => Buffer.div dLdy x

private def reciprocal {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (pointwiseContract "inv" elementwise .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.inv fun x dLdy =>
      let negative := Buffer.neg dLdy
      let quotient := Buffer.div negative x
      Buffer.releaseThen negative <| Buffer.releaseThen quotient <| Buffer.div quotient x

/-- The smooth logarithm surrogate `log (softplus x + ε)` for an `ε` that stays positive after
conversion to the input dtype. Nothing here checks that condition. -/
private def safeLog {s : Shape} : Kernel [.value s, .frozen []] [.value s] :=
  ofResult (pointwiseContract "safe_log" elementwise .torchLeanTape) fun t name (x, ε, ()) => do
    let ε := ε.item
    let n ← SomeBuffer.numelU32 s
    Tape.unary t name x s s
      (fun x =>
        let epsBuf := Buffer.full n ε (Buffer.dtype x)
        let sp := Tape.softplusBuf x n
        let denom := Buffer.add sp epsBuf
        let y := Buffer.log denom
        Buffer.releaseThen epsBuf <| Buffer.releaseThen sp <| Buffer.releaseThen denom y)
      fun x dLdy =>
        let epsBuf := Buffer.full n ε (Buffer.dtype x)
        let sp := Tape.softplusBuf x n
        let denom := Buffer.add sp epsBuf
        let sig := Buffer.sigmoid x
        let dlog := Buffer.div sig denom
        Buffer.releaseThen epsBuf <| Buffer.releaseThen sp <| Buffer.releaseThen denom <|
          Buffer.releaseThen sig <| Buffer.releaseThen dlog <| Buffer.mul dLdy dlog

/-- Sigmoid with the native input-based pullback, both from the vendor math library's
exponential. -/
private def sigmoid {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "sigmoid" elementwise .backendVJP) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.sigmoid Buffer.sigmoidBackward

/-- Tanh with the native input-based pullback, both from the vendor math library. -/
private def tanh {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "tanh" elementwise .backendVJP) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.tanh Buffer.tanhBackward

/-- SiLU as one node. ATen evaluates the split-exponential value and the local pullback, both with
the vendor math library's exponential. -/
private def silu {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "silu" elementwise .backendVJP) fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.silu Buffer.siluBackward

/-- The contract of the `gelu` entry, with the qualifications of its value and VJP. -/
private def geluContract : Backend.KernelContract :=
  let c := primitiveContract
    "gelu"
    elementwise
    ("GELU's staged formula in `Activation.geluTanh`'s order with its constants, using CUDA " ++
      "vendor transcendental operations; bit equality with the Lean reference, which is the " ++
      "specification, is not guaranteed (binary32 sample: 12 of 15 IR inputs equal, " ++
      "3 one ulp away).")
    ("GELU VJP in `Activation.geluTanhDeriv`'s order with the vendor `exp`, keeping the " ++
      "squared reciprocal cosh where it is subnormal or zero instead of `tanhDeriv`'s replay, " ++
      "checked through runtime autograd tests.")
    .backendVJP .notApplicable
  { c with
    value.qualification :=
      "staged formula with CUDA vendor transcendental operations; bit equality with the Lean " ++
        "reference, which is the specification, is not guaranteed (binary32 sample: " ++
        "12 of 15 IR inputs equal, 3 one ulp away)"
    vjp.qualification :=
      "`Activation.geluTanhDeriv`'s order with the CUDA vendor `exp` and without " ++
        "`tanhDeriv`'s replay where the squared reciprocal cosh is subnormal or zero, so not " ++
        "bit-exact to the Lean reference, which is the specification" }

/-- Tanh-approximate GELU as one node, which need not be one GPU kernel launch: the staged formula
of `Activation.geluTanh` (its order and constants) with vendor transcendental operations, bit
equality with the Lean reference not guaranteed, and the pullback (`staged_gelu_backward`) in the
order of `Activation.geluTanhDeriv` with the vendor `exp` and no `tanh`, except that where
the squared reciprocal cosh of `u = β (x + κ x³)` is subnormal or zero (`|u|` above about
`64 ln 2` in binary32 and `512 ln 2` in binary64) it keeps that square, which
`TorchLean.Numeric.tanhDeriv` replaces by `4 q^4 / (1 + q^4)^2` at the rounded
`q = exp (-|u| / 2)`, rounded once. On a 15-input binary32
sample, 12 outputs agree with the reference and three differ by one ulp, which is neither a global
bound nor a gradient measurement. -/
private def gelu {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult geluContract fun t name (x, ()) =>
    Tape.unary t name x s s Buffer.gelu Buffer.geluBackward

/-- Stable softplus `max x 0 + log1p (exp (-|x|))` with the vendor math library's `exp` and
`log1p`. Its pullback multiplies by the CUDA sigmoid, which calls the vendor `exp`. -/
private def softplus {s : Shape} : Kernel [.value s] [.value s] :=
  ofResult (vendorMathContract "softplus" elementwise .torchLeanTape) fun t name (x, ()) => do
    let n ← SomeBuffer.numelU32 s
    Tape.unary t name x s s (fun x => Tape.softplusBuf x n) fun x dLdy =>
      let dy := Buffer.sigmoid x
      Buffer.releaseThen dy <| Buffer.mul dLdy dy

/-! ## Normalization, softmax, losses and the affine layer -/

/-- Weighted logits cross-entropy along `dim` with its two native pullbacks. -/
private def crossEntropy {s : Shape} (dim : Fin s.rank) (mean : Bool) :
    Kernel [.value s, .value s] [.value .scalar] :=
  ofResult (primitiveContract "cross_entropy" empty
      "Weighted log-probabilities retain original logits for overflow reconstruction."
      "TorchLean selects both pullbacks; ATen evaluates their numerical formulas."
      .backendVJP .implementationDefined) fun t name (logits, target, ()) => do
    Tape.validateU32Dimensions "cross_entropy" s.toArray
    let some axis32 := Bytes.toUInt32? dim.val | throw SomeBuffer.tooLarge
    if mean then
      checkDivCount "cross_entropy" (Buffer.dtype (← Tape.requireValue t logits s))
        (s.size / s.extent dim)
    Tape.binary t name logits target s s Shape.scalar
      (fun logits target => Buffer.crossEntropy logits target s.toArray axis32 mean)
      (fun logits target seed => Buffer.crossEntropyBackward logits target seed s.toArray
        axis32 mean)

/-- The layer normalization node of a `(seqLen, embedDim)` buffer over its last axis. The native
primitive evaluates the forward formula and TorchLean's pullback. The node keeps the normalized
input and the inverse standard deviation for the pullback, so both passes use the caller's
epsilon. -/
private def layerNormNode (seqLen embedDim : Nat) (t : Tape SomeBuffer) (name : String)
    (xId gammaId betaId : Nat) (epsilon : Float) : Result (Tape SomeBuffer × Nat) := do
  let some rows32 := Bytes.toUInt32? seqLen | throw SomeBuffer.tooLarge
  let some cols32 := Bytes.toUInt32? embedDim | throw SomeBuffer.tooLarge
  let x ← Tape.requireValue t xId (.dim seqLen (.dim embedDim .scalar))
  let gamma ← Tape.requireValue t gammaId (.dim embedDim .scalar)
  let beta ← Tape.requireValue t betaId (.dim embedDim .scalar)
  let (y, xHat, invStd) := Buffer.layerNorm x gamma beta rows32 cols32 epsilon
  let outShape : Shape := .dim seqLen (.dim embedDim .scalar)
  let node : Node SomeBuffer :=
    { name := some name
      value := { s := outShape, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
        (t.getNode? gammaId).any (·.requiresGrad) ||
        (t.getNode? betaId).any (·.requiresGrad)
      parents := #[xId, gammaId, betaId]
      cleanup := #[xHat, invStd].map SomeBuffer.ofWorkspace
      backward := fun dLdyAny => do
        let dLdy ← Tape.requireGrad dLdyAny outShape
        checkDivCount "layer_norm" (Buffer.dtype dLdy.buf) embedDim
        let (dx, dGamma, dBeta) := Buffer.layerNormBackward dLdy.buf xHat invStd gamma rows32 cols32
        pure #[
          (xId, { s := outShape, buf := dx }),
          (gammaId, { s := .dim embedDim .scalar, buf := dGamma }),
          (betaId, { s := .dim embedDim .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/-- The contract of the `layerNorm` entry, with the qualification of its value. -/
private def layerNormContract : Backend.KernelContract :=
  let c := primitiveContract
    "layer_norm"
    empty
    "LayerNorm follows the per-row normalization contract."
    "LayerNorm VJP calls the native backward primitive."
    .backendVJP .implementationDefined
  { c with
    value.qualification :=
      "rows normalized in binary64 in the steps of `TorchLean.Normalization.normalizedRows` with \
        ATen's row sums, " ++
        "and binary32 results rounded once before the binary32 affine map, so not bit-exact to " ++
        "the Lean reference, which is the specification" }

/-- Layer normalization over the last axis of `x`, its leading axes folded row-major into the
`seqLen` rows, as `Op.layerNorm`'s rule reads it: the input viewed as the `(seqLen, embedDim)`
matrix, the matrix node, and the view back. When `x` is that matrix there are no views. Each row
runs the steps of `Normalization.normalizedRows` in binary64 with ATen's row sums. A binary32
row's normalized values and inverse standard deviation are rounded once to binary32 before the
binary32 affine map, so the words can differ from the rule's. -/
private def layerNorm {seqLen embedDim : Nat} {x : Shape} :
    Kernel [.value x, .value [embedDim], .value [embedDim], .frozen []] [.value x] :=
  ofResult layerNormContract fun t name (xId, gammaId, betaId, epsilon, ()) => do
    let matrix : Shape := [seqLen, embedDim]
    if x = matrix then
      layerNormNode seqLen embedDim t name xId gammaId betaId epsilon.item
    else
      let (t, rows) ← reshapeNode x matrix t "reshape" xId
      let (t, y) ← layerNormNode seqLen embedDim t name rows gammaId betaId epsilon.item
      reshapeNode matrix x t "reshape" y

/-- The contract of the `batchNorm` and `batchNormRunning` entries. -/
private def batchNormContract : Backend.KernelContract :=
  primitiveContract
    "batch_norm"
    empty
    "BatchNorm follows the per-channel normalization contract, with batch or running statistics."
    "Lean composes the BatchNorm VJP from LibTorch numerical primitives."
    .torchLeanTape .implementationDefined

/-- Batch normalization over every axis after the channel axis, with
`Normalization.normalizedRows`'s formula: each centered row is divided by its scale before it is
squared. The row reductions see the spatial axes folded into one contiguous dimension, a view of the
storage that works at any spatial rank. -/
private def batchNorm {channels : Nat} {spatial : Shape} :
    Kernel [.value (.dim channels spatial), .value [channels], .value [channels], .frozen []]
      [.value (.dim channels spatial)] :=
  ofResult batchNormContract fun t name (xId, gammaId, betaId, epsilon, ()) => do
    let epsilon := epsilon.item
    let some rows32 := Bytes.toUInt32? channels | throw SomeBuffer.tooLarge
    let cols : Nat := Shape.size spatial
    let some cols32 := Bytes.toUInt32? cols | throw SomeBuffer.tooLarge
    let xShape : Shape := .dim channels spatial
    let x ← Tape.requireValue t xId xShape
    checkDivCount "batch_norm" (Buffer.dtype x) cols
    let gamma ← Tape.requireValue t gammaId (.dim channels .scalar)
    let beta ← Tape.requireValue t betaId (.dim channels .scalar)
    -- As `Normalization.normalizedRows` without recentering: the row less its mean, divided by its
    -- scale (the IEEE maximum of its largest magnitude and `sqrt |epsilon|`), then by
    -- `std = sqrt (max (m + epsilon / scale / scale) 0)` for the mean `m` of its squares, each
    -- row sum divided once by the number of positions. `reduceMaxByRow` skips a NaN, which changes
    -- no output: a centered row has a NaN only when its mean is NaN or infinite, and then every
    -- output of the row is NaN either way.
    let sum1 := Buffer.reduceSumByRow x rows32 cols32
    let mean := Buffer.divCount sum1 cols.toUInt64
    let meanB := Buffer.broadcastVecToCols mean rows32 cols32
    let centered := Buffer.sub x meanB
    let magnitude := Buffer.abs centered
    let largest := Buffer.reduceMaxByRow magnitude rows32 cols32
    let epsVec := Buffer.full rows32 epsilon (Buffer.dtype x)
    let epsAbs := Buffer.abs epsVec
    let epsRoot := Buffer.sqrtClampNegative epsAbs
    let scale := Buffer.max largest epsRoot
    let scaleB := Buffer.broadcastVecToCols scale rows32 cols32
    let scaled := Buffer.div centered scaleB
    let scaled2 := Buffer.mul scaled scaled
    let squareSum := Buffer.reduceSumByRow scaled2 rows32 cols32
    let meanSquare := Buffer.divCount squareSum cols.toUInt64
    let epsScaled := Buffer.div epsVec scale
    let epsScaled2 := Buffer.div epsScaled scale
    let shifted := Buffer.add meanSquare epsScaled2
    let std := Buffer.sqrtClampNegative shifted
    let stdB := Buffer.broadcastVecToCols std rows32 cols32
    let xHat := Buffer.div scaled stdB
    let gammaB := Buffer.broadcastVecToCols gamma rows32 cols32
    let betaB := Buffer.broadcastVecToCols beta rows32 cols32
    let xHatGamma := Buffer.mul xHat gammaB
    let y := Buffer.releaseManyThen
      #[sum1, mean, meanB, centered, magnitude, largest, epsVec, epsAbs, epsRoot, scaleB,
        scaled, scaled2, squareSum, meanSquare, epsScaled, epsScaled2, shifted, stdB, betaB,
        xHatGamma] (Buffer.add xHatGamma betaB)
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := xShape, buf := y }
        requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
          (t.getNode? gammaId).any (·.requiresGrad) ||
          (t.getNode? betaId).any (·.requiresGrad)
        parents := #[xId, gammaId, betaId]
        cleanup := #[scale, std, xHat, gammaB].map SomeBuffer.ofWorkspace
        backward := fun dLdyAny => do
          let dLdy ← Tape.requireGrad dLdyAny xShape
          -- dBeta and dGamma sum over the folded spatial axis.
          let dBeta := Buffer.reduceSumByRow dLdy.buf rows32 cols32
          let dGammaPointwise := Buffer.mul dLdy.buf xHat
          let dGamma := Buffer.releaseThen dGammaPointwise <|
            Buffer.reduceSumByRow dGammaPointwise rows32 cols32
          let dXhat := Buffer.mul dLdy.buf gammaB
          let sumDXhat := Buffer.reduceSumByRow dXhat rows32 cols32
          let dXhatXhat := Buffer.mul dXhat xHat
          let sumDXhatXhat := Buffer.releaseThen dXhatXhat <|
            Buffer.reduceSumByRow dXhatXhat rows32 cols32
          -- As `BatchNorm.normalizedBackward`: invStd * ((dXhat - m1) - xHat * m2), with m1 and
          -- m2 the row sums of dXhat and dXhat * xHat, each divided once by the number of
          -- positions, and invStd = 1 / std / scale.
          let mean1 := Buffer.divCount sumDXhat cols.toUInt64
          let mean2 := Buffer.divCount sumDXhatXhat cols.toUInt64
          let mean1B := Buffer.broadcastVecToCols mean1 rows32 cols32
          let mean2B := Buffer.broadcastVecToCols mean2 rows32 cols32
          let centeredDXhat := Buffer.sub dXhat mean1B
          let xHatMean2 := Buffer.mul xHat mean2B
          let term := Buffer.sub centeredDXhat xHatMean2
          let stdInv := Buffer.inv std
          let invStd := Buffer.div stdInv scale
          let invStdB := Buffer.broadcastVecToCols invStd rows32 cols32
          let dxRaw := Buffer.mul invStdB term
          let dx :=
            Buffer.releaseThen dXhat <| Buffer.releaseThen sumDXhat <|
              Buffer.releaseThen sumDXhatXhat <| Buffer.releaseThen mean1 <|
                Buffer.releaseThen mean2 <| Buffer.releaseThen mean1B <|
                  Buffer.releaseThen mean2B <| Buffer.releaseThen centeredDXhat <|
                    Buffer.releaseThen xHatMean2 <| Buffer.releaseThen term <|
                      Buffer.releaseThen stdInv <| Buffer.releaseThen invStd <|
                        Buffer.releaseThen invStdB dxRaw
          pure #[
            (xId, { s := xShape, buf := dx }),
            (gammaId, { s := .dim channels .scalar, buf := dGamma }),
            (betaId, { s := .dim channels .scalar, buf := dBeta })
          ] }
    pure (t.addNode node)

/-- Batch normalization along `dim` with frozen running statistics
(`Tensor.batchNorm x dim gamma beta (.running mean var) epsilon`). The denominator is
`sqrt (max (max var 0 + epsilon) 0)` per channel, with clamps that keep a NaN (`Buffer.relu`,
`Buffer.sqrtClampNegative`), and the result is `(x - mean) / denominator * gamma + beta` with each
channel vector broadcast along `dim`. The statistics are uploaded in the input's dtype, and neither
they nor epsilon receive a gradient. The node keeps the normalized input and the broadcast `gamma`
and denominator for the pullback. A failed upload is the entry's error. -/
private def batchNormRunning {s : Shape} (dim : Fin s.rank) :
    Kernel [.value s, .value [s.extent dim], .value [s.extent dim], .frozen [s.extent dim],
      .frozen [s.extent dim], .frozen []] [.value s] where
  record t name _ := fun (xId, gammaId, betaId, mean, var, epsilon, ()) => do
    let channels := s.extent dim
    let checked : Result (UInt32 × Buffer × Buffer × Buffer) := do
      Tape.validateU32Dimensions "batch_norm" s.toArray
      let some channels32 := Bytes.toUInt32? channels | throw SomeBuffer.tooLarge
      pure (channels32, ← Tape.requireValue t xId s,
        ← Tape.requireValue t gammaId [channels], ← Tape.requireValue t betaId [channels])
    match checked with
    | .error message => pure (t, .error message)
    | .ok (channels32, x, gamma, beta) =>
      let dtype := Buffer.dtype x
      try
        let meanBuf ← Buffer.ofFloatArrayIO (mean.to FloatArray) dtype
        let varBuf ← tryCatch (Buffer.ofFloatArrayIO (var.to FloatArray) dtype)
          fun e => throw (Buffer.releaseThen meanBuf e)
        let inDims := #[channels]
        let outDims := s.toArray
        let axisMap := Array.ofFn fun j : Fin s.rank => if j = dim then 1 else 0
        let spread := fun v => Buffer.broadcastTo v inDims outDims axisMap
        let varClamped := Buffer.relu varBuf
        let epsVec := Buffer.full channels32 epsilon.item dtype
        let varEps := Buffer.add varClamped epsVec
        let denominator := Buffer.sqrtClampNegative varEps
        let meanB := spread meanBuf
        let centered := Buffer.sub x meanB
        let denominatorB := spread denominator
        let xHat := Buffer.div centered denominatorB
        let gammaB := spread gamma
        let betaB := spread beta
        let scaled := Buffer.mul xHat gammaB
        let y := Buffer.releaseManyThen
          #[meanBuf, varBuf, varClamped, epsVec, varEps, denominator, meanB, centered, betaB,
            scaled]
          (Buffer.add scaled betaB)
        let node : Node SomeBuffer :=
          { name := some name
            value := { s, buf := y }
            requiresGrad := (t.getNode? xId).any (·.requiresGrad) ||
              (t.getNode? gammaId).any (·.requiresGrad) ||
              (t.getNode? betaId).any (·.requiresGrad)
            parents := #[xId, gammaId, betaId]
            cleanup := #[xHat, gammaB, denominatorB].map SomeBuffer.ofWorkspace
            backward := fun dLdyAny => do
              let dLdy ← Tape.requireGrad dLdyAny s
              let weighted := Buffer.mul dLdy.buf gammaB
              let dx := Buffer.releaseThen weighted (Buffer.div weighted denominatorB)
              let normalized := Buffer.mul dLdy.buf xHat
              let dGamma := Buffer.releaseThen normalized <|
                Buffer.reduceFromBroadcastTo normalized inDims outDims axisMap
              let dBeta := Buffer.reduceFromBroadcastTo dLdy.buf inDims outDims axisMap
              pure #[
                (xId, { s, buf := dx }),
                (gammaId, { s := [channels], buf := dGamma }),
                (betaId, { s := [channels], buf := dBeta })
              ] }
        let (t, id) := t.addNode node
        pure (t, .ok (id, ()))
      catch e => pure (t, .error (toString e))
  contract? := batchNormContract

/-- Adjacent-swap depths that reorder the axes so that axis `i` of the result is axis `order[i]` of
the input: each output position in turn receives its axis by swaps toward the front. -/
private def swapDepths (order : List Nat) : List Nat :=
  go 0 (List.range order.length) order
where
  go (i : Nat) (current : List Nat) : List Nat → List Nat
    | [] => []
    | a :: rest =>
        let j := current.idxOf a
        (List.range (j - i)).map (fun k => j - 1 - k) ++
          go (i + 1) (current.take i ++ [a] ++ (current.drop i).erase a) rest

/--
A row operation along `axis` whose pullback uses the saved output. Adjacent swaps move the axis
last, the row kernels run on the moved buffer, and the reverse swaps move the result back. The
pullback moves the cotangent the same way. Along the last axis there is no swap. Otherwise the node
keeps a moved copy of the output for the pullback and releases it with the tape. Empty tensors are
copied.
-/
@[inline] private def rowOp {s : Shape} (t : Tape SomeBuffer) (name : String) (xId : Nat)
    (axis : Nat) (forward : Buffer → UInt32 → UInt32 → Buffer.WithWorkspace)
    (backward : Buffer → Buffer → UInt32 → UInt32 → Buffer) : Result (Tape SomeBuffer × Nat) := do
  -- With no coordinates, both the result and its cotangent have the same empty shape.
  if Shape.size s == 0 then
    return ← Tape.unary t name xId s s Buffer.copy (fun _ gradient => Buffer.copy gradient)
  let (moved, depths) := if h : axis < s.rank then
      let σ := Fin.cycleIcc (⟨axis, h⟩ : Fin s.rank) ⟨s.rank - 1, Nat.sub_one_lt_of_lt h⟩
      (s.permute σ, swapDepths ((List.finRange s.rank).map fun i => (σ i).val))
    else (s, [])
  let (rows32, cols32) ← foldRowsColsLastAxis moved
  let steps ← Tape.Indexing.checkedSwapSteps depths
  let back ← Tape.Indexing.checkedSwapSteps depths.reverse
  let x ← Tape.requireValue t xId s
  let xMoved := Tape.Indexing.permuteBorrowed x s steps
  let yOwned := forward xMoved rows32 cols32
  let yMoved :=
    Tape.Indexing.releasePermutedThen steps xMoved (yOwned.releaseWorkspaceThen yOwned.value)
  let y := Tape.Indexing.permuteBorrowed yMoved moved back
  let node : Node SomeBuffer :=
    { name := some name
      value := { s := s, buf := y }
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      cleanup := if steps.isEmpty then #[] else #[SomeBuffer.ofWorkspace yMoved]
      backward := fun dLdyAny => do
        let dLdy ← Tape.requireGrad dLdyAny s
        let dMoved := Tape.Indexing.permuteBorrowed dLdy.buf s steps
        let dxMoved :=
          Tape.Indexing.releasePermutedThen steps dMoved (backward yMoved dMoved rows32 cols32)
        pure #[(xId, { s := s, buf := Tape.Indexing.permuteOwned dxMoved moved back })] }
  pure (t.addNode node)

/-- Softmax along `dim` with the pivoted pullback. -/
private def softmax {s : Shape} (dim : Fin s.rank) : Kernel [.value s] [.value s] :=
  ofResult (primitiveContract "softmax" (full "TorchLeanTest.CUDA.Softmax")
      "Softmax kernels follow the row/axis normalization contract."
      "Softmax VJPs are checked through runtime autograd tests."
      .torchLeanTape .implementationDefined) fun t name (x, ()) =>
    rowOp (s := s) t name x dim.val Tape.rowSoftmaxForward Tape.rowSoftmaxBackward

/-- Stable log-softmax along `dim`. -/
private def logSoftmax {s : Shape} (dim : Fin s.rank) : Kernel [.value s] [.value s] :=
  ofResult (primitiveContract "log_softmax" (full "TorchLeanTest.CUDA.Softmax")
      "Log-softmax kernels follow the stable row/axis normalization contract."
      "Log-softmax VJPs are checked through runtime autograd tests."
      .torchLeanTape .implementationDefined) fun t name (x, ()) =>
    rowOp (s := s) t name x dim.val Tape.rowLogSoftmaxForward Tape.rowLogSoftmaxBackward

/-- The affine map `W x + b` for `W : (outDim, inDim)`, `x : inDim` and `b : outDim`. The node
keeps the product `W x` as workspace. -/
private def linear {inDim outDim : Nat} :
    Kernel [.value [outDim, inDim], .value [outDim], .value [inDim]] [.value [outDim]] :=
  ofResult (primitiveContract "linear" empty
      "Linear layer kernels follow the matvec/matmul plus bias contract."
      ("The weight cotangent is the native product of the seed and the input, the input " ++
        "cotangent the transposed weight times the seed, and the bias cotangent a copy of the " ++
        "seed.")
      .torchLeanTape .implementationDefined) fun t name (wId, bId, xId, ()) => do
    let some out32 := Bytes.toUInt32? outDim | throw SomeBuffer.tooLarge
    let some in32 := Bytes.toUInt32? inDim | throw SomeBuffer.tooLarge
    let one32 : UInt32 := 1
    let wBuf ← Tape.requireValue t wId (.dim outDim (.dim inDim .scalar))
    let bBuf ← Tape.requireValue t bId (.dim outDim .scalar)
    let xBuf ← Tape.requireValue t xId (.dim inDim .scalar)
    let wx := Buffer.bmm wBuf xBuf one32 out32 in32 one32
    let yBuf := Buffer.add wx bBuf
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := .dim outDim .scalar, buf := yBuf }
        requiresGrad := (t.getNode? wId).any (·.requiresGrad) ||
          (t.getNode? bId).any (·.requiresGrad) ||
          (t.getNode? xId).any (·.requiresGrad)
        parents := #[wId, bId, xId]
        cleanup := #[wx].map SomeBuffer.ofWorkspace
        backward := fun dLdyAny => do
          let dLdy ← Tape.requireGrad dLdyAny (.dim outDim .scalar)
          let g := dLdy.buf
          let dW := Buffer.bmm g xBuf one32 out32 one32 in32
          let db := Buffer.copy g
          let dx := Buffer.bmm wBuf g one32 in32 out32 one32 (transposeA := 1)
          pure #[
              (wId, { s := .dim outDim (.dim inDim .scalar), buf := dW })
            , (bId, { s := .dim outDim .scalar, buf := db })
            , (xId, { s := .dim inDim .scalar, buf := dx }) ] }
    pure (t.addNode node)

/-- Mean-squared error with `"mean"` reduction. The node keeps the difference of its inputs as
workspace for the pullback. -/
private def mseLoss {s : Shape} : Kernel [.value s, .value s] [.value .scalar] :=
  ofResult (primitiveContract "mse_loss" (forwardOnly "TorchLeanTest.CUDA.Trainer")
      "MSE loss follows the mean squared residual contract."
      ("The VJP is the native slope of `TorchLean.Tensor.meanSquaredErrorBackward` times the seed, \
        negated " ++
        "for the target.")
      .torchLeanTape .implementationDefined) fun t name (yhatId, targetId, ()) => do
    let yhat ← Tape.requireValue t yhatId s
    let target ← Tape.requireValue t targetId s
    checkDivCount "mse_loss" (Buffer.dtype yhat) (Tensor.meanDenominator s)
    let diff := Buffer.sub yhat target
    -- `Tensor.meanSquaredError`, divided by one when empty (`Buffer.mseMean`).
    let count := (Tensor.meanDenominator s).toUInt64
    let mean := Buffer.mseMean diff count
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := Shape.scalar, buf := mean }
        requiresGrad := (t.getNode? yhatId).any (·.requiresGrad) ||
          (t.getNode? targetId).any (·.requiresGrad)
        parents := #[yhatId, targetId]
        cleanup := #[diff].map SomeBuffer.ofWorkspace
        backward := fun dLdyAny => do
          let dLdy ← Tape.requireGrad dLdyAny Shape.scalar
          -- `Tensor.meanSquaredErrorBackward` times the seed (`Buffer.mseSlope`).
          let gBroad := broadcastScalarToShape dLdy.buf s
          let slope := Buffer.mseSlope diff count
          let dYhat := Buffer.mul slope gBroad
          let dTarget := Buffer.releaseThen gBroad <|
            Buffer.releaseThen slope <|
              Buffer.scale dYhat (-1.0)
          pure #[
            (yhatId, { s := s, buf := dYhat }),
            (targetId, { s := s, buf := dTarget })
          ] }
    pure (t.addNode node)

/-- `Op.huberLoss mean` with the target and the threshold frozen: `Huber.term` at each entry
(`Buffer.huberTerms`), summed by `reduceSum` and, for the mean, divided once by the count
(`divCount`). The node keeps the difference for the pullback, the slope of
`Tensor.huberLossBackward` (`Buffer.huberSlope`) times the seed. A failed upload is the error. -/
private def huberLoss {s : Shape} (mean : Bool) :
    Kernel [.value s, .frozen s, .frozen []] [.value .scalar] where
  record t name _ := fun (predictionId, target, delta, ()) => do
    let prediction : Result Buffer := do
      let prediction ← Tape.requireValue t predictionId s
      if mean then
        checkDivCount "huber_loss" (Buffer.dtype prediction) (Tensor.meanDenominator s)
      pure prediction
    match prediction with
    | .error message => pure (t, .error message)
    | .ok prediction =>
      try
        let targetBuf ← Buffer.ofFloatArrayIO (target.to FloatArray)
          (Buffer.dtype prediction)
        let threshold := delta.item
        let count := (Tensor.meanDenominator s).toUInt64
        let (terms, diff) := Buffer.releaseThen targetBuf
          (Buffer.huberTerms prediction targetBuf threshold, Buffer.sub prediction targetBuf)
        let total := Buffer.releaseThen terms (Buffer.reduceSum terms)
        let loss := if mean then Buffer.releaseThen total (Buffer.divCount total count) else total
        let node : Node SomeBuffer :=
          { name := some name
            value := { s := Shape.scalar, buf := loss }
            requiresGrad := (t.getNode? predictionId).any (·.requiresGrad)
            parents := #[predictionId]
            cleanup := #[diff].map SomeBuffer.ofWorkspace
            backward := fun dLdyAny => do
              let dLdy ← Tape.requireGrad dLdyAny Shape.scalar
              let gBroad := broadcastScalarToShape dLdy.buf s
              let slope := Buffer.huberSlope diff threshold
              let scaled :=
                if mean then Buffer.releaseThen slope (Buffer.divCount slope count) else slope
              let dPrediction := Buffer.releaseThen gBroad <| Buffer.releaseThen scaled <|
                Buffer.mul scaled gBroad
              pure #[(predictionId, { s := s, buf := dPrediction })] }
        let (t, id) := t.addNode node
        pure (t, .ok (id, ()))
      catch e => pure (t, .error (toString e))
  contract? := primitiveContract "huber_loss" (full "TorchLeanTest.CUDA.Trainer")
    "Huber loss follows TorchLean.Huber.term at each entry, then the sum or the mean."
    "Huber VJP is checked by CUDA runtime coverage."
    .torchLeanTape .implementationDefined

/-! ## Shapes and broadcasting -/

private def reshape {s t : Shape} : Kernel [.value s] [.value t] :=
  ofResult (viewContract "reshape" empty .torchLeanTape) (configuredFormats := true)
      fun tape name (x, ()) =>
    reshapeNode s t tape name x

/-- The axes permuted by adjacent swaps of the buffer. The pullback applies the swaps to the
cotangent in reverse order, and the identity permutation copies. -/
private def permute {s : Shape} (perm : Equiv.Perm (Fin s.rank)) (out : Shape) :
    Kernel [.value s] [.value out] :=
  ofResult (viewContract "permute" empty .torchLeanTape) fun t name (x, ()) => do
    let depths := swapDepths ((List.finRange s.rank).map fun i => (perm i).val)
    unless s.permute perm == out do
      throw "autograd: cuda: permute: the output shape is not the permuted input shape"
    let steps ← Tape.Indexing.checkedSwapSteps depths
    Tape.unary t name x s out
      (fun x => if steps.isEmpty then Buffer.copy x else Tape.Indexing.permuteBorrowed x s steps)
      fun _x dLdy =>
        if steps.isEmpty then Buffer.copy dLdy
        else Tape.Indexing.permuteBorrowed dLdy out steps.reverse

/-- Broadcast to `t`, with a pullback that sums the cotangent over the broadcast axes natively. -/
@[inline] private def broadcastNode {s t : Shape} (cb : Shape.CanBroadcastTo s t)
    (tape : Tape SomeBuffer) (name : String) (x : Nat) : Result (Tape SomeBuffer × Nat) :=
  let inDims := s.toArray
  let outDims := t.toArray
  let axisMap := Broadcast.axisMap cb
  Tape.unary tape name x s t (fun x => Buffer.broadcastTo x inDims outDims axisMap)
    fun _x dLdy => Buffer.reduceFromBroadcastTo dLdy inDims outDims axisMap

private def broadcast {s t : Shape} (cb : Shape.CanBroadcastTo s t) :
    Kernel [.value s] [.value t] :=
  ofResult (viewContract "broadcast" empty .backendVJP) fun tape name (x, ()) =>
    broadcastNode cb tape name x

/-! ## Matrix products -/

/-- The batched product of `[batch, m, n]` and `[batch, n, p]` buffers, with the scale kept inside
each contraction when there is one. The pullbacks are the products with the other operand
transposed, scaled the same way. -/
@[inline] private def bmm (batch m n p : Nat) (scale : Option Float) (t : Tape SomeBuffer)
    (a b : Nat) :
    Result (Tape SomeBuffer × Nat) := do
  let some b32 := Bytes.toUInt32? batch | throw SomeBuffer.tooLarge
  let some m32 := Bytes.toUInt32? m | throw SomeBuffer.tooLarge
  let some n32 := Bytes.toUInt32? n | throw SomeBuffer.tooLarge
  let some p32 := Bytes.toUInt32? p | throw SomeBuffer.tooLarge
  let σ₁ : Shape := .dim batch (.dim m (.dim n .scalar))
  let σ₂ : Shape := .dim batch (.dim n (.dim p .scalar))
  let τ : Shape := .dim batch (.dim m (.dim p .scalar))
  Tape.binary t "bmm" a b σ₁ σ₂ τ
    (fun a b => match scale with
      | none => Buffer.bmm a b b32 m32 n32 p32
      | some scale => Buffer.scaledProduct a b b32 m32 n32 p32 scale)
    fun a b dLdy =>
      let dA := match scale with
        | none => Buffer.bmm dLdy b b32 m32 p32 n32 (transposeB := 1)
        | some scale => Buffer.scaledProduct dLdy b b32 m32 p32 n32 scale (transposeB := true)
      let dB := match scale with
        | none => Buffer.bmm a dLdy b32 n32 m32 p32 (transposeA := 1)
        | some scale => Buffer.scaledProduct a dLdy b32 n32 m32 p32 scale (transposeA := true)
      (dA, dB)

/-- The broadcast matrix product as six recorded nodes: both operands broadcast to the common batch
prefix, both viewed as `[batch.size, rows, cols]`, the batched product, and the view back to
`batch ++ [m, p]`. -/
private def matmulProgram {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (scale : Option Float)
    (t : Tape SomeBuffer) (a b : Nat) : Result (Tape SomeBuffer × Nat) := do
  let (t, commonA) ← broadcastNode
    (Shape.CanBroadcastTo.concat_right [m, n] broadcastA) t
      "broadcastTo" a
  let (t, commonB) ← broadcastNode
    (Shape.CanBroadcastTo.concat_right [n, p] broadcastB) t
      "broadcastTo" b
  let (t, flatA) := ← reshapeNode (batch ++ [m, n]) [batch.size, m, n] t "reshape" commonA
  let (t, flatB) := ← reshapeNode (batch ++ [n, p]) [batch.size, n, p] t "reshape" commonB
  let (t, flatOut) ← bmm batch.size m n p scale t flatA flatB
  reshapeNode [batch.size, m, p] (batch ++ [m, p]) t "reshape" flatOut

/-- `matmulProgram` with vector operands, as `Op.matmul`'s rule reads them: a left vector (`lv`)
viewed as a one-row matrix and a right vector (`rv`) as a one-column matrix, the program, and the
result viewed without the axis of each vector. Two matrices get the program alone. -/
private def matmulViews {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (scale : Option Float)
    (t : Tape SomeBuffer) (a b : Nat) : Result (Tape SomeBuffer × Nat) := do
  let (t, a) ← if lv then
      reshapeNode (Shape.matmulLeft lv batchA m n) (batchA ++ [m, n]) t "reshape" a
    else pure (t, a)
  let (t, b) ← if rv then
      reshapeNode (Shape.matmulRight rv batchB n p) (batchB ++ [n, p]) t "reshape" b
    else pure (t, b)
  let (t, c) ← matmulProgram (m := m) (n := n) (p := p) broadcastA broadcastB scale t a b
  if lv || rv then
    reshapeNode (batch ++ [m, p]) (Shape.matmulOut lv rv batch m p) t "reshape" c
  else pure (t, c)

private def matmul {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    Kernel [.value (Shape.matmulLeft lv batchA m n), .value (Shape.matmulRight rv batchB n p)]
      [.value (Shape.matmulOut lv rv batch m p)] :=
  ofResult (primitiveContract "matmul" (full "TorchLeanTest.CUDA.Graph")
      "Matrix products agree with the row-major runtime contract."
      "Backward products are checked through autograd/runtime parity."
      .torchLeanTape .implementationDefined) fun t _ (a, b, ()) =>
    matmulViews (m := m) (n := n) (p := p) lv rv broadcastA broadcastB none t a b

/-- `matmul` with the frozen scale kept inside each contraction (`Buffer.scaledProduct`) and
conservative range recovery from the original operands: its range-loss flag costs at most one
device synchronization, and a flagged product may run reference coefficient arithmetic on the CPU.
Neither this mixed execution path nor its VJP promises a fixed-left accumulation schedule. -/
private def scaledMatmul {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    Kernel [.value (Shape.matmulLeft lv batchA m n), .value (Shape.matmulRight rv batchB n p),
        .frozen []] [.value (Shape.matmulOut lv rv batch m p)] :=
  ofResult (primitiveContract "scaled_matmul" (full "TorchLeanTest.CUDA.Graph")
      "ATen evaluates scaled matrix products; flagged range loss replays original finite operands."
      "TorchLean records scaled matrix cotangents with the same range-recovery boundary."
      .torchLeanTape .implementationDefined) fun t _ (a, b, scale, ()) =>
    matmulViews (m := m) (n := n) (p := p) lv rv broadcastA broadcastB (some scale.item) t a b

/-! ## Convolution and pooling

The pooling and transposed-convolution kernels take one channel-first sample `(C, spatial...)` with
one to three spatial axes, the window parameters as dimension arrays, padding equal before and
after, unit dilation and one group. Pooling folds the axes before the spatial ones into `C`, one
channel when there are none. The convolution kernel takes a batch `(N, C, spatial...)`, PyTorch's
packed grouped kernel, and dilation and padding before and after each axis. An input whose batch
axis stands for several leading axes, or for none, is viewed as that batch. The pooling and
transposed-convolution checks keep the order and texts of the native spatial operations: rank,
kernel and stride, the channels, then each array against the `UInt32` ABI, then the output size.

The convolution has no entry outside its attributes' domain (`groupedConv`: another layout, a
spatial rank outside one to three, a zero stride, dilation or group count). Its recording checks
the kernel, the channels, each array and the output size in that order. A native failure of its
forward call is the entry's error, while a native failure in the pooling and
transposed-convolution kernels aborts the process. -/

/-- The spatial rank, kernel and stride checks of the native spatial operations. -/
private def checkSpatial (opName : String) (d : Nat) (kernel stride : List Nat) :
    Result Unit := do
  if d = 0 then
    throw s!"autograd: cuda: {opName}: d=0 is not supported"
  if d > 3 then
    throw s!"autograd: cuda: {opName}: spatial rank must be at most 3"
  if kernel.any (· == 0) then
    throw s!"autograd: cuda: {opName}: kernel_size must be > 0"
  if stride.any (· == 0) then
    throw s!"autograd: cuda: {opName}: stride must be > 0"

/-- The spatial metadata as native arrays, axis sizes checked before element counts. -/
private def spatialArrays (opName : String) (spatial kernel stride padding : List Nat) :
    Result (Array Nat × Array Nat × Array Nat × Array Nat) := do
  let arrays := (spatial.toArray, kernel.toArray, stride.toArray, padding.toArray)
  Tape.validateU32Dimensions opName arrays.1
  Tape.validateU32Dimensions opName arrays.2.1
  Tape.validateU32Dimensions opName arrays.2.2.1
  Tape.validateU32Dimensions opName arrays.2.2.2
  let _ ← SomeBuffer.numelU32 spatial
  let _ ← SomeBuffer.numelU32 kernel
  pure arrays

/-- The output's spatial and full element counts against the `UInt32` ABI, the spatial axes
after the `lead` leading ones. -/
private def checkOutput (lead : Nat) (out : Shape) : Result Unit := do
  let _ ← SomeBuffer.numelU32 (out.drop lead)
  let _ ← SomeBuffer.numelU32 out

/-- Whether pooling attributes are the native ones: the trailing axes, the axes before them folding
into the native channel count (one channel when there are none), padding equal before and after,
unit dilation, floor mode, one window parameter per spatial axis. -/
private def nativePool {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) : Bool :=
  let d := kernel.length
  d ≤ s.rank && dims.map (·.val) == (List.range s.rank).drop (s.rank - d) && padLow == padHigh &&
    dilation == List.replicate d 1 && !ceilMode && stride.length == d && padLow.length == d

/-- A pooling node from its forward and backward kernels on the native metadata. -/
@[inline] private def poolNode {s : Shape} (out : Shape) (opName : String)
    (kernel stride padding : List Nat)
    (forward : Buffer → Array Nat → Array Nat → Array Nat → Array Nat → UInt32 → Buffer)
    (backward : Buffer → Buffer → Array Nat → Array Nat → Array Nat → Array Nat → UInt32 → Buffer)
    (t : Tape SomeBuffer) (name : String) (xId : Nat) : Result (Tape SomeBuffer × Nat) := do
  let lead := s.rank - kernel.length
  checkSpatial opName kernel.length kernel stride
  let some inC32 := Bytes.toUInt32? (s.take lead).prod | throw SomeBuffer.tooLarge
  let (inSpatial, kernelArr, strideArr, paddingArr) ←
    spatialArrays opName (s.drop lead) kernel stride padding
  checkOutput lead out
  let xBuf ← Tape.requireValue t xId s
  let output ← SomeBuffer.validate
    { s := out, buf := forward xBuf inSpatial kernelArr strideArr paddingArr inC32 }
  let node : Node SomeBuffer :=
    { name := some name
      value := output
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← Tape.requireGrad dLdyAny out
        let dx := backward xBuf dLdy.buf inSpatial kernelArr strideArr paddingArr inC32
        pure #[(xId, { s := s, buf := dx })] }
  pure (t.addNode node)

/-- Max pooling that skips padded cells and keeps each window's first maximum in row-major order,
ordering `0` above `-0` as `Tensor.pool .max` does, or its last NaN if it has one. -/
private def maxPool {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Kernel [.value s] [.value (s.pool dims kernel stride padLow padHigh dilation ceilMode)] :=
  ofResult (primitiveContract "max_pool" (forwardOnly "TorchLeanTest.CUDA.Graph")
      "LibTorch CUDA `max_pool` follows the channel-first runtime contract."
      "LibTorch CUDA `max_pool` VJP calls the native backward primitive."
      .backendVJP .implementationDefined) fun t name (x, ()) =>
    poolNode (s := s) (s.pool dims kernel stride padLow padHigh dilation ceilMode) "max_pool"
      kernel stride padLow Buffer.maxPool Buffer.maxPoolBackward t name x

/-- Average pooling counting the padding (PyTorch's `count_include_pad`). -/
private def avgPool {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Kernel [.value s] [.value (s.pool dims kernel stride padLow padHigh dilation ceilMode)] :=
  ofResult (primitiveContract "avg_pool" (forwardOnly "TorchLeanTest.CUDA.Graph")
      "Average-pooling follows the channel-first window contract."
      "Average-pooling VJP calls the native backward primitive."
      .backendVJP .implementationDefined) fun t name (x, ()) =>
    poolNode (s := s) (s.pool dims kernel stride padLow padHigh dilation ceilMode) "avg_pool"
      kernel stride padLow Buffer.avgPool (fun _x dLdy => Buffer.avgPoolBackward dLdy) t
      name x

/-- Smooth max pooling. `beta` must be finite and nonzero in the input dtype, where conversion can
underflow a nonzero value to zero or overflow a finite value to infinity. -/
private def smoothMaxPool {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    Kernel [.value s, .frozen []]
      [.value (s.pool dims kernel stride padLow padHigh dilation ceilMode)] :=
  ofResult (primitiveContract "smooth_max_pool" empty
      ("Smooth max pooling uses finite nonzero beta and max/min shifts " ++
        "(0 in the forward if infinite).")
      "The native backward rule uses the shifted window weights."
      .backendVJP .implementationDefined) fun t name (xId, beta, ()) => do
    let beta := beta.item
    let out := s.pool dims kernel stride padLow padHigh dilation ceilMode
    let lead := s.rank - kernel.length
    checkSpatial "smooth_max_pool" kernel.length kernel stride
    let some inC32 := Bytes.toUInt32? (s.take lead).prod | throw SomeBuffer.tooLarge
    let (inSpatial, kernelArr, strideArr, paddingArr) ←
      spatialArrays "smooth_max_pool" (s.drop lead) kernel stride padLow
    checkOutput lead out
    let xBuf ← Tape.requireValue t xId s
    let effectiveBeta := if Buffer.dtype xBuf == .float32 then beta.toFloat32.toFloat else beta
    if !effectiveBeta.isFinite || effectiveBeta == 0 then
      throw "autograd: cuda: smooth_max_pool: beta must be finite and nonzero"
    let output ← SomeBuffer.validate
      { s := out
        buf := Buffer.smoothMaxPool xBuf beta inSpatial kernelArr strideArr paddingArr
          inC32 }
    let node : Node SomeBuffer :=
      { name := some name
        value := output
        requiresGrad := (t.getNode? xId).any (·.requiresGrad)
        parents := #[xId]
        backward := fun dLdyAny => do
          let dLdy ← Tape.requireGrad dLdyAny out
          let dx := Buffer.smoothMaxPoolBackward xBuf dLdy.buf beta inSpatial kernelArr strideArr
            paddingArr inC32
          pure #[(xId, { s := s, buf := dx })] }
    pure (t.addNode node)

/-- Whether convolution attributes are those of the native grouped convolution: PyTorch's
channel-first layout, the input `(N, C, spatial...)` and the packed kernel
`(O, C / groups, window...)`, one to three spatial axes, at least one group, and positive strides
and dilations. The padding may differ before and after an axis. -/
private def groupedConv {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride dilation : Fin r → Nat) (groups : Nat) : Bool :=
  1 ≤ r && r ≤ 3 && s.rank == r + 2 && k.rank == r + 2 && 1 ≤ groups &&
    (List.finRange (r + 2)).all (fun i => (dims.input i).val == i.val &&
      (dims.kernel i).val == i.val) &&
    (List.finRange r).all fun j => 1 ≤ stride j && 1 ≤ dilation j

/-- Whether transposed-convolution attributes are the native ones: one sample in PyTorch's
channel-first layout with the kernel `(inC, outC, window...)` whose `inC` is the input's channel
count, padding equal before and after, unit dilation, no output padding and one group. -/
private def nativeConvTranspose {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (padLow padHigh dilation outputPadding : Fin r → Nat) (groups : Nat) : Bool :=
  let kernelAxis := fun i : Fin (r + 2) => if i.val = 0 then 1 else if i.val = 1 then 0 else i.val
  s.headD 0 == 1 && s.rank == r + 2 && k.rank == r + 2 && groups == 1 &&
    k.headD 0 == s.getD 1 0 &&
    (List.finRange (r + 2)).all (fun i => (dims.input i).val == i.val &&
      (dims.kernel i).val == kernelAxis i) &&
    (List.finRange r).all fun j => padLow j == padHigh j && dilation j == 1 && outputPadding j == 0

/-- A transposed-convolution node. The sample's buffer is the unbatched `(inC, spatial...)` input
of the native kernel, and the native backward returns the kernel, bias and input gradients. -/
@[inline] private def convNode {s k : Shape} (out : Shape) (opName : String) (inC outC : Nat)
    (kernelId biasId inputId : Nat) (stride padding : List Nat)
    (forward : Buffer → Buffer → Buffer → Array Nat → Array Nat → Array Nat → Array Nat →
      UInt32 → UInt32 → Buffer)
    (backward : Buffer → Buffer → Buffer → Array Nat → Array Nat → Array Nat → Array Nat →
      UInt32 → UInt32 → Buffer × Buffer × Buffer)
    (t : Tape SomeBuffer) (name : String) : Result (Tape SomeBuffer × Nat) := do
  let window := k.drop 2
  checkSpatial opName window.length window stride
  let some inC32 := Bytes.toUInt32? inC | throw SomeBuffer.tooLarge
  let some outC32 := Bytes.toUInt32? outC | throw SomeBuffer.tooLarge
  let (inSpatial, kernelArr, strideArr, paddingArr) ←
    spatialArrays opName (s.drop 2) window stride padding
  checkOutput 1 out
  let kernelBuf ← Tape.requireValue t kernelId k
  let biasBuf ← Tape.requireValue t biasId (.dim outC .scalar)
  let inputBuf ← Tape.requireValue t inputId s
  let output ← SomeBuffer.validate
    { s := out
      buf := forward inputBuf kernelBuf biasBuf inSpatial kernelArr strideArr paddingArr inC32
        outC32 }
  let node : Node SomeBuffer :=
    { name := some name
      value := output
      requiresGrad := (t.getNode? kernelId).any (·.requiresGrad) ||
        (t.getNode? biasId).any (·.requiresGrad) ||
        (t.getNode? inputId).any (·.requiresGrad)
      parents := #[kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← Tape.requireGrad dLdyAny out
        let (dKernel, dBias, dInput) :=
          backward inputBuf kernelBuf dLdy.buf inSpatial kernelArr strideArr paddingArr inC32
            outC32
        pure #[
          (kernelId, { s := k, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := s, buf := dInput })
        ] }
  pure (t.addNode node)

/-- The convolution of the batch `s = N :: C :: spatial` with the packed kernel
`k = O :: C / groups :: window` (`Buffer.conv`), with `groups` feature groups, dilation and padding
before and after each spatial axis. The pullback (`Buffer.convBackward`) returns the kernel, bias
and input gradients. A native failure of the forward call is the entry's error. -/
private def conv {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) :
    Kernel [.value k, .value [k.extent (dims.kernel ⟨0, by omega⟩)], .value s]
      [.value (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1)] where
  record t name _ := fun (kernelId, biasId, inputId, ()) => do
    let out := s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1
    let biasShape : Shape := [k.extent (dims.kernel ⟨0, by omega⟩)]
    let strides := (List.ofFn stride).toArray
    let before := (List.ofFn padLow).toArray
    let after := (List.ofFn padHigh).toArray
    let dilations := (List.ofFn dilation).toArray
    let checked : Result (UInt32 × Buffer × Buffer × Buffer) := do
      checkSpatial "conv" r (k.drop 2) strides.toList
      let inC := s.getD 1 0
      let outC := k.headD 0
      unless inC % groups == 0 && outC % groups == 0 && k.getD 1 0 * groups == inC do
        throw s!"autograd: cuda: conv: {groups} groups do not divide {inC} input and {outC} \
          output channels with a kernel of {k.getD 1 0} input channels"
      let some groups32 := Bytes.toUInt32? groups | throw SomeBuffer.tooLarge
      for metadata in [s.toArray, k.toArray, strides, before, after, dilations] do
        Tape.validateU32Dimensions "conv" metadata
      -- The spatial products first, so that no channel count of zero hides an oversized one.
      let _ ← SomeBuffer.numelU32 (s.drop 2)
      let _ ← SomeBuffer.numelU32 (k.drop 2)
      let _ ← SomeBuffer.numelU32 s
      let _ ← SomeBuffer.numelU32 k
      checkOutput 2 out
      pure (groups32, ← Tape.requireValue t kernelId k, ← Tape.requireValue t biasId biasShape,
        ← Tape.requireValue t inputId s)
    match checked with
    | .error message => pure (t, .error message)
    | .ok (groups32, kernelBuf, biasBuf, inputBuf) =>
      try
        let y ← Buffer.conv inputBuf kernelBuf biasBuf s.toArray k.toArray strides before after
          dilations groups32
        match SomeBuffer.validate { s := out, buf := y } with
        | .error message => pure (t, .error message)
        | .ok output =>
          let node : Node SomeBuffer :=
            { name := some name
              value := output
              requiresGrad := (t.getNode? kernelId).any (·.requiresGrad) ||
                (t.getNode? biasId).any (·.requiresGrad) ||
                (t.getNode? inputId).any (·.requiresGrad)
              parents := #[kernelId, biasId, inputId]
              backward := fun dLdyAny => do
                let dLdy ← Tape.requireGrad dLdyAny out
                let (dKernel, dBias, dInput) :=
                  Buffer.convBackward inputBuf kernelBuf dLdy.buf s.toArray k.toArray strides
                    before after dilations groups32
                pure #[
                  (kernelId, { s := k, buf := dKernel }),
                  (biasId, { s := biasShape, buf := dBias }),
                  (inputId, { s := s, buf := dInput })
                ] }
          let (t, id) := t.addNode node
          pure (t, .ok (id, ()))
      catch e => pure (t, .error (toString e))
  contract? :=
    let c := primitiveContract "conv" (full "TorchLeanTest.CUDA.Graph")
      "Convolution follows the generic channel-first runtime contract."
      "Convolution VJP calls the native backward primitive."
      .backendVJP .implementationDefined
    some { c with
      vjp.qualification :=
        "`+0` gradients at an empty output, where the Lean rule multiplies the absent cotangent " ++
          "entries, read as zero, by the weights (a NaN input gradient at an infinite weight), " ++
          "and the same products past a nonempty output are untested at infinite weights" }

/-- The convolution of an input `x` whose batch axis stands for several leading axes, as
`Op.conv`'s rule reads it: `x` viewed at the one-batch shape `s` (the leading axes folded row-major
in place), `conv`'s node, and its output viewed at `y`. When `x` is `s` and `y` is `conv`'s output
shape there are no views. -/
private def convLeading {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (x y : Shape) :
    Kernel [.value k, .value [k.extent (dims.kernel ⟨0, by omega⟩)], .value x] [.value y] where
  record t name dtype := fun (kernelId, biasId, inputId, ()) => do
    let entry := conv dims stride padLow padHigh dilation groups
    let out := s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1
    if x = s && y = out then
      entry.record t name dtype (kernelId, biasId, inputId, ())
    else
      match reshapeNode x s t "reshape" inputId with
      | .error message => pure (t, .error message)
      | .ok (viewed, input) =>
        let (convolved, result) ← entry.record viewed name dtype (kernelId, biasId, input, ())
        match result >>= fun (z, ()) => reshapeNode out y convolved "reshape" z with
        | .error message => pure (t, .error message)
        | .ok (t, id) => pure (t, .ok (id, ()))
  contract? := (conv dims stride padLow padHigh dilation groups).contract?

/-- The contract of the `convTranspose` entry, with the qualification of its VJP. -/
private def convTransposeContract : Backend.KernelContract :=
  let c := primitiveContract
    "conv_transpose"
    empty
    "Transpose convolution follows the generic channel-first runtime contract."
    "Transpose-convolution VJP calls the native backward primitive."
    .backendVJP .implementationDefined
  { c with
    vjp.qualification :=
      "`+0` kernel and input gradients at an empty output, where the Lean rule multiplies the " ++
        "absent cotangent entries, read as zero, by the input and kernel entries (NaN at an " ++
        "infinite entry), and the same products past a nonempty output are untested" }

/-- Channel-first transposed convolution with the kernel in PyTorch's `(inC, outC, kernel...)`
layout. At an empty output the native backward (`Buffer.convTransposeBackward`) returns `+0` kernel
and input gradients, where `Op.convTranspose`'s rule multiplies the absent cotangent entries, read
as zero, by the input and kernel entries (NaN at an infinite entry). Past a nonempty output the
rule makes the same products, and the CUDA suite has no transposed-convolution row. -/
private def convTranspose {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    Kernel [.value k, .value [groups * k.extent (dims.kernel ⟨0, by omega⟩)], .value s]
      [.value (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)] :=
  ofResult convTransposeContract fun t name (kernelId, biasId, inputId, ()) =>
    convNode (s := s) (k := k)
      (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups) "conv_transpose"
      (k.headD 0) (k.getD 1 0)
      kernelId biasId inputId (List.ofFn stride) (List.ofFn padLow)
      Buffer.convTranspose Buffer.convTransposeBackward t name

/-! ## Attention -/

/-- Attention after its score product, with conservative host range recovery in its pullback
(`Buffer.attentionValuesBackward`): the softmax weights of the scores (hard-masked when a mask is
given), times the multiplier, times the value. The saved probabilities are the node's only
workspace, and the input buffers stay borrowed tape parents. -/
private def attentionFromScores {batch : Shape} {rows keys width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) :
    Kernel [.value (batch ++ [rows, keys]), .value (batch ++ [keys, width]),
      .value (batch ++ [rows, keys])] [.value (batch ++ [rows, width])] :=
  ofResult (primitiveContract "attention" empty
      "ATen executes row softmax, an optional multiplier and matrix multiplication."
      "TorchLean retains saved probabilities and original cotangents through the centered pullback."
      .torchLeanTape .implementationDefined) fun t name (scoresId, valueId, multiplierId, ()) => do
    let some b := Bytes.toUInt32? batch.size | throw SomeBuffer.tooLarge
    let some r := Bytes.toUInt32? rows | throw SomeBuffer.tooLarge
    let some k := Bytes.toUInt32? keys | throw SomeBuffer.tooLarge
    let some w := Bytes.toUInt32? width | throw SomeBuffer.tooLarge
    let some rowCount := Bytes.toUInt32? (batch.size * rows) | throw SomeBuffer.tooLarge
    let some scoreCount := Bytes.toUInt32? (batch.size * rows * keys) | throw SomeBuffer.tooLarge
    let some _ := Bytes.toUInt32? (batch.size * keys * width) | throw SomeBuffer.tooLarge
    let some outputCount := Bytes.toUInt32? (batch.size * rows * width) | throw SomeBuffer.tooLarge
    let scoreShape := batch ++ [rows, keys]
    let valueShape := batch ++ [keys, width]
    let outputShape := batch ++ [rows, width]
    let scores ← Tape.requireValue t scoresId scoreShape
    let value ← Tape.requireValue t valueId valueShape
    let multiplier ← Tape.requireValue t multiplierId scoreShape
    unless scores.dtype == value.dtype && scores.dtype == multiplier.dtype do
      throw "attention: inputs must have the same buffer dtype"
    let probabilities := if scoreCount == 0 then Buffer.zeros scoreCount scores.dtype else
      match mask with
      | none =>
          let normalized := Tape.rowSoftmaxForward scores rowCount k
          normalized.releaseWorkspaceThen normalized.value
      | some mask =>
          let source := Buffer.ofFloatArray (Convert.flattenBoolMask mask) scores.dtype
          let allowed :=
            Buffer.broadcastTo source #[rows, keys] #[batch.size, rows, keys] #[0, 1, 2]
          Buffer.releaseThen source <| Buffer.releaseThen allowed <|
            Buffer.hardMaskedSoftmaxByRow scores allowed rowCount k
    let mixed := Buffer.mul probabilities multiplier
    let output := Buffer.releaseThen mixed <|
      if outputCount == 0 then Buffer.zeros outputCount scores.dtype else
        Buffer.bmm mixed value b r k w
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := outputShape, buf := output }
        requiresGrad := #[scoresId, valueId, multiplierId].any fun id =>
          (t.getNode? id).any (·.requiresGrad)
        parents := #[scoresId, valueId, multiplierId]
        cleanup := #[SomeBuffer.ofWorkspace probabilities]
        backward := fun seed => do
          let gradient ← Tape.requireGrad seed outputShape
          let (ds, dv, dm) := Buffer.attentionValuesBackward probabilities value multiplier
            gradient.buf b r k w
          pure #[(scoresId, { s := scoreShape, buf := ds }),
            (valueId, { s := valueShape, buf := dv }),
            (multiplierId, { s := scoreShape, buf := dm })] }
    pure (t.addNode node)

/-- The native multi-head attention node of `batch` sequences laid out as `inputShape`, the steps of
`MultiHeadAttention.forward` with query and key/value `x`: the projections `Q = x Wq`, `K = x Wk`,
`V = x Wv` over the rows of every sequence, the heads split to `(batch, numHeads, n, headDim)`, the
attention of every head, and the heads combined and projected by `Wo`. As in
`MultiHeadAttention.forward` and its pullback, a blocked entry has a zero softmax numerator, a fully
blocked row has zero weights, and the products multiply zero weights by nonfinite entries into NaN.
On a row whose allowed scores give a NaN denominator, a blocked weight stays zero where the
reference weight is NaN. Lean composes the forward and the backward and keeps the probabilities on
the tape, while LibTorch runs the numerical primitives without an autograd graph. The host mask is
copied to the device. -/
private def multiheadAttentionNode
  {n numHeads dModel headDim : Nat} (_hSeq : n ≠ 0)
  (batch : Nat) (_hBatch : batch ≠ 0) (inputShape : Shape)
  (t : Tape SomeBuffer) (name : String) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n])) :
  IO (Result (Tape SomeBuffer × Nat)) := (do
  let one32 : UInt32 := 1
  let depth1 : UInt32 := 1
  let some n32 := Bytes.toUInt32? n | throw SomeBuffer.tooLarge
  let some dModel32 := Bytes.toUInt32? dModel | throw SomeBuffer.tooLarge
  let some head32 := Bytes.toUInt32? headDim | throw SomeBuffer.tooLarge
  let projDim : Nat := numHeads * headDim
  let some proj32 := Bytes.toUInt32? projDim | throw SomeBuffer.tooLarge
  let rows : Nat := batch * n
  let some rows32 := Bytes.toUInt32? rows | throw SomeBuffer.tooLarge
  let batchHeads : Nat := batch * numHeads
  let some batchHeads32 := Bytes.toUInt32? batchHeads | throw SomeBuffer.tooLarge
  let wq ← ExceptT.mk (pure <| Tape.requireValue (t := t) wqId (.dim dModel (.dim projDim .scalar)))
  let wk ← ExceptT.mk (pure <| Tape.requireValue (t := t) wkId (.dim dModel (.dim projDim .scalar)))
  let wv ← ExceptT.mk (pure <| Tape.requireValue (t := t) wvId (.dim dModel (.dim projDim .scalar)))
  let wo ← ExceptT.mk (pure <| Tape.requireValue (t := t) woId (.dim projDim (.dim dModel .scalar)))
  let x ← ExceptT.mk (pure <| Tape.requireValue (t := t) xId inputShape)
  -- Flatten the leading batch into the row axis for the shared projections.
  let Q := Buffer.bmm x wq one32 rows32 dModel32 proj32
  let K := Buffer.bmm x wk one32 rows32 dModel32 proj32
  let V := Buffer.bmm x wv one32 rows32 dModel32 proj32
  -- Split heads:
  --   `(batch,n,projDim)` views as `(batch,n,numHeads,headDim)`, then swaps to
  --   `(batch,numHeads,n,headDim)`. The first two axes are folded into the BMM batch axis.
  let dimsView : Array Nat := #[batch, n, numHeads, headDim]
  let dimsHead : Array Nat := #[batch, numHeads, n, headDim]
  let Qh := Buffer.releaseThen Q <| Buffer.swapAdjacentAtDepth Q dimsView depth1
  let Kh := Buffer.releaseThen K <| Buffer.swapAdjacentAtDepth K dimsView depth1
  let Vh := Buffer.releaseThen V <| Buffer.swapAdjacentAtDepth V dimsView depth1
  let scaleDenom : Float := if headDim = 0 then 1.0 else Float.sqrt (Float.ofNat headDim)
  let scale : Float := 1.0 / scaleDenom
  -- Optional mask: `mask[i,j]=true` means allowed.
  let maskB : Option Buffer :=
    match mask with
    | none => none
    | some m =>
        let mF := Buffer.ofFloatArray
          (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m) (Buffer.dtype Qh)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[batchHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        let maskB := Buffer.broadcastTo mF inDims outDims axisMap
        some (Buffer.releaseThen mF maskB)
  let result ← liftM <| IO.lazyPure fun _ =>
    Buffer.attentionForward Qh Kh Vh maskB batchHeads32 n32 head32 scale
  for buffer in maskB.toArray do
    discard <| liftM (Buffer.releaseIO buffer)
  let (outHeads, probabilities) ← match result with
    | .ok output => pure output
    | .error message =>
        for buffer in #[Qh, Kh, Vh] do
          discard <| liftM (Buffer.releaseIO buffer)
        throw message
  -- Combine heads and fold the leading axes back into `rows` for the output projection.
  let concat := Buffer.releaseThen outHeads <|
    Buffer.swapAdjacentAtDepth outHeads dimsHead depth1
  let y := Buffer.bmm concat wo one32 rows32 proj32 dModel32
  let node : Node SomeBuffer :=
    { name := some name
      value := { s := inputShape, buf := y }
      requiresGrad := (t.getNode? wqId).any (·.requiresGrad) ||
        (t.getNode? wkId).any (·.requiresGrad) ||
        (t.getNode? wvId).any (·.requiresGrad) ||
        (t.getNode? woId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wqId, wkId, wvId, woId, xId]
      cleanup := #[Qh, Kh, Vh, probabilities, concat].map SomeBuffer.ofWorkspace
      backward := fun dLdyAny => do
        let dLdy ← Tape.requireGrad dLdyAny inputShape
        -- The pullbacks of the output projection `y = concat @ wo`.
        let dConcat :=
          Buffer.bmm dLdy.buf wo one32 rows32 dModel32 proj32 (transposeB := 1)
        let dWo :=
          Buffer.bmm concat dLdy.buf one32 proj32 rows32 dModel32 (transposeA := 1)
        let dOutHeads := Buffer.swapAdjacentAtDepth dConcat dimsView depth1
        let (dQh, dKh, dVh) ←
          match Buffer.attentionBackward Qh Kh Vh probabilities dOutHeads
              batchHeads32 n32 head32 scale with
          | .ok (dq, dk, dv) =>
              pure (Buffer.releaseThen dOutHeads dq, dk, dv)
          | .error message => .error message
        -- Undo the head permutation and view each projection gradient as `(rows, projDim)`.
        let dQ := Buffer.releaseThen dQh <| Buffer.swapAdjacentAtDepth dQh dimsHead depth1
        let dK := Buffer.releaseThen dKh <| Buffer.swapAdjacentAtDepth dKh dimsHead depth1
        let dV := Buffer.releaseThen dVh <| Buffer.swapAdjacentAtDepth dVh dimsHead depth1
        -- The pullbacks of the projections `Q = x @ wq`, `K = x @ wk` and `V = x @ wv`.
        let dxQ := Buffer.bmm dQ wq one32 rows32 proj32 dModel32 (transposeB := 1)
        let dxK := Buffer.bmm dK wk one32 rows32 proj32 dModel32 (transposeB := 1)
        let dxV := Buffer.bmm dV wv one32 rows32 proj32 dModel32 (transposeB := 1)
        let dxQK := Buffer.add dxQ dxK
        let dxRaw := Buffer.add dxQK dxV
        let dx := Buffer.releaseThen dxQ <| Buffer.releaseThen dxK <|
          Buffer.releaseThen dxV <| Buffer.releaseThen dxQK dxRaw
        let dWq := Buffer.bmm x dQ one32 dModel32 rows32 proj32 (transposeA := 1)
        let dWk := Buffer.bmm x dK one32 dModel32 rows32 proj32 (transposeA := 1)
        let dWv := Buffer.bmm x dV one32 dModel32 rows32 proj32 (transposeA := 1)
        let dWv := Buffer.releaseThen dConcat <| Buffer.releaseThen dQ <|
          Buffer.releaseThen dK <| Buffer.releaseThen dV dWv
        pure #[
          (xId,  { s := inputShape, buf := dx }),
          (wqId, { s := .dim dModel (.dim projDim .scalar), buf := dWq }),
          (wkId, { s := .dim dModel (.dim projDim .scalar), buf := dWk }),
          (wvId, { s := .dim dModel (.dim projDim .scalar), buf := dWv }),
          (woId, { s := .dim projDim (.dim dModel .scalar), buf := dWo })
        ] }
  pure (t.addNode node) : ExceptT String IO (Tape SomeBuffer × Nat)).run

/-- Multi-head self-attention of the sequences along the leading axes `batch` as one node: one
native product per projection over all `batch.size * n` rows, then the attention of the
`batch.size * numHeads` heads. The value and the input's gradient of a sequence depend on that
sequence alone. Each projection gradient is one native product summed over all rows at once, with
no per-sequence partial sums. With no leading axes it is the node of one sequence. It emits the
input's gradient before the four projections', and a leading axis of extent zero is an error. -/
private def multiheadAttention {batch : Shape} {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (mask : Option (Tensor Bool [n, n])) :
    Kernel [.value [dModel, numHeads * headDim], .value [dModel, numHeads * headDim],
      .value [dModel, numHeads * headDim], .value [numHeads * headDim, dModel],
      .value (batch ++ [n, dModel])] [.value (batch ++ [n, dModel])] where
  record t name _ := fun (wq, wk, wv, wo, x, ()) => do
    if hb : batch.size ≠ 0 then
      let result ← try
          multiheadAttentionNode (n := n) (numHeads := numHeads) (dModel := dModel)
            (headDim := headDim) h1 batch.size hb (batch ++ [n, dModel]) t name wq wk wv wo x
            mask
        catch e => pure (.error (toString e))
      pure <| match result with
        | .ok (recorded, id) => (recorded, .ok (id, ()))
        | .error message => (t, .error message)
    else pure (t, .error s!"autograd: cuda: {name}: a leading axis has extent zero")
  contract? := Backend.LibTorch.multiheadAttention

/-! ## Reductions, concatenation and slicing -/

/-- The sum over every axis by the flat native sum, which with deterministic algorithms enabled is
the tree of `LibTorch.reduceSumTree`. The pullback broadcasts the scalar cotangent. -/
private def reduceSumAll {s : Shape} (dims : List (Fin s.rank)) :
    Kernel [.value s] [.value (s.reduce dims false)] :=
  ofResult (reductionContract "reduce_sum" elementwise .torchLeanTape) fun t name (x, ()) =>
    Tape.unary t name x s (s.reduce dims false) Buffer.reduceSum
      fun _x dLdy => broadcastScalarToShape dLdy s

/-- The sum over one axis by the native axis sum. The pullback broadcasts the cotangent back. -/
private def reduceSumAxis {s : Shape} (axis : Fin s.rank) :
    Kernel [.value s] [.value (s.reduce [axis] false)] :=
  ofResult (reduceSumAll (s := s) [axis]).contract? fun t name (x, ()) => do
    let some axis32 := Bytes.toUInt32? axis.val | throw SomeBuffer.tooLarge
    let dims := s.toArray
    Tape.unary t name x s (s.reduce [axis] false) (fun x => Buffer.reduceSumAxis x dims axis32)
      fun _x dLdy =>
        let (inDims, outDims, axisMap) := Broadcast.afterSumArgs s axis.val
        Buffer.broadcastTo dLdy inDims outDims axisMap

/-- The contract of the `mean` and `meanAll` entries. -/
private def meanContract : Backend.KernelContract :=
  reductionContract "reduce_mean" (forwardOnly "TorchLeanTest.CUDA.Stress")
    .torchLeanTape

/-- The mean over one axis: the native axis sum divided once by the axis extent (by one for an
empty axis, as `Op.mean` divides an empty fiber by one), and each entry of the broadcast cotangent
divided once by the same count. -/
private def mean {s : Shape} (axis : Fin s.rank) :
    Kernel [.value s] [.value (s.reduce [axis] false)] :=
  ofResult meanContract fun t name (x, ()) => do
    let some axis32 := Bytes.toUInt32? axis.val | throw SomeBuffer.tooLarge
    let dims := s.toArray
    let extent := s.extent axis
    let divisor := if extent = 0 then 1 else extent
    checkDivCount "mean" (Buffer.dtype (← Tape.requireValue t x s)) divisor
    let count := divisor.toUInt64
    Tape.unary t name x s (s.reduce [axis] false)
      (fun x =>
        let sum := Buffer.reduceSumAxis x dims axis32
        Buffer.releaseThen sum <| Buffer.divCount sum count)
      fun _x dLdy =>
        let (inDims, outDims, axisMap) := Broadcast.afterSumArgs s axis.val
        let dLdx := Buffer.broadcastTo dLdy inDims outDims axisMap
        Buffer.releaseThen dLdx <| Buffer.divCount dLdx count

/-- The mean over every axis: the flat native sum divided once by the element count, by one for an
empty tensor (`Buffer.reduceMean`), and each entry of the broadcast scalar cotangent divided once
by the same count. -/
private def meanAll {s : Shape} (dims : List (Fin s.rank)) :
    Kernel [.value s] [.value (s.reduce dims false)] :=
  ofResult meanContract fun t name (x, ()) => do
    checkDivCount "mean" (Buffer.dtype (← Tape.requireValue t x s)) (Tensor.meanDenominator s)
    Tape.unary t name x s (s.reduce dims false) Buffer.reduceMean
      fun _x dLdy =>
        let dLdx := broadcastScalarToShape dLdy s
        Buffer.releaseThen dLdx <| Buffer.divCount dLdx (Tensor.meanDenominator s).toUInt64

/-- Two inputs concatenated along the first axis. Each input's pullback is its block of the
cotangent. -/
private def cat {s : Shape} (dim : Fin s.rank) (n m : Nat) :
    Kernel [.value (s.set dim n), .value (s.set dim m)]
      [.value (s.set dim [n, m].sum)] :=
  ofResult (viewContract "concat" empty .torchLeanTape) (configuredFormats := true)
      fun t name (a, b, ()) => do
    let inner : Nat := Shape.size (s.eraseIdx dim)
    let some nLen32 := Bytes.toUInt32? (n * inner) | throw SomeBuffer.tooLarge
    let some mLen32 := Bytes.toUInt32? (m * inner) | throw SomeBuffer.tooLarge
    let some nmLen32 := Bytes.toUInt32? (n * inner + m * inner) | throw SomeBuffer.tooLarge
    Tape.binary t name a b (s.set dim n) (s.set dim m)
      (s.set dim [n, m].sum)
      (fun a b => Buffer.concat a b nLen32 mLen32)
      fun _a _b dLdy =>
        (Buffer.slice dLdy nmLen32 0 nLen32, Buffer.slice dLdy nmLen32 nLen32 mLen32)

/-- The entries `start .. stop - 1` along the first axis of `s` as one node of shape `out`. The
pullback pads the cotangent with zeros on both sides. -/
@[inline] private def sliceFront (s out : Shape) (start stop : Nat) (t : Tape SomeBuffer)
    (name : String) (x : Nat) : Result (Tape SomeBuffer × Nat) := do
    let n := s.headD 0
    let len := stop - start
    -- An empty slice reads nothing, so its start, which may lie beyond the axis, becomes 0.
    let start := if len = 0 then 0 else start
    let inner : Nat := Shape.size s.tail
    let nTot : Nat := n * inner
    let startOff : Nat := start * inner
    let lenTot : Nat := len * inner
    let rightTot : Nat := nTot - (startOff + lenTot)
    let some nTot32 := Bytes.toUInt32? nTot | throw SomeBuffer.tooLarge
    let some start32 := Bytes.toUInt32? startOff | throw SomeBuffer.tooLarge
    let some len32 := Bytes.toUInt32? lenTot | throw SomeBuffer.tooLarge
    let some right32 := Bytes.toUInt32? rightTot | throw SomeBuffer.tooLarge
    let some midLen32 := Bytes.toUInt32? (startOff + lenTot) | throw SomeBuffer.tooLarge
    Tape.unary t name x s out
      (fun x => Buffer.slice x nTot32 start32 len32)
      fun _x dLdy =>
        let left := Buffer.zerosLike dLdy start32
        let right := Buffer.zerosLike dLdy right32
        let leftPadded := Buffer.concat left dLdy start32 len32
        Buffer.releaseThen left <|
          Buffer.releaseThen right <|
            Buffer.releaseThen leftPadded <|
              Buffer.concat leftPadded right midLen32 right32

/-- The entries `start .. stop - 1` along `d`, none when `stop ≤ start`. Along the first axis this
is one node, which also runs buffers of a configured scalar format. Along another axis it is three
nodes: adjacent swaps move the axis to the front, the slice cuts it there and the reverse swaps
move it back, with pullbacks that move and pad the cotangent exactly. The swaps read native values,
so that path takes native binary32 and binary64 buffers only and rejects a buffer of a configured
scalar format before the first swap. -/
private def slice {s : Shape} (d : Fin s.rank) (start stop : Nat) :
    Kernel [.value s] [.value (s.slice [d] [start] [stop] [1])] :=
  ofResult (viewContract "slice" (forwardOnly "TorchLeanTest.CUDA.Graph") .torchLeanTape)
      (configuredFormats := d.val == 0)
      fun t name (x, ()) => do
    let out := s.slice [d] [start] [stop] [1]
    if d.val = 0 then sliceFront s out start stop t name x
    else
      if let some stored := t.getValue? x then
        if (Buffer.format? stored.buf).isSome then
          throw "autograd: configured GPU arithmetic does not support `slice` along an axis other \
            than the first; use CPU"
      let depths := Tape.Indexing.moveAxisToFrontDepths d.val
      let front : Shape := .dim (s.extent d) (s.eraseIdx d)
      let cut : Shape := .dim (stop - start) (s.eraseIdx d)
      unless depths.foldl Shape.swapAdjacentAtDepth s == front &&
          depths.reverse.foldl Shape.swapAdjacentAtDepth cut == out do
        throw "autograd: cuda: slice: swap schedule does not produce the output shape"
      let steps ← Tape.Indexing.checkedSwapSteps depths
      let (t, moved) ← Tape.unary t "permute" x s front
        (fun x => Tape.Indexing.permuteBorrowed x s steps)
        fun _x dLdy => Tape.Indexing.permuteBorrowed dLdy front steps.reverse
      let (t, sliced) ← sliceFront front cut start stop t name moved
      Tape.unary t "permute" sliced cut out
        (fun x => Tape.Indexing.permuteBorrowed x cut steps.reverse)
        fun _x dLdy => Tape.Indexing.permuteBorrowed dLdy out steps

/-! ## Gather, one index along an axis, and scatter

The gather and scatter entries compute on the host, by the coordinate map of `Tensor.gather`, one
row-major position per entry: the position an output entry reads or an update lands on. They move
single entries with the native row kernels. The entry of `indexAt` moves the axis to the front and
reads one row. -/

/-- The row-major position in the operand that each entry of
`Tensor.gather x dims batchDims indices` reads: at an output coordinate `(ct, cr)`, the coordinate
`h.source (indices ct) ct cr` of `Tensor.gather_apply`, linearized, in row-major order of the
output. There is one position per output entry, and the operand is not read. -/
private def gatherPositions {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims)
    (indices : Tensor (Einops.AxisTuple s.extent dims) t) : Array Nat :=
  Array.ofFn fun i : Fin (Shape.size
      (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))) =>
    let parts := Shape.Coord.appendEquiv t
      (s.eraseAxes (dims ++ batchDims.map Prod.fst)) (Shape.Coord.unlinearize i)
    (Shape.Coord.linearize (h.source (indices parts.1) parts.1 parts.2)).val

/-- The contract of the `gather` and `indexAt` entries. -/
private def gatherContract : Backend.KernelContract :=
  primitiveContract
    "gather"
    (full "TorchLeanTest.CUDA.Indexing")
    "Each output entry copies the operand entry at its index tuple and batch coordinates."
    ("Op.gather's VJP adds output gradients into zeros at the read positions, including " ++
      "repeated positions; Op.indexAt's copies its cotangent into the selected slice, +0 " ++
      "elsewhere.")
    .torchLeanTape .implementationDefined

/-- Gather as one node: each output entry copies the entry of `x` at the position the rule reads
(`gatherPositions`). The pullback adds the cotangent into zeros at those positions with the native
scatter-add in output order. With deterministic reductions the native scatter-add is an ordered
fold (an assumption about the native implementation in `csrc/libtorch/torchlean.cpp` that ATen
does not guarantee), so on finite `Float32` values the pullback has the bits of the rule's. Without
deterministic reductions, repeated positions add in an implementation-defined order. -/
private def gather {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims) :
    Kernel [.value s, .index (dims.map s.extent) t]
      [.value (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))] :=
  ofResult gatherContract fun tape name (x, indices, ()) => do
    let out := t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)
    let rows ← SomeBuffer.numelU32 s
    let count ← SomeBuffer.numelU32 out
    let sources := gatherPositions dims batchDims h (Tensor.map (Op.axisTuples _ dims) indices)
    Tape.unary tape name x s out (fun x => Buffer.gatherRows x rows 1 sources count)
      fun _ gradient =>
        let zeros := Buffer.zerosLike gradient rows
        Buffer.releaseThen zeros (Buffer.scatterAddRows zeros gradient rows 1 sources count)

/-- One index along `dim`, the axis removed, as one node: adjacent swaps move the axis to the front
and the native row gather reads the row `index`. The pullback places the cotangent between blocks
of zeros in that layout, as the slice entry does, and moves the axis back. On native buffers the
value and the node's local pullback are copies with the bits of the rule's, negative zeros
included, though the tape may add other contributions to the parent's gradient. -/
private def indexAt {s : Shape} (dim : Fin s.rank) (index : Fin (s.extent dim)) :
    Kernel [.value s] [.value (s.eraseIdx dim)] :=
  ofResult gatherContract fun t name (xId, ()) => do
    let axis := dim.val
    let outShape := s.eraseIdx axis
    let x ← Tape.requireValue (t := t) xId s
    let some rows32 := Bytes.toUInt32? (s.extent dim) | throw SomeBuffer.tooLarge
    let cols32 ← SomeBuffer.numelU32 outShape
    let _ ← SomeBuffer.numelU32 s
    let some before32 := Bytes.toUInt32? (index.val * cols32.toNat) | throw SomeBuffer.tooLarge
    let some through32 := Bytes.toUInt32? ((index.val + 1) * cols32.toNat)
      | throw SomeBuffer.tooLarge
    let some after32 := Bytes.toUInt32? ((rows32.toNat - index.val - 1) * cols32.toNat)
      | throw SomeBuffer.tooLarge
    let steps ← Tape.Indexing.checkedSwapSteps (Tape.Indexing.moveAxisToFrontDepths axis)
    let reverseSteps := steps.reverse
    let one32 : UInt32 := 1
    let indices : Array Nat := #[index.val]
    let moved := Tape.Indexing.permuteBorrowed x s steps
    let gathered := Buffer.gatherRows moved rows32 cols32 indices one32
    let y := Tape.Indexing.releasePermutedThen steps moved gathered
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := outShape, buf := y }
        requiresGrad := (t.getNode? xId).any (·.requiresGrad)
        parents := #[xId]
        backward := fun seed => do
          let gradient ← Tape.requireGrad seed outShape
          let before := Buffer.zerosLike gradient.buf before32
          let after := Buffer.zerosLike gradient.buf after32
          let leftPadded := Buffer.concat before gradient.buf before32 cols32
          let frontDx := Buffer.releaseThen before <|
            Buffer.releaseThen after <|
              Buffer.releaseThen leftPadded <|
                Buffer.concat leftPadded after through32 after32
          let dx := Tape.Indexing.permuteOwned frontDx (.dim (s.extent dim) outShape)
            reverseSteps
          pure #[(xId, { s := s, buf := dx })] }
    pure (t.addNode node)

/-- The sum scatter as one node: each update entry is added to the entry of `base` it lands on, the
position the gather with no batch pairs reads (`gatherPositions`, the gather of the rule's update
pullback `Tensor.scatterUpdatesBackward`), by the native scatter-add in update order. With
deterministic reductions that is an ordered fold from the entry of `base` (the same assumption about
the native implementation as `gather`'s pullback), so on finite `Float32` values the result has the
bits of the rule's. Without deterministic reductions, repeated positions add in an
implementation-defined order. The pullback copies the cotangent to `base` and reads it at the
landing positions for the updates. -/
private def scatterSum {s t : Shape} (dims : List (Fin s.rank)) (h : dims.Nodup) :
    Kernel [.value s, .index (dims.map s.extent) t, .value (t ++ s.eraseAxes dims)] [.value s] :=
  ofResult (primitiveContract "scatter_add" (full "TorchLeanTest.CUDA.Indexing")
      "Indexed source values accumulate into the base tensor, including repeated indices."
      "The VJP gathers output gradients at the source indices and preserves the base gradient."
      .torchLeanTape .implementationDefined) fun tape name (base, indices, updates, ()) => do
    let updateShape := t ++ s.eraseAxes dims
    let rows ← SomeBuffer.numelU32 s
    let count ← SomeBuffer.numelU32 updateShape
    let targets := gatherPositions dims [] ⟨by simpa using h, by simp, by simp⟩
      (Tensor.map (Op.axisTuples _ dims) indices)
    Tape.binary tape name base updates s updateShape s
      (fun base updates => Buffer.scatterAddRows base updates rows 1 targets count)
      fun _ _ gradient => (Buffer.copy gradient, Buffer.gatherRows gradient rows 1 targets count)

/-! ## Real Fourier transforms and the diagonal selective scan

The native transforms run along the last axis of a `[batch, n]` layout and the scans along the
first axis of a `[seqLen, state]` layout. Other attributes have no entry. -/

/-- The packed real transform along the last axis with its real-linear adjoint. Spectral layers
compose it with channel mixing and `irfft`. -/
private def rfft {s : Shape} (dim : Fin s.rank) (n : Nat) :
    Kernel [.value (s.set dim n)] [.value (s.set dim (n / 2 + 1) ++ [2])] :=
  ofResult (primitiveContract "fft_fno" (full "TorchLeanTest.CUDA.FFT")
      "Packed rFFT/irFFT follow the documented half-spectrum contract."
      ("Lean composes the real-linear transform adjoints and spectral channel mixing, " ++
        "propagating gradients to the input and both weight components.")
      .torchLeanTape .implementationDefined) fun t name (x, ()) => do
    if n = 0 then throw "autograd: rfft1d: transform length must be positive"
    let some batch32 := Bytes.toUInt32? (s.headD 0) | throw SomeBuffer.tooLarge
    let some n32 := Bytes.toUInt32? n | throw SomeBuffer.tooLarge
    Tape.unary t name x (s.set dim n) (s.set dim (n / 2 + 1) ++ [2])
      (fun x => Buffer.rfftPacked x batch32 n32)
      (fun _ gradient => Buffer.rfftAdjoint gradient batch32 n32)

/-- The normalized packed inverse along the last axis with its packed-coordinate adjoint. -/
private def irfft {s : Shape} (dim : Fin s.rank) (n : Nat) :
    Kernel [.value (s.set dim (n / 2 + 1) ++ [2])] [.value (s.set dim n)] :=
  ofResult (rfft (s := s) dim n).contract? fun t name (x, ()) => do
    if n = 0 then throw "autograd: irfft1d: transform length must be positive"
    let some batch32 := Bytes.toUInt32? (s.headD 0) | throw SomeBuffer.tooLarge
    let some n32 := Bytes.toUInt32? n | throw SomeBuffer.tooLarge
    Tape.unary t name x (s.set dim (n / 2 + 1) ++ [2]) (s.set dim n)
      (fun x => Buffer.irfftPacked x batch32 n32)
      (fun _ gradient => Buffer.irfftAdjoint gradient batch32 n32)

/-- The diagonal scan `h t = a t * h (t - 1) + b t * x t` along the first axis, the coefficients
shared across time or one row per step, with all four input gradients. -/
private def selectiveScan {s : Shape} (dim : Fin s.rank) (sharedA sharedB : Bool) :
    Kernel [.value (Tensor.scanCoefficients s dim sharedA),
      .value (Tensor.scanCoefficients s dim sharedB), .value s, .value (s.eraseIdx dim)]
      [.value s] :=
  ofResult (primitiveContract "selective_scan" (full "TorchLeanTest.CUDA.SelectiveScan")
      "Shared and token-dependent coefficients follow the diagonal recurrence contract."
      ("Reverse recurrence differentiates coefficients, inputs, and the initial state; shared " ++
        "coefficient cotangents are added from `+0` in increasing time.")
      .backendVJP .implementationDefined) fun t name (aId, bId, xId, initialId, ()) => do
    let perStep := !sharedA
    let seqLen := s.headD 0
    let state := s.getD 1 0
    let some time32 := Bytes.toUInt32? seqLen | throw SomeBuffer.tooLarge
    let some state32 := Bytes.toUInt32? state | throw SomeBuffer.tooLarge
    let coefficientShape := Tensor.scanCoefficients s dim sharedA
    let a ← Tape.requireValue t aId coefficientShape
    let b ← Tape.requireValue t bId coefficientShape
    let x ← Tape.requireValue t xId s
    let initial ← Tape.requireValue t initialId (s.eraseIdx dim)
    let output := if perStep
      then Buffer.selectiveScanDiagVar a b x initial time32 state32
      else Buffer.selectiveScanDiag a b x initial time32 state32
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := s, buf := output }
        requiresGrad := (t.getNode? aId).any (·.requiresGrad) ||
          (t.getNode? bId).any (·.requiresGrad) ||
          (t.getNode? xId).any (·.requiresGrad) ||
          (t.getNode? initialId).any (·.requiresGrad)
        parents := #[aId, bId, xId, initialId]
        backward := fun gradient => do
          let checked ← Tape.requireGrad gradient s
          let (da, db, dx, dInitial) := if perStep
            then Buffer.selectiveScanDiagVarBackward a b x initial output checked.buf time32 state32
            else Buffer.selectiveScanDiagBackward a b x initial output checked.buf time32 state32
          pure #[
            (aId, { s := coefficientShape, buf := da }),
            (bId, { s := coefficientShape, buf := db }),
            (xId, { s := s, buf := dx }),
            (initialId, { s := s.eraseIdx dim, buf := dInitial })] }
    return t.addNode node

/-! ## Distillation, the top-k mask and random tensors -/

/-- The mean cross-entropy of the student's logits against the teacher's along `dim`. Both are
parents, but only the student receives a contribution because the teacher is a detached input. -/
private def distill {s : Shape} (dim : Fin s.rank) :
    Kernel [.value s, .detached s] [.value .scalar] :=
  ofResult (primitiveContract "distill" empty
      "Teacher-logit cross-entropy with a separated log-softmax normalizer."
      "Student VJP uses logit differences near the teacher and a far-field probability difference."
      .backendVJP .implementationDefined) fun t name (studentId, teacherId, ()) => do
    Tape.validateU32Dimensions "distill" s.toArray
    let some axis32 := Bytes.toUInt32? dim.val | throw SomeBuffer.tooLarge
    let student ← Tape.requireValue t studentId s
    let teacher ← Tape.requireValue t teacherId s
    checkDivCount "distill" (Buffer.dtype student) (s.size / s.extent dim)
    let loss := Buffer.distill student teacher s.toArray axis32
    let node : Node SomeBuffer :=
      { name := some name
        value := { s := Shape.scalar, buf := loss }
        requiresGrad := (t.getNode? studentId).any (·.requiresGrad)
        parents := #[studentId, teacherId]
        backward := fun seed => do
          let seed ← Tape.requireGrad seed Shape.scalar
          let gradient := Buffer.distill student teacher s.toArray axis32 (gradient := true)
          let scale := broadcastScalarToShape seed.buf s
          let result := Buffer.releaseManyThen #[gradient, scale] (Buffer.mul gradient scale)
          pure #[(studentId, { s := s, buf := result })] }
    pure (t.addNode node)

/-- `Tensor.detach`: the parent's buffer as a borrowed value that receives no gradient. The node
owns no storage, and its parent receives no contribution. -/
private def detach {s : Shape} : Kernel [.detached s] [.value s] :=
  ofResult none (configuredFormats := true) fun t name (xId, ()) => do
    let x ← Tape.requireValue t xId s
    pure <| t.addNode
      { name := some name
        value := { s, buf := x }
        ownsValue := false
        requiresGrad := false
        parents := #[xId]
        backward := fun _ => .ok #[] }

/-- The top-`k` mask along the last axis (`MoE.topKMask`) of a detached input: a node whose
parent receives no contribution. Buffers of a configured scalar format are rejected. -/
private def topKMask {s : Shape} (k : Nat) : Kernel [.detached s] [.value s] :=
  ofResult (forwardOnlyContract "top_k_mask" (forwardOnly "TorchLeanTest.CUDA.MoE")
      ("The kernel compares every pair of entries in a row and counts, so the mask is exact and " ++
        "equals `TorchLean.MoE.topKMaskInnermost`, ties to the lower index.")) fun t name
          (xId, ()) => do
    if let some stored := t.getValue? xId then
      if (Buffer.format? stored.buf).isSome then
        throw "autograd: configured GPU arithmetic does not support `top_k_mask`; use CPU"
    let x ← Tape.requireValue t xId s
    let mask ←
      if Shape.size s == 0 then pure (Buffer.copy x)
      else do
        let (rows, cols) ← foldRowsColsLastAxis s
        pure (Buffer.topKMaskByRow x rows cols (Nat.min k cols.toNat).toUInt32)
    pure <| t.addNode
      { name := some name
        value := { s := s, buf := mask }
        requiresGrad := false
        parents := #[xId]
        backward := fun _ => .ok #[] }

/-- A random node without parents or gradient: `draw` allocates its buffer in the session's
format from the checked element count, and a failed format read or draw returns the tape with its
error. -/
@[inline] private def randomNode {ins : List Port} {s : Shape}
    (contract : Backend.KernelContract)
    (draw : UInt32 → LibTorch.Dtype → Args ins → IO Buffer) :
    Kernel ins [.value s] where
  record t name dtype args := do
    match Bytes.toUInt32? (Shape.size s) with
    | none => pure (t, .error SomeBuffer.tooLarge)
    | some count =>
        let drawn : Except String Buffer ←
          try pure (Except.ok (← draw count (← dtype) args))
          catch e => pure (Except.error (toString e))
        match drawn with
        | .error message => pure (t, .error message)
        | .ok buf =>
            let (t, id) := Tape.leaf t { s, buf } (name := some name) (requiresGrad := false)
            pure (t, .ok (id, ()))
  configuredFormats := true
  contract? := some contract

/-- A tensor of the draws `d / 2^32` keyed by `key` (`Buffer.randUniformIO`), each `d` a 32-bit
word, rounded into the session's dtype, so a float32 draw can be `1`. -/
private def randUniform {s : Shape} (key : UInt64) : Kernel [] [.value s] :=
  randomNode (forwardOnlyContract "rand_uniform"
      (forwardOnly "TorchLeanTest.CUDA.Stress")
      "LibTorch CUDA deterministic random-uniform buffers follow the seeded runtime contract.")
      fun count dtype () =>
    Buffer.randUniformIO count key dtype

/-- A keep mask drawn from `key`: one with the frozen keep probability, zero otherwise. -/
private def bernoulliMask {s : Shape} (key : UInt64) : Kernel [.frozen []] [.value s] :=
  randomNode (forwardOnlyContract "bernoulli_mask"
      (forwardOnly "TorchLeanTest.CUDA.Stress")
      "LibTorch CUDA deterministic Bernoulli masks follow the seeded runtime contract.")
      fun count dtype (probability, ()) =>
    Buffer.bernoulliMaskIO count probability.item key dtype

/-- Dropout with a uniform sample and a frozen drop probability `p` (`Op.dropout`): one
`Buffer.dropout` call in each pass, an IEEE division by `1 - p` at a kept entry and `x * 0` at the
others. Every entry is `+0` when `p ≥ 1`, and the sample is a parent that receives no contribution,
as `distill`'s teacher is. -/
private def dropout {s : Shape} : Kernel [.value s, .detached s, .frozen []] [.value s] :=
  ofResult (pointwiseContract "dropout" elementwise .torchLeanTape)
      fun t name (xId, sampleId, p, ()) => do
    let x ← Tape.requireValue t xId s
    let sample ← Tape.requireValue t sampleId s
    let node : Node SomeBuffer :=
      { name := some name
        value := { s, buf := Buffer.dropout x sample p.item }
        requiresGrad := (t.getNode? xId).any (·.requiresGrad)
        parents := #[xId, sampleId]
        backward := fun seed => do
          let seed ← Tape.requireGrad seed s
          pure #[(xId, { s, buf := Buffer.dropout seed.buf sample p.item })] }
    pure (t.addNode node)

end Kernel

/-- The CUDA implementation of `op`, or `none` where LibTorch has none. -/
def Kernel.ofOp {ins outs : List Port} : Op ins outs → Option (Kernel ins outs)
  | .add => some Kernel.add
  | .sub => some Kernel.sub
  | .mul => some Kernel.mul
  | .div => some Kernel.div
  | .neg => some Kernel.neg
  | .scale => some Kernel.scale
  | .abs => some Kernel.abs
  | .sqrt .raw => some Kernel.sqrt
  | .sqrt .clampNegative => some Kernel.sqrtClampNegative
  | .clamp => some Kernel.clamp
  | .max => some Kernel.max
  | .min => some Kernel.min
  | .relu => some Kernel.relu
  | .exp => some Kernel.exp
  | .sin => some Kernel.sin
  | .cos => some Kernel.cos
  | .log => some Kernel.log
  | .reciprocal => some Kernel.reciprocal
  | .safeLog => some Kernel.safeLog
  | .sigmoid => some Kernel.sigmoid
  | .tanh => some Kernel.tanh
  | .silu => some Kernel.silu
  | .gelu => some Kernel.gelu
  | .softplus => some Kernel.softplus
  | .crossEntropy dim mean => some (Kernel.crossEntropy dim mean)
  | @Op.layerNorm seqLen embedDim _ _ _ x _ _ =>
      some (Kernel.layerNorm (seqLen := seqLen) (embedDim := embedDim) (x := x))
  | .batchNorm _ => some Kernel.batchNorm
  | .batchNormRunning dim => some (Kernel.batchNormRunning dim)
  | .softmax dim => some (Kernel.softmax dim)
  | .logSoftmax dim => some (Kernel.logSoftmax dim)
  | @Op.linear .scalar _ _ => some Kernel.linear
  | .mseLoss => some Kernel.mseLoss
  | .huberLoss mean => some (Kernel.huberLoss mean)
  | .attentionFromScores mask => some (Kernel.attentionFromScores mask)
  | .multiheadAttention h1 mask => some (Kernel.multiheadAttention h1 mask)
  | .reshape _ => some Kernel.reshape
  | .detach => some Kernel.detach
  | .gather dims batchDims h => some (Kernel.gather dims batchDims h)
  | .indexAt dim index => some (Kernel.indexAt dim index)
  | .scatter .sum dims h => some (Kernel.scatterSum dims h)
  | .distill dim => some (Kernel.distill dim)
  | @Op.topKMask s dim k => if dim.val + 1 == s.rank then some (Kernel.topKMask k) else none
  | .randUniform key => some (Kernel.randUniform key)
  | .bernoulliMask key => some (Kernel.bernoulliMask key)
  | .dropout => some Kernel.dropout
  | @Op.rfft s dim n _ =>
      if s.rank == 2 && dim.val == 1 then some (Kernel.rfft dim n) else none
  | @Op.irfft s dim n _ =>
      if s.rank == 2 && dim.val == 1 then some (Kernel.irfft dim n) else none
  | @Op.selectiveScan s dim sharedA sharedB =>
      if s.rank == 2 && dim.val == 0 && sharedA == sharedB then
        some (Kernel.selectiveScan dim sharedA sharedB)
      else none
  | @Op.reduce s .sum dims false _ =>
      if dims.map (·.val) == List.range s.rank then some (Kernel.reduceSumAll dims)
      else match dims with
        | [axis] => some (Kernel.reduceSumAxis axis)
        | _ => none
  | @Op.mean s dims false =>
      if dims.map (·.val) == List.range s.rank then some (Kernel.meanAll dims)
      else match dims with
        | [axis] => some (Kernel.mean axis)
        | _ => none
  | .cat dim [n, m] => if dim.val == 0 then some (Kernel.cat dim n m) else none
  | .slice [d] [start] [stop] [1] _ => some (Kernel.slice d start stop)
  | .permute perm out _ => some (Kernel.permute perm out)
  | @Op.broadcast s t dims _ .scalar =>
      if hcb : Shape.CanBroadcastTo s t then
        if dims = hcb.dims then some (Kernel.broadcast hcb) else none
      else none
  | .matmul lv rv _ _ broadcastA broadcastB => some (Kernel.matmul lv rv broadcastA broadcastB)
  | .scaledMatmul lv rv _ _ broadcastA broadcastB =>
      some (Kernel.scaledMatmul lv rv broadcastA broadcastB)
  | .pool .max dims kernel stride padLow padHigh dilation ceilMode =>
      if Kernel.nativePool dims kernel stride padLow padHigh dilation ceilMode then
        some (Kernel.maxPool dims kernel stride padLow padHigh dilation ceilMode)
      else none
  | .pool (.avg true) dims kernel stride padLow padHigh dilation ceilMode =>
      if Kernel.nativePool dims kernel stride padLow padHigh dilation ceilMode then
        some (Kernel.avgPool dims kernel stride padLow padHigh dilation ceilMode)
      else none
  | .smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode =>
      if Kernel.nativePool dims kernel stride padLow padHigh dilation ceilMode then
        some (Kernel.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode)
      else none
  | @Op.conv _ _ _ dims stride padLow padHigh dilation groups _ x y _ _ _ =>
      if Kernel.groupedConv dims stride dilation groups then
        some (Kernel.convLeading dims stride padLow padHigh dilation groups x y)
      else none
  | .convTranspose dims stride padLow padHigh outputPadding dilation groups =>
      if Kernel.nativeConvTranspose dims padLow padHigh dilation outputPadding groups then
        some (Kernel.convTranspose dims stride padLow padHigh outputPadding dilation groups)
      else none
  | _ => none

/-- The LibTorch CUDA implementation table as the planner sees it: the capsule of each entry's
contract for its operator, under provider `libTorch` on device `cuda`. -/
def implementation : Backend.Implementation :=
  Backend.Implementation.ofContracts "libtorch" .libTorch .cuda fun op =>
    (Kernel.ofOp op).bind (·.contract?)

/-- `implementation` is the LibTorch CUDA table named `libtorch`, of provider `libTorch` on
device `cuda`. -/
theorem implementation_identity :
    implementation.name = "libtorch" ∧ implementation.provider = .libTorch ∧
      implementation.device = .cuda :=
  ⟨rfl, rfl, rfl⟩

namespace Tape

/-- Record the registered operator `op` on the CUDA tape through its entry, under the node label
`name`, with `dtype` the session's buffer format: the extended tape with the outputs, or the given
tape with the error. An operator without an entry is an error. -/
def apply {ins outs : List Port} (t : Tape SomeBuffer) (name : String) (op : Op ins outs)
    (args : Args ins) (dtype : IO LibTorch.Dtype) :
    IO (Tape SomeBuffer × Result (Args outs)) :=
  match Kernel.ofOp op with
  | some k => k.record t name dtype args
  | none => pure (t, .error s!"torch: cuda: `{name}` is unsupported by LibTorch")

end Tape

end LibTorch
end TorchLean
