/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Kernel.Graph
public import TorchLean.NN.IR.Builder
public import TorchLean.NN.Kernel.Function
public import TorchLeanTest.Harness.CUDA

/-!
# Graph evaluation through the CUDA table

`TorchLean.IR.Graph.runBuffers` runs a graph of registered operators by their entries in
`LibTorch.Kernel.ofOp`. These checks build small graphs with `TorchLean.IR.Graph.Builder`, whose
results are exact in binary32, run them and compare them with the expected values: a slice along an
inner axis, an empty slice, the product of a vector with a batch of matrices, max pooling with a
batch axis before the channel axis and without a channel axis, average pooling with two leading
axes, and smooth max pooling of two entries at the subnormal `beta = 2 ^ -149` and at infinite
pivots. A layer normalization over two trailing axes (the view, the matrix layer normalization and
the view back), with the default constants and with given ones, is compared bit for bit with the
entry on the matrix of those axes, and `Op.layerNorm` of three axes with the same graph over the
matrix of its rows. A graph whose later node has no entry is rejected before any node runs. On the
tape, an empty slice along an inner axis and one of a configured binary128 buffer along the first
axis, both with the start beyond the axis, give empty values and zero pullbacks of the input's shape
(and format).

A convolution with two groups, dilation two, stride two and padding one before and none after runs
on a batch, on one sample without a batch axis and behind two leading axes, with its packed kernel
and bias as constants. On the tape its kernel, bias and input pullbacks are the hand-computed sums,
and an input without entries whose padding makes the output nonempty multiplies the padding zeros
by the weights. Rank-2 and rank-3 convolutions with per-axis geometry and the symmetric one-sample
convolution match the CPU rule in value and pullbacks, and the eager API's convolution on a CUDA
session matches it on a CPU session. Malformed native metadata is an IO error. Batch normalization
with running statistics (a negative variance clamped to zero, a NaN variance kept) and `detach` run
in graphs and on the tape, where `x + detach x` sends the gradient through the first use only.
Every value compared with a hand-computed one or with the CPU rule is exact in binary32.

The runner reads an index constant (a gather), a value buffer at a frozen port (a scaled product of
a vector and a matrix), refuses a graph with a later node without an entry or a later missing input
or constant before any node runs, runs a custom program and refuses one without its program or of
another precision, and runs the general entries (a layer normalization of three axes, vector
operands of matrix products, convolutions behind several leading axes and without a batch axis). On
the tape those entries' values and pullbacks equal the matrix entry's or the CPU rule's.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean.Test.CUDA.Graph

open Harness

/-- The binary32 buffer of `xs` at shape `s`. -/
def buffer (s : Shape) (xs : Array Float) : IO LibTorch.SomeBuffer := do
  pure { s, buf := ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk xs) }

/-- Whether two buffers have the same dtype and the same bytes. -/
def sameWords (a b : LibTorch.Buffer) : IO Bool := do
  pure (LibTorch.Buffer.dtype a == LibTorch.Buffer.dtype b &&
    (← LibTorch.Buffer.toBytesIO a) ==
      (← LibTorch.Buffer.toBytesIO b))

/-- The value buffer `buf` of shape `s` as a resident value. -/
def valueOf (s : Shape) (buf : LibTorch.Buffer) : IR.Resident :=
  ⟨⟨.value, s⟩, buf⟩

/-- The words of output `k` of node `j`, a value buffer. -/
def wordsAt (values : Array (List IR.Resident)) (j k : Nat) : IO FloatArray := do
  match values[j]? >>= (·[k]?) with
  | some ⟨⟨.value, _⟩, buf⟩ => LibTorch.Buffer.toFloatArrayIO buf
  | _ => throw <| IO.userError s!"operator graph: no value at node {j}, output {k}"

/-- The words of node `j`'s first output after running the built operator graph `g` with `payload`
on the value buffers `inputs`, graph input `i` read from `inputs[i]`. -/
def runOutput (g : Except String IR.Graph) (payload : IR.Payload Float32)
    (inputs : Array LibTorch.SomeBuffer) (j : Nat) : IO FloatArray := do
  let g ← IO.ofExcept (g.mapError fun e => s!"graph: build: {e}")
  wordsAt (← g.runBuffers payload (inputs.map fun b => valueOf b.s b.buf)) j 0

/-- The operator graph of a layer normalization of `[2, 3, 4]` over its two trailing axes, with its
payload: the input viewed as the `(2, 12)` matrix, `Op.layerNorm` of that matrix with the constants
`gamma`, `beta` and `epsilon`, and the view back. -/
def layerNormForm (gamma beta : Tensor Float32 [12]) (epsilon : Float32) :
    IR.Graph × IR.Payload Float32 :=
  ({ nodes := #[.input 0 (.value [2, 3, 4]),
      .op ⟨_, _, Op.reshape (s := [2, 3, 4]) (t := [2, 12]) (by decide)⟩ #[⟨0, 0⟩],
      .const (.value [12]), .const (.value [12]), .const (.frozen []),
      .op ⟨_, _, Op.layerNorm (seqLen := 2) (embedDim := 12) (by decide) (by decide) (lead := [2])
        (x := [2, 12]) (Shape.size_vector _).symm rfl⟩ #[⟨1, 0⟩, ⟨2, 0⟩, ⟨3, 0⟩, ⟨4, 0⟩],
      .op ⟨_, _, Op.reshape (s := [2, 12]) (t := [2, 3, 4]) (by decide)⟩ #[⟨5, 0⟩]],
     outputs := #[⟨6, 0⟩] },
   { const? := fun j => match j with
      | 2 => some ⟨.value [12], gamma⟩
      | 3 => some ⟨.value [12], beta⟩
      | 4 => some ⟨.frozen [], Tensor.scalar epsilon⟩
      | _ => none })

/-- The numbers `0, 1, ..., n - 1`. -/
def iota (n : Nat) : Array Float := Array.ofFn fun i : Fin n => i.val.toFloat

/-- The slice `1:3` along the middle axis of `[2, 3, 4]`, against the entries it keeps. -/
def checkSlice : IO Unit := do
  let (b, x) := ({} : IR.Graph.Builder).input 0 (.value [2, 3, 4])
  let (b, ys) := b.op (Op.slice (s := [2, 3, 4]) [⟨1, by decide⟩] [1] [3] [1] (by decide)) [x]
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [2, 3, 4] (iota 24)] 1
  let expected := Id.run do
    let mut out := #[]
    for b in [0:2] do
      for j in [1:3] do
        for k in [0:4] do
          out := out.push (b * 12 + j * 4 + k).toFloat
    pure out
  assertFloatArrayEq "graph: slice along axis 1" got (FloatArray.mk expected)

/-- The product of the vector `[1, 2, 3]` with each matrix of a `[2, 3, 4]` batch, against the
sums of the products. -/
def checkVectorMatmul : IO Unit := do
  let (g, x) := ({} : IR.Graph.Builder).input 0 (.value [3])
  let (g, y) := g.input 1 (.value [2, 3, 4])
  let (g, zs) := g.op (Op.matmul (batchA := []) (batchB := [2]) (batch := [2]) (m := 1) (n := 3)
    (p := 4) true false (fun _ => rfl) (fun h => nomatch h) (by decide)
    (Shape.CanBroadcastTo.refl [2])) [x, y]
  let a : Array Float := #[1, 2, 3]
  let b := iota 24
  let got ← runOutput (g.build zs.toArray) {} #[← buffer [3] a, ← buffer [2, 3, 4] b] 2
  let expected := Array.ofFn fun i : Fin 8 =>
    let batch := i.val / 4
    let col := i.val % 4
    (List.range 3).foldl (fun acc j => acc + a[j]! * b[batch * 12 + j * 4 + col]!) 0
  assertFloatArrayEq "graph: vector times a batch of matrices" got (FloatArray.mk expected)

/-- Max pooling with `2 × 2` windows and stride two over the last two axes of `[2, 2, 4, 4]`,
where a batch axis precedes the channel axis. -/
def checkBatchedMaxPool : IO Unit := do
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 2, 4, 4])
  let (b, ys) := b.op (Op.pool (s := [2, 2, 4, 4]) .max [⟨2, by decide⟩, ⟨3, by decide⟩] [2, 2]
    [2, 2] [0, 0] [0, 0] [1, 1] false) [r]
  let x := Array.ofFn fun i : Fin 64 => (i.val * 37 % 64).toFloat
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [2, 2, 4, 4] x] 1
  let expected := Array.ofFn fun i : Fin 16 =>
    let plane := i.val / 4
    let row := i.val % 4 / 2
    let col := i.val % 2
    [0, 1].foldl (fun acc dr => [0, 1].foldl (fun acc dc =>
      max acc x[plane * 16 + (2 * row + dr) * 4 + 2 * col + dc]!) acc) (-1)
  assertFloatArrayEq "graph: max pooling with a batch axis" got (FloatArray.mk expected)

/-- Layer normalization over the two trailing axes of `[2, 3, 4]` with unit scale, zero bias and
the default epsilon, bit for bit equal to the `layer_norm` tape entry on `[2, 12]`. -/
def checkLayerNormDefaults : IO Unit := do
  let x := Array.ofFn fun i : Fin 24 => (i.val * 7 % 24).toFloat / 4
  let (g, payload) :=
    layerNormForm (Tensor.full _ 1) (Tensor.full _ 0) normalizationEpsilon
  let got ← runOutput (.ok g) payload #[← buffer [2, 3, 4] x] 6
  let (t, xId) := Tape.leaf .empty (← buffer [2, 12] x)
  let (t, gammaId) := Tape.leaf t (← buffer [12] (Array.replicate 12 1))
  let (t, betaId) := Tape.leaf t (← buffer [12] (Array.replicate 12 0))
  let epsilon : Tensor Float [] := Tensor.scalar (normalizationEpsilon : Float32).toFloat
  let (t, result) ← LibTorch.Tape.apply t "layer_norm"
    (.layerNorm (by decide) (by decide) (lead := [2]) (x := [2, 12]) (Shape.size_vector _).symm
      rfl)
    (xId, gammaId, betaId, epsilon, ()) (pure .float32)
  let (id, ()) ← IO.ofExcept result
  let some value := t.getValue? id | throw <| IO.userError "graph: no layer_norm node"
  let expected ← LibTorch.Buffer.toFloatArrayIO value.buf
  unless got.size == expected.size &&
      (List.range got.size).all fun i => (got.get! i).toBits == (expected.get! i).toBits do
    throw <| IO.userError "graph: layer_norm over trailing axes differs from the matrix entry"

/-- Max pooling over both axes of `[4, 4]`, without a channel axis, and average pooling of
`[2, 2, 4, 4]` with its two leading axes folded. -/
def checkPoolsWithoutChannel : IO Unit := do
  let x := Array.ofFn fun i : Fin 16 => (i.val * 5 % 16).toFloat
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [4, 4])
  let (b, ys) := b.op (Op.pool (s := [4, 4]) .max [⟨0, by decide⟩, ⟨1, by decide⟩] [2, 2] [2, 2]
    [0, 0] [0, 0] [1, 1] false) [r]
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [4, 4] x] 1
  let expected := Array.ofFn fun i : Fin 4 =>
    [0, 1].foldl (fun acc dr => [0, 1].foldl (fun acc dc =>
      max acc x[(2 * (i.val / 2) + dr) * 4 + 2 * (i.val % 2) + dc]!) acc) (-1)
  assertFloatArrayEq "graph: max pooling without a channel axis" got (FloatArray.mk expected)
  let y := Array.ofFn fun i : Fin 64 => (i.val * 13 % 64).toFloat
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 2, 4, 4])
  let (b, ys) := b.op (Op.pool (s := [2, 2, 4, 4]) (.avg true) [⟨2, by decide⟩, ⟨3, by decide⟩]
    [2, 2] [2, 2] [0, 0] [0, 0] [1, 1] false) [r]
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [2, 2, 4, 4] y] 1
  let expected := Array.ofFn fun i : Fin 16 =>
    let plane := i.val / 4
    let row := i.val % 4 / 2
    let col := i.val % 2
    ([0, 1].foldl (fun acc dr => [0, 1].foldl (fun acc dc =>
      acc + y[plane * 16 + (2 * row + dr) * 4 + 2 * col + dc]!) acc) 0) / 4
  assertFloatArrayEq "graph: average pooling with two leading axes" got (FloatArray.mk expected)

/-- The binary32 word of `Op.smoothMaxPool` at `beta` over the one window of `[v, w]`. -/
def smoothMaxPoolWord (beta : Float32) (v w : Float) : IO UInt32 := do
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [2])
  let (b, c) := b.const (.frozen [])
  let (b, ys) := b.op (Op.smoothMaxPool (s := [2]) [⟨0, by decide⟩] [2] [2] [0] [0] [1] false)
    [r, c]
  let got ← runOutput (b.build ys.toArray)
    { const? := fun j => if j = 1 then some ⟨.frozen [], Tensor.scalar beta⟩ else none }
    #[← buffer [2] #[v, w]] 2
  pure (got.get! 0).toFloat32.toBits

/-- Smooth max pooling of two entries: `[0, -∞]` at the subnormal `beta = 2 ^ -149` and `[0, ∞]`
at `-2 ^ -149` give `0`, which a product with the overflowing `1 / beta` would turn into NaN, and
the infinite pivots of `[∞, 1]` and `[-∞, -∞]` at `beta = 1` give `∞` and `-∞`. -/
def checkSmoothMaxPool : IO Unit := do
  let inf : Float := 1.0 / 0.0
  let tiny := Float32.ofBits 1
  for (beta, v, w, expected) in ([(tiny, 0, -inf, 0x00000000), (-tiny, 0, inf, 0x00000000),
      (1, inf, 1, 0x7F800000), (1, -inf, -inf, 0xFF800000)] :
      List (Float32 × Float × Float × UInt32)) do
    let got ← smoothMaxPoolWord beta v w
    unless got == expected do
      throw <| IO.userError s!"graph: smooth max pooling of [{v}, {w}] at beta {beta}: got \
        0x{got.toBitVec.toHex}, expected 0x{expected.toBitVec.toHex}"

/-- An empty slice whose start lies beyond its stop, and a layer normalization over two axes with a
payload, against the matrix entry with the flattened scale and bias. -/
def checkEdges : IO Unit := do
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [4])
  let (b, ys) := b.op (Op.slice (s := [4]) [⟨0, by decide⟩] [5] [3] [1] (by decide)) [r]
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [4] (iota 4)] 1
  unless got.size == 0 do throw <| IO.userError "graph: an empty slice has entries"
  let gammaValues : Array Float := #[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
  let betaValues : Array Float := #[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
  let gamma : Tensor Float32 [3, 4] := (Tensor.from
    (#[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3] : Array Float32)).reshape
    [3, 4] (by decide)
  let beta : Tensor Float32 [3, 4] := (Tensor.from
    (#[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] : Array Float32)).reshape [3, 4] (by decide)
  let x := Array.ofFn fun i : Fin 24 => (i.val * 11 % 24).toFloat / 8
  let (g, payload) :=
    layerNormForm (gamma.reshape [12] (by decide)) (beta.reshape [12] (by decide)) (1 / 64)
  let got ← runOutput (.ok g) payload #[← buffer [2, 3, 4] x] 6
  let (t, xId) := Tape.leaf .empty (← buffer [2, 12] x)
  let (t, gammaId) := Tape.leaf t (← buffer [12] gammaValues)
  let (t, betaId) := Tape.leaf t (← buffer [12] betaValues)
  let (t, result) ← LibTorch.Tape.apply t "layer_norm"
    (.layerNorm (by decide) (by decide) (lead := [2]) (x := [2, 12]) (Shape.size_vector _).symm
      rfl)
    (xId, gammaId, betaId, Tensor.scalar (1 / 64 : Float), ()) (pure .float32)
  let (id, ()) ← IO.ofExcept result
  let some expected := t.getValue? id | throw <| IO.userError "graph: no layer_norm node"
  let expected ← LibTorch.Buffer.toFloatArrayIO expected.buf
  unless got.size == expected.size &&
      (List.range got.size).all fun i => (got.get! i).toBits == (expected.get! i).toBits do
    throw <| IO.userError "graph: layer_norm with a payload differs from the matrix entry"

/-- An empty slice along an inner axis whose start lies beyond the axis, on the tape: no entries,
and a pullback of zeros at the input's shape. -/
def checkEmptySlicePullback : IO Unit := do
  let s : Shape := [2, 4, 3]
  let d : Fin s.rank := ⟨1, by decide⟩
  let (t, x) := Tape.leaf .empty (← buffer s (iota 24))
  let (t, result) ← LibTorch.Tape.apply t "slice"
    (.slice (s := s) [d] [5] [3] [1] (by decide)) (x, ()) (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "graph: no empty slice node"
  let values ← LibTorch.Buffer.toFloatArrayIO value.buf
  unless value.s == [2, 0, 3] && values.size == 0 do
    throw <| IO.userError "graph: the empty slice has entries"
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  let some dx := grads[x]? | throw <| IO.userError "graph: no gradient for the sliced input"
  let dxValues ← LibTorch.Buffer.toFloatArrayIO dx.buf
  unless dx.s == s && dxValues.size == 24 && (List.range 24).all (fun i => dxValues.get! i == 0) do
    throw <| IO.userError "graph: the empty slice's pullback is not zeros of the input shape"

/-- A configured binary128 buffer sliced along its first axis with the start beyond the stop: an
empty value in the same format and a pullback of zeros of the input's shape and format. -/
def checkConfiguredEmptySlice : IO Unit := do
  -- The configured words stay on the CUDA tape with the encoded dtype, as `Recording.run` sets it.
  let transfer : TensorTransfer (FloatLib.Floats.ExecFloat.Binary 15 112) :=
    { (inferInstance : TensorTransfer
        (FloatLib.Floats.ExecFloat.Binary 15 112)) with dtype? := some .encoded }
  let session ← Session.create
    (α := FloatLib.Floats.ExecFloat.Binary 15 112) { device := .cuda }
  let x ← session.input (Tensor.full [4] (1 : FloatLib.Floats.ExecFloat.Binary 15 112))
    (requiresGrad := true)
  let t ← session.cudaTape
  let d : Fin (Shape.rank [4]) := ⟨0, by decide⟩
  let (t, result) ← LibTorch.Tape.apply t "slice"
    (.slice (s := [4]) [d] [5] [3] [1] (by decide)) (x.id, ()) (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "graph: no configured empty slice"
  let some input := t.getValue? x.id | throw <| IO.userError "graph: no configured input"
  let format := LibTorch.Buffer.format? input.buf
  unless value.s == [0] && format.isSome &&
      LibTorch.Buffer.format? value.buf == format do
    throw <| IO.userError "graph: the configured empty slice lost its shape or format"
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  let some dx := grads[x.id]? | throw <| IO.userError "graph: no gradient for the configured input"
  let zeros := LibTorch.Buffer.zerosLike input.buf
  unless dx.s == [4] && LibTorch.Buffer.format? dx.buf == format &&
      (← LibTorch.Buffer.toEncodedIO dx.buf) ==
        (← LibTorch.Buffer.toEncodedIO zeros) do
    throw <| IO.userError "graph: the configured empty slice's pullback is not zeros"

/-- A dense IR kernel entry `(o, c, t)`: `o + c + t + 1` inside output channel `o`'s group (input
channels `2o`, `2o + 1`), `100` outside it, which the packed kernel never reads. -/
def convWeight : List Nat → Float
  | o :: c :: t :: _ => if c / 2 = o then Float.ofNat (o + c + t + 1) else 100
  | _ => 0

/-- The input `x[n, c, j]` of the convolution checks, small integers. -/
def convInput (n c j : Nat) : Float := Float.ofNat ((n * 20 + c * 5 + j) % 7) - 3

/-- The input position read by output position `p` at window cell `t`, if it is not padding. -/
def convSource (p t : Nat) : Option Nat :=
  let q := 2 * p + 2 * t
  if q = 0 || q - 1 ≥ 5 then none else some (q - 1)

/-- `y[n, o, p] = b[o] + Σ w[o, c, t] * x[n, c, convSource p t]` over `o`'s group and `t < 2`. -/
def convValue (n o p : Nat) : Float :=
  (List.range 2).foldl (fun acc c => (List.range 2).foldl (fun acc t =>
    match convSource p t with
    | some j => acc + convWeight [o, 2 * o + c, t] * convInput n (2 * o + c) j
    | none => acc) acc) (if o = 0 then 0.5 else -1)

/-- The graph of an input of shape `x`, the packed kernel `(2, 2, 2)` and the bias as constants 1
and 2 (`convConstants`), and the convolution `o` over them. -/
def ofConv {x y : Shape} (o : Op [.value [2, 2, 2], .value [2], .value x] [.value y]) :
    Except String IR.Graph :=
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value x)
  let (b, k) := b.const (.value [2, 2, 2])
  let (b, c) := b.const (.value [2])
  let (b, ys) := b.op o [k, c, r]
  b.build ys.toArray

/-- The packed kernel `(o, c, t) ↦ convWeight [o, 2o + c, t]` of the two groups and the bias
`[0.5, -1]`, at constants 1 and 2. -/
def convConstants : IR.Payload Float32 :=
  { const? := fun
      | 1 => some ⟨.value [2, 2, 2], Tensor.generate [2, 2, 2] fun i => match i with
          | o :: c :: t :: _ => (convWeight [o, 2 * o + c, t]).toFloat32
          | _ => 0⟩
      | 2 => some ⟨.value [2], Tensor.from (#[0.5, -1] : Array Float32)⟩
      | _ => none }

/-- The convolution with two groups, stride two, padding one before and dilation two, on a batch
of two samples, on one sample without a batch axis and behind the leading axes `[1, 2]`, against
`convValue`. -/
def checkConv : IO Unit := do
  let x := Array.ofFn fun i : Fin 40 => convInput (i.val / 20) (i.val % 20 / 5) (i.val % 5)
  let expected := Array.ofFn fun i : Fin 8 => convValue (i.val / 4) (i.val % 4 / 2) (i.val % 2)
  let dims : Shape.ConvDims [2, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  let got ← runOutput (ofConv (Op.conv dims (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 2)
    2 (lead := [2]) (by decide) rfl rfl)) convConstants #[← buffer [2, 4, 5] x] 3
  assertFloatArrayEq "graph: grouped dilated conv of a batch" got (FloatArray.mk expected)
  let dims : Shape.ConvDims [1, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  let got ← runOutput (ofConv (Op.conv dims (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 2)
    2 (lead := []) (by decide) rfl rfl)) convConstants #[← buffer [4, 5] (x.extract 0 20)] 3
  assertFloatArrayEq "graph: conv without a batch axis" got (FloatArray.mk (expected.extract 0 4))
  let dims : Shape.ConvDims [2, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  let got ← runOutput (ofConv (Op.conv dims (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 2)
    2 (lead := [1, 2]) (by decide) rfl rfl)) convConstants #[← buffer [1, 2, 4, 5] x] 3
  assertFloatArrayEq "graph: conv behind two leading axes" got (FloatArray.mk expected)

/-- The convolution entry on the tape with the packed kernel `(2, 2, 2)`, seeded with its own
value `g`: the kernel gradient `Σ g[n, o, p] * x[n, 2o + c, j]`, the bias gradient `Σ g[n, o, p]`
and the input gradient `Σ g[n, o, p] * w[o, c, t]`, over the positions `j = convSource p t`. -/
def checkConvPullback : IO Unit := do
  let x := Array.ofFn fun i : Fin 40 => convInput (i.val / 20) (i.val % 20 / 5) (i.val % 5)
  let packed := Array.ofFn fun i : Fin 8 =>
    convWeight [i.val / 4, 2 * (i.val / 4) + i.val % 4 / 2, i.val % 2]
  let (t, kernelId) := Tape.leaf .empty (← buffer [2, 2, 2] packed)
  let (t, biasId) := Tape.leaf t (← buffer [2] #[0.5, -1])
  let (t, inputId) := Tape.leaf t (← buffer [2, 4, 5] x)
  let dims : Shape.ConvDims [2, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  let (t, result) ← LibTorch.Tape.apply t "conv"
    (.conv dims (fun _ => 2) (fun _ => 1) (fun _ => 0) (fun _ => 2) 2 (Shape.size_vector _).symm
      (Shape.expandAxis_extent _ _).symm (Shape.conv_eq_expandAxis_batch _ _ _ _ _ _))
    (kernelId, biasId, inputId, ()) (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "graph: no conv node"
  let g ← LibTorch.Buffer.toFloatArrayIO value.buf
  let expected := Array.ofFn fun i : Fin 8 => convValue (i.val / 4) (i.val % 4 / 2) (i.val % 2)
  assertFloatArrayEq "tape: grouped dilated conv" g (FloatArray.mk expected)
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  let read := fun (id : Nat) (what : String) => do
    let some v := grads[id]? | throw <| IO.userError s!"tape: no conv {what} gradient"
    LibTorch.Buffer.toFloatArrayIO v.buf
  let cells := (List.range 2).flatMap fun n => (List.range 2).flatMap fun p =>
    (List.range 2).filterMap fun t => (convSource p t).map fun j => (n, p, t, j)
  let gAt := fun n o p => expected[n * 4 + o * 2 + p]!
  let dKernel := Array.ofFn fun i : Fin 8 =>
    let (o, c, t) := (i.val / 4, i.val % 4 / 2, i.val % 2)
    cells.foldl (fun acc (n, p, t', j) =>
      if t' = t then acc + gAt n o p * convInput n (2 * o + c) j else acc) 0
  let dBias := Array.ofFn fun o : Fin 2 =>
    (List.range 2).foldl (fun acc n =>
      (List.range 2).foldl (fun acc p => acc + gAt n o.val p) acc) 0
  let dInput := Array.ofFn fun i : Fin 40 =>
    let (n, c, j) := (i.val / 20, i.val % 20 / 5, i.val % 5)
    let o := c / 2
    cells.foldl (fun acc (n', p, t, j') =>
      if n' = n && j' = j then acc + gAt n o p * convWeight [o, c, t] else acc) 0
  assertFloatArrayEq "tape: conv kernel pullback" (← read kernelId "kernel") (FloatArray.mk dKernel)
  assertFloatArrayEq "tape: conv bias pullback" (← read biasId "bias") (FloatArray.mk dBias)
  assertFloatArrayEq "tape: conv input pullback" (← read inputId "input") (FloatArray.mk dInput)

/-- An input without entries whose padding makes the output nonempty. The windows read only padding
zeros, and their products with the infinite weight make both outputs NaN (zero times infinity). -/
def checkConvPaddedEmpty : IO Unit := do
  let inf : Float := 1 / 0
  let (t, kernelId) := Tape.leaf .empty (← buffer [1, 1, 1] #[inf])
  let (t, biasId) := Tape.leaf t (← buffer [1] #[0])
  let (t, inputId) := Tape.leaf t (← buffer [1, 1, 0] #[])
  let dims : Shape.ConvDims [1, 1, 0] [1, 1, 1] 1 := Shape.ConvDims.channelsFirst rfl
  let (t, result) ← LibTorch.Tape.apply t "conv"
    (.conv dims (fun _ => 1) (fun _ => 1) (fun _ => 1) (fun _ => 1) 1 (Shape.size_vector _).symm
      (Shape.expandAxis_extent _ _).symm (Shape.conv_eq_expandAxis_batch _ _ _ _ _ _))
    (kernelId, biasId, inputId, ()) (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "tape: no padded empty conv node"
  let got ← LibTorch.Buffer.toFloatArrayIO value.buf
  unless got.size == 2 && (got.get! 0).isNaN && (got.get! 1).isNaN do
    throw <| IO.userError s!"tape: a padded empty conv with an infinite weight gave {got}"

/-- The convolution operator `op` with kernel `k`, a bias of `c` entries, input `s` and output `y`,
on the CPU tape by its rule and on the CUDA tape by its entry, with the packed kernel `w`, the bias
`b` and the input `x` given as functions of the index: the CUDA value and its kernel, bias and
input pullbacks for the cotangent of the value itself equal the CPU ones. The cases use small
integers, so every sum is exact in binary32 in any order. -/
def compareConvOp (tag : String) {k s y : Shape} {c : Nat}
    (op : Op [.value k, .value [c], .value s] [.value y]) (w b x : List Nat → Float) :
    IO Unit := do
  let biasShape : Shape := [c]
  let wT : Tensor Float k := Tensor.generate k w
  let bT : Tensor Float biasShape := Tensor.generate biasShape b
  let xT : Tensor Float s := Tensor.generate s x
  let cpu : Result (Array (Array Float)) := do
    let t : Tape (SomeTensor Float) := .empty
    let (t, kId) := Tape.leaf (t := t) (.ofTensor wT)
    let (t, bId) := Tape.leaf (t := t) (.ofTensor bT)
    let (t, xId) := Tape.leaf (t := t) (.ofTensor xT)
    let (t, yId, ()) ← Tape.apply t "conv" op (kId, bId, xId, ())
    let some y := t.getValue? yId | throw "no CPU conv value"
    let grads ← Tape.backwardDenseAll t yId y
    let some dk := grads[kId]? | throw "no CPU kernel gradient"
    let some db := grads[bId]? | throw "no CPU bias gradient"
    let some dx := grads[xId]? | throw "no CPU input gradient"
    pure (#[y, dk, db, dx].map fun v => Tensor.Rep.data v.tensor)
  let expected ← IO.ofExcept (cpu.mapError fun e => s!"{tag}: CPU rule: {e}")
  let upload := fun (sh : Shape) (t : FloatArray) => do
    pure ({ s := sh, buf := ← LibTorch.Buffer.ofFloatArrayIO t } :
      LibTorch.SomeBuffer)
  let leaf := Tape.leaf (V := LibTorch.SomeBuffer)
  let (t, kernelId) := leaf .empty (← upload k (FloatArray.mk (Tensor.Rep.data wT)))
  let (t, biasId) := leaf t (← upload biasShape (FloatArray.mk (Tensor.Rep.data bT)))
  let (t, inputId) := leaf t (← upload s (FloatArray.mk (Tensor.Rep.data xT)))
  let (t, result) ← LibTorch.Tape.apply t "conv" op (kernelId, biasId, inputId, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError s!"{tag}: no CUDA conv node"
  assertFloatArrayEq s!"{tag}: value" (← LibTorch.Buffer.toFloatArrayIO value.buf)
    (FloatArray.mk expected[0]!)
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  for (id, what, j) in [(kernelId, "kernel", 1), (biasId, "bias", 2), (inputId, "input", 3)] do
    let some g := grads[id]? | throw <| IO.userError s!"{tag}: no {what} gradient"
    assertFloatArrayEq s!"{tag}: {what} pullback"
      (← LibTorch.Buffer.toFloatArrayIO g.buf) (FloatArray.mk expected[j]!)

/-- `compareConvOp` for `Op.conv` with batch axis `dims.input 0` alone (`lead` its extent). -/
def compareConv (tag : String) {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (w b x : List Nat → Float) : IO Unit :=
  compareConvOp tag (Op.conv dims stride padLow padHigh dilation groups (Shape.size_vector _).symm
    (Shape.expandAxis_extent _ _).symm (Shape.conv_eq_expandAxis_batch _ _ _ _ _ _)) w b x

/-- A small integer of an index list, depending on every coordinate and on `salt`. -/
def mix (salt : Nat) (i : List Nat) : Float :=
  Float.ofNat ((i.foldl (fun acc j => acc * 5 + j) salt) % 7) - 3

/-- Rank-2 and rank-3 convolutions with different extents, strides, paddings and dilations per
axis (two groups of two output channels in rank 2), and the symmetric one-sample convolution the
eager API records, each against the CPU rule. -/
def checkConvAgainstRule : IO Unit := do
  let dims2 : Shape.ConvDims [2, 4, 5, 4] [4, 2, 2, 3] 2 := Shape.ConvDims.channelsFirst rfl
  compareConv "conv rank 2" dims2 (fun j => [2, 1].getD j.val 1) (fun j => [1, 0].getD j.val 0)
    (fun j => [0, 2].getD j.val 0) (fun j => [1, 2].getD j.val 1) 2 (mix 1) (mix 2) (mix 3)
  let dims3 : Shape.ConvDims [2, 2, 3, 4, 3] [2, 2, 2, 1, 2] 3 := Shape.ConvDims.channelsFirst rfl
  compareConv "conv rank 3" dims3 (fun j => [1, 2, 1].getD j.val 1)
    (fun j => [0, 1, 1].getD j.val 0) (fun j => [1, 0, 0].getD j.val 0)
    (fun j => [2, 1, 1].getD j.val 1) 1 (mix 4) (mix 5) (mix 6)
  let dims : Shape.ConvDims [1, 2, 4, 4] [3, 2, 3, 3] 2 := Shape.ConvDims.channelsFirst rfl
  compareConv "conv one sample, symmetric" dims (fun _ => 1) (fun _ => 1) (fun _ => 1)
    (fun _ => 1) 1 (mix 7) (mix 8) (mix 9)

/-- The eager API's `Op.conv` of one unbatched sample (one group, unit dilation, symmetric padding)
on a CPU session and on a CUDA session, which records it as a reshape, the convolution and a
reshape in one transaction. The value and the kernel, bias and input gradients for the cotangent
of the value itself are equal. -/
def checkEagerConv : IO Unit := do
  let kernel : Tensor Nat [2] := Tensor.from (#[3, 2] : Array Nat)
  let stride : Tensor Nat [2] := Tensor.from (#[1, 2] : Array Nat)
  let padding : Tensor Nat [2] := Tensor.from (#[1, 1] : Array Nat)
  let inSpatial : Tensor Nat [2] := Tensor.from (#[4, 5] : Array Nat)
  let run := fun (device : Backend.Device) => do
    let session ← Session.create (α := Float) { device }
    let w ← session.input (Tensor.generate (3 :: 2 :: Tensor.to kernel (List Nat))
      (mix 10)) (requiresGrad := true)
    let b ← session.input (Tensor.generate [3] (mix 11)) (requiresGrad := true)
    let x ← session.input (Tensor.generate (2 :: Tensor.to inSpatial (List Nat))
      (mix 12)) (requiresGrad := true)
    let y ← (Ops.apply
      (m := SessionM Float) (α := Float)
      (.conv (Shape.ConvDims.channelsFirst (n := 1) (c := 2) (o := 3) (i := 2)
          (spatial := inSpatial.to (List Nat)) (kernel := kernel.to (List Nat)) (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) 1 (lead := []) rfl rfl
        (congrArg (List.drop 1)
          (Shape.conv_channelsFirst (n := 1) inSpatial kernel stride padding)).symm)
      (w, b, x, ())).run session
    let value ← session.getValue y
    let grads ← session.backwardDenseAll y value
    pure #[Tensor.Rep.data value,
      Tensor.Rep.data (← Session.grad grads w),
      Tensor.Rep.data (← Session.grad grads b),
      Tensor.Rep.data (← Session.grad grads x)]
  let cpu ← run .cpu
  let cuda ← run .cuda
  for (what, j) in [("value", 0), ("kernel pullback", 1), ("bias pullback", 2),
      ("input pullback", 3)] do
    assertFloatArrayEq s!"eager conv: {what}" (FloatArray.mk cuda[j]!) (FloatArray.mk cpu[j]!)

/-- Metadata the typed entry never sends, at the native boundary: a kernel claiming `2^32 - 1`
input channels per group of `2^32 - 1` groups over an input without channels is an IO error. -/
def checkConvMalformed : IO Unit := do
  let empty ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk #[])
  try
    let _ ← LibTorch.Buffer.conv empty empty empty #[1, 0, 1] #[0, 4294967295, 1]
      #[1] #[0] #[0] #[1] 4294967295
    throw <| IO.userError "native conv: malformed channel groups were accepted"
  catch e =>
    unless (e.toString.splitOn "invalid channel groups").length > 1 do
      throw <| IO.userError s!"native conv: unexpected error for malformed groups: {e}"

/-- Running mean of the three channels along axis 1. -/
def runningMean : Array Float := #[1, -1, 0]
/-- Running variance of the three channels: after the clamp of `-5` at zero, `var + 1` is
`4, 1, 16`, so every quotient is exact. -/
def runningVar : Array Float := #[3, -5, 15]
/-- Scale of the three channels. -/
def bnGamma : Array Float := #[2, 1, 3]
/-- Bias of the three channels. -/
def bnBeta : Array Float := #[0.5, 0, -1]
/-- The denominators `√(max var 0 + 1)` of the three channels, with epsilon `1`. -/
def bnDenominator : Array Float := #[2, 1, 4]

/-- Batch normalization of `[2, 3]` along axis 1 with these running statistics and epsilon `1`,
in a graph and on the tape, against the hand-computed value and input, scale and bias pullbacks.
A NaN running variance gives NaN in its channel and leaves the other channels unchanged. -/
def checkBatchNormRunning : IO Unit := do
  let x : Array Float := #[3, 4, 8, -1, -3, -4]
  let normalized := Array.ofFn fun i : Fin 6 =>
    (x[i.val]! - runningMean[i.val % 3]!) / bnDenominator[i.val % 3]!
  let expected := Array.ofFn fun i : Fin 6 =>
    normalized[i.val]! * bnGamma[i.val % 3]! + bnBeta[i.val % 3]!
  let payload : IR.Payload Float32 :=
    { const? := fun
        | 1 => some ⟨.value [3], Tensor.from (#[2, 1, 3] : Array Float32)⟩
        | 2 => some ⟨.value [3], Tensor.from (#[0.5, 0, -1] : Array Float32)⟩
        | 3 => some ⟨.frozen [3], Tensor.from (#[1, -1, 0] : Array Float32)⟩
        | 4 => some ⟨.frozen [3], Tensor.from (#[3, -5, 15] : Array Float32)⟩
        | 5 => some ⟨.frozen [], Tensor.scalar 1⟩
        | _ => none }
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 3])
  let (b, gamma) := b.const (.value [3])
  let (b, beta) := b.const (.value [3])
  let (b, mean) := b.const (.frozen [3])
  let (b, var) := b.const (.frozen [3])
  let (b, eps) := b.const (.frozen [])
  let (b, ys) := b.op (Op.batchNormRunning (s := [2, 3]) ⟨1, by decide⟩)
    [r, gamma, beta, mean, var, eps]
  let source ← buffer [2, 3] x
  let got ← runOutput (b.build ys.toArray) payload #[source] 6
  assertFloatArrayEq "graph: batch_norm with running statistics" got (FloatArray.mk expected)
  let (t, xId) := Tape.leaf .empty source
  let (t, gammaId) := Tape.leaf t (← buffer [3] bnGamma)
  let (t, betaId) := Tape.leaf t (← buffer [3] bnBeta)
  let (t, result) ← LibTorch.Tape.apply t "batch_norm"
    (.batchNormRunning (s := [2, 3]) ⟨1, by decide⟩)
    (xId, gammaId, betaId, (Tensor.from (#[1, -1, 0] : Array Float) : Tensor Float [3]),
      (Tensor.from (#[3, -5, 15] : Array Float) : Tensor Float [3]), Tensor.scalar 1, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "tape: no batch_norm node"
  assertFloatArrayEq "tape: batch_norm with running statistics"
    (← LibTorch.Buffer.toFloatArrayIO value.buf) (FloatArray.mk expected)
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  let read := fun (id : Nat) (what : String) => do
    let some v := grads[id]? | throw <| IO.userError s!"tape: no batch_norm {what} gradient"
    LibTorch.Buffer.toFloatArrayIO v.buf
  let dx := Array.ofFn fun i : Fin 6 =>
    expected[i.val]! * bnGamma[i.val % 3]! / bnDenominator[i.val % 3]!
  let dGamma := Array.ofFn fun c : Fin 3 =>
    expected[c.val]! * normalized[c.val]! + expected[3 + c.val]! * normalized[3 + c.val]!
  let dBeta := Array.ofFn fun c : Fin 3 => expected[c.val]! + expected[3 + c.val]!
  assertFloatArrayEq "tape: batch_norm input pullback" (← read xId "input") (FloatArray.mk dx)
  assertFloatArrayEq "tape: batch_norm scale pullback" (← read gammaId "scale")
    (FloatArray.mk dGamma)
  assertFloatArrayEq "tape: batch_norm bias pullback" (← read betaId "bias") (FloatArray.mk dBeta)
  -- A NaN running variance stays NaN through both clamps.
  let (t, result) ← LibTorch.Tape.apply t "batch_norm"
    (.batchNormRunning (s := [2, 3]) ⟨1, by decide⟩)
    (xId, gammaId, betaId, (Tensor.from (#[1, -1, 0] : Array Float) : Tensor Float [3]),
      (Tensor.from (#[0 / 0, -5, 15] : Array Float) : Tensor Float [3]), Tensor.scalar 1, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "tape: no NaN batch_norm node"
  let got ← LibTorch.Buffer.toFloatArrayIO value.buf
  unless (got.get! 0).isNaN && (got.get! 3).isNaN &&
      [1, 2, 4, 5].all (fun i => got.get! i == expected[i]!) do
    throw <| IO.userError s!"tape: a NaN running variance gave {got}"

/-- `detach` returns its parent's values in a graph, and on the tape its node borrows the parent's
buffer and needs no gradient. -/
def checkDetach : IO Unit := do
  let x : Array Float := #[1.5, -2, 0]
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [3])
  let (b, ys) := b.op (Op.detach (s := [3])) [r]
  let got ← runOutput (b.build ys.toArray) {} #[← buffer [3] x] 1
  assertFloatArrayEq "graph: detach" got (FloatArray.mk x)
  let (t, xId) := Tape.leaf .empty (← buffer [3] x)
  let (t, result) ← LibTorch.Tape.apply t "detach" (.detach (s := [3])) (xId, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some node := t.getNode? y | throw <| IO.userError "tape: no detach node"
  unless !node.requiresGrad && !node.ownsValue && node.parents == #[xId] do
    throw <| IO.userError "tape: the detach node owns its value or needs a gradient"
  assertFloatArrayEq "tape: detach"
    (← LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk x)
  -- `x + detach x` with `x` requiring a gradient: only the first use contributes, `dx = 1`.
  let (t, xId) := Tape.leaf .empty (← buffer [3] x)
  let (t, result) ← LibTorch.Tape.apply t "detach" (.detach (s := [3])) (xId, ())
    (pure .float32)
  let (d, ()) ← IO.ofExcept result
  let (t, result) ← LibTorch.Tape.apply t "add" (.add (s := [3])) (xId, d, ())
    (pure .float32)
  let (z, ()) ← IO.ofExcept result
  let grads ← IO.ofExcept
    (Tape.backwardDenseAll t z (← buffer [3] #[1, 1, 1]))
  let some dx := grads[xId]? | throw <| IO.userError "tape: no gradient through x + detach x"
  assertFloatArrayEq "tape: x + detach x"
    (← LibTorch.Buffer.toFloatArrayIO dx.buf) (FloatArray.mk #[1, 1, 1])

/-- Max pooling of `[1, 2, 2, 2, 2]` over its four trailing axes with a unit window, and the flip
of its first axis after it, which has no CUDA entry. -/
def pooledFlip : Except String IR.Graph :=
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [1, 2, 2, 2, 2])
  let (b, ys) := b.op (Op.pool (s := [1, 2, 2, 2, 2]) .max
    [⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ⟨4, by decide⟩] [1, 1, 1, 1] [1, 1, 1, 1]
    [0, 0, 0, 0] [0, 0, 0, 0] [1, 1, 1, 1] false) [r]
  let (b, zs) := b.op (Op.flip (s := [1, 2, 2, 2, 2]) [⟨0, by decide⟩]) ys
  b.build zs.toArray

/-- Operator graphs that read an index constant (a gather of the rows in reverse order), that run
`Op.layerNorm` of three axes (over the last axis of `[2, 3, 4]`) and give the bits of the same layer
normalization of the `(6, 4)` matrix of its rows, and that take the scaled product of a vector and
a matrix whose scale is a value buffer at the frozen port. The operator runner reports a later node
without an entry before any node runs. -/
def checkOperatorRunner : IO Unit := do
  let (g, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 3])
  let (g, index) := g.const (.index [2] [2])
  let (g, ys) := g.op (Op.gather (s := [2, 3]) (t := [2]) [⟨0, by decide⟩] []
    ⟨by decide, by decide, by decide⟩) [r, index]
  let got ← runOutput (g.build ys.toArray)
    { const? := fun j => if j = 1 then
        some ⟨.index [2] [2], Tensor.from (#[1, 0] : Array (Fin 2))⟩ else none }
    #[← buffer [2, 3] (iota 6)] 2
  assertFloatArrayEq "graph: gather of the rows in reverse order" got
    (FloatArray.mk #[3, 4, 5, 0, 1, 2])
  let x := Array.ofFn fun i : Fin 24 => (i.val * 5 % 24).toFloat / 4
  let constants : IR.Payload Float32 :=
    { const? := fun
        | 1 => some ⟨.value [4], Tensor.full _ 1⟩
        | 2 => some ⟨.value [4], Tensor.full _ 0⟩
        | 3 => some ⟨.frozen [], Tensor.scalar normalizationEpsilon⟩
        | _ => none }
  let (g, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 3, 4])
  let (g, gamma) := g.const (.value [4])
  let (g, beta) := g.const (.value [4])
  let (g, eps) := g.const (.frozen [])
  let (g, ys) := g.op (Op.layerNorm (seqLen := 6) (embedDim := 4) (by decide) (by decide)
    (lead := [2, 3]) (x := [2, 3, 4]) (by decide) rfl) [r, gamma, beta, eps]
  let general ← runOutput (g.build ys.toArray) constants #[← buffer [2, 3, 4] x] 4
  let (g, r) := ({} : IR.Graph.Builder).input 0 (.value [2, 3, 4])
  let (g, gamma) := g.const (.value [4])
  let (g, beta) := g.const (.value [4])
  let (g, eps) := g.const (.frozen [])
  let (g, rows) := g.op (Op.reshape (s := [2, 3, 4]) (t := [6, 4]) (by decide)) [r]
  let (g, ys) := g.op (Op.layerNorm (seqLen := 6) (embedDim := 4) (by decide) (by decide)
    (lead := [6]) (x := [6, 4]) (Shape.size_vector _).symm rfl) (rows ++ [gamma, beta, eps])
  let (g, zs) := g.op (Op.reshape (s := [6, 4]) (t := [2, 3, 4]) (by decide)) ys
  let matrix ← runOutput (g.build zs.toArray) constants #[← buffer [2, 3, 4] x] 6
  unless general.size == matrix.size &&
      (List.range general.size).all fun i => (general.get! i).toBits == (matrix.get! i).toBits do
    throw <| IO.userError "graph: layer_norm of three axes differs from the matrix of its rows"
  let (g, va) := ({} : IR.Graph.Builder).input 0 (.value [3])
  let (g, vb) := g.input 1 (.value [3, 2])
  let (g, scale) := g.input 2 (.value [])
  let (g, ys) := g.op (Op.scaledMatmul (batchA := []) (batchB := []) (batch := []) (m := 1)
    (n := 3) (p := 2) true false (fun _ => rfl) (fun h => nomatch h)
    (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])) [va, vb, scale]
  let a : Array Float := #[1, 2, 3]
  let b := (iota 6).map (· - 2)
  let got ← runOutput (g.build ys.toArray) {}
    #[← buffer [3] a, ← buffer [3, 2] b, ← buffer [] #[0.5]] 3
  let expected := Array.ofFn fun i : Fin 2 =>
    0.5 * (List.range 3).foldl (fun acc j => acc + a[j]! * b[j * 2 + i.val]!) 0
  assertFloatArrayEq "graph: scaled product of a vector and a matrix" got (FloatArray.mk expected)
  let op ← IO.ofExcept pooledFlip
  let source ← buffer [1, 2, 2, 2, 2] (iota 16)
  try
    let _ ← op.runBuffers {} #[⟨⟨.value, source.s⟩, source.buf⟩]
    throw <| IO.userError "operator graph: a node without an entry was accepted"
  catch e =>
    unless (e.toString.splitOn "node 2: no CUDA entry").length > 1 do
      throw <| IO.userError s!"operator graph: a node ran before every node was checked: {e}"

/-- The run of `g` on `inputs` is refused with a message containing `text`. -/
def expectRefusal (tag : String) (g : IR.Graph) (payload : IR.Payload Float32)
    (inputs : Array IR.Resident) (text : String) : IO Unit := do
  let refused ← try
    let _ ← g.runBuffers payload inputs
    pure none
  catch e => pure (some e.toString)
  match refused with
  | some message =>
      unless (message.splitOn text).length > 1 do
        throw <| IO.userError s!"{tag}: unexpected refusal: {message}"
  | none => throw <| IO.userError s!"{tag}: accepted"

/-- The scaled product of `[2, 3]` and `[3, 2]` with its scale read from graph input `2` at
`scale`. -/
def ofScaledMatmul (scale : Port) : IR.Graph :=
  { nodes := #[.input 0 (.value [2, 3]), .input 1 (.value [3, 2]), .input 2 scale,
      .op ⟨_, _, Op.scaledMatmul (batchA := []) (batchB := []) (batch := []) (m := 2) (n := 3)
        (p := 2) false false (fun h => nomatch h) (fun h => nomatch h)
        (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])⟩ #[⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩]],
    outputs := #[⟨3, 0⟩] }

/-- The entries of a node's only output when it is a Boolean value. -/
def boolsOf : Option (List IR.Resident) → Option (Array Bool)
  | some [⟨⟨.bool, _⟩, t⟩] => some (Tensor.Rep.data t)
  | _ => none

/-- The operator runner's ports: a frozen scale supplied on the host, the same scale supplied as a
value buffer to a frozen and to a value input node, a detached input supplied detached and as a
value, and Boolean values of an input node and of a constant give the expected values. Refused:
a host tensor supplied to a value input node, a frozen or detached node feeding a value port, a
sort (the table has no entry for an operator of several outputs, so no CUDA run reads a second
output), and a later missing input or constant, reported before an earlier pooling node of spatial
rank four records. -/
def checkResidentPaths : IO Unit := do
  let a := iota 6
  let b := (iota 6).map (· - 2)
  let xa ← buffer [2, 3] a
  let xb ← buffer [3, 2] b
  let half ← buffer [] #[0.5]
  let expected := FloatArray.mk <| Array.ofFn fun i : Fin 4 =>
    0.5 * (List.range 3).foldl (fun acc j => acc + a[i.val / 2 * 3 + j]! * b[j * 2 + i.val % 2]!) 0
  for (tag, scale, supplied) in [("frozen scale on the host", Port.frozen [],
        (⟨⟨.frozen, []⟩, Tensor.scalar 0.5⟩ : IR.Resident)),
      ("value buffer read at a frozen input", Port.frozen [], valueOf [] half.buf),
      ("value input feeding a frozen port", Port.value [], valueOf [] half.buf)] do
    let values ← (ofScaledMatmul scale).runBuffers {}
      #[valueOf [2, 3] xa.buf, valueOf [3, 2] xb.buf,
      supplied]
    assertFloatArrayEq s!"operator graph: {tag}" (← wordsAt values 3 0) expected
  let x ← buffer [3] #[1.5, -2, 0]
  let detached : IR.Graph :=
    { nodes := #[.input 0 (.detached [3]), .op ⟨_, _, Op.detach (s := [3])⟩ #[⟨0, 0⟩]],
      outputs := #[⟨1, 0⟩] }
  for supplied in [(⟨⟨.detached, [3]⟩, x.buf⟩ : IR.Resident), valueOf [3] x.buf] do
    let values ← detached.runBuffers {} #[supplied]
    assertFloatArrayEq "operator graph: detached input" (← wordsAt values 1 0)
      (FloatArray.mk #[1.5, -2, 0])
  let mask : Tensor Bool [2] := Tensor.from (#[true, false] : Array Bool)
  let flags : Tensor Bool [2] := Tensor.from (#[false, true] : Array Bool)
  let booleans : IR.Graph :=
    { nodes := #[.input 0 (.bool [2]), .const (.bool [2])], outputs := #[⟨0, 0⟩, ⟨1, 0⟩] }
  let values ← booleans.runBuffers { const? := fun j => if j = 1 then some ⟨.bool [2], mask⟩
    else none } #[⟨⟨.bool, [2]⟩, flags⟩]
  unless boolsOf values[0]? == some #[false, true] && boolsOf values[1]? == some #[true, false] do
    throw <| IO.userError "operator graph: Boolean values changed"
  expectRefusal "host tensor for a value input" (ofScaledMatmul (.value []))
    {} #[valueOf [2, 3] xa.buf, valueOf [3, 2] xb.buf, ⟨⟨.frozen, []⟩, Tensor.scalar 0.5⟩]
    "where"
  expectRefusal "frozen node feeding a value port"
    { nodes := #[.input 0 (.frozen [3]), .op ⟨_, _, Op.relu (s := [3])⟩ #[⟨0, 0⟩]],
      outputs := #[] } {} #[⟨⟨.frozen, [3]⟩, Tensor.full [3] 1⟩] "does not accept"
  expectRefusal "detached node feeding a value port"
    { nodes := #[.input 0 (.detached [3]), .op ⟨_, _, Op.relu (s := [3])⟩ #[⟨0, 0⟩]],
      outputs := #[] } {} #[⟨⟨.detached, [3]⟩, x.buf⟩] "does not accept"
  expectRefusal "sort"
    { nodes := #[.input 0 (.value [3]),
        .op ⟨_, _, Op.sort (s := [3]) ⟨0, by decide⟩ false⟩ #[⟨0, 0⟩]],
      outputs := #[⟨1, 1⟩] } {} #[valueOf [3] x.buf] "node 1: no CUDA entry for sort"
  let op ← IO.ofExcept pooledFlip
  let op := { op with nodes := op.nodes.pop, outputs := #[] }
  let source ← buffer [1, 2, 2, 2, 2] (iota 16)
  expectRefusal "later missing input" { op with nodes := op.nodes.push (.input 1 (.value [3])) }
    {} #[valueOf source.s source.buf] "node 2: missing input 1"
  expectRefusal "later missing constant" { op with nodes := op.nodes.push (.const (.value [3])) }
    {} #[valueOf source.s source.buf] "node 2: missing constant"

/-- The checked program of `x + 1` at binary32. -/
def addOne : Except String (Kernel.Elementwise (fun x : Float32 => x + 1)) := by
  prepare_tensor_function

/-- A custom operation `x + 1` in a graph, and three custom nodes refused before they run: one
without its program, one whose program declares another precision, and one of no input whose
program reads input `0` (no CUDA source). -/
def checkCustom : IO Unit := do
  let program ← IO.ofExcept (addOne.map (·.program))
  let (b, r) := ({} : IR.Graph.Builder).input 0 (.value [3])
  let (b, ys) := b.op (Op.custom "addOne" [[3]] [3]) [r]
  let op ← IO.ofExcept (b.build ys.toArray)
  let payload : IR.Payload Float32 :=
    { custom? := fun j => if j = 1 then some program else none }
  let source ← buffer [3] #[1, 2, -0.5]
  assertFloatArrayEq "graph: custom x + 1" (← runOutput (.ok op) payload #[source] 1)
    (FloatArray.mk #[2, 3, 0.5])
  expectRefusal "custom without its program" op {} #[valueOf [3] source.buf] "missing checked body"
  let wide := { program with scalar := { program.scalar with precision := .native .binary64 } }
  expectRefusal "custom of another precision" op
    { custom? := fun j => if j = 1 then some wide else none } #[valueOf [3] source.buf]
    "node 1: custom addOne: resident buffers require native binary32"
  expectRefusal "custom without CUDA source"
    { nodes := #[.op ⟨_, _, Op.custom "addOne" [] [3]⟩ #[]], outputs := #[⟨0, 0⟩] }
    { custom? := fun j => if j = 0 then some program else none } #[] "is not declared"

/-- The value and the input, scale and bias pullbacks of layer normalization `op` of an input of
shape `s` on the CUDA tape, for the cotangent of the value itself. -/
def layerNormOnTape {s : Shape} (op : Op [.value s, .value [4], .value [4], .frozen []] [.value s])
    (x : Array Float) : IO (Array LibTorch.Buffer) := do
  let leaf := Tape.leaf (V := LibTorch.SomeBuffer)
  let (t, xId) := leaf .empty (← buffer s x)
  let (t, gammaId) := leaf t (← buffer [4] #[0.5, 1, 1.5, 2])
  let (t, betaId) := leaf t (← buffer [4] #[-1, 0, 1, 2])
  let (t, result) ← LibTorch.Tape.apply t "layer_norm" op
    (xId, gammaId, betaId, Tensor.scalar (1 / 64 : Float), ()) (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError "tape: no layer_norm node"
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  let mut out := #[value.buf]
  for id in [xId, gammaId, betaId] do
    let some g := grads[id]? | throw <| IO.userError "tape: no layer_norm gradient"
    out := out.push g.buf
  pure out

/-- The matrix product `op` of `a` and `b` (functions of the index with small integer values) on
the CPU tape by its rule and on the CUDA tape by its entry: the value and both pullbacks for the
cotangent of the value itself are equal. -/
def compareProduct (tag : String) {sa sb so : Shape}
    (op : Op [.value sa, .value sb] [.value so]) (a b : List Nat → Float) : IO Unit := do
  let aT : Tensor Float sa := Tensor.generate sa a
  let bT : Tensor Float sb := Tensor.generate sb b
  let cpu : Result (Array (Array Float)) := do
    let t : Tape (SomeTensor Float) := .empty
    let (t, aId) := Tape.leaf (t := t) (.ofTensor aT)
    let (t, bId) := Tape.leaf (t := t) (.ofTensor bT)
    let (t, yId, ()) ← Tape.apply t "matmul" op (aId, bId, ())
    let some y := t.getValue? yId | throw "no CPU product"
    let grads ← Tape.backwardDenseAll t yId y
    let some da := grads[aId]? | throw "no CPU left gradient"
    let some db := grads[bId]? | throw "no CPU right gradient"
    pure (#[y, da, db].map fun v => Tensor.Rep.data v.tensor)
  let expected ← IO.ofExcept (cpu.mapError fun e => s!"{tag}: CPU rule: {e}")
  let leaf := Tape.leaf (V := LibTorch.SomeBuffer)
  let (t, aId) := leaf .empty (← buffer sa (Tensor.Rep.data aT))
  let (t, bId) := leaf t (← buffer sb (Tensor.Rep.data bT))
  let (t, result) ← LibTorch.Tape.apply t "matmul" op (aId, bId, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError s!"{tag}: no CUDA product"
  assertFloatArrayEq s!"{tag}: value" (← LibTorch.Buffer.toFloatArrayIO value.buf)
    (FloatArray.mk expected[0]!)
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  for (id, what, j) in [(aId, "left", 1), (bId, "right", 2)] do
    let some g := grads[id]? | throw <| IO.userError s!"{tag}: no {what} gradient"
    assertFloatArrayEq s!"{tag}: {what} pullback"
      (← LibTorch.Buffer.toFloatArrayIO g.buf) (FloatArray.mk expected[j]!)

/-- The scaled matrix product `op` of `a` and `b` with the scale `0.5`, as `compareProduct`. -/
def compareScaledProduct (tag : String) {sa sb so : Shape}
    (op : Op [.value sa, .value sb, .frozen []] [.value so]) (a b : List Nat → Float) :
    IO Unit := do
  let aT : Tensor Float sa := Tensor.generate sa a
  let bT : Tensor Float sb := Tensor.generate sb b
  let scale : Tensor Float [] := Tensor.scalar 0.5
  let cpu : Result (Array (Array Float)) := do
    let t : Tape (SomeTensor Float) := .empty
    let (t, aId) := Tape.leaf (t := t) (.ofTensor aT)
    let (t, bId) := Tape.leaf (t := t) (.ofTensor bT)
    let (t, yId, ()) ← Tape.apply t "scaled_matmul" op (aId, bId, scale, ())
    let some y := t.getValue? yId | throw "no CPU product"
    let grads ← Tape.backwardDenseAll t yId y
    let some da := grads[aId]? | throw "no CPU left gradient"
    let some db := grads[bId]? | throw "no CPU right gradient"
    pure (#[y, da, db].map fun v => Tensor.Rep.data v.tensor)
  let expected ← IO.ofExcept (cpu.mapError fun e => s!"{tag}: CPU rule: {e}")
  let leaf := Tape.leaf (V := LibTorch.SomeBuffer)
  let (t, aId) := leaf .empty (← buffer sa (Tensor.Rep.data aT))
  let (t, bId) := leaf t (← buffer sb (Tensor.Rep.data bT))
  let (t, result) ← LibTorch.Tape.apply t "scaled_matmul" op (aId, bId, scale, ())
    (pure .float32)
  let (y, ()) ← IO.ofExcept result
  let some value := t.getValue? y | throw <| IO.userError s!"{tag}: no CUDA product"
  assertFloatArrayEq s!"{tag}: value" (← LibTorch.Buffer.toFloatArrayIO value.buf)
    (FloatArray.mk expected[0]!)
  let grads ← IO.ofExcept (Tape.backwardDenseAll t y value)
  for (id, what, j) in [(aId, "left", 1), (bId, "right", 2)] do
    let some g := grads[id]? | throw <| IO.userError s!"{tag}: no {what} gradient"
    assertFloatArrayEq s!"{tag}: {what} pullback"
      (← LibTorch.Buffer.toFloatArrayIO g.buf) (FloatArray.mk expected[j]!)

/-- Whether layer normalization `general` of an input of shape `s` and the matrix entry `matrix` on
the `(seqLen, embedDim)` view of the same entries give the same words, value and pullbacks. -/
def sameLayerNorm {s m : Shape}
    (general : Op [.value s, .value [4], .value [4], .frozen []] [.value s])
    (matrix : Op [.value m, .value [4], .value [4], .frozen []] [.value m]) (x : Array Float) :
    IO Bool := do
  let a ← layerNormOnTape general x
  let b ← layerNormOnTape matrix x
  pure (a.size == 4 && b.size == 4 && (← (a.zip b).allM fun (u, v) => sameWords u v))

/-- The general entries on the tape. A layer normalization of `[2, 3, 4]` over its last axis and one
of a vector give the words of the matrix entry on `[6, 4]` and on `[1, 4]`, value and pullbacks.
Matrix products with a vector on the left (also behind a broadcast batch prefix), on the right and
on both sides, three scaled ones, and convolutions behind two leading axes of extents two and
three, behind `[1, 2]` and without a batch axis match the CPU rule in value and pullbacks. A
convolution behind two leading axes whose recording fails after its first view returns the tape it
was given. -/
def checkGeneralEntries : IO Unit := do
  let x := Array.ofFn fun i : Fin 24 => (i.val * 7 % 24).toFloat / 4
  unless (← sameLayerNorm (Op.layerNorm (seqLen := 6) (embedDim := 4) (by decide) (by decide)
      (lead := [2, 3]) (x := [2, 3, 4]) (by decide) rfl)
      (Op.layerNorm (seqLen := 6) (embedDim := 4) (by decide) (by decide) (lead := [6])
        (x := [6, 4]) (Shape.size_vector _).symm rfl) x) do
    throw <| IO.userError "tape: layer_norm of three axes differs from the matrix entry"
  unless (← sameLayerNorm (Op.layerNorm (seqLen := 1) (embedDim := 4) (by decide) (by decide)
      (lead := []) (x := [4]) rfl rfl)
      (Op.layerNorm (seqLen := 1) (embedDim := 4) (by decide) (by decide) (lead := [1])
        (x := [1, 4]) (Shape.size_vector _).symm rfl) #[0.5, -1, 2, 0.25]) do
    throw <| IO.userError "tape: layer_norm of a vector differs from the matrix entry"
  compareProduct "matmul vector-batch" (Op.matmul (batchA := []) (batchB := [2]) (batch := [2])
    (m := 1) (n := 3) (p := 4) true false (fun _ => rfl) (fun h => nomatch h) (by decide)
    (Shape.CanBroadcastTo.refl [2])) (mix 13) (mix 14)
  compareProduct "matmul batch-vector" (Op.matmul (batchA := [2]) (batchB := []) (batch := [2])
    (m := 2) (n := 3) (p := 1) false true (fun h => nomatch h) (fun _ => rfl)
    (Shape.CanBroadcastTo.refl [2]) (by decide)) (mix 15) (mix 16)
  compareProduct "matmul vector-vector" (Op.matmul (batchA := []) (batchB := []) (batch := [])
    (m := 1) (n := 3) (p := 1) true true (fun _ => rfl) (fun _ => rfl)
    (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])) (mix 17) (mix 18)
  compareProduct "matmul broadcast vectors-batch" (Op.matmul (batchA := [1]) (batchB := [3])
    (batch := [3]) (m := 1) (n := 2) (p := 4) true false (fun _ => rfl) (fun h => nomatch h)
    (by decide) (Shape.CanBroadcastTo.refl [3])) (mix 25) (mix 26)
  compareScaledProduct "scaled matmul vector-batch" (Op.scaledMatmul (batchA := [])
    (batchB := [2]) (batch := [2]) (m := 1) (n := 3) (p := 4) true false (fun _ => rfl)
    (fun h => nomatch h) (by decide) (Shape.CanBroadcastTo.refl [2])) (mix 27) (mix 28)
  compareScaledProduct "scaled matmul batch-broadcast vectors" (Op.scaledMatmul (batchA := [2])
    (batchB := [1]) (batch := [2]) (m := 2) (n := 3) (p := 1) false true (fun h => nomatch h)
    (fun _ => rfl) (Shape.CanBroadcastTo.refl [2]) (by decide)) (mix 29) (mix 30)
  compareScaledProduct "scaled matmul vector-vector" (Op.scaledMatmul (batchA := [])
    (batchB := []) (batch := []) (m := 1) (n := 3) (p := 1) true true (fun _ => rfl)
    (fun _ => rfl) (Shape.CanBroadcastTo.refl []) (Shape.CanBroadcastTo.refl [])) (mix 34) (mix 35)
  let dims : Shape.ConvDims [2, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  compareConvOp "conv behind [1, 2]" (Op.conv dims (fun _ => 2) (fun _ => 1)
    (fun _ => 0) (fun _ => 2) 2 (lead := [1, 2]) (by decide) rfl rfl) (mix 19) (mix 20) (mix 21)
  -- Three input channels do not split into two groups: the recording fails after the input's view
  -- and returns the tape it was given.
  let leaf := Tape.leaf (V := LibTorch.SomeBuffer)
  let (t, kernelId) := leaf .empty (← buffer [2, 2, 2] (iota 8))
  let (t, biasId) := leaf t (← buffer [2] #[0, 0])
  let (t, inputId) := leaf t (← buffer [1, 2, 3, 5] (iota 30))
  let dims3 : Shape.ConvDims [2, 3, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  let (t', result) ← LibTorch.Tape.apply t "conv" (Op.conv dims3 (fun _ => 1)
    (fun _ => 0) (fun _ => 0) (fun _ => 1) 2 (lead := [1, 2]) (by decide) rfl rfl)
    (kernelId, biasId, inputId, ()) (pure .float32)
  match result with
  | .ok _ => throw <| IO.userError "tape: a conv of three channels in two groups recorded"
  | .error message =>
      unless (message.splitOn "groups do not divide").length > 1 do
        throw <| IO.userError s!"tape: unexpected conv failure: {message}"
  let mut intact := t'.size == t.size
  for id in [kernelId, biasId, inputId] do
    match t'.getValue? id, t.getValue? id with
    | some a, some b => intact := intact && a.s == b.s && (← sameWords a.buf b.buf)
    | _, _ => intact := false
  unless intact do
    throw <| IO.userError "tape: a failed conv behind two leading axes changed the tape"
  let dims : Shape.ConvDims [6, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  compareConvOp "conv behind [2, 3]" (Op.conv dims (fun _ => 2) (fun _ => 1)
    (fun _ => 0) (fun _ => 2) 2 (lead := [2, 3]) (by decide) rfl rfl) (mix 31) (mix 32) (mix 33)
  let dims : Shape.ConvDims [1, 4, 5] [2, 2, 2] 1 := Shape.ConvDims.channelsFirst rfl
  compareConvOp "conv without a batch axis" (Op.conv dims (fun _ => 1) (fun _ => 1)
    (fun _ => 1) (fun _ => 1) 2 (lead := []) (by decide) rfl rfl) (mix 22) (mix 23) (mix 24)

/-- Run every graph and tape check of this file. -/
def run : IO Unit := do
  checkSlice
  checkVectorMatmul
  checkBatchedMaxPool
  checkPoolsWithoutChannel
  checkSmoothMaxPool
  checkLayerNormDefaults
  checkEdges
  checkEmptySlicePullback
  checkConfiguredEmptySlice
  checkConv
  checkConvPullback
  checkConvPaddedEmpty
  checkConvAgainstRule
  checkEagerConv
  checkConvMalformed
  checkBatchNormRunning
  checkDetach
  checkOperatorRunner
  checkResidentPaths
  checkCustom
  checkGeneralEntries
  IO.println "  graph evaluation: registered operators run by their CUDA entries"
  IO.println "  operator graphs: operator-runner checks passed"

end TorchLean.Test.CUDA.Graph
