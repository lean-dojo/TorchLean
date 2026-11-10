/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Reduce.Backward

/-!
# Convolution and pooling on any axes

`conv x w dims stride padLow padHigh inputDilation dilation groups batchGroups` is XLA's
`ConvGeneralDilated` at any number `r` of spatial axes. The dimension numbers `dims` say which
axes of the input hold the batch, the features and the window, and which axes of the kernel hold
its output and input features and the window. The output keeps the layout of the input. Each
output entry is one left fold from zero over the kernel's input features and then its window in
row-major order, and each step adds `xv * w _`, where `xv` is the input entry under the window cell
or zero at a padding or input-dilation hole.

`convTranspose` is PyTorch's transposed convolution with the same dimension numbers, and
`pool op x dims kernel stride padLow padHigh dilation ceilMode` is PyTorch's pooling over the axes
`dims` (`ReduceWindow` for sums, maxima and minima, and the average with or without the padding in
the count). The other axes of a pooled tensor are kept, so no axis plays the role of a channel.
`smoothMaxPool` is a log-sum-exp over each window of `pool`, and `adaptivePool` is PyTorch's
adaptive pooling into `outputSize` bins along the axes `dims`. Both convolutions have input and
kernel gradients (those of `conv` without input dilation or batch groups), and `pool` and
`smoothMaxPool` have directional derivatives and input gradients.

We wrote `conv` by hand from the `ConvGeneralDilated` section of XLA's operation semantics
(`https://openxla.org/xla/operation_semantics`), and `convTranspose` and `pool` from the PyTorch
2.8 pages of `torch.nn.ConvTranspose1d`, `torch.nn.MaxPool1d` and `torch.nn.AvgPool1d`. The
adaptive bins are ATen's `start_index` and `end_index` (`aten/src/ATen/native/AdaptivePooling.h`
at v2.8.0). A maximum pool selects the last NaN of a window, or its first extremum when it holds
no NaN, as torch 2.8.0 does, with two differences. We put `-0` below `0`, and torch keeps whichever
comes first: on `[-0, 0]` its `max_pool1d` and `adaptive_max_pool1d` give `-0` at index `0`, and
we give `0` at index `1`. A window of padding only gives `0`, where torch's `max_pool1d` gives
`-inf` (`[5, 7]` with kernel 2, stride 1, padding 1 and dilation 3).
Over two axes torch's adaptive average divides a bin's sum by each bin
extent in turn, where `adaptivePool` divides once by the cell count. A binary32 `6 × 6` input of
zeros with a `5` at a corner, pooled to `2 × 2`, gives `(5 / 3) / 3 = 0.55555552` in torch and
`5 / 9 = 0.5555556` here.

* `Shape.ConvDims` holds the dimension numbers and `ConvDims.channelsFirst` PyTorch's layout.
* `Shape.conv`, `Shape.convTranspose` and `Shape.pool` are the output shapes.
* `conv_apply`, `convTranspose_apply` and the `pool_*_apply` lemmas read an output entry as its
  fold.
* `poolFast` and `poolBackwardFast` compute `pool` and `poolBackward` in one pass (`@[csimp]`).
* `convFast`, `convBackwardInputFast` and `convBackwardWeightFast` compute `conv` and its
  gradients with their shapes, axis tables and window indices built once (`@[csimp]`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (rank rank_eq_length size_eq_prod)

namespace TorchLean.Shape

/-! ## Coordinates read at positions -/

/-- The index of a coordinate of `List.ofFn g` along axis `k` is below `g k`. -/
theorem Coord.index_lt_ofFn {n : Nat} (g : Fin n → Nat) (c : Coord (List.ofFn g)) (k : Fin n) :
    Coord.index c k.val < g k := by
  have hk : k.val < Shape.rank (List.ofFn g) := by simp [Shape.rank_eq_length]
  simpa [extent] using Coord.index_lt c hk

/-- The coordinate of `s` whose index along the axis `e p` is `f p`. -/
def Coord.ofPositions {s : Shape} {n : Nat} (e : Fin n ≃ Fin s.rank)
    (f : (p : Fin n) → Fin (s.extent (e p))) : s.Coord :=
  Coord.ofFun fun a => Fin.cast (by rw [Equiv.apply_symm_apply]) (f (e.symm a))

/-- The index of `Coord.ofPositions e f` along the axis `e p` is `f p`. -/
@[simp] theorem Coord.index_ofPositions {s : Shape} {n : Nat} (e : Fin n ≃ Fin s.rank)
    (f : (p : Fin n) → Fin (s.extent (e p))) (p : Fin n) :
    Coord.index (Coord.ofPositions e f) (e p).val = (f p).val := by
  have key : ∀ q, q = p → (f q).val = (f p).val := by
    rintro _ rfl
    rfl
  rw [Coord.ofPositions, Coord.index_ofFun]
  simpa using key _ (e.symm_apply_apply p)

/-- The coordinate of `s` whose index along the axis `e p` is `index p`, or `none` when an index
is out of range. -/
def Coord.ofIndices? {s : Shape} {n : Nat} (e : Fin n ≃ Fin s.rank) (index : Fin n → Nat) :
    Option s.Coord :=
  if h : ∀ p, index p < s.extent (e p) then some (Coord.ofPositions e fun p => ⟨index p, h p⟩)
  else none

/-! ## Dimension numbers -/

/-- Dimension numbers of a convolution of an input of shape `s` with a kernel of shape `t` over
`r` spatial axes, in XLA's convention. Position `0` is the batch axis of the input and the output
feature axis of the kernel, position `1` the feature axis of the input and the input feature axis
of the kernel, and position `i + 2` spatial axis `i` of both. -/
structure ConvDims (s t : Shape) (r : Nat) where
  /-- The axis of the input at each position. -/
  input : Fin (r + 2) ≃ Fin s.rank
  /-- The axis of the kernel at each position. -/
  kernel : Fin (r + 2) ≃ Fin t.rank

/-- PyTorch's layout: the input is `(N, C, spatial...)` and the kernel `(O, I, kernel...)`. -/
def ConvDims.channelsFirst {n c o i : Nat} {spatial kernel : List Nat}
    (h : kernel.length = spatial.length) :
    ConvDims (n :: c :: spatial) (o :: i :: kernel) spatial.length where
  input := finCongr (by simp [Shape.rank_eq_length])
  kernel := finCongr (by simp [Shape.rank_eq_length, h])

/-- The extent of the output of `conv` at a position of its dimension numbers: the input batch
divided into `batchGroups`, the kernel's output features, and the window count along each spatial
axis of the input dilated by `inputDilation`. -/
def convExtent (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => s.extent (dims.input ⟨0, by omega⟩) / batchGroups
  | ⟨1, _⟩ => t.extent (dims.kernel ⟨0, by omega⟩)
  | ⟨i + 2, h⟩ =>
      slidingWindowOutDimDilated
        (dilatedKernelExtent (s.extent (dims.input ⟨i + 2, h⟩)) (inputDilation ⟨i, by omega⟩))
        (t.extent (dims.kernel ⟨i + 2, h⟩)) (stride ⟨i, by omega⟩) (dilation ⟨i, by omega⟩)
        (padLow ⟨i, by omega⟩) (padHigh ⟨i, by omega⟩)

/-- The output shape of `conv`, in the layout of the input. -/
def conv (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) : Shape :=
  List.ofFn fun a =>
    convExtent s t dims stride padLow padHigh inputDilation dilation batchGroups (dims.input.symm a)

/-- The window a convolution folds over: the kernel's input features, then its spatial axes. -/
def convWindow (t : Shape) {s : Shape} {r : Nat} (dims : ConvDims s t r) : Shape :=
  List.ofFn fun p : Fin (r + 1) => t.extent (dims.kernel p.succ)

/-- The index of an output coordinate of `conv` at a position of the dimension numbers is below
the output extent there. -/
theorem index_conv_lt {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    {stride padLow padHigh inputDilation dilation : Fin r → Nat} {batchGroups : Nat}
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (p : Fin (r + 2)) :
    Coord.index o (dims.input p).val <
      convExtent s t dims stride padLow padHigh inputDilation dilation batchGroups p := by
  have := Coord.index_lt_ofFn _ o (dims.input p)
  rwa [Equiv.symm_apply_apply] at this

/-- The output feature of an output coordinate of `conv`, an index of the kernel's output
feature axis. -/
def ConvDims.outFeature {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    {stride padLow padHigh inputDilation dilation : Fin r → Nat} {batchGroups : Nat}
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord) :
    Fin (t.extent (dims.kernel ⟨0, by omega⟩)) :=
  ⟨Coord.index o (dims.input ⟨1, by omega⟩).val, index_conv_lt dims o ⟨1, by omega⟩⟩

/-- The kernel coordinate read at the output feature `f` and the window coordinate `k`. -/
def ConvDims.kernelCoord {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (f : Fin (t.extent (dims.kernel ⟨0, by omega⟩))) (k : (t.convWindow dims).Coord) : t.Coord :=
  Coord.ofPositions dims.kernel fun
    | ⟨0, _⟩ => f
    | ⟨q + 1, h⟩ => ⟨Coord.index k q, Coord.index_lt_ofFn _ k ⟨q, by omega⟩⟩

/-- The index along the input axis at position `p` that `conv` reads at the output coordinate
`o` and the window coordinate `k`, before the input-dilation division: the batch group of the
output feature and the batch index, the feature group and the window feature, or the window
position `o * stride + k * dilation` less the low padding. -/
def ConvDims.inputOffset {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) : Fin (r + 2) → Nat
  | ⟨0, _⟩ =>
      Coord.index o (dims.input ⟨1, by omega⟩).val /
          (t.extent (dims.kernel ⟨0, by omega⟩) / batchGroups) *
        (s.extent (dims.input ⟨0, by omega⟩) / batchGroups) +
      Coord.index o (dims.input ⟨0, by omega⟩).val
  | ⟨1, _⟩ =>
      Coord.index o (dims.input ⟨1, by omega⟩).val /
          (t.extent (dims.kernel ⟨0, by omega⟩) / groups) *
        t.extent (dims.kernel ⟨1, by omega⟩) + Coord.index k 0
  | ⟨i + 2, h⟩ =>
      Coord.index o (dims.input ⟨i + 2, h⟩).val * stride ⟨i, by omega⟩ +
        Coord.index k (i + 1) * dilation ⟨i, by omega⟩ - padLow ⟨i, by omega⟩

/-- Whether the window cell at position `p` lies on an input entry: past the low padding and on
the input-dilation grid along a spatial axis, always along the batch and feature axes. -/
def ConvDims.inputAligned {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) : Fin (r + 2) → Bool
  | ⟨0, _⟩ => true
  | ⟨1, _⟩ => true
  | ⟨i + 2, h⟩ =>
      decide (padLow ⟨i, by omega⟩ ≤
          Coord.index o (dims.input ⟨i + 2, h⟩).val * stride ⟨i, by omega⟩ +
            Coord.index k (i + 1) * dilation ⟨i, by omega⟩) &&
        decide ((Coord.index o (dims.input ⟨i + 2, h⟩).val * stride ⟨i, by omega⟩ +
            Coord.index k (i + 1) * dilation ⟨i, by omega⟩ - padLow ⟨i, by omega⟩) %
              inputDilation ⟨i, by omega⟩ = 0)

/-- The index along the input axis at position `p` that `conv` reads: the offset, divided by the
input dilation along a spatial axis. -/
def ConvDims.inputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) : Fin (r + 2) → Nat
  | ⟨0, h⟩ => dims.inputOffset stride padLow padHigh inputDilation dilation groups batchGroups o k
      ⟨0, h⟩
  | ⟨1, h⟩ => dims.inputOffset stride padLow padHigh inputDilation dilation groups batchGroups o k
      ⟨1, h⟩
  | ⟨i + 2, h⟩ =>
      dims.inputOffset stride padLow padHigh inputDilation dilation groups batchGroups o k
        ⟨i + 2, h⟩ / inputDilation ⟨i, by omega⟩

/-- The dimension numbers with the kernel's two feature axes exchanged. A transposed convolution
in PyTorch's layout `(I, O / groups, kernel...)` takes `(ConvDims.channelsFirst h).swapKernel`. -/
def ConvDims.swapKernel {s t : Shape} {r : Nat} (dims : ConvDims s t r) : ConvDims s t r where
  input := dims.input
  kernel := (Equiv.swap ⟨0, by omega⟩ ⟨1, by omega⟩).trans dims.kernel

/-- The extent of a transposed convolution along one spatial axis: `(n - 1) * stride +
dilation * (kernel - 1) + 1 + outputPadding - padLow - padHigh`, and `0` for an empty input or
kernel or a zero stride. -/
def convTransposeOutDim (n kernel stride padLow padHigh outputPadding dilation : Nat) : Nat :=
  if n = 0 || kernel = 0 || stride = 0 then 0
  else (n - 1) * stride + dilation * (kernel - 1) + 1 + outputPadding - padLow - padHigh

/-- The extent of the output of `convTranspose` at a position of its dimension numbers: the batch,
`groups` times the kernel's output features, and `convTransposeOutDim` along each spatial axis. -/
def convTransposeExtent (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => s.extent (dims.input ⟨0, by omega⟩)
  | ⟨1, _⟩ => groups * t.extent (dims.kernel ⟨0, by omega⟩)
  | ⟨i + 2, h⟩ =>
      convTransposeOutDim (s.extent (dims.input ⟨i + 2, h⟩)) (t.extent (dims.kernel ⟨i + 2, h⟩))
        (stride ⟨i, by omega⟩) (padLow ⟨i, by omega⟩) (padHigh ⟨i, by omega⟩)
        (outputPadding ⟨i, by omega⟩) (dilation ⟨i, by omega⟩)

/-- The output shape of `convTranspose`, in the layout of the input. -/
def convTranspose (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) : Shape :=
  List.ofFn fun a =>
    convTransposeExtent s t dims stride padLow padHigh outputPadding dilation groups
      (dims.input.symm a)

/-- The window a transposed convolution folds over: the kernel's input features of one group, then
its spatial axes. -/
def convTransposeWindow (s t : Shape) {r : Nat} (dims : ConvDims s t r) (groups : Nat) : Shape :=
  t.extent (dims.kernel ⟨1, by omega⟩) / groups ::
    List.ofFn fun i : Fin r => t.extent (dims.kernel i.succ.succ)

/-- The index of an output coordinate of `convTranspose` at a position of the dimension numbers is
below the output extent there. -/
theorem index_convTranspose_lt {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    {stride padLow padHigh outputPadding dilation : Fin r → Nat} {groups : Nat}
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (p : Fin (r + 2)) :
    Coord.index o (dims.input p).val <
      convTransposeExtent s t dims stride padLow padHigh outputPadding dilation groups p := by
  have := Coord.index_lt_ofFn _ o (dims.input p)
  rwa [Equiv.symm_apply_apply] at this

/-! ## Folded batch axes -/

/-- The shape `s` with its axis `a` replaced in place by the axes of `lead`: a convolution's batch
axis standing for the leading axes `lead`, folded row-major. -/
def expandAxis (s : Shape) (a : Fin s.rank) (lead : Shape) : Shape :=
  s.take a ++ lead ++ s.drop (a + 1)

/-- Replacing an axis by axes with as many entries keeps the number of entries. -/
theorem size_expandAxis {s : Shape} {a : Fin s.rank} {lead : Shape}
    (h : s.extent a = lead.size) : (s.expandAxis a lead).size = s.size := by
  have ha : a.val < s.length := a.isLt.trans_eq s.rank_eq_length
  conv_rhs => rw [← List.take_append_drop a.val s, List.drop_eq_getElem_cons ha]
  simp only [expandAxis, Shape.size_eq_prod, List.prod_append, List.prod_cons]
  rw [← Shape.size_eq_prod lead, ← h, Nat.mul_assoc]
  rfl

/-- A shape given as `s` with an axis replaced by axes with as many entries has as many entries
as `s`. -/
theorem size_eq_of_expandAxis {s x lead : Shape} {a : Fin s.rank} (h : s.extent a = lead.size)
    (hx : x = s.expandAxis a lead) : x.size = s.size := by
  rw [hx]
  exact size_expandAxis h

/-- Replacing an axis by itself gives the shape back. -/
@[simp] theorem expandAxis_extent (s : Shape) (a : Fin s.rank) :
    s.expandAxis a [s.extent a] = s := by
  have ha : a.val < s.length := a.isLt.trans_eq s.rank_eq_length
  conv_rhs => rw [← List.take_append_drop a.val s, List.drop_eq_getElem_cons ha]
  simp [expandAxis, extent]

/-! ## Gradient windows -/

/-- `conv` keeps the rank of its input. -/
@[simp] theorem rank_conv (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) :
    (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).rank = s.rank := by
  simp [conv, Shape.rank_eq_length]

/-- The axes of the output of `conv` at each position: those of the input. -/
def ConvDims.output {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) :
    Fin (r + 2) ≃
      Fin (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).rank :=
  dims.input.trans (finCongr (rank_conv s t dims stride padLow padHigh inputDilation dilation
    batchGroups).symm)

/-- The extent of the output of `conv` at a position is `convExtent` there. -/
@[simp] theorem extent_conv_output {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat)
    (p : Fin (r + 2)) :
    (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).extent
        (dims.output stride padLow padHigh inputDilation dilation batchGroups p) =
      convExtent s t dims stride padLow padHigh inputDilation dilation batchGroups p := by
  simp only [ConvDims.output, conv, extent, List.getElem_ofFn]
  exact congrArg (convExtent s t dims stride padLow padHigh inputDilation dilation batchGroups)
    (dims.input.symm_apply_apply p)

/-- Without batch groups, the batch axis of the output of `conv` has the extent of the input's. -/
theorem extent_conv_batch {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) :
    (s.conv t dims stride padLow padHigh inputDilation dilation 1).extent
        (dims.output stride padLow padHigh inputDilation dilation 1 ⟨0, by omega⟩) =
      s.extent (dims.input ⟨0, by omega⟩) := by
  rw [extent_conv_output, convExtent, Nat.div_one]

/-- Without batch groups, replacing the batch axis of the output shape of `conv` by one axis of the
input's batch extent gives that shape back. -/
theorem conv_eq_expandAxis_batch {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) :
    s.conv t dims stride padLow padHigh inputDilation dilation 1 =
      (s.conv t dims stride padLow padHigh inputDilation dilation 1).expandAxis
        (dims.output stride padLow padHigh inputDilation dilation 1 ⟨0, by omega⟩)
        [s.extent (dims.input ⟨0, by omega⟩)] := by
  rw [← extent_conv_batch dims stride padLow padHigh inputDilation dilation, expandAxis_extent]

/-- Replacing the batch axis of a convolution's output (without batch groups) by axes with as many
entries as the input's batch axis keeps its number of entries. -/
theorem size_conv_expandAxis {s t : Shape} {r : Nat} {dims : ConvDims s t r}
    {stride padLow padHigh inputDilation dilation : Fin r → Nat} {lead y : Shape}
    (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hy : y = (s.conv t dims stride padLow padHigh inputDilation dilation 1).expandAxis
      (dims.output stride padLow padHigh inputDilation dilation 1 ⟨0, by omega⟩) lead) :
    (s.conv t dims stride padLow padHigh inputDilation dilation 1).size = y.size :=
  (size_eq_of_expandAxis ((extent_conv_batch dims stride padLow padHigh inputDilation
    dilation).trans hlead) hy).symm

/-- `convTranspose` keeps the rank of its input. -/
@[simp] theorem rank_convTranspose (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).rank = s.rank := by
  simp [convTranspose, Shape.rank_eq_length]

/-- The axes of the output of `convTranspose` at each position: those of the input. -/
def ConvDims.transposeOutput {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    Fin (r + 2) ≃
      Fin (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).rank :=
  dims.input.trans (finCongr (rank_convTranspose s t dims stride padLow padHigh outputPadding
    dilation groups).symm)

/-- The extent of the output of `convTranspose` at a position is `convTransposeExtent` there. -/
@[simp] theorem extent_convTranspose_output {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (p : Fin (r + 2)) :
    (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).extent
        (dims.transposeOutput stride padLow padHigh outputPadding dilation groups p) =
      convTransposeExtent s t dims stride padLow padHigh outputPadding dilation groups p := by
  simp only [ConvDims.transposeOutput, convTranspose, extent, List.getElem_ofFn]
  exact congrArg (convTransposeExtent s t dims stride padLow padHigh outputPadding dilation groups)
    (dims.input.symm_apply_apply p)

/-- The window of the input gradient of `conv`: the output features of one group, then the
kernel's spatial axes. -/
def convBackwardInputWindow (t : Shape) {s : Shape} {r : Nat} (dims : ConvDims s t r)
    (groups : Nat) : Shape :=
  t.extent (dims.kernel ⟨0, by omega⟩) / groups ::
    List.ofFn fun i : Fin r => t.extent (dims.kernel i.succ.succ)

/-- The window of the weight gradient of `conv`: the batch, then the output's spatial axes. -/
def convBackwardWeightWindow (s t : Shape) {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) : Shape :=
  convExtent s t dims stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩ ::
    List.ofFn fun i : Fin r =>
      convExtent s t dims stride padLow padHigh (fun _ => 1) dilation 1 i.succ.succ

/-- The window of the input gradient of `convTranspose`: the kernel's output features, then its
spatial axes. -/
def convTransposeBackwardInputWindow (t : Shape) {s : Shape} {r : Nat} (dims : ConvDims s t r) :
    Shape :=
  t.extent (dims.kernel ⟨0, by omega⟩) ::
    List.ofFn fun i : Fin r => t.extent (dims.kernel i.succ.succ)

/-- The window of the weight gradient of `convTranspose`: the batch, then the input's spatial
axes. -/
def convTransposeBackwardWeightWindow (s : Shape) {t : Shape} {r : Nat} (dims : ConvDims s t r) :
    Shape :=
  s.extent (dims.input ⟨0, by omega⟩) ::
    List.ofFn fun i : Fin r => s.extent (dims.input i.succ.succ)

/-- Whether the window cell `k` of the input gradient of `conv` lands the input coordinate `c` on
an output position: along every spatial position `c + padLow - k * dilation` is a natural number
divisible by the nonzero stride. -/
def ConvDims.backwardInputAligned {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (k : (t.convBackwardInputWindow dims groups).Coord) : Bool :=
  decide (∀ j : Fin r, stride j ≠ 0 ∧
    Coord.index k (j + 1) * dilation j ≤ Coord.index c (dims.input j.succ.succ).val + padLow j ∧
    (Coord.index c (dims.input j.succ.succ).val + padLow j -
      Coord.index k (j + 1) * dilation j) % stride j = 0)

/-- The cotangent indices that the window cell `k` of the input gradient of `conv` reads for the
input coordinate `c`, by position of `dims.output`. -/
def ConvDims.backwardInputOutputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (k : (t.convBackwardInputWindow dims groups).Coord) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => Coord.index c (dims.input ⟨0, by omega⟩).val
  | ⟨1, _⟩ => Coord.index c (dims.input ⟨1, by omega⟩).val /
      t.extent (dims.kernel ⟨1, by omega⟩) * (t.extent (dims.kernel ⟨0, by omega⟩) / groups) +
        k.1.val
  | ⟨j + 2, h⟩ => (Coord.index c (dims.input ⟨j + 2, h⟩).val + padLow ⟨j, by omega⟩ -
      Coord.index k (j + 1) * dilation ⟨j, by omega⟩) / stride ⟨j, by omega⟩

/-- The kernel indices that the window cell `k` of the input gradient of `conv` reads for the
input coordinate `c`, by position of `dims.kernel`. -/
def ConvDims.backwardInputKernelIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (groups : Nat) (c : s.Coord) (k : (t.convBackwardInputWindow dims groups).Coord) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => Coord.index c (dims.input ⟨1, by omega⟩).val /
      t.extent (dims.kernel ⟨1, by omega⟩) * (t.extent (dims.kernel ⟨0, by omega⟩) / groups) +
        k.1.val
  | ⟨1, _⟩ => Coord.index c (dims.input ⟨1, by omega⟩).val % t.extent (dims.kernel ⟨1, by omega⟩)
  | ⟨j + 2, _⟩ => Coord.index k (j + 1)

/-- Whether no spatial index `o * stride + k * dilation` that the kernel coordinate `w` of the
weight gradient of `conv` reads at the output position `o` lies in the low padding. -/
def ConvDims.backwardWeightAbovePadLow {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) : Bool :=
  decide (∀ j : Fin r, padLow j ≤ Coord.index o (j + 1) * stride j +
    Coord.index w (dims.kernel j.succ.succ).val * dilation j)

/-- The input indices that the kernel coordinate `w` of the weight gradient of `conv` reads at the
output position `o`, by position of `dims.input`. -/
def ConvDims.backwardWeightInputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => o.1.val
  | ⟨1, _⟩ => Coord.index w (dims.kernel ⟨0, by omega⟩).val /
      (t.extent (dims.kernel ⟨0, by omega⟩) / groups) * t.extent (dims.kernel ⟨1, by omega⟩) +
        Coord.index w (dims.kernel ⟨1, by omega⟩).val
  | ⟨j + 2, h⟩ => Coord.index o (j + 1) * stride ⟨j, by omega⟩ +
      Coord.index w (dims.kernel ⟨j + 2, h⟩).val * dilation ⟨j, by omega⟩ - padLow ⟨j, by omega⟩

/-- The cotangent indices that the kernel coordinate `w` of the weight gradient of `conv` reads at
the output position `o`, by position of `dims.output`. -/
def ConvDims.backwardWeightOutputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => o.1.val
  | ⟨1, _⟩ => Coord.index w (dims.kernel ⟨0, by omega⟩).val
  | ⟨j + 2, _⟩ => Coord.index o (j + 1)

/-- Whether no spatial index `c * stride + k * dilation` that the window cell `k` of the input
gradient of `convTranspose` reads for the input coordinate `c` lies in the low padding. -/
def ConvDims.transposeBackwardInputAbovePadLow {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (c : s.Coord)
    (k : (t.convTransposeBackwardInputWindow dims).Coord) : Bool :=
  decide (∀ j : Fin r, padLow j ≤ Coord.index c (dims.input j.succ.succ).val * stride j +
    Coord.index k (j + 1) * dilation j)

/-- The kernel indices that the window cell `k` of the input gradient of `convTranspose` reads for
the input coordinate `c`, by position of `dims.kernel`. -/
def ConvDims.transposeBackwardInputKernelIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (groups : Nat) (c : s.Coord) (k : (t.convTransposeBackwardInputWindow dims).Coord) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => k.1.val
  | ⟨1, _⟩ => Coord.index c (dims.input ⟨1, by omega⟩).val /
      (s.extent (dims.input ⟨1, by omega⟩) / groups) * (t.extent (dims.kernel ⟨1, by omega⟩) /
        groups) + Coord.index c (dims.input ⟨1, by omega⟩).val %
          (s.extent (dims.input ⟨1, by omega⟩) / groups)
  | ⟨j + 2, _⟩ => Coord.index k (j + 1)

/-- The cotangent indices that the window cell `k` of the input gradient of `convTranspose` reads
for the input coordinate `c`, by position of `dims.transposeOutput`. -/
def ConvDims.transposeBackwardInputOutputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (k : (t.convTransposeBackwardInputWindow dims).Coord) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => Coord.index c (dims.input ⟨0, by omega⟩).val
  | ⟨1, _⟩ => Coord.index c (dims.input ⟨1, by omega⟩).val /
      (s.extent (dims.input ⟨1, by omega⟩) / groups) * t.extent (dims.kernel ⟨0, by omega⟩) +
        k.1.val
  | ⟨j + 2, h⟩ => Coord.index c (dims.input ⟨j + 2, h⟩).val * stride ⟨j, by omega⟩ +
      Coord.index k (j + 1) * dilation ⟨j, by omega⟩ - padLow ⟨j, by omega⟩

/-- Whether no spatial index `c * stride + k * dilation` that the kernel coordinate `w` of the
weight gradient of `convTranspose` reads at the input position `c` lies in the low padding. -/
def ConvDims.transposeBackwardWeightAbovePadLow {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (w : t.Coord)
    (c : (s.convTransposeBackwardWeightWindow dims).Coord) : Bool :=
  decide (∀ j : Fin r, padLow j ≤ Coord.index c (j + 1) * stride j +
    Coord.index w (dims.kernel j.succ.succ).val * dilation j)

/-- The input indices that the kernel coordinate `w` of the weight gradient of `convTranspose`
reads at the input position `c`, by position of `dims.input`. -/
def ConvDims.transposeBackwardWeightInputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (groups : Nat) (w : t.Coord) (c : (s.convTransposeBackwardWeightWindow dims).Coord) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => c.1.val
  | ⟨1, _⟩ => Coord.index w (dims.kernel ⟨1, by omega⟩).val /
      (t.extent (dims.kernel ⟨1, by omega⟩) / groups) * (s.extent (dims.input ⟨1, by omega⟩) /
        groups) + Coord.index w (dims.kernel ⟨1, by omega⟩).val %
          (t.extent (dims.kernel ⟨1, by omega⟩) / groups)
  | ⟨j + 2, _⟩ => Coord.index c (j + 1)

/-- The cotangent indices that the kernel coordinate `w` of the weight gradient of `convTranspose`
reads at the input position `c`, by position of `dims.transposeOutput`. -/
def ConvDims.transposeBackwardWeightOutputIndex {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (w : t.Coord)
    (c : (s.convTransposeBackwardWeightWindow dims).Coord) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => c.1.val
  | ⟨1, _⟩ => Coord.index w (dims.kernel ⟨1, by omega⟩).val /
      (t.extent (dims.kernel ⟨1, by omega⟩) / groups) * t.extent (dims.kernel ⟨0, by omega⟩) +
        Coord.index w (dims.kernel ⟨0, by omega⟩).val
  | ⟨j + 2, h⟩ => Coord.index c (j + 1) * stride ⟨j, by omega⟩ +
      Coord.index w (dims.kernel ⟨j + 2, h⟩).val * dilation ⟨j, by omega⟩ - padLow ⟨j, by omega⟩

/-! ## Bias and its gradient -/

/-- A bias of one entry per output feature broadcasts along the feature axis of the output of
`conv`. -/
theorem ConvDims.broadcastDims_conv {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) :
    BroadcastDims [t.extent (dims.kernel ⟨0, by omega⟩)]
      (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups)
      (fun _ => dims.output stride padLow padHigh inputDilation dilation batchGroups
        ⟨1, by omega⟩) where
  strictMono i j hij := by
    have hi := i.isLt
    have hj := j.isLt
    simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at hi hj
    exact absurd hij (by rw [Fin.lt_def]; omega)
  extent_eq i := Or.inr (by
    rw [extent_conv_output]
    match i with
    | ⟨0, _⟩ => rfl)

/-- A bias of one entry per output feature broadcasts along the feature axis of the output of
`convTranspose`. -/
theorem ConvDims.broadcastDims_convTranspose {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    BroadcastDims [groups * t.extent (dims.kernel ⟨0, by omega⟩)]
      (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups)
      (fun _ => dims.transposeOutput stride padLow padHigh outputPadding dilation groups
        ⟨1, by omega⟩) where
  strictMono i j hij := by
    have hi := i.isLt
    have hj := j.isLt
    simp only [Shape.rank_eq_length, List.length_cons, List.length_nil] at hi hj
    exact absurd hij (by rw [Fin.lt_def]; omega)
  extent_eq i := Or.inr (by
    rw [extent_convTranspose_output]
    match i with
    | ⟨0, _⟩ => rfl)

/-- Summing over every axis but `d` leaves the extent of `d`. -/
theorem reduce_otherAxes_singleton (s : Shape) (d : Fin s.rank) :
    s.reduce (s.otherAxes [d]) false = [s.extent d] := by
  have hother : s.otherAxes (s.otherAxes [d]) = s.sortedAxes [d] :=
    List.filter_congr fun a _ => by simp
  rw [reduce, Bool.cond_false, eraseAxes, hother, sortedAxes_singleton]
  rfl

/-- Summing the output of `conv` over every axis but its feature axis leaves one entry per output
feature, the shape of the bias. -/
theorem reduce_conv_otherAxes {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (batchGroups : Nat) :
    (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).reduce
        ((s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).otherAxes
          [dims.output stride padLow padHigh inputDilation dilation batchGroups ⟨1, by omega⟩])
        false =
      [t.extent (dims.kernel ⟨0, by omega⟩)] := by
  rw [reduce_otherAxes_singleton, extent_conv_output]
  rfl

/-- Summing the output of `convTranspose` over every axis but its feature axis leaves one entry per
output feature, the shape of the bias. -/
theorem reduce_convTranspose_otherAxes {s t : Shape} {r : Nat} (dims : ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).reduce
        ((s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).otherAxes
          [dims.transposeOutput stride padLow padHigh outputPadding dilation groups ⟨1, by omega⟩])
        false =
      [groups * t.extent (dims.kernel ⟨0, by omega⟩)] := by
  rw [reduce_otherAxes_singleton, extent_convTranspose_output]
  rfl

/-! ## Pooling windows -/

/-- The number of pooling windows along one axis of extent `n`: PyTorch's floor or ceil formula
for a window of `kernel` cells spaced by `dilation`, moved by `stride`, over the axis padded by
`padLow` and `padHigh`. In ceil mode the count is rounded up and a last window that starts in the
high padding is dropped, as in PyTorch. An empty axis or kernel, a zero stride, or padding beyond
half the kernel gives no window. We compared the count with torch 2.8.0's `max_pool1d` and
`avg_pool1d` at extents `1` to `8`, kernels `1` to `4`, strides `1` to `3`, equal padding `0` to
`2`, both modes and (for the maximum) dilations `1` to `3`. The counts agree at the 832 and 352
cases torch accepts, and at each of the others torch raises and this gives `0`. -/
def poolExtent (n kernel stride padLow padHigh dilation : Nat) (ceilMode : Bool) : Nat :=
  if n = 0 || kernel = 0 || stride = 0 || padLow > kernel / 2 || padHigh > kernel / 2 then 0
  else if ceilMode then
    if n + padLow + padHigh + (stride - 1) < dilatedKernelExtent kernel dilation then 0
    else
      let count :=
        (n + padLow + padHigh + (stride - 1) - dilatedKernelExtent kernel dilation) / stride + 1
      if (count - 1) * stride ≥ n + padLow then count - 1 else count
  else slidingWindowOutDimDilated n kernel stride dilation padLow padHigh

/-- The output shape of `pool` over the axes `dims`, which keeps every other axis. -/
def pool (s : Shape) (dims : List (Fin s.rank)) (kernel stride padLow padHigh dilation : List Nat)
    (ceilMode : Bool) : Shape :=
  List.ofFn fun a =>
    poolExtent (s.extent a) (valueAt dims kernel 1 a) (valueAt dims stride 1 a)
      (valueAt dims padLow 0 a) (valueAt dims padHigh 0 a) (valueAt dims dilation 1 a) ceilMode

/-- The window of `pool`: the kernel extent along each listed axis and one elsewhere. -/
def poolWindow (s : Shape) (dims : List (Fin s.rank)) (kernel : List Nat) : Shape :=
  List.ofFn fun a : Fin s.rank => valueAt dims kernel 1 a

/-- The position along axis `a` of the padded input of the window cell `k` of the output
coordinate `o`: `o * stride + k * dilation`. -/
def poolPosition {s : Shape} (dims : List (Fin s.rank)) (kernel stride padLow padHigh dilation :
    List Nat) (ceilMode : Bool)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) (a : Fin s.rank) : Nat :=
  Coord.index o a.val * valueAt dims stride 1 a + Coord.index k a.val * valueAt dims dilation 1 a

/-- An output position of `poolExtent` comes from a nonempty kernel, and its window starts inside
the padded input. -/
theorem kernel_ne_zero_and_mul_stride_lt_of_lt_poolExtent
    {n kernel stride padLow padHigh dilation out : Nat} {ceilMode : Bool}
    (h : out < poolExtent n kernel stride padLow padHigh dilation ceilMode) :
    kernel ≠ 0 ∧ out * stride < n + padLow + padHigh := by
  have valid : ¬(n = 0 || kernel = 0 || stride = 0 || padLow > kernel / 2 ||
      padHigh > kernel / 2) = true → kernel ≠ 0 ∧ stride ≠ 0 ∧
        1 ≤ dilatedKernelExtent kernel dilation := by
    intro hA
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hA
    refine ⟨hA.1.1.1.2, hA.1.1.2, ?_⟩
    unfold dilatedKernelExtent
    simp only [hA.1.1.1.2, ite_false]
    omega
  unfold poolExtent at h
  split_ifs at h with hA hC hB
  · omega
  · omega
  · obtain ⟨hk, hs, heff⟩ := valid hA
    refine ⟨hk, ?_⟩
    generalize hq : (n + padLow + padHigh + (stride - 1) - dilatedKernelExtent kernel dilation) /
      stride = q at h
    have h1 : q * stride ≤
        n + padLow + padHigh + (stride - 1) - dilatedKernelExtent kernel dilation :=
      hq ▸ Nat.div_mul_le_self _ _
    dsimp only at h
    simp only [Nat.add_sub_cancel] at h
    split_ifs at h with hD
    · have h2 := Nat.mul_le_mul_right stride (show out + 1 ≤ q by omega)
      rw [Nat.add_mul, one_mul] at h2
      omega
    · have h2 := Nat.mul_le_mul_right stride (show out ≤ q by omega)
      omega
  · obtain ⟨hk, hs, heff⟩ := valid hA
    refine ⟨hk, ?_⟩
    unfold slidingWindowOutDimDilated at h
    dsimp only at h
    split_ifs at h with hB
    · omega
    · have h1 := Nat.div_mul_le_self
        (n + padLow + padHigh - dilatedKernelExtent kernel dilation) stride
      have h2 := Nat.mul_le_mul_right stride (show out ≤
        (n + padLow + padHigh - dilatedKernelExtent kernel dilation) / stride by omega)
      simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hB
      omega

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s t : Shape}

/-! ## Convolution -/

/-- The input coordinate under the window cell `k` of the output coordinate `o` of `conv`, or
`none` at a padding or input-dilation hole. -/
def convSource {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) : Option s.Coord :=
  if h : ∀ p, dims.inputAligned stride padLow padHigh inputDilation dilation batchGroups o k p =
        true ∧
      dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p <
        s.extent (dims.input p) then
    some (Shape.Coord.ofPositions dims.input fun p =>
      ⟨dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p,
        (h p).2⟩)
  else none

/-- The input entry under the window cell `k` of the output coordinate `o`, or zero at a padding
or input-dilation hole. -/
def convInput [Zero α] (x : Tensor α s) {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) : α :=
  if h : ∀ p, dims.inputAligned stride padLow padHigh inputDilation dilation batchGroups o k p =
        true ∧
      dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p <
        s.extent (dims.input p) then
    x (Shape.Coord.ofPositions dims.input fun p =>
      ⟨dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p,
        (h p).2⟩)
  else 0

/-- Convolve `x` with the kernel `w` (XLA's `ConvGeneralDilated`): the output entry at `o` adds,
from zero, over the kernel's input features and then its window in row-major order, the input
entry under each window cell times the kernel entry. A padding or input-dilation hole contributes
zero times the kernel entry. The output feature `f` reads the input features of its group
`f / (O / groups)` and the batch of its batch group `f / (O / batchGroups)`. `Op.check` accepts
an `Op.conv` only at positive `groups` dividing `O` and an input with `groups` times the kernel's
input features. -/
def conv [Add α] [Mul α] [Zero α] (x : Tensor α s) (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh : Fin r → Nat)
    (inputDilation dilation : Fin r → Nat := fun _ => 1) (groups batchGroups : Nat := 1) :
    Tensor α (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups) :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size (t.convWindow dims))).foldl
      (fun acc i =>
        acc + convInput x dims stride padLow padHigh inputDilation dilation groups batchGroups o
            (Shape.Coord.unlinearize i) *
          w (dims.kernelCoord (dims.outFeature o) (Shape.Coord.unlinearize i)))
      0

/-- An entry of `conv` adds the products of the input entries under the window and the kernel
entries to zero, in row-major order of the kernel's input features and window. -/
@[simp] theorem conv_apply [Add α] [Mul α] [Zero α] (x : Tensor α s) (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh inputDilation dilation : Fin r → Nat)
    (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord) :
    conv x w dims stride padLow padHigh inputDilation dilation groups batchGroups o =
      (List.finRange (Shape.size (t.convWindow dims))).foldl
        (fun acc i =>
          acc + convInput x dims stride padLow padHigh inputDilation dilation groups batchGroups o
              (Shape.Coord.unlinearize i) *
            w (dims.kernelCoord (dims.outFeature o) (Shape.Coord.unlinearize i)))
        0 :=
  Rep.get_ofFn _ o

/-! ## Transposed convolution -/

/-- The kernel input feature that `convTranspose` reads for the output feature `f < groups * O`
and the window feature `j < I / groups`: `j` within the group `f / O`, below `I`. -/
theorem convTranspose_kernelIn_lt {groups O I f j : Nat} (hf : f < groups * O)
    (hj : j < I / groups) : f / O * (I / groups) + j < I := by
  have hO := Nat.pos_of_lt_mul_left hf
  have hg : f / O < groups := (Nat.div_lt_iff_lt_mul hO).mpr hf
  have hle : (f / O + 1) * (I / groups) ≤ groups * (I / groups) := Nat.mul_le_mul_right _ hg
  have hdiv : groups * (I / groups) ≤ I := Nat.mul_div_le I groups
  rw [Nat.succ_mul] at hle
  omega

/-- The kernel coordinate that `convTranspose` reads at the output coordinate `o` and the window
coordinate `k`: the output feature within its group, the input feature of the group, and the
window cell. -/
def convTransposeKernel {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r)
    {stride padLow padHigh outputPadding dilation : Fin r → Nat} {groups : Nat}
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : t.Coord :=
  have hf : Shape.Coord.index o (dims.input ⟨1, by omega⟩).val <
      groups * t.extent (dims.kernel ⟨0, by omega⟩) := Shape.index_convTranspose_lt dims o _
  Shape.Coord.ofPositions dims.kernel fun
    | ⟨0, _⟩ => ⟨Shape.Coord.index o (dims.input ⟨1, by omega⟩).val %
        t.extent (dims.kernel ⟨0, by omega⟩),
        Nat.mod_lt _ (Nat.pos_of_lt_mul_left hf)⟩
    | ⟨1, _⟩ => ⟨Shape.Coord.index o (dims.input ⟨1, by omega⟩).val /
          t.extent (dims.kernel ⟨0, by omega⟩) * (t.extent (dims.kernel ⟨1, by omega⟩) / groups) +
        k.1.val, convTranspose_kernelIn_lt hf k.1.isLt⟩
    | ⟨q + 2, _⟩ => ⟨Shape.Coord.index k (q + 1),
        Shape.Coord.index_lt_ofFn (fun i : Fin r => t.extent (dims.kernel i.succ.succ)) k.2
          ⟨q, by omega⟩⟩

/-- The index along the input axis at position `p` that `convTranspose` reads, before the stride
division along a spatial axis: the batch, the input feature `g * (C / groups) + j` of the group
`g` of the output feature, or `o + padLow - k * dilation`. -/
def convTransposeOffset {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => Shape.Coord.index o (dims.input ⟨0, by omega⟩).val
  | ⟨1, _⟩ => Shape.Coord.index o (dims.input ⟨1, by omega⟩).val /
      t.extent (dims.kernel ⟨0, by omega⟩) * (s.extent (dims.input ⟨1, by omega⟩) / groups) +
        k.1.val
  | ⟨i + 2, h⟩ => Shape.Coord.index o (dims.input ⟨i + 2, h⟩).val + padLow ⟨i, by omega⟩ -
      Shape.Coord.index k (i + 1) * dilation ⟨i, by omega⟩

/-- Whether the window cell at position `p` of a transposed convolution lands on an input entry:
always along the batch and feature axes, and along a spatial axis when `k * dilation ≤ o + padLow`
and the stride divides the difference. -/
def convTransposeAligned {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : Fin (r + 2) → Bool
  | ⟨0, _⟩ => true
  | ⟨1, _⟩ => true
  | ⟨i + 2, h⟩ =>
      decide (Shape.Coord.index k (i + 1) * dilation ⟨i, by omega⟩ ≤
          Shape.Coord.index o (dims.input ⟨i + 2, h⟩).val + padLow ⟨i, by omega⟩) &&
        decide (convTransposeOffset dims stride padLow padHigh outputPadding dilation groups o k
          ⟨i + 2, h⟩ % stride ⟨i, by omega⟩ = 0)

/-- The index along the input axis at position `p` that `convTranspose` reads. -/
def convTransposeIndex {s t : Shape} {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : Fin (r + 2) → Nat
  | ⟨0, h⟩ => convTransposeOffset dims stride padLow padHigh outputPadding dilation groups o k
      ⟨0, h⟩
  | ⟨1, h⟩ => convTransposeOffset dims stride padLow padHigh outputPadding dilation groups o k
      ⟨1, h⟩
  | ⟨i + 2, h⟩ => convTransposeOffset dims stride padLow padHigh outputPadding dilation groups o k
      ⟨i + 2, h⟩ / stride ⟨i, by omega⟩

/-- The input coordinate that lands on the output coordinate `o` of a transposed convolution
through the window cell `k`, or `none` where no input entry lands. -/
def convTransposeSource {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : Option s.Coord :=
  if h : ∀ p, convTransposeAligned dims stride padLow padHigh outputPadding dilation groups o k p =
        true ∧
      convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p <
        s.extent (dims.input p) then
    some (Shape.Coord.ofPositions dims.input fun p =>
      ⟨convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p, (h p).2⟩)
  else none

/-- The input entry that the window cell `k` of the output coordinate `o` of a transposed
convolution reads, or zero where no input entry lands. -/
def convTransposeInput [Zero α] (x : Tensor α s) {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) : α :=
  if h : ∀ p, convTransposeAligned dims stride padLow padHigh outputPadding dilation groups o k p =
        true ∧
      convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p <
        s.extent (dims.input p) then
    x (Shape.Coord.ofPositions dims.input fun p =>
      ⟨convTransposeIndex dims stride padLow padHigh outputPadding dilation groups o k p, (h p).2⟩)
  else 0

/-- The transposed convolution of `x` with the kernel `w`, PyTorch's `conv_transpose` with
asymmetric padding: the output entry at `o` adds, from zero, over the input features of the
group of its output feature and then the kernel window in row-major order, the input entry that
lands on `o` through each window cell times the kernel entry. Where no input entry lands, the term
is zero times the kernel entry. The kernel's output feature axis holds `O / groups` features.
`Op.check` accepts an `Op.convTranspose` only at positive `groups` dividing the input's feature
extent, which must also be the kernel's input-feature extent. -/
def convTranspose [Add α] [Mul α] [Zero α] (x : Tensor α s) (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh : Fin r → Nat)
    (outputPadding : Fin r → Nat := fun _ => 0) (dilation : Fin r → Nat := fun _ => 1)
    (groups : Nat := 1) :
    Tensor α (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups) :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size (s.convTransposeWindow t dims groups))).foldl
      (fun acc i =>
        acc + convTransposeInput x dims stride padLow padHigh outputPadding dilation groups o
            (Shape.Coord.unlinearize i) *
          w (convTransposeKernel dims o (Shape.Coord.unlinearize i)))
      0

/-- An entry of `convTranspose` adds the products of the input entries landing on it and the
kernel entries to zero, in row-major order of the group's input features and the window. -/
@[simp] theorem convTranspose_apply [Add α] [Mul α] [Zero α] (x : Tensor α s) (w : Tensor α t)
    {r : Nat} (dims : Shape.ConvDims s t r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord) :
    convTranspose x w dims stride padLow padHigh outputPadding dilation groups o =
      (List.finRange (Shape.size (s.convTransposeWindow t dims groups))).foldl
        (fun acc i =>
          acc + convTransposeInput x dims stride padLow padHigh outputPadding dilation groups o
              (Shape.Coord.unlinearize i) *
            w (convTransposeKernel dims o (Shape.Coord.unlinearize i)))
        0 :=
  Rep.get_ofFn _ o

/-! ## Convolution gradients -/

/-- The entry of `x` whose index along the axis `e p` is `index p` for every position `p`, or zero
when an index is out of range. -/
def getOrZero [Zero α] (x : Tensor α s) {n : Nat} (e : Fin n ≃ Fin s.rank) (index : Fin n → Nat) :
    α :=
  if h : ∀ p, index p < s.extent (e p) then
    x (Shape.Coord.ofPositions e fun p => ⟨index p, h p⟩)
  else 0

/-- The gradient of `conv` without input dilation or batch groups with respect to its input, for
the cotangent `dy`. The entry at `c` folds from zero over the output features of the group of its
feature and then the kernel window in row-major order. Where the window cell `k` lands `c` on the
output position `(c + padLow - k * dilation) / stride` it adds the entry of `dy` there, or zero
past the output, times the kernel entry. Elsewhere it adds zero. -/
def convBackwardInput [Add α] [Mul α] [Zero α] (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) : Tensor α s :=
  Rep.ofFn fun c =>
    (List.finRange (Shape.size (t.convBackwardInputWindow dims groups))).foldl
      (fun acc i =>
        let k : (t.convBackwardInputWindow dims groups).Coord := Shape.Coord.unlinearize i
        let feature := Shape.Coord.index c (dims.input ⟨1, by omega⟩).val
        let inner := t.extent (dims.kernel ⟨1, by omega⟩)
        let out := feature / inner * (t.extent (dims.kernel ⟨0, by omega⟩) / groups) + k.1.val
        if ∀ j : Fin r, stride j ≠ 0 ∧
            Shape.Coord.index k (j + 1) * dilation j ≤
              Shape.Coord.index c (dims.input j.succ.succ).val + padLow j ∧
            (Shape.Coord.index c (dims.input j.succ.succ).val + padLow j -
              Shape.Coord.index k (j + 1) * dilation j) % stride j = 0 then
          acc + getOrZero dy (dims.output stride padLow padHigh (fun _ => 1) dilation 1)
              (fun
                | ⟨0, _⟩ => Shape.Coord.index c (dims.input ⟨0, by omega⟩).val
                | ⟨1, _⟩ => out
                | ⟨j + 2, h⟩ => (Shape.Coord.index c (dims.input ⟨j + 2, h⟩).val +
                    padLow ⟨j, by omega⟩ - Shape.Coord.index k (j + 1) * dilation ⟨j, by omega⟩) /
                      stride ⟨j, by omega⟩) *
            getOrZero w dims.kernel
              (fun
                | ⟨0, _⟩ => out
                | ⟨1, _⟩ => feature % inner
                | ⟨j + 2, _⟩ => Shape.Coord.index k (j + 1))
        else acc + 0)
      0

/-- The gradient of `conv` without input dilation or batch groups with respect to its kernel, for
the cotangent `dy`. The entry at the kernel coordinate `(f, j, k)` adds to zero, over the batch and
then the output positions `o` in row-major order, the input entry under the window cell `k` of `o`
(zero at a hole) times the entry of `dy` at `o` and the output feature `f`. -/
def convBackwardWeight [Add α] [Mul α] [Zero α] (x : Tensor α s) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) : Tensor α t :=
  Rep.ofFn fun w =>
    (List.finRange (Shape.size (s.convBackwardWeightWindow t dims stride padLow padHigh
        dilation))).foldl
      (fun acc i =>
        let o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord :=
          Shape.Coord.unlinearize i
        let out := Shape.Coord.index w (dims.kernel ⟨0, by omega⟩).val
        acc +
          (if ∀ j : Fin r, padLow j ≤ Shape.Coord.index o (j + 1) * stride j +
              Shape.Coord.index w (dims.kernel j.succ.succ).val * dilation j then
            getOrZero x dims.input
              (fun
                | ⟨0, _⟩ => o.1.val
                | ⟨1, _⟩ => out / (t.extent (dims.kernel ⟨0, by omega⟩) / groups) *
                    t.extent (dims.kernel ⟨1, by omega⟩) +
                      Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val
                | ⟨j + 2, h⟩ => Shape.Coord.index o (j + 1) * stride ⟨j, by omega⟩ +
                    Shape.Coord.index w (dims.kernel ⟨j + 2, h⟩).val * dilation ⟨j, by omega⟩ -
                      padLow ⟨j, by omega⟩)
          else 0) *
            getOrZero dy (dims.output stride padLow padHigh (fun _ => 1) dilation 1)
              (fun
                | ⟨0, _⟩ => o.1.val
                | ⟨1, _⟩ => out
                | ⟨j + 2, _⟩ => Shape.Coord.index o (j + 1)))
      0

/-- The gradient of `convTranspose` with respect to its input, for the cotangent `dy`. The entry
at `c` folds from zero over the kernel's output features and then its window in row-major order.
Where the window cell `k` of `c` is not in the low padding it adds the kernel entry times the entry
of `dy` at `c * stride + k * dilation - padLow` (zero past the output). Elsewhere it skips. -/
def convTransposeBackwardInput [Add α] [Mul α] [Zero α] (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat)
    (dy : Tensor α (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups)) :
    Tensor α s :=
  Rep.ofFn fun c =>
    (List.finRange (Shape.size (t.convTransposeBackwardInputWindow dims))).foldl
      (fun acc i =>
        let k : (t.convTransposeBackwardInputWindow dims).Coord := Shape.Coord.unlinearize i
        let feature := Shape.Coord.index c (dims.input ⟨1, by omega⟩).val
        let size := s.extent (dims.input ⟨1, by omega⟩) / groups
        if ∀ j : Fin r, padLow j ≤ Shape.Coord.index c (dims.input j.succ.succ).val * stride j +
            Shape.Coord.index k (j + 1) * dilation j then
          acc + getOrZero w dims.kernel
              (fun
                | ⟨0, _⟩ => k.1.val
                | ⟨1, _⟩ => feature / size * (t.extent (dims.kernel ⟨1, by omega⟩) / groups) +
                    feature % size
                | ⟨j + 2, _⟩ => Shape.Coord.index k (j + 1)) *
            getOrZero dy (dims.transposeOutput stride padLow padHigh outputPadding dilation groups)
              (fun
                | ⟨0, _⟩ => Shape.Coord.index c (dims.input ⟨0, by omega⟩).val
                | ⟨1, _⟩ => feature / size * t.extent (dims.kernel ⟨0, by omega⟩) + k.1.val
                | ⟨j + 2, h⟩ => Shape.Coord.index c (dims.input ⟨j + 2, h⟩).val *
                    stride ⟨j, by omega⟩ + Shape.Coord.index k (j + 1) * dilation ⟨j, by omega⟩ -
                      padLow ⟨j, by omega⟩)
        else acc)
      0

/-- The gradient of `convTranspose` with respect to its kernel, for the cotangent `dy`. The entry at
the kernel coordinate `(f, j, k)` folds from zero over the batch and then the input positions `c`
in row-major order. Where the window cell `k` of `c` is not in the low padding it adds the input
entry at `c` times the entry of `dy` at `c * stride + k * dilation - padLow` (zero past the output).
Elsewhere it skips. -/
def convTransposeBackwardWeight [Add α] [Mul α] [Zero α] (x : Tensor α s) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat)
    (dy : Tensor α (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups)) :
    Tensor α t :=
  Rep.ofFn fun w =>
    (List.finRange (Shape.size (s.convTransposeBackwardWeightWindow dims))).foldl
      (fun acc i =>
        let c : (s.convTransposeBackwardWeightWindow dims).Coord := Shape.Coord.unlinearize i
        let inner := Shape.Coord.index w (dims.kernel ⟨1, by omega⟩).val
        let size := t.extent (dims.kernel ⟨1, by omega⟩) / groups
        if ∀ j : Fin r, padLow j ≤ Shape.Coord.index c (j + 1) * stride j +
            Shape.Coord.index w (dims.kernel j.succ.succ).val * dilation j then
          acc + getOrZero x dims.input
              (fun
                | ⟨0, _⟩ => c.1.val
                | ⟨1, _⟩ => inner / size * (s.extent (dims.input ⟨1, by omega⟩) / groups) +
                    inner % size
                | ⟨j + 2, _⟩ => Shape.Coord.index c (j + 1)) *
            getOrZero dy (dims.transposeOutput stride padLow padHigh outputPadding dilation groups)
              (fun
                | ⟨0, _⟩ => c.1.val
                | ⟨1, _⟩ => inner / size * t.extent (dims.kernel ⟨0, by omega⟩) +
                    Shape.Coord.index w (dims.kernel ⟨0, by omega⟩).val
                | ⟨j + 2, h⟩ => Shape.Coord.index c (j + 1) * stride ⟨j, by omega⟩ +
                    Shape.Coord.index w (dims.kernel ⟨j + 2, h⟩).val * dilation ⟨j, by omega⟩ -
                      padLow ⟨j, by omega⟩)
        else acc)
      0

/-! ## Pooling -/

/-- The reduction of `pool`. `.avg countIncludePad` divides the sum by the number of window cells
in the padded input, or with `countIncludePad := false` in the input itself. -/
inductive PoolOp
  /-- Sum, padding adding zero. -/
  | sum
  /-- Maximum, skipping padding. -/
  | max
  /-- Minimum, skipping padding. -/
  | min
  /-- Average, PyTorch's `count_include_pad`. -/
  | avg (countIncludePad : Bool)
  deriving DecidableEq, Repr

section Pool

variable [Context α] {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- The input coordinate under the window cell `k` of the output coordinate `o`, or `none` in the
padding. -/
def poolSource (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) : Option s.Coord :=
  if h : ∀ a, Shape.valueAt dims padLow 0 a ≤ Shape.poolPosition dims kernel stride padLow padHigh
        dilation ceilMode o k a ∧
      Shape.poolPosition dims kernel stride padLow padHigh dilation ceilMode o k a -
        Shape.valueAt dims padLow 0 a < s.extent a then
    some (Shape.Coord.ofFun fun a =>
      ⟨Shape.poolPosition dims kernel stride padLow padHigh dilation ceilMode o k a -
        Shape.valueAt dims padLow 0 a, (h a).2⟩)
  else none

/-- The input entry under the window cell `k` of the output coordinate `o`, or zero in the
padding. -/
def poolInput (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) : α :=
  match poolSource o k with
  | none => 0
  | some c => x c

/-- The sum over the window of the output coordinate `o`, from zero in row-major order, with
padding adding zero. -/
def poolSum (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : α :=
  (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
    (fun acc i => acc + poolInput x o (Shape.Coord.unlinearize i)) 0

/-- The number of cells of the window of `o` that an average divides by: those in the padded input
when `countIncludePad`, those in the input otherwise. -/
def poolCount (countIncludePad : Bool)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : Nat :=
  (List.finRange (Shape.size (s.poolWindow dims kernel))).countP fun i =>
    if countIncludePad then
      decide (∀ a, Shape.poolPosition dims kernel stride padLow padHigh dilation ceilMode o
        (Shape.Coord.unlinearize i) a <
          s.extent a + Shape.valueAt dims padLow 0 a + Shape.valueAt dims padHigh 0 a)
    else (poolSource o (Shape.Coord.unlinearize i)).isSome

/-- The first cell of every window of a pool lies in the padded input, so `poolCount true` is
positive. -/
theorem poolCount_true_pos
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    0 < poolCount true o := by
  have hout : ∀ a : Fin s.rank, Shape.valueAt dims kernel 1 a ≠ 0 ∧
      Shape.Coord.index o a.val * Shape.valueAt dims stride 1 a <
        s.extent a + Shape.valueAt dims padLow 0 a + Shape.valueAt dims padHigh 0 a :=
    fun a => Shape.kernel_ne_zero_and_mul_stride_lt_of_lt_poolExtent
      (Shape.Coord.index_lt_ofFn _ o a)
  have hpos : ∀ a, 0 < (s.poolWindow dims kernel).extent a := by
    intro a
    have ha : a.val < s.rank := by simpa [Shape.poolWindow, Shape.rank_eq_length] using a.isLt
    simpa [Shape.poolWindow, Shape.extent, Nat.pos_iff_ne_zero] using (hout ⟨a.val, ha⟩).1
  let first : (s.poolWindow dims kernel).Coord := Shape.Coord.ofFun fun a => ⟨0, hpos a⟩
  unfold poolCount
  simp only [ite_true]
  rw [List.countP_pos_iff]
  refine ⟨Shape.Coord.linearize first, List.mem_finRange _, ?_⟩
  rw [Shape.Coord.unlinearize_linearize, decide_eq_true_eq]
  intro a
  have hidx : Shape.Coord.index first a.val = 0 := by
    simp [first, Shape.Coord.index_ofFun]
  rw [Shape.poolPosition, hidx, zero_mul, add_zero]
  exact (hout a).2

/-- Whether `v` replaces `b` as the selected entry: when it is larger for a maximum or smaller for
a minimum, and always when it is a NaN (an entry not equal to itself). As in the IEEE 754-2019
`maximum` and `minimum`, `-0` is below `0`, with the signs read through `Context.copysign 1`. -/
def poolWins (largest : Bool) (v b : α) : Bool :=
  if largest then Context.gtBool v b || (!(Context.gtBool b v) &&
    (!(v == v) || (b == b && Context.gtBool (Context.copysign 1 v) (Context.copysign 1 b))))
  else Context.gtBool b v || (!(Context.gtBool v b) &&
    (!(v == v) || (b == b && Context.gtBool (Context.copysign 1 b) (Context.copysign 1 v))))

/-- The input coordinate a maximum (`largest`) or minimum pool selects in each window, skipping
padding: the first extremum in row-major window order, with `-0` below `0`, where a NaN wins over
numbers and a later NaN replaces it. A window that holds no input entry selects nothing. -/
def poolIndices (largest : Bool) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor (Option s.Coord) (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
      (fun best i =>
        match poolSource o (Shape.Coord.unlinearize i), best with
        | none, _ => best
        | some c, none => some c
        | some c, some b => if poolWins largest (x c) (x b) then some c else best)
      none

/-- Pool `x` over the axes `dims` with windows of `kernel` cells spaced by `dilation`, moved by
`stride` over the input padded by `padLow` and `padHigh`, in PyTorch's floor or ceil mode. A sum
adds the window to zero in row-major order, with padding adding zero, and an average divides that
sum once by the count `poolCount` (`Context.divNat`). A maximum or minimum reads the entry
`poolIndices` selects, or gives zero for a window of padding only. -/
def pool (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  match op with
  | .sum => Rep.ofFn fun o => poolSum x o
  | .avg countIncludePad => Rep.ofFn fun o =>
      Context.divNat (poolSum x o) (poolCount countIncludePad o)
  | .max => Rep.ofFn fun o =>
      match poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => x c
  | .min => Rep.ofFn fun o =>
      match poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => x c

variable (x : Tensor α s) (dims kernel stride padLow padHigh dilation ceilMode)

/-- An entry of a sum pool adds its window to zero in row-major order. -/
@[simp] theorem pool_sum_apply
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    pool .sum x dims kernel stride padLow padHigh dilation ceilMode o =
      (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
        (fun acc i => acc + poolInput x o (Shape.Coord.unlinearize i)) 0 :=
  Rep.get_ofFn _ o

/-- An entry of an average pool divides the sum of its window once by the count of its cells. -/
@[simp] theorem pool_avg_apply (countIncludePad : Bool)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    pool (.avg countIncludePad) x dims kernel stride padLow padHigh dilation ceilMode o =
      Context.divNat ((List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
          (fun acc i => acc + poolInput x o (Shape.Coord.unlinearize i)) 0)
        (poolCount countIncludePad o) :=
  Rep.get_ofFn _ o

/-- An entry of a maximum pool reads the input at the coordinate `poolIndices` selects. -/
@[simp] theorem pool_max_apply
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    pool .max x dims kernel stride padLow padHigh dilation ceilMode o =
      match poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => x c :=
  Rep.get_ofFn _ o

/-- An entry of a minimum pool reads the input at the coordinate `poolIndices` selects. -/
@[simp] theorem pool_min_apply
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    pool .min x dims kernel stride padLow padHigh dilation ceilMode o =
      match poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => x c :=
  Rep.get_ofFn _ o

/-- The directional derivative of `pool op` at `x` along `dx`. Sums and averages are linear, so it
pools `dx`, and a maximum or minimum pool reads `dx` at the coordinate `poolIndices` selects in
`x`, which at a tie is the derivative of the selected branch and not of the pool. -/
def poolJVP (op : PoolOp) (x dx : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  match op with
  | .sum => pool .sum dx dims kernel stride padLow padHigh dilation ceilMode
  | .avg countIncludePad => pool (.avg countIncludePad) dx dims kernel stride padLow padHigh
      dilation ceilMode
  | .max => Rep.ofFn fun o =>
      match poolIndices true x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => dx c
  | .min => Rep.ofFn fun o =>
      match poolIndices false x dims kernel stride padLow padHigh dilation ceilMode o with
      | none => 0
      | some c => dx c

/-- The gradient of `pool op` with respect to its input, for the cotangent `dy`. A maximum or
minimum pool adds each output entry of `dy` into the coordinate it selected, in row-major order
of the output. A sum or average adds each entry of `dy`, divided by the count for an average, into
every input coordinate of its window, in row-major order of the output and then the window. -/
def poolBackward (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) : Tensor α s :=
  match op with
  | .max => Rep.ofFn fun c =>
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          if poolIndices true x dims kernel stride padLow padHigh dilation ceilMode
              (Shape.Coord.unlinearize i) = some c then acc + dy (Shape.Coord.unlinearize i)
          else acc)
        0
  | .min => Rep.ofFn fun c =>
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          if poolIndices false x dims kernel stride padLow padHigh dilation ceilMode
              (Shape.Coord.unlinearize i) = some c then acc + dy (Shape.Coord.unlinearize i)
          else acc)
        0
  | .sum => Rep.ofFn fun c =>
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
            (fun acc j =>
              if poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j) =
                  some c then acc + dy (Shape.Coord.unlinearize i)
              else acc)
            acc)
        0
  | .avg countIncludePad => Rep.ofFn fun c =>
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
            (fun acc j =>
              if poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j) =
                  some c then
                acc + Context.divNat (dy (Shape.Coord.unlinearize i))
                  (poolCount countIncludePad (Shape.Coord.unlinearize i))
              else acc)
            acc)
        0

end Pool

/-! ## Smooth maximum pooling -/

section SmoothMaxPool

variable [Context α] {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- The pivot of a smooth maximum pool at the output coordinate `o`: the maximum of the window for
`beta > 0` and its minimum otherwise, folded from the first window cell over every cell, with
padding read as zero. -/
def smoothMaxPoolPivot (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : α :=
  (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
    (fun pivot i =>
      if beta > 0 then Max.max pivot (poolInput x o (Shape.Coord.unlinearize i))
      else Min.min pivot (poolInput x o (Shape.Coord.unlinearize i)))
    (if h : 0 < Shape.size (s.poolWindow dims kernel) then
      poolInput x o (Shape.Coord.unlinearize ⟨0, h⟩)
    else 0)

/-- The shift of a smooth maximum pool at `o`: its pivot, or zero when the pivot is infinite,
where `∞ - ∞` would be NaN. -/
def smoothMaxPoolShift (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : α :=
  if Context.isInf (smoothMaxPoolPivot beta x o) then 0 else smoothMaxPoolPivot beta x o

/-- The sum over the window of `o` of `exp (beta * (v - shift))`, from zero in row-major order. -/
def smoothMaxPoolSum (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (shift : α) : α :=
  (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
    (fun acc i => acc + MathFunctions.exp (beta *
      (poolInput x o (Shape.Coord.unlinearize i) - shift)))
    0

/-- Smooth maximum pooling, the log-sum-exp `beta⁻¹ log Σ exp (beta * v)` over each window of
`pool`, evaluated as `shift + log (Σ exp (beta * (v - shift))) / beta` with the shift of
`smoothMaxPoolShift`, the pivot of `smoothMaxPoolPivot` or zero when that is infinite. Padding is
read as zero. -/
def smoothMaxPool (beta : α) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  Rep.ofFn fun o =>
    let shift := smoothMaxPoolShift beta x o
    -- Dividing by `beta` rounds once, and at a subnormal `beta` the reciprocal `1 / beta`
    -- overflows, so a window whose sum is `1` would give `log 1 * (1 / beta) = 0 * ∞`, NaN.
    shift + MathFunctions.log (smoothMaxPoolSum beta x o shift) / beta

/-- The directional derivative of `smoothMaxPool` at `x` along `dx`: the sum over each window of
the softmax weight `exp (beta * (v - pivot)) / Σ` of each cell times the cell of `dx`. At an
infinite pivot, where no derivative exists, the weights are NaN. -/
def smoothMaxPoolJVP (beta : α) (x dx : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  Rep.ofFn fun o =>
    let pivot := smoothMaxPoolPivot beta x o
    let sum := smoothMaxPoolSum beta x o pivot
    (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
      (fun acc i => acc + (MathFunctions.exp (beta *
          (poolInput x o (Shape.Coord.unlinearize i) - pivot)) / sum) *
        poolInput dx o (Shape.Coord.unlinearize i))
      0

/-- The gradient of `smoothMaxPool` for the cotangent `dy`: each output entry of `dy`, times the
softmax weight of each cell of its window, is added into the input coordinate of that cell, in
row-major order of the output and then the window. At an infinite pivot, where no
derivative exists, the weights are NaN. -/
def smoothMaxPoolBackward (beta : α) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) : Tensor α s :=
  Rep.ofFn fun c =>
    (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
        ceilMode))).foldl
      (fun acc i =>
        let o := Shape.Coord.unlinearize i
        let pivot := smoothMaxPoolPivot beta x o
        let sum := smoothMaxPoolSum beta x o pivot
        (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
          (fun acc j =>
            if poolSource o (Shape.Coord.unlinearize j) = some c then
              acc + dy o * (MathFunctions.exp (beta *
                (poolInput x o (Shape.Coord.unlinearize j) - pivot)) / sum)
            else acc)
          acc)
      0

/-- An entry of a smooth maximum pool is its shift plus the logarithm of its sum of shifted
exponentials divided by `beta`. -/
@[simp] theorem smoothMaxPool_apply (beta : α) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPool beta x dims kernel stride padLow padHigh dilation ceilMode o =
      let shift := smoothMaxPoolShift beta x o
      shift + MathFunctions.log (smoothMaxPoolSum beta x o shift) / beta :=
  Rep.get_ofFn _ o

end SmoothMaxPool

/-! ## Adaptive pooling -/

/-- The start of bin `i` of `out` bins over an axis of extent `n`: `⌊i * n / out⌋`. -/
def adaptiveStart (n out i : Nat) : Nat :=
  i * n / out

/-- The end of bin `i` of `out` bins over an axis of extent `n`: `⌈(i + 1) * n / out⌉`. -/
def adaptiveEnd (n out i : Nat) : Nat :=
  ((i + 1) * n + out - 1) / out

/-- Every adaptive-pooling bin is nonempty when the input and output extents are positive.

Both hypotheses are needed: an empty input axis gives `start = end = 0`, and `out = 0` makes both
quotients zero under Lean's total division. -/
theorem adaptiveStart_lt_adaptiveEnd (n out i : Nat) (hn : n ≠ 0) (hout : out ≠ 0) :
    adaptiveStart n out i < adaptiveEnd n out i := by
  unfold adaptiveStart adaptiveEnd
  have hle : i * n / out * out ≤ i * n := Nat.div_mul_le_self _ _
  rw [Nat.lt_iff_add_one_le, Nat.le_div_iff_mul_le (Nat.pos_of_ne_zero hout), Nat.add_mul,
    Nat.one_mul, Nat.succ_mul]
  generalize i * n = m at *
  generalize m / out * out = q at *
  have hpos : 1 ≤ n := Nat.pos_of_ne_zero hn
  have hstep : m + out + 1 ≤ m + n + out := by
    rw [Nat.add_right_comm m n out]
    exact Nat.add_le_add_left hpos _
  exact le_trans (Nat.add_le_add_right hle out) (Nat.le_sub_of_add_le hstep)

/-- The output shape of `adaptivePool`: `outputSize` bins along each listed axis, every other axis
kept. -/
def _root_.TorchLean.Shape.adaptivePool (s : Shape) (dims : List (Fin s.rank))
    (outputSize : List Nat) :
    Shape :=
  List.ofFn fun a => Shape.valueAt dims outputSize (s.extent a) a

/-- The bin of the output coordinate `o` of an adaptive pool: its width along each axis. -/
def adaptiveBin {s : Shape} {dims : List (Fin s.rank)} {outputSize : List Nat}
    (o : (s.adaptivePool dims outputSize).Coord) : Shape :=
  List.ofFn fun a : Fin s.rank =>
    adaptiveEnd (s.extent a) (Shape.valueAt dims outputSize (s.extent a) a)
        (Shape.Coord.index o a.val) -
      adaptiveStart (s.extent a) (Shape.valueAt dims outputSize (s.extent a) a)
        (Shape.Coord.index o a.val)

/-- The input coordinate at the cell `k` of the bin of `o`, if it lies in the input. -/
def adaptiveSource {dims : List (Fin s.rank)} {outputSize : List Nat}
    (o : (s.adaptivePool dims outputSize).Coord) (k : (adaptiveBin o).Coord) : Option s.Coord :=
  if h : ∀ a, adaptiveStart (s.extent a) (Shape.valueAt dims outputSize (s.extent a) a)
        (Shape.Coord.index o a.val) + Shape.Coord.index k a.val < s.extent a then
    some (Shape.Coord.ofFun fun a =>
      ⟨adaptiveStart (s.extent a) (Shape.valueAt dims outputSize (s.extent a) a)
        (Shape.Coord.index o a.val) + Shape.Coord.index k a.val, h a⟩)
  else none

/-- The input entry at the cell `k` of the bin of `o`, or zero outside the input. -/
def adaptiveInput [Zero α] (x : Tensor α s) {dims : List (Fin s.rank)} {outputSize : List Nat}
    (o : (s.adaptivePool dims outputSize).Coord) (k : (adaptiveBin o).Coord) : α :=
  match adaptiveSource o k with
  | none => 0
  | some c => x c

/-- The input coordinate an adaptive maximum (`largest`) or minimum pool selects in each bin: the
first extremum in row-major bin order, with `-0` below `0`, a NaN winning over numbers and a later
NaN replacing it. -/
def adaptiveIndices [Context α] (largest : Bool) (x : Tensor α s) (dims : List (Fin s.rank))
    (outputSize : List Nat) : Tensor (Option s.Coord) (s.adaptivePool dims outputSize) :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size (adaptiveBin o))).foldl
      (fun best i =>
        match adaptiveSource o (Shape.Coord.unlinearize i), best with
        | none, _ => best
        | some c, none => some c
        | some c, some b => if poolWins largest (x c) (x b) then some c else best)
      none

/-- PyTorch's adaptive pooling over the axes `dims`: `outputSize` bins along each listed axis,
bin `i` covering `⌊i * n / out⌋` to `⌈(i + 1) * n / out⌉`. A sum or average folds the bin from
zero in row-major order (an average divides once by the number of its cells). A maximum or minimum
reads the entry `adaptiveIndices` selects, and is zero on an empty bin. Torch 2.8.0 also averages
`[0, 1, 2, 3, 4]` into `[0.5, 2, 3.5]`. No test runs this definition against torch, since the
importer lowers `adaptive_avg_pool` to slices, sums and divisions of its own. -/
def adaptivePool [Context α] (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (outputSize : List Nat) : Tensor α (s.adaptivePool dims outputSize) :=
  Rep.ofFn fun o =>
    match op with
    | .sum => (List.finRange (Shape.size (adaptiveBin o))).foldl
        (fun acc i => acc + adaptiveInput x o (Shape.Coord.unlinearize i)) 0
    | .avg _ => Context.divNat ((List.finRange (Shape.size (adaptiveBin o))).foldl
        (fun acc i => acc + adaptiveInput x o (Shape.Coord.unlinearize i)) 0)
          (Shape.size (adaptiveBin o))
    | .max =>
        match adaptiveIndices true x dims outputSize o with
        | none => 0
        | some c => x c
    | .min =>
        match adaptiveIndices false x dims outputSize o with
        | none => 0
        | some c => x c

/-! ## Reading the operators -/

section Reads

variable {r : Nat}

/-- `convInput` reads the input at `convSource`, and is zero at a hole. -/
theorem convInput_eq_convSource [Zero α] (x : Tensor α s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) :
    convInput x dims stride padLow padHigh inputDilation dilation groups batchGroups o k =
      match convSource dims stride padLow padHigh inputDilation dilation groups batchGroups o k with
      | none => 0
      | some c => x c := by
  unfold convInput convSource
  split <;> rfl

/-- `convTransposeInput` reads the input at `convTransposeSource`, and is zero where no input
entry lands. -/
theorem convTransposeInput_eq_convTransposeSource [Zero α] (x : Tensor α s)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat)
    (o : (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups).Coord)
    (k : (s.convTransposeWindow t dims groups).Coord) :
    convTransposeInput x dims stride padLow padHigh outputPadding dilation groups o k =
      match convTransposeSource dims stride padLow padHigh outputPadding dilation groups o k with
      | none => 0
      | some c => x c := by
  unfold convTransposeInput convTransposeSource
  split <;> rfl

/-- `getOrZero` reads `x` at `Shape.Coord.ofIndices? e index`, and is zero at `none`. -/
theorem getOrZero_eq_ofIndices? [Zero α] (x : Tensor α s) {n : Nat} (e : Fin n ≃ Fin s.rank)
    (index : Fin n → Nat) :
    getOrZero x e index =
      match Shape.Coord.ofIndices? e index with
      | none => 0
      | some c => x c := by
  unfold getOrZero Shape.Coord.ofIndices?
  by_cases h : ∀ p, index p < s.extent (e p) <;> simp [h]

/-- An entry of the input gradient of `conv` is its fold over the group's output features and the
window. -/
theorem convBackwardInput_apply [Add α] [Mul α] [Zero α] (w : Tensor α t)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) (c : s.Coord) :
    convBackwardInput w dims stride padLow padHigh dilation groups dy c =
      (List.finRange (Shape.size (t.convBackwardInputWindow dims groups))).foldl
        (fun acc i =>
          if dims.backwardInputAligned stride padLow dilation groups c
              (Shape.Coord.unlinearize i) then
            acc + getOrZero dy (dims.output stride padLow padHigh (fun _ => 1) dilation 1)
                (dims.backwardInputOutputIndex stride padLow dilation groups c
                  (Shape.Coord.unlinearize i)) *
              getOrZero w dims.kernel
                (dims.backwardInputKernelIndex groups c (Shape.Coord.unlinearize i))
          else acc + 0)
        0 := by
  rw [convBackwardInput, Rep.get_ofFn]
  congr 1
  funext acc i
  simp only [Shape.ConvDims.backwardInputAligned, decide_eq_true_eq]
  split_ifs
  · congr 2
  · rfl

/-- An entry of the weight gradient of `conv` is its fold over the batch and output positions. -/
theorem convBackwardWeight_apply [Add α] [Mul α] [Zero α] (x : Tensor α s)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) (w : t.Coord) :
    convBackwardWeight x dims stride padLow padHigh dilation groups dy w =
      (List.finRange (Shape.size (s.convBackwardWeightWindow t dims stride padLow padHigh
          dilation))).foldl
        (fun acc i =>
          acc +
            (if dims.backwardWeightAbovePadLow stride padLow padHigh dilation w
                (Shape.Coord.unlinearize i) then
              getOrZero x dims.input
                (dims.backwardWeightInputIndex stride padLow padHigh dilation groups w
                  (Shape.Coord.unlinearize i))
            else 0) *
              getOrZero dy (dims.output stride padLow padHigh (fun _ => 1) dilation 1)
                (dims.backwardWeightOutputIndex stride padLow padHigh dilation w
                  (Shape.Coord.unlinearize i)))
        0 := by
  rw [convBackwardWeight, Rep.get_ofFn]
  congr 1
  funext acc i
  simp only [Shape.ConvDims.backwardWeightAbovePadLow, decide_eq_true_eq]
  split_ifs <;> congr 2

/-- An entry of the input gradient of `convTranspose` is its fold over the kernel's output features
and window. -/
theorem convTransposeBackwardInput_apply [Add α] [Mul α] [Zero α] (w : Tensor α t)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat)
    (dy : Tensor α (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups))
    (c : s.Coord) :
    convTransposeBackwardInput w dims stride padLow padHigh outputPadding dilation groups dy c =
      (List.finRange (Shape.size (t.convTransposeBackwardInputWindow dims))).foldl
        (fun acc i =>
          if dims.transposeBackwardInputAbovePadLow stride padLow dilation c
              (Shape.Coord.unlinearize i) then
            acc + getOrZero w dims.kernel
                (dims.transposeBackwardInputKernelIndex groups c (Shape.Coord.unlinearize i)) *
              getOrZero dy
                (dims.transposeOutput stride padLow padHigh outputPadding dilation groups)
                (dims.transposeBackwardInputOutputIndex stride padLow dilation groups c
                  (Shape.Coord.unlinearize i))
          else acc)
        0 := by
  rw [convTransposeBackwardInput, Rep.get_ofFn]
  congr 1
  funext acc i
  simp only [Shape.ConvDims.transposeBackwardInputAbovePadLow, decide_eq_true_eq]
  split_ifs <;> congr 2

/-- An entry of the weight gradient of `convTranspose` is its fold over the batch and input
positions. -/
theorem convTransposeBackwardWeight_apply [Add α] [Mul α] [Zero α] (x : Tensor α s)
    (dims : Shape.ConvDims s t r) (stride padLow padHigh outputPadding dilation : Fin r → Nat)
    (groups : Nat)
    (dy : Tensor α (s.convTranspose t dims stride padLow padHigh outputPadding dilation groups))
    (w : t.Coord) :
    convTransposeBackwardWeight x dims stride padLow padHigh outputPadding dilation groups dy w =
      (List.finRange (Shape.size (s.convTransposeBackwardWeightWindow dims))).foldl
        (fun acc i =>
          if dims.transposeBackwardWeightAbovePadLow stride padLow dilation w
              (Shape.Coord.unlinearize i) then
            acc + getOrZero x dims.input
                (dims.transposeBackwardWeightInputIndex groups w (Shape.Coord.unlinearize i)) *
              getOrZero dy
                (dims.transposeOutput stride padLow padHigh outputPadding dilation groups)
                (dims.transposeBackwardWeightOutputIndex stride padLow dilation groups w
                  (Shape.Coord.unlinearize i))
          else acc)
        0 := by
  rw [convTransposeBackwardWeight, Rep.get_ofFn]
  congr 1
  funext acc i
  simp only [Shape.ConvDims.transposeBackwardWeightAbovePadLow, decide_eq_true_eq]
  split_ifs <;> congr 2

variable [Context α] {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- `poolInput` is the input entry at `poolSource`, or zero in the padding. -/
theorem poolInput_def (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord)
    (k : (s.poolWindow dims kernel).Coord) :
    poolInput x o k =
      match poolSource o k with
      | none => 0
      | some c => x c :=
  rfl

/-- An entry of the gradient of a maximum pool adds the cotangent entries whose window selected
it. -/
theorem poolBackward_max_apply (x : Tensor α s)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    poolBackward .max x dims kernel stride padLow padHigh dilation ceilMode dy c =
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          if poolIndices true x dims kernel stride padLow padHigh dilation ceilMode
              (Shape.Coord.unlinearize i) = some c then acc + dy (Shape.Coord.unlinearize i)
          else acc)
        0 :=
  Rep.get_ofFn _ c

/-- An entry of the gradient of a minimum pool adds the cotangent entries whose window selected
it. -/
theorem poolBackward_min_apply (x : Tensor α s)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    poolBackward .min x dims kernel stride padLow padHigh dilation ceilMode dy c =
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          if poolIndices false x dims kernel stride padLow padHigh dilation ceilMode
              (Shape.Coord.unlinearize i) = some c then acc + dy (Shape.Coord.unlinearize i)
          else acc)
        0 :=
  Rep.get_ofFn _ c

/-- An entry of the gradient of a sum pool adds the cotangent entry of every window containing
it, once for each of its cells there. -/
theorem poolBackward_sum_apply (x : Tensor α s)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    poolBackward .sum x dims kernel stride padLow padHigh dilation ceilMode dy c =
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
            (fun acc j =>
              if poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j) =
                  some c then acc + dy (Shape.Coord.unlinearize i)
              else acc)
            acc)
        0 :=
  Rep.get_ofFn _ c

/-- An entry of the gradient of an average pool adds the cotangent entry of every window containing
it divided by the window's count. -/
theorem poolBackward_avg_apply (countIncludePad : Bool) (x : Tensor α s)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    poolBackward (.avg countIncludePad) x dims kernel stride padLow padHigh dilation ceilMode dy c =
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
            (fun acc j =>
              if poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j) =
                  some c then
                acc + Context.divNat (dy (Shape.Coord.unlinearize i))
                  (poolCount countIncludePad (Shape.Coord.unlinearize i))
              else acc)
            acc)
        0 :=
  Rep.get_ofFn _ c

/-- The pivot of a smooth maximum pool is its fold over the window. -/
theorem smoothMaxPoolPivot_def (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolPivot beta x o =
      (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
        (fun pivot i =>
          if beta > 0 then Max.max pivot (poolInput x o (Shape.Coord.unlinearize i))
          else Min.min pivot (poolInput x o (Shape.Coord.unlinearize i)))
        (if h : 0 < Shape.size (s.poolWindow dims kernel) then
          poolInput x o (Shape.Coord.unlinearize ⟨0, h⟩)
        else 0) :=
  rfl

/-- The shift of a smooth maximum pool is its pivot, or zero when the pivot is infinite. -/
theorem smoothMaxPoolShift_def (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolShift beta x o =
      if Context.isInf (smoothMaxPoolPivot beta x o) then 0 else smoothMaxPoolPivot beta x o :=
  rfl

/-- The sum of shifted exponentials of a smooth maximum pool is its fold over the window. -/
theorem smoothMaxPoolSum_def (beta : α) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) (shift : α) :
    smoothMaxPoolSum beta x o shift =
      (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
        (fun acc i => acc + MathFunctions.exp (beta *
          (poolInput x o (Shape.Coord.unlinearize i) - shift)))
        0 :=
  rfl

/-- An entry of the directional derivative of a smooth maximum pool is its weighted fold over the
window. -/
theorem smoothMaxPoolJVP_apply (beta : α) (x dx : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) :
    smoothMaxPoolJVP beta x dx dims kernel stride padLow padHigh dilation ceilMode o =
      let pivot := smoothMaxPoolPivot beta x o
      let sum := smoothMaxPoolSum beta x o pivot
      (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
        (fun acc i => acc + (MathFunctions.exp (beta *
            (poolInput x o (Shape.Coord.unlinearize i) - pivot)) / sum) *
          poolInput dx o (Shape.Coord.unlinearize i))
        0 :=
  Rep.get_ofFn _ o

/-- An entry of the gradient of a smooth maximum pool is its fold over the windows containing it. -/
theorem smoothMaxPoolBackward_apply (beta : α) (x : Tensor α s)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) (c : s.Coord) :
    smoothMaxPoolBackward beta x dims kernel stride padLow padHigh dilation ceilMode dy c =
      (List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation
          ceilMode))).foldl
        (fun acc i =>
          let o := Shape.Coord.unlinearize i
          let pivot := smoothMaxPoolPivot beta x o
          let sum := smoothMaxPoolSum beta x o pivot
          (List.finRange (Shape.size (s.poolWindow dims kernel))).foldl
            (fun acc j =>
              if poolSource o (Shape.Coord.unlinearize j) = some c then
                acc + dy o * (MathFunctions.exp (beta *
                  (poolInput x o (Shape.Coord.unlinearize j) - pivot)) / sum)
              else acc)
            acc)
        0 :=
  Rep.get_ofFn _ c

end Reads

/-! ## Pooling in one pass

`pool` and `poolBackward` state each entry by itself. Run as written, a maximum or minimum pool
builds the whole `poolIndices` tensor again for each entry, and the gradient searches every output
for each input coordinate. The `@[csimp]` theorems below run them as `poolFast` and
`poolBackwardFast`, which build `poolIndices` once and add each entry of the cotangent into its
window in one pass over the output. -/

section PoolFast

variable [Context α]

/-- `pool`, building the `poolIndices` tensor of a maximum or minimum pool once for all entries. -/
def poolFast (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  match op with
  | .sum => Rep.ofFn fun o => poolSum x o
  | .avg countIncludePad => Rep.ofFn fun o =>
      Context.divNat (poolSum x o) (poolCount countIncludePad o)
  | .max =>
      let selected := poolIndices true x dims kernel stride padLow padHigh dilation ceilMode
      Rep.ofFn fun o =>
        match selected o with
        | none => 0
        | some c => x c
  | .min =>
      let selected := poolIndices false x dims kernel stride padLow padHigh dilation ceilMode
      Rep.ofFn fun o =>
        match selected o with
        | none => 0
        | some c => x c

/-- `pool` runs as `poolFast`. -/
@[csimp] theorem poolFast_eq : @pool = @poolFast := by
  funext α _ s _ op x dims kernel stride padLow padHigh dilation ceilMode
  cases op <;> rfl

/-- Add `value` into the row-major entry of `target` in `acc`, if there is a target. -/
def addInto (acc : Array α) (target : Option s.Coord) (value : α) : Array α :=
  match target with
  | none => acc
  | some c => acc.modify (Shape.Coord.linearize c) (· + value)

omit [Storage α] in
/-- `addInto` adds `value` to the entry of `c` exactly when the target is `c`. -/
private theorem getElem?_addInto (acc : Array α) (target : Option s.Coord) (value : α)
    (c : s.Coord) :
    (addInto acc target value)[(Shape.Coord.linearize c : Nat)]? =
      acc[(Shape.Coord.linearize c : Nat)]?.map fun a =>
        if target = some c then a + value else a := by
  cases target with
  | none => simp [addInto]
  | some t =>
    by_cases h : t = c
    · subst h
      simp [addInto, Array.getElem?_modify]
    · have hne : (Shape.Coord.linearize t : Nat) ≠ Shape.Coord.linearize c :=
        fun e => h (Shape.Coord.linearize_injective (Fin.ext e))
      simp [addInto, Array.getElem?_modify, hne, h]

/-- An entry of a left fold whose every step acts on that entry by `cell` is the fold of `cell`. -/
private theorem getElem?_foldl_of_step {β γ : Type} (k : Nat) (step : Array γ → β → Array γ)
    (cell : γ → β → γ) (hstep : ∀ acc b, (step acc b)[k]? = acc[k]?.map fun a => cell a b)
    (l : List β) (acc : Array γ) :
    (l.foldl step acc)[k]? = acc[k]?.map fun a => l.foldl cell a := by
  induction l generalizing acc with
  | nil => simp
  | cons b l ih => simp [ih, hstep, Option.map_map, Function.comp_def]

omit [Storage α] in
/-- An entry of a fold of `addInto` adds, in order, the values whose target is its coordinate. -/
private theorem getElem?_foldl_addInto {β : Type} (target : β → Option s.Coord) (value : β → α)
    (c : s.Coord) (l : List β) (acc : Array α) :
    (l.foldl (fun acc b => addInto acc (target b) (value b)) acc)[
        (Shape.Coord.linearize c : Nat)]? =
      acc[(Shape.Coord.linearize c : Nat)]?.map fun a =>
        l.foldl (fun a b => if target b = some c then a + value b else a) a :=
  getElem?_foldl_of_step _ _ _ (fun acc b => getElem?_addInto acc (target b) (value b) c) l acc

omit [Storage α] in
/-- An entry of a nested fold of `addInto` adds, in order, the values whose target is its
coordinate. -/
private theorem getElem?_foldl_foldl_addInto {β δ : Type} (target : β → δ → Option s.Coord)
    (value : β → α) (c : s.Coord) (l : List β) (m : List δ) (acc : Array α) :
    (l.foldl (fun acc b => m.foldl (fun acc d => addInto acc (target b d) (value b)) acc) acc)[
        (Shape.Coord.linearize c : Nat)]? =
      acc[(Shape.Coord.linearize c : Nat)]?.map fun a =>
        l.foldl (fun a b => m.foldl (fun a d => if target b d = some c then a + value b else a) a)
          a :=
  getElem?_foldl_of_step _ _ _
    (fun acc b => getElem?_foldl_addInto (target b) (fun _ => value b) c m acc) l acc

/-- `poolBackward` in one pass over the output. Each entry of `dy` is added into the coordinate its
window selected for a maximum or minimum pool, and into every input coordinate of its window for a
sum or average pool, in the order of `poolBackward`. -/
def poolBackwardFast (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (dy : Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode)) : Tensor α s :=
  let outputs :=
    List.finRange (Shape.size (s.pool dims kernel stride padLow padHigh dilation ceilMode))
  let window := List.finRange (Shape.size (s.poolWindow dims kernel))
  let zeros : Array α := Array.replicate (Shape.size s) 0
  let sums := match op with
    | .max =>
        let selected := poolIndices true x dims kernel stride padLow padHigh dilation ceilMode
        outputs.foldl (fun acc i =>
          addInto acc (selected (Shape.Coord.unlinearize i)) (dy (Shape.Coord.unlinearize i)))
          zeros
    | .min =>
        let selected := poolIndices false x dims kernel stride padLow padHigh dilation ceilMode
        outputs.foldl (fun acc i =>
          addInto acc (selected (Shape.Coord.unlinearize i)) (dy (Shape.Coord.unlinearize i)))
          zeros
    | .sum =>
        outputs.foldl (fun acc i =>
          window.foldl (fun acc j =>
            addInto acc (poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j))
              (dy (Shape.Coord.unlinearize i))) acc) zeros
    | .avg countIncludePad =>
        outputs.foldl (fun acc i =>
          let share := Context.divNat (dy (Shape.Coord.unlinearize i))
            (poolCount countIncludePad (Shape.Coord.unlinearize i))
          window.foldl (fun acc j =>
            addInto acc (poolSource (Shape.Coord.unlinearize i) (Shape.Coord.unlinearize j))
              share) acc) zeros
  Rep.ofFlatFn fun k => sums.getD k 0

/-- `poolBackward` runs as `poolBackwardFast`. -/
@[csimp] theorem poolBackwardFast_eq : @poolBackward = @poolBackwardFast := by
  funext α _ s _ op x dims kernel stride padLow padHigh dilation ceilMode dy
  apply Rep.ext
  intro c
  have hzero : (Array.replicate (Shape.size s) (0 : α))[(Shape.Coord.linearize c : Nat)]? =
      some 0 := by
    simp
  cases op with
  | max =>
    simp only [poolBackwardFast, Rep.get_ofFlatFn, Array.getD_eq_getD_getElem?,
      getElem?_foldl_addInto, hzero, Option.map_some, Option.getD_some, poolBackward_max_apply]
  | min =>
    simp only [poolBackwardFast, Rep.get_ofFlatFn, Array.getD_eq_getD_getElem?,
      getElem?_foldl_addInto, hzero, Option.map_some, Option.getD_some, poolBackward_min_apply]
  | sum =>
    simp only [poolBackwardFast, Rep.get_ofFlatFn, Array.getD_eq_getD_getElem?,
      getElem?_foldl_foldl_addInto, hzero, Option.map_some, Option.getD_some,
      poolBackward_sum_apply]
  | avg countIncludePad =>
    simp only [poolBackwardFast, Rep.get_ofFlatFn, Array.getD_eq_getD_getElem?,
      getElem?_foldl_foldl_addInto, hzero, Option.map_some, Option.getD_some,
      poolBackward_avg_apply]

end PoolFast

/-! ## Convolution and its gradients in one pass

`conv`, `convBackwardInput` and `convBackwardWeight` read each entry through coordinates of shapes
computed from the dimension numbers. Run as written, each step computes the output and window
shapes again and builds the coordinates of the entries it reads. The `@[csimp]` theorems below run
them as `convFast`, `convBackwardInputFast` and `convBackwardWeightFast`, which compute the shapes,
the axis tables and the indices of every window cell once and read each entry at its row-major
offset. Each output entry is the same fold as in the operator it computes, step for step. -/

section ConvFast

variable {r : Nat}

/-- Whether the window cell at position `p` lies on an input entry, as
`Shape.ConvDims.inputAligned`, read from the indices `oi` of an output entry and `ki` of a window
cell by position. -/
def convAlignedAt (stride padLow inputDilation dilation : Fin r → Nat) (oi : Fin (r + 2) → Nat)
    (ki : Fin (r + 1) → Nat) : Fin (r + 2) → Bool
  | ⟨0, _⟩ => true
  | ⟨1, _⟩ => true
  | ⟨i + 2, h⟩ =>
      decide (padLow ⟨i, by omega⟩ ≤
          oi ⟨i + 2, h⟩ * stride ⟨i, by omega⟩ + ki ⟨i + 1, by omega⟩ * dilation ⟨i, by omega⟩) &&
        decide ((oi ⟨i + 2, h⟩ * stride ⟨i, by omega⟩ +
              ki ⟨i + 1, by omega⟩ * dilation ⟨i, by omega⟩ - padLow ⟨i, by omega⟩) %
            inputDilation ⟨i, by omega⟩ = 0)

/-- The index along the input axis at position `p` that `conv` reads, as
`Shape.ConvDims.inputIndex`, read from the indices `oi` of an output entry and `ki` of a window cell
by position. -/
def convIndexAt (dims : Shape.ConvDims s t r) (stride padLow inputDilation dilation : Fin r → Nat)
    (groups batchGroups : Nat) (oi : Fin (r + 2) → Nat) (ki : Fin (r + 1) → Nat) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ =>
      oi ⟨1, by omega⟩ / (t.extent (dims.kernel ⟨0, by omega⟩) / batchGroups) *
        (s.extent (dims.input ⟨0, by omega⟩) / batchGroups) + oi ⟨0, by omega⟩
  | ⟨1, _⟩ =>
      oi ⟨1, by omega⟩ / (t.extent (dims.kernel ⟨0, by omega⟩) / groups) *
        t.extent (dims.kernel ⟨1, by omega⟩) + ki ⟨0, by omega⟩
  | ⟨i + 2, h⟩ =>
      (oi ⟨i + 2, h⟩ * stride ⟨i, by omega⟩ + ki ⟨i + 1, by omega⟩ * dilation ⟨i, by omega⟩ -
        padLow ⟨i, by omega⟩) / inputDilation ⟨i, by omega⟩

/-- `conv` with the output and window shapes, the axis tables and the indices of every window cell
computed once, each step reading the input and the kernel at their row-major offsets. -/
def convFast [Add α] [Mul α] [Zero α] (x : Tensor α s) (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh : Fin r → Nat)
    (inputDilation dilation : Fin r → Nat := fun _ => 1) (groups batchGroups : Nat := 1) :
    Tensor α (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups) :=
  let out := s.conv t dims stride padLow padHigh inputDilation dilation batchGroups
  let window := t.convWindow dims
  let cells : Array (Array Nat) := Array.ofFn fun i : Fin (Shape.size window) =>
    Array.ofFn fun q : Fin (r + 1) => Shape.Coord.index (Shape.Coord.unlinearize i) q.val
  let kernelOffsets : Array Nat := Array.ofFn fun i : Fin (Shape.size window) =>
    (List.ofFn fun q : Fin (r + 1) =>
      (cells.getD i #[]).getD q 0 * t.stride (dims.kernel q.succ)).sum
  let extents : Array Nat := Array.ofFn fun p : Fin (r + 2) => s.extent (dims.input p)
  let strides : Array Nat := Array.ofFn fun p : Fin (r + 2) => s.stride (dims.input p)
  let strideV := Array.ofFn stride
  let padLowV := Array.ofFn padLow
  let inputDilationV := Array.ofFn inputDilation
  let dilationV := Array.ofFn dilation
  let strideAt : Fin r → Nat := fun j => strideV.getD j 0
  let padLowAt : Fin r → Nat := fun j => padLowV.getD j 0
  let inputDilationAt : Fin r → Nat := fun j => inputDilationV.getD j 0
  let dilationAt : Fin r → Nat := fun j => dilationV.getD j 0
  let extentAt : Fin (r + 2) → Nat := fun p => extents.getD p 0
  let strideOf : Fin (r + 2) → Nat := fun p => strides.getD p 0
  let featureStride := t.stride (dims.kernel ⟨0, by omega⟩)
  Rep.ofFn (s := out) fun o =>
    let ov : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
      Shape.Coord.index o (dims.input p).val
    let oi : Fin (r + 2) → Nat := fun p => ov.getD p 0
    let feature := Shape.Coord.index o (dims.input ⟨1, by omega⟩).val
    (List.finRange (Shape.size window)).foldl
      (fun acc (i : Fin (Shape.size window)) =>
        let kv := cells.getD i #[]
        let ki : Fin (r + 1) → Nat := fun q => kv.getD q 0
        let aligned := convAlignedAt strideAt padLowAt inputDilationAt dilationAt oi ki
        let index := convIndexAt dims strideAt padLowAt inputDilationAt dilationAt groups
          batchGroups oi ki
        let xv := if ∀ p, aligned p = true ∧ index p < extentAt p then
            let n := (List.ofFn fun p => index p * strideOf p).sum
            if h : n < Shape.size s then x.getFlat ⟨n, h⟩ else 0
          else 0
        let n := feature * featureStride + kernelOffsets.getD i 0
        acc + xv * (if h : n < Shape.size t then w.getFlat ⟨n, h⟩ else 0))
      0

/-- The row-major index of `Shape.Coord.ofPositions e f` adds each index times the stride of its
axis. -/
private theorem linearize_ofPositions_val {n : Nat} (e : Fin n ≃ Fin s.rank)
    (f : (p : Fin n) → Fin (s.extent (e p))) :
    (Shape.Coord.linearize (Shape.Coord.ofPositions e f)).val =
      (List.ofFn fun p => (f p).val * s.stride (e p)).sum := by
  rw [Shape.Coord.linearize_val_eq_sum, List.sum_ofFn, ← Equiv.sum_comp e]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Shape.Coord.val_get, Shape.Coord.index_ofPositions]

/-- `convInput` reads the input at the row-major offset of the indices `convIndexAt`. -/
private theorem convInput_eq_getFlat [Zero α] (x : Tensor α s) (dims : Shape.ConvDims s t r)
    (stride padLow padHigh inputDilation dilation : Fin r → Nat) (groups batchGroups : Nat)
    (o : (s.conv t dims stride padLow padHigh inputDilation dilation batchGroups).Coord)
    (k : (t.convWindow dims).Coord) :
    convInput x dims stride padLow padHigh inputDilation dilation groups batchGroups o k =
      if ∀ p, convAlignedAt stride padLow inputDilation dilation
            (fun p => Shape.Coord.index o (dims.input p).val) (fun q => Shape.Coord.index k q.val)
            p = true ∧
          convIndexAt dims stride padLow inputDilation dilation groups batchGroups
            (fun p => Shape.Coord.index o (dims.input p).val) (fun q => Shape.Coord.index k q.val)
            p < s.extent (dims.input p) then
        if h : (List.ofFn fun p => convIndexAt dims stride padLow inputDilation dilation groups
            batchGroups (fun p => Shape.Coord.index o (dims.input p).val)
            (fun q => Shape.Coord.index k q.val) p * s.stride (dims.input p)).sum <
              Shape.size s then
          x.getFlat ⟨_, h⟩
        else 0
      else 0 := by
  have hA : ∀ p, convAlignedAt stride padLow inputDilation dilation
      (fun p => Shape.Coord.index o (dims.input p).val) (fun q => Shape.Coord.index k q.val) p =
        dims.inputAligned stride padLow padHigh inputDilation dilation batchGroups o k p := by
    rintro ⟨_ | _ | j, hp⟩ <;> rfl
  have hI : ∀ p, convIndexAt dims stride padLow inputDilation dilation groups batchGroups
      (fun p => Shape.Coord.index o (dims.input p).val) (fun q => Shape.Coord.index k q.val) p =
        dims.inputIndex stride padLow padHigh inputDilation dilation groups batchGroups o k p := by
    rintro ⟨_ | _ | j, hp⟩ <;> rfl
  simp only [hA, hI]
  unfold convInput
  split_ifs with hc hn
  · change Rep.getFlat x (Shape.Coord.linearize _) = _
    exact congrArg x.getFlat (Fin.ext (linearize_ofPositions_val dims.input _))
  · exact absurd ((linearize_ofPositions_val dims.input
      (fun p => ⟨_, (hc p).2⟩)) ▸ (Shape.Coord.linearize _).isLt) hn
  · rfl

/-- The kernel entry at `kernelCoord f k` is the entry at its row-major offset. -/
private theorem get_kernelCoord [Zero α] (w : Tensor α t) (dims : Shape.ConvDims s t r)
    (f : Fin (t.extent (dims.kernel ⟨0, by omega⟩))) (k : (t.convWindow dims).Coord) :
    w (dims.kernelCoord f k) =
      if h : f.val * t.stride (dims.kernel ⟨0, by omega⟩) +
          (List.ofFn fun q : Fin (r + 1) =>
            Shape.Coord.index k q.val * t.stride (dims.kernel q.succ)).sum < Shape.size t then
        w.getFlat ⟨_, h⟩
      else 0 := by
  have hval : (Shape.Coord.linearize (dims.kernelCoord f k)).val =
      f.val * t.stride (dims.kernel ⟨0, by omega⟩) +
        (List.ofFn fun q : Fin (r + 1) =>
          Shape.Coord.index k q.val * t.stride (dims.kernel q.succ)).sum := by
    rw [Shape.ConvDims.kernelCoord, linearize_ofPositions_val, List.ofFn_succ, List.sum_cons]
    rfl
  split_ifs with h
  · change Rep.getFlat w (Shape.Coord.linearize _) = _
    exact congrArg w.getFlat (Fin.ext hval)
  · exact absurd (hval ▸ (Shape.Coord.linearize _).isLt) h

/-- `conv` runs as `convFast`. -/
@[csimp] theorem convFast_eq : @conv = @convFast := by
  funext α _ s t _ _ _ x w r dims stride padLow padHigh inputDilation dilation groups batchGroups
  apply Rep.ext
  intro o
  rw [conv_apply]
  simp only [convFast, Rep.get_ofFn, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
    Fin.is_lt, ↓reduceDIte, Option.getD_some, Fin.eta]
  congr 1
  funext acc i
  rw [convInput_eq_getFlat, get_kernelCoord]
  rfl

/-- The entry of `x` at the row-major offset of `index` for the strides `strides` of the axes by
position, or zero when an index is not below its extent in `extents`. -/
def getOrZeroAt [Zero α] (x : Tensor α s) {n : Nat} (extents strides : Fin n → Nat)
    (index : Fin n → Nat) : α :=
  if ∀ p, index p < extents p then
    if h : (List.ofFn fun p => index p * strides p).sum < Shape.size s then x.getFlat ⟨_, h⟩
    else 0
  else 0

/-- `getOrZero x e` is `getOrZeroAt` at the extents and strides of the axes `e p`. -/
theorem getOrZero_eq_getOrZeroAt [Zero α] (x : Tensor α s) {n : Nat} (e : Fin n ≃ Fin s.rank)
    (index : Fin n → Nat) :
    getOrZero x e index =
      getOrZeroAt x (fun p => s.extent (e p)) (fun p => s.stride (e p)) index := by
  unfold getOrZero getOrZeroAt
  split_ifs with hc hn
  · change Rep.getFlat x (Shape.Coord.linearize _) = _
    exact congrArg x.getFlat (Fin.ext (linearize_ofPositions_val e _))
  · exact absurd ((linearize_ofPositions_val e (fun p => ⟨_, hc p⟩)) ▸
      (Shape.Coord.linearize _).isLt) hn
  · rfl

/-- `Shape.ConvDims.backwardInputAligned`, read from the indices `ci` of an input entry and `ki` of
a window cell by position. -/
def convBackwardInputAlignedAt (stride padLow dilation : Fin r → Nat) (ci : Fin (r + 2) → Nat)
    (ki : Fin (r + 1) → Nat) : Bool :=
  decide (∀ j : Fin r, stride j ≠ 0 ∧ ki j.succ * dilation j ≤ ci j.succ.succ + padLow j ∧
    (ci j.succ.succ + padLow j - ki j.succ * dilation j) % stride j = 0)

/-- `Shape.ConvDims.backwardInputOutputIndex`, read from the indices `ci` of an input entry and `ki`
of a window cell by position, for `inner` input features of the kernel and `outPerGroup` output
features in a group. -/
def convBackwardInputOutputIndexAt (stride padLow dilation : Fin r → Nat) (inner outPerGroup : Nat)
    (ci : Fin (r + 2) → Nat) (ki : Fin (r + 1) → Nat) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => ci ⟨0, by omega⟩
  | ⟨1, _⟩ => ci ⟨1, by omega⟩ / inner * outPerGroup + ki ⟨0, by omega⟩
  | ⟨j + 2, h⟩ =>
      (ci ⟨j + 2, h⟩ + padLow ⟨j, by omega⟩ - ki ⟨j + 1, by omega⟩ * dilation ⟨j, by omega⟩) /
        stride ⟨j, by omega⟩

/-- `Shape.ConvDims.backwardInputKernelIndex`, read from the indices `ci` of an input entry and `ki`
of a window cell by position, for `inner` input features of the kernel and `outPerGroup` output
features in a group. -/
def convBackwardInputKernelIndexAt (inner outPerGroup : Nat) (ci : Fin (r + 2) → Nat)
    (ki : Fin (r + 1) → Nat) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => ci ⟨1, by omega⟩ / inner * outPerGroup + ki ⟨0, by omega⟩
  | ⟨1, _⟩ => ci ⟨1, by omega⟩ % inner
  | ⟨j + 2, _⟩ => ki ⟨j + 1, by omega⟩

/-- `convBackwardInput` with the cotangent and window shapes, the axis tables and the indices of
every window cell computed once, each step reading the cotangent and the kernel at their row-major
offsets. -/
def convBackwardInputFast [Add α] [Mul α] [Zero α] (w : Tensor α t) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) : Tensor α s :=
  let out := s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1
  let window := t.convBackwardInputWindow dims groups
  let cells : Array (Array Nat) := Array.ofFn fun i : Fin (Shape.size window) =>
    Array.ofFn fun q : Fin (r + 1) => Shape.Coord.index (Shape.Coord.unlinearize i) q.val
  let outExtents : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
    out.extent (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p)
  let outStrides : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
    out.stride (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p)
  let kernelExtents : Array Nat := Array.ofFn fun p : Fin (r + 2) => t.extent (dims.kernel p)
  let kernelStrides : Array Nat := Array.ofFn fun p : Fin (r + 2) => t.stride (dims.kernel p)
  let strideV := Array.ofFn stride
  let padLowV := Array.ofFn padLow
  let dilationV := Array.ofFn dilation
  let strideAt : Fin r → Nat := fun j => strideV.getD j 0
  let padLowAt : Fin r → Nat := fun j => padLowV.getD j 0
  let dilationAt : Fin r → Nat := fun j => dilationV.getD j 0
  let outExtentAt : Fin (r + 2) → Nat := fun p => outExtents.getD p 0
  let outStrideAt : Fin (r + 2) → Nat := fun p => outStrides.getD p 0
  let kernelExtentAt : Fin (r + 2) → Nat := fun p => kernelExtents.getD p 0
  let kernelStrideAt : Fin (r + 2) → Nat := fun p => kernelStrides.getD p 0
  let inner := t.extent (dims.kernel ⟨1, by omega⟩)
  let outPerGroup := t.extent (dims.kernel ⟨0, by omega⟩) / groups
  Rep.ofFn fun c =>
    let cv : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
      Shape.Coord.index c (dims.input p).val
    let ci : Fin (r + 2) → Nat := fun p => cv.getD p 0
    (List.finRange (Shape.size window)).foldl
      (fun acc (i : Fin (Shape.size window)) =>
        let kv := cells.getD i #[]
        let ki : Fin (r + 1) → Nat := fun q => kv.getD q 0
        if convBackwardInputAlignedAt strideAt padLowAt dilationAt ci ki then
          acc + getOrZeroAt dy outExtentAt outStrideAt
              (convBackwardInputOutputIndexAt strideAt padLowAt dilationAt inner outPerGroup ci
                ki) *
            getOrZeroAt w kernelExtentAt kernelStrideAt
              (convBackwardInputKernelIndexAt inner outPerGroup ci ki)
        else acc + 0)
      0

/-- `convBackwardInputAlignedAt` at the indices of `c` and `k` is `backwardInputAligned`. -/
private theorem convBackwardInputAlignedAt_eq (dims : Shape.ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (k : (t.convBackwardInputWindow dims groups).Coord) :
    convBackwardInputAlignedAt stride padLow dilation
        (fun p => Shape.Coord.index c (dims.input p).val) (fun q => Shape.Coord.index k q.val) =
      dims.backwardInputAligned stride padLow dilation groups c k :=
  rfl

/-- `convBackwardInputOutputIndexAt` at the indices of `c` and `k` is
`backwardInputOutputIndex`. -/
private theorem convBackwardInputOutputIndexAt_eq (dims : Shape.ConvDims s t r)
    (stride padLow dilation : Fin r → Nat) (groups : Nat) (c : s.Coord)
    (k : (t.convBackwardInputWindow dims groups).Coord) :
    convBackwardInputOutputIndexAt stride padLow dilation (t.extent (dims.kernel ⟨1, by omega⟩))
        (t.extent (dims.kernel ⟨0, by omega⟩) / groups)
        (fun p => Shape.Coord.index c (dims.input p).val) (fun q => Shape.Coord.index k q.val) =
      dims.backwardInputOutputIndex stride padLow dilation groups c k := by
  funext p
  rcases p with ⟨_ | _ | j, hp⟩ <;> rfl

/-- `convBackwardInputKernelIndexAt` at the indices of `c` and `k` is
`backwardInputKernelIndex`. -/
private theorem convBackwardInputKernelIndexAt_eq (dims : Shape.ConvDims s t r) (groups : Nat)
    (c : s.Coord) (k : (t.convBackwardInputWindow dims groups).Coord) :
    convBackwardInputKernelIndexAt (t.extent (dims.kernel ⟨1, by omega⟩))
        (t.extent (dims.kernel ⟨0, by omega⟩) / groups)
        (fun p => Shape.Coord.index c (dims.input p).val) (fun q => Shape.Coord.index k q.val) =
      dims.backwardInputKernelIndex groups c k := by
  funext p
  rcases p with ⟨_ | _ | j, hp⟩ <;> rfl

/-- `convBackwardInput` runs as `convBackwardInputFast`. -/
@[csimp] theorem convBackwardInputFast_eq : @convBackwardInput = @convBackwardInputFast := by
  funext α _ s t _ _ _ w r dims stride padLow padHigh dilation groups dy
  apply Rep.ext
  intro c
  rw [convBackwardInput_apply]
  simp only [convBackwardInputFast, Rep.get_ofFn, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
    Fin.is_lt, ↓reduceDIte, Option.getD_some, Fin.eta]
  congr 1
  funext acc i
  rw [getOrZero_eq_getOrZeroAt, getOrZero_eq_getOrZeroAt, ← convBackwardInputAlignedAt_eq,
    ← convBackwardInputOutputIndexAt_eq, ← convBackwardInputKernelIndexAt_eq]

/-- `Shape.ConvDims.backwardWeightAbovePadLow`, read from the indices `wi` of a kernel entry and
`oi` of a window cell by position. -/
def convBackwardWeightAbovePadLowAt (stride padLow dilation : Fin r → Nat)
    (wi : Fin (r + 2) → Nat) (oi : Fin (r + 1) → Nat) : Bool :=
  decide (∀ j : Fin r, padLow j ≤ oi j.succ * stride j + wi j.succ.succ * dilation j)

/-- `Shape.ConvDims.backwardWeightInputIndex`, read from the indices `wi` of a kernel entry and `oi`
of a window cell by position, for `outPerGroup` output features in a group and `inner` input
features of the kernel. -/
def convBackwardWeightInputIndexAt (stride padLow dilation : Fin r → Nat) (outPerGroup inner : Nat)
    (wi : Fin (r + 2) → Nat) (oi : Fin (r + 1) → Nat) : Fin (r + 2) → Nat
  | ⟨0, _⟩ => oi ⟨0, by omega⟩
  | ⟨1, _⟩ => wi ⟨0, by omega⟩ / outPerGroup * inner + wi ⟨1, by omega⟩
  | ⟨j + 2, h⟩ =>
      oi ⟨j + 1, by omega⟩ * stride ⟨j, by omega⟩ + wi ⟨j + 2, h⟩ * dilation ⟨j, by omega⟩ -
        padLow ⟨j, by omega⟩

/-- `Shape.ConvDims.backwardWeightOutputIndex`, read from the indices `wi` of a kernel entry and
`oi` of a window cell by position. -/
def convBackwardWeightOutputIndexAt (wi : Fin (r + 2) → Nat) (oi : Fin (r + 1) → Nat) :
    Fin (r + 2) → Nat
  | ⟨0, _⟩ => oi ⟨0, by omega⟩
  | ⟨1, _⟩ => wi ⟨0, by omega⟩
  | ⟨j + 2, _⟩ => oi ⟨j + 1, by omega⟩

/-- `convBackwardWeight` with the cotangent and window shapes, the axis tables and the indices of
every window cell computed once, each step reading the input and the cotangent at their row-major
offsets. -/
def convBackwardWeightFast [Add α] [Mul α] [Zero α] (x : Tensor α s) {r : Nat}
    (dims : Shape.ConvDims s t r) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (dy : Tensor α (s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1)) : Tensor α t :=
  let out := s.conv t dims stride padLow padHigh (fun _ => 1) dilation 1
  let window := s.convBackwardWeightWindow t dims stride padLow padHigh dilation
  let cells : Array (Array Nat) := Array.ofFn fun i : Fin (Shape.size window) =>
    Array.ofFn fun q : Fin (r + 1) => Shape.Coord.index (Shape.Coord.unlinearize i) q.val
  let outExtents : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
    out.extent (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p)
  let outStrides : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
    out.stride (dims.output stride padLow padHigh (fun _ => 1) dilation 1 p)
  let inputExtents : Array Nat := Array.ofFn fun p : Fin (r + 2) => s.extent (dims.input p)
  let inputStrides : Array Nat := Array.ofFn fun p : Fin (r + 2) => s.stride (dims.input p)
  let strideV := Array.ofFn stride
  let padLowV := Array.ofFn padLow
  let dilationV := Array.ofFn dilation
  let strideAt : Fin r → Nat := fun j => strideV.getD j 0
  let padLowAt : Fin r → Nat := fun j => padLowV.getD j 0
  let dilationAt : Fin r → Nat := fun j => dilationV.getD j 0
  let outExtentAt : Fin (r + 2) → Nat := fun p => outExtents.getD p 0
  let outStrideAt : Fin (r + 2) → Nat := fun p => outStrides.getD p 0
  let inputExtentAt : Fin (r + 2) → Nat := fun p => inputExtents.getD p 0
  let inputStrideAt : Fin (r + 2) → Nat := fun p => inputStrides.getD p 0
  let outPerGroup := t.extent (dims.kernel ⟨0, by omega⟩) / groups
  let inner := t.extent (dims.kernel ⟨1, by omega⟩)
  Rep.ofFn fun w =>
    let wv : Array Nat := Array.ofFn fun p : Fin (r + 2) =>
      Shape.Coord.index w (dims.kernel p).val
    let wi : Fin (r + 2) → Nat := fun p => wv.getD p 0
    (List.finRange (Shape.size window)).foldl
      (fun acc (i : Fin (Shape.size window)) =>
        let ov := cells.getD i #[]
        let oi : Fin (r + 1) → Nat := fun q => ov.getD q 0
        acc +
          (if convBackwardWeightAbovePadLowAt strideAt padLowAt dilationAt wi oi then
            getOrZeroAt x inputExtentAt inputStrideAt
              (convBackwardWeightInputIndexAt strideAt padLowAt dilationAt outPerGroup inner wi
                oi)
          else 0) *
            getOrZeroAt dy outExtentAt outStrideAt (convBackwardWeightOutputIndexAt wi oi))
      0

/-- `convBackwardWeightAbovePadLowAt` at the indices of `w` and `o` is
`backwardWeightAbovePadLow`. -/
private theorem convBackwardWeightAbovePadLowAt_eq (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) :
    convBackwardWeightAbovePadLowAt stride padLow dilation
        (fun p => Shape.Coord.index w (dims.kernel p).val) (fun q => Shape.Coord.index o q.val) =
      dims.backwardWeightAbovePadLow stride padLow padHigh dilation w o :=
  rfl

/-- `convBackwardWeightInputIndexAt` at the indices of `w` and `o` is
`backwardWeightInputIndex`. -/
private theorem convBackwardWeightInputIndexAt_eq (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) :
    convBackwardWeightInputIndexAt stride padLow dilation
        (t.extent (dims.kernel ⟨0, by omega⟩) / groups) (t.extent (dims.kernel ⟨1, by omega⟩))
        (fun p => Shape.Coord.index w (dims.kernel p).val) (fun q => Shape.Coord.index o q.val) =
      dims.backwardWeightInputIndex stride padLow padHigh dilation groups w o := by
  funext p
  rcases p with ⟨_ | _ | j, hp⟩ <;> rfl

/-- `convBackwardWeightOutputIndexAt` at the indices of `w` and `o` is
`backwardWeightOutputIndex`. -/
private theorem convBackwardWeightOutputIndexAt_eq (dims : Shape.ConvDims s t r)
    (stride padLow padHigh dilation : Fin r → Nat) (w : t.Coord)
    (o : (s.convBackwardWeightWindow t dims stride padLow padHigh dilation).Coord) :
    convBackwardWeightOutputIndexAt (fun p => Shape.Coord.index w (dims.kernel p).val)
        (fun q => Shape.Coord.index o q.val) =
      dims.backwardWeightOutputIndex stride padLow padHigh dilation w o := by
  funext p
  rcases p with ⟨_ | _ | j, hp⟩ <;> rfl

/-- `convBackwardWeight` runs as `convBackwardWeightFast`. -/
@[csimp] theorem convBackwardWeightFast_eq : @convBackwardWeight = @convBackwardWeightFast := by
  funext α _ s t _ _ _ x r dims stride padLow padHigh dilation groups dy
  apply Rep.ext
  intro w
  rw [convBackwardWeight_apply]
  simp only [convBackwardWeightFast, Rep.get_ofFn, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
    Fin.is_lt, ↓reduceDIte, Option.getD_some, Fin.eta]
  congr 1
  funext acc i
  rw [getOrZero_eq_getOrZeroAt, getOrZero_eq_getOrZeroAt, ← convBackwardWeightAbovePadLowAt_eq,
    ← convBackwardWeightInputIndexAt_eq, ← convBackwardWeightOutputIndexAt_eq]

end ConvFast

/-! ## Reading adaptive pools -/

section AdaptiveReads

variable {dims : List (Fin s.rank)} {outputSize : List Nat}

/-- `adaptiveInput` is the input entry at `adaptiveSource`, or zero outside the input. -/
theorem adaptiveInput_def [Zero α] (x : Tensor α s) (o : (s.adaptivePool dims outputSize).Coord)
    (k : (adaptiveBin o).Coord) :
    adaptiveInput x o k =
      match adaptiveSource o k with
      | none => 0
      | some c => x c :=
  rfl

variable [Context α] (x : Tensor α s) (o : (s.adaptivePool dims outputSize).Coord)

/-- An entry of an adaptive sum pool adds its bin to zero in row-major order. -/
theorem adaptivePool_sum_apply :
    adaptivePool .sum x dims outputSize o =
      (List.finRange (Shape.size (adaptiveBin o))).foldl
        (fun acc i => acc + adaptiveInput x o (Shape.Coord.unlinearize i)) 0 :=
  Rep.get_ofFn _ o

/-- An entry of an adaptive average pool divides the sum of its bin once by its number of cells. -/
theorem adaptivePool_avg_apply (countIncludePad : Bool) :
    adaptivePool (.avg countIncludePad) x dims outputSize o =
      Context.divNat ((List.finRange (Shape.size (adaptiveBin o))).foldl
          (fun acc i => acc + adaptiveInput x o (Shape.Coord.unlinearize i)) 0)
        (Shape.size (adaptiveBin o)) :=
  Rep.get_ofFn _ o

/-- An entry of an adaptive maximum pool reads the input at the coordinate `adaptiveIndices`
selects. -/
theorem adaptivePool_max_apply :
    adaptivePool .max x dims outputSize o =
      match adaptiveIndices true x dims outputSize o with
      | none => 0
      | some c => x c :=
  Rep.get_ofFn _ o

/-- An entry of an adaptive minimum pool reads the input at the coordinate `adaptiveIndices`
selects. -/
theorem adaptivePool_min_apply :
    adaptivePool .min x dims outputSize o =
      match adaptiveIndices false x dims outputSize o with
      | none => 0
      | some c => x c :=
  Rep.get_ofFn _ o

end AdaptiveReads

end TorchLean.Tensor
