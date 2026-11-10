/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.GraphM -- shake: keep
public import TorchLean.NN.Ops.Pure -- shake: keep
public import TorchLean.NN.Ops.Tensor -- shake: keep
public import TorchLean.NN.Session.Config
public import TorchLean.Tensor.Layers.Conv.Window

/-!
# Operation-Polymorphic Tensor Programs

`Program` is model code abstract over a tensor-operation interpreter. With the eager interpreter,
operations execute immediately and record a dynamic tape. With the typed-graph recorder, which the
`autograd` transforms supply, they record reusable shape-indexed SSA data.

`Program.Data` also accepts non-differentiable tensor inputs with their own element type, such as
token indices. `map` applies an operation over a specified prefix shape, and `linear`,
convolution, pooling, normalization and attention accept those prefixes without separate batched
variants.
-/

@[expose] public section

open TorchLean.Shape (rank size size_append)

namespace TorchLean
namespace nn.functional

open TorchLean.Tensor
open Ops

/-! ## Prefix-polymorphic derived operations -/

/-- Apply a single-sample operation independently over an arbitrary prefix shape. -/
def map {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (leadingShape : Shape) {s t : Shape}
    (x : Ref (m := m) (α := α) (leadingShape ++ s))
    (f : Ref (m := m) (α := α) s → m (Ref (m := m) (α := α) t)) :
    m (Ref (m := m) (α := α) (leadingShape ++ t)) := do
  let xFlat ← Ops.apply (m := m) (α := α)
      (.reshape (s := leadingShape ++ s) (t := Shape.dim leadingShape.size s) (by
      simp [Shape.size_append])) (x, ())
  let yFlat ← Ops.mapBatch (m := m) (α := α) f xFlat
  Ops.apply (m := m) (α := α)
      (.reshape (s := Shape.dim leadingShape.size t) (t := leadingShape ++ t) (by
      simp [Shape.size_append])) (yFlat, ())

/-- Affine transformation of the final axis, independently over any prefix shape. -/
def linear {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {inDim outDim : Nat}
    (weight : Ref (m := m) (α := α) [outDim, inDim])
    (bias : Ref (m := m) (α := α) [outDim])
    (input : Ref (m := m) (α := α) (leadingShape ++ [inDim])) :
    m (Ref (m := m) (α := α) (leadingShape ++ [outDim])) :=
  -- One sample at a time (`lead := []`): the LibTorch kernel table and the IR tangent rules
  -- implement `Op.linear` only at that lead.
  map (m := m) (α := α) leadingShape input fun x =>
    Ops.apply (m := m) (α := α) (.linear (lead := [])) (weight, bias, x, ())

/-- Rank-polymorphic convolution over channels-first inputs with any prefix shape. -/
def conv {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weight : Ref (m := m) (α := α)
      (outC :: inC :: kernel.to (List Nat)))
    (bias : Ref (m := m) (α := α) [outC])
    (input : Ref (m := m) (α := α)
      (leadingShape ++ (inC :: inSpatial.to (List Nat)))) :
    m (Ref (m := m) (α := α)
      (leadingShape ++ (outC ::
          (Conv.outSpatialDilated inSpatial kernel stride
              (Tensor.ones [d]) padding padding).to (List Nat)))) :=
  -- Each sample is the convolution's input with its batch axis of one left out (`lead := []`).
  map (m := m) (α := α) leadingShape input (fun x ↦
    Ops.apply (m := m) (α := α)
      (.conv (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := outC) (i := inC)
          (spatial := inSpatial.to (List Nat)) (kernel := kernel.to (List Nat)) (by simp))
        (fun j => stride.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j))
        (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 1) 1 (lead := []) rfl rfl
        (congrArg (List.drop 1)
          (Shape.conv_channelsFirst (n := 1) inSpatial kernel stride padding)).symm)
      (weight, bias, x, ()))

/-- Rank-polymorphic transpose convolution over channels-first inputs with any prefix shape. -/
def convTranspose {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d inC outC : Nat}
    {kernel stride padding inSpatial : Tensor Nat [d]}
    (weight : Ref (m := m) (α := α)
      (inC :: outC :: kernel.to (List Nat)))
    (bias : Ref (m := m) (α := α) [outC])
    (input : Ref (m := m) (α := α)
      (leadingShape ++ (inC :: inSpatial.to (List Nat)))) :
    m (Ref (m := m) (α := α)
      (leadingShape ++ (outC ::
          (ConvTranspose.outSpatial inSpatial kernel stride padding).to (List Nat)))) :=
  -- Each sample gets a batch axis of one, as PyTorch does for an unbatched input. The reshape, the
  -- transposed convolution and the unbatching are one tape transaction: on success the tape keeps
  -- the three nodes, on failure it is restored to where it started.
  map (m := m) (α := α) leadingShape input (fun x ↦
    Ops.atomic (m := m) (α := α) do
      let batched ← Ops.apply (m := m) (α := α)
        (.reshape (t := 1 :: inC :: inSpatial.to (List Nat)) (Nat.one_mul _).symm) (x, ())
      let y ← Ops.apply (m := m) (α := α)
        (.convTranspose (Shape.ConvDims.channelsFirst (n := 1) (c := inC) (o := inC) (i := outC)
            (spatial := inSpatial.to (List Nat)) (kernel := kernel.to (List Nat))
            (by simp)).swapKernel
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j)) (fun _ => 0) (fun _ => 1) 1)
        (weight, Ref.cast (m := m) (α := α) (by rw [Nat.one_mul]; rfl) bias,
          batched, ())
      Ops.apply (m := m) (α := α)
        (.reshape ((congrArg Shape.size
          (Shape.convTranspose_channelsFirst (n := 1) inSpatial kernel stride padding)).trans
            (Nat.one_mul _)))
        (y, ()))

/-- Rank-polymorphic max pooling over channels-first inputs with any prefix shape. -/
def maxPool {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (input : Ref (m := m) (α := α)
      (leadingShape ++ (channels :: inSpatial.to (List Nat)))) :
    m (Ref (m := m) (α := α)
      (leadingShape ++ (channels ::
        (Pool.outSpatial inSpatial kernel stride padding).to (List Nat)))) :=
  map (m := m) (α := α) leadingShape input (fun x ↦
    Ref.cast (m := m) (α := α)
        (Shape.pool_tail_channelsFirst channels inSpatial kernel stride padding) <$>
      Ops.apply (m := m) (α := α)
        (.pool .max (List.finRange (Shape.rank (channels :: inSpatial.to (List Nat)))).tail
          (kernel.to (List Nat)) (stride.to (List Nat)) (padding.to (List Nat))
          (padding.to (List Nat)) (List.replicate d 1) false)
        (x, ()))

/-- Rank-polymorphic average pooling over channels-first inputs with any prefix shape. -/
def avgPool {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (input : Ref (m := m) (α := α)
      (leadingShape ++ (channels :: inSpatial.to (List Nat)))) :
    m (Ref (m := m) (α := α)
      (leadingShape ++ (channels ::
        (Pool.outSpatial inSpatial kernel stride padding).to (List Nat)))) :=
  map (m := m) (α := α) leadingShape input (fun x ↦
    Ref.cast (m := m) (α := α)
        (Shape.pool_tail_channelsFirst channels inSpatial kernel stride padding) <$>
      Ops.apply (m := m) (α := α)
        (.pool (.avg true) (List.finRange (Shape.rank (channels :: inSpatial.to (List Nat)))).tail
          (kernel.to (List Nat)) (stride.to (List Nat)) (padding.to (List Nat))
          (padding.to (List Nat)) (List.replicate d 1) false)
        (x, ()))

/-- Batched rank-polymorphic smooth maximum pooling over channels-first inputs. -/
def smoothMaxPool {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (input : Ref (m := m) (α := α)
      (leadingShape ++ (channels :: inSpatial.to (List Nat)))) (temp : α) :
    m (Ref (m := m) (α := α)
      (leadingShape ++ (channels ::
        (Pool.outSpatial inSpatial kernel stride padding).to (List Nat)))) :=
  map (m := m) (α := α) leadingShape input (fun x ↦
    Ref.cast (m := m) (α := α)
        (Shape.pool_tail_channelsFirst channels inSpatial kernel stride padding) <$>
      Ops.apply (m := m) (α := α)
        (.smoothMaxPool (List.finRange (Shape.rank (channels :: inSpatial.to (List Nat)))).tail
          (kernel.to (List Nat)) (stride.to (List Nat)) (padding.to (List Nat))
          (padding.to (List Nat)) (List.replicate d 1) false)
        (x, .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar temp)),
          ()))

/--
Layer normalization over the final axis of an arbitrary tensor.

Backends expose a matrix-shaped fused primitive. The wrapper gives that primitive its
rank-polymorphic TorchLean interface by flattening the leading axes once and restoring them after
the operation. Empty leading axes produce the unique empty tensor without invoking a backend
kernel that requires a positive row count.
-/
def layerNorm {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : Ref (m := m) (α := α) (leading ++ [width]))
    (gamma beta : Ref (m := m) (α := α) [width])
    (epsilon : α := normalizationEpsilon) :
    m (Ref (m := m) (α := α) (leading ++ [width])) :=
  match hRows : leading.size with
  | 0 => Ops.const (m := m) (α := α)
      (s := leading ++ [width]) (Tensor.full (leading ++ [width]) (0 : α))
  | rows + 1 => do
      let matrixShape : Shape := [rows + 1, width]
      let xMatrix ← Ops.apply (m := m) (α := α)
          (.reshape (s := leading ++ [width]) (t := matrixShape) (by
          simp [matrixShape, hRows])) (x, ())
      let yMatrix ← Ops.apply (m := m) (α := α)
          (.layerNorm (seqLen := rows + 1) (embedDim := width) (Nat.succ_pos rows) hWidth
            (lead := [rows + 1]) (x := [rows + 1, width]) (Shape.size_vector _).symm rfl)
          (xMatrix, gamma, beta, .data
            (Ops.dataConst (m := m) (α := α) (Tensor.scalar epsilon)), ())
      Ops.apply (m := m) (α := α)
          (.reshape (s := matrixShape) (t := leading ++ [width]) (by
          simp [matrixShape, hRows])) (yMatrix, ())

/-- Self-attention over any prefix shape, with an optional output-feature bias. -/
def attention {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {n numHeads dModel headDim : Nat} (hN : n ≠ 0)
    (wq wk wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
    (x : Ref (m := m) (α := α) (leadingShape ++ [n, dModel]))
    (mask : Option (Tensor Bool [n, n]) := none)
    (outputBias : Option (Ref (m := m) (α := α) [dModel]) := none) :
    m (Ref (m := m) (α := α) (leadingShape ++ [n, dModel])) := do
  let output ← do
    match batchEq : leadingShape.size with
    | 0 =>
        Ops.const (m := m) (α := α)
          (s := leadingShape ++ [n, dModel])
          (Tensor.full (leadingShape ++ [n, dModel]) (0 : α))
    | batch + 1 => do
        let xFlat ← Ops.apply (m := m) (α := α)
            (.reshape (s := leadingShape ++ [n, dModel]) (t := [batch + 1, n, dModel]) (by
            simp [Shape.size_append, batchEq])) (x, ())
        let yFlat ← Ops.apply (m := m) (α := α)
          (.multiheadAttention (batch := [batch + 1]) (n := n) (numHeads := numHeads)
            (dModel := dModel) (headDim := headDim) hN mask)
          (wq, wk, wv, wo, xFlat, ())
        Ops.apply (m := m) (α := α)
            (.reshape (s := [batch + 1, n, dModel]) (t := leadingShape ++ [n, dModel]) (by
            simp [Shape.size_append, batchEq])) (yFlat, ())
  match outputBias with
  | none => pure output
  | some bias => do
      let cb : Shape.CanBroadcastTo [dModel] (leadingShape ++ [n, dModel]) := by
        simpa only [List.append_assoc, List.cons_append, List.nil_append] using
          Shape.canBroadcastTo_concat_right (leadingShape ++ [n]) [dModel]
      let biasFull ← Ops.apply (m := m) (α := α)
        (.broadcast cb.dims cb.broadcastDims) (bias, ())
      Ops.apply (m := m) (α := α) .add (output, biasFull, ())

/-- An execution-polymorphic differentiable tensor program. -/
abbrev _root_.TorchLean.Program (α : Type) [Storage α] [Context α] (ss : List Shape) (τ : Shape) :
    Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s ↦ Ref (m := m) (α := α) s) ss (m (Ref (m := m) (α := α) τ))

/--
An execution-polymorphic program with differentiable tensors followed by non-differentiable data
tensors. The data tensors have their own element type `β`, so a token model can use `Fin vocab`,
which makes an out-of-range token unrepresentable.
-/
abbrev _root_.TorchLean.Program.Data (α β : Type) [Storage α] [Storage β]
    [Context α]
    (ss dataSs : List Shape) (τ : Shape) : Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s ↦ Ref (m := m) (α := α) s) ss
      (CurriedRef (fun s ↦ Ops.DataRef
        (m := m) (α := α) β s) dataSs (m (Ref (m := m) (α := α) τ)))

end nn.functional
end TorchLean
