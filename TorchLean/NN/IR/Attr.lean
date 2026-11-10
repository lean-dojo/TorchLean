/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Port

/-!
# Attribute records of operator nodes

The per-axis window of pooling (`WindowConfig`) and the shape metadata of convolution
(`ConvConfig`), used by the model recorder. Their shape contracts are in
`TorchLean.NN.IR.OpContracts`, and the recording of registered operators (`TorchLean.NN.IR.Record`)
builds the windows of its pooling and convolution steps.

`Values.get` reads output `k` of a list of values. The typed graphs of `TorchLean.NN.IR.OpGraph`
read each argument of a step with it.
-/

@[expose] public section

namespace TorchLean

/-- Per-axis geometry for pooling and convolution operators.

All three tensors describe the same spatial suffix of the input tensor. With one entry per axis, a
frontend cannot silently impose one stride or padding on every axis.
-/
structure IR.WindowConfig where
  /-- Number of spatial axes. -/
  spatialRank : Nat
  /-- Window extent along each spatial axis. -/
  kernel : Tensor Nat [spatialRank]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [spatialRank]
  /-- Symmetric padding along each spatial axis. -/
  padding : Tensor Nat [spatialRank]
  deriving Repr, DecidableEq

/-- Shape metadata for an arbitrary-dimensional convolution.

Axes before `channelAxis` are preserved and mapped independently. The channel axis takes the extent
`outChannels`, and each later axis is spatial, with the kernel, stride and padding of the inherited
`WindowConfig` and the `dilation` and `paddingAfter` below.
-/
structure IR.ConvConfig extends WindowConfig where
  /-- Spacing between kernel elements along each spatial axis. -/
  dilation : Tensor Nat [spatialRank] :=
    Tensor.dim fun _ => Tensor.scalar 1
  /-- Zero padding after the input along each spatial axis.

  The inherited `padding` field is the padding before the input. Storing both sides lets one
  record describe asymmetric padding at any rank, with no rank-specific convolution variant. -/
  paddingAfter : Tensor Nat [spatialRank] := padding
  /-- Number of channel groups. `1` is an ordinary dense convolution. -/
  groups : Nat := 1
  /-- Axis containing the input channels. -/
  channelAxis : Nat
  /-- Expected extent of the input-channel axis. -/
  inChannels : Nat
  /-- Extent of the output-channel axis. -/
  outChannels : Nat
  deriving Repr, DecidableEq

/-- The output `k` of a list of values. -/
def Values.get {α : Type} [Storage α] : {ps : List Port} → Values α ps → (k : Fin ps.length) →
    (ps.get k).Carrier α
  | _ :: _, (x, _), ⟨0, _⟩ => x
  | _ :: _, (_, xs), ⟨k + 1, h⟩ => Values.get xs ⟨k, Nat.lt_of_succ_lt_succ h⟩

end TorchLean
