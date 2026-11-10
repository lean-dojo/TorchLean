/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the convolutions and poolings

Each lemma here states one arm of `Op.rule?` for `conv`, `convTranspose`, `pool` or `smoothMaxPool`,
and `Op.rule?_conv_batch` gives the rule of `conv` over one batch axis.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.conv`. -/
theorem rule?_conv [Op.Capabilities α] {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat) {lead x y : Shape}
    (hlead : s.extent (dims.input ⟨0, by omega⟩) = lead.size)
    (hx : x = s.expandAxis (dims.input ⟨0, by omega⟩) lead)
    (hy : y = (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
      (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩) lead) :
    (Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.reshape
            (Tensor.conv (Tensor.reshape xs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx)) xs.1
                dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
              Tensor.broadcast xs.2.1
                (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1) (fun _ =>
                  dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
                (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1))
            _ (Shape.size_conv_expandAxis hlead hy), ())
        backward := fun xs g =>
          (Tensor.convBackwardWeight (Tensor.reshape xs.2.2.1 _
              (Shape.size_eq_of_expandAxis hlead hx)) dims stride padLow padHigh dilation groups
              (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm),
            Tensor.castShape
                (Tensor.reduce .sum
                  (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm)
                (Shape.otherAxes (s.conv k dims stride padLow padHigh (fun _ =>
                1) dilation 1) [dims.output stride padLow padHigh (fun _ =>
                1) dilation 1 ⟨1, by omega⟩]) false)
              (Shape.reduce_conv_otherAxes dims stride padLow padHigh (fun _ => 1) dilation 1),
            Tensor.reshape (Tensor.convBackwardInput xs.1 dims stride padLow padHigh dilation groups
              (Tensor.reshape g.1 _ (Shape.size_conv_expandAxis hlead hy).symm)) _
              (Shape.size_eq_of_expandAxis hlead hx).symm, ())
        jvp? := some fun xs dxs =>
          let bias (b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]) :=
            Tensor.broadcast b (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1)
              (fun _ => dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
              (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1)
          (Tensor.reshape
            ((Tensor.conv (Tensor.reshape dxs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx)) xs.1
                  dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
                bias (Tensor.full _ (0 : α))) +
              (Tensor.conv (Tensor.reshape xs.2.2.1 _ (Shape.size_eq_of_expandAxis hlead hx))
                  dxs.1 dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
                bias dxs.2.1))
            _ (Shape.size_conv_expandAxis hlead hy), ()) } :=
  rfl

/-- The rule of `Op.conv` over one batch axis: the input at `s` and the result at the output shape,
the batch axis standing for itself. -/
theorem rule?_conv_batch [Op.Capabilities α] {s k : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (hlead : s.extent (dims.input ⟨0, by omega⟩) =
      Shape.size [s.extent (dims.input ⟨0, by omega⟩)])
    (hx : s = s.expandAxis (dims.input ⟨0, by omega⟩) [s.extent (dims.input ⟨0, by omega⟩)])
    (hy : s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1 =
      (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1).expandAxis
        (dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨0, by omega⟩)
        [s.extent (dims.input ⟨0, by omega⟩)]) :
    (Op.conv dims stride padLow padHigh dilation groups hlead hx hy).rule? (α := α) = some
      { forward := fun xs =>
          ((Tensor.conv xs.2.2.1 xs.1 dims stride padLow padHigh (fun _ =>
              1) dilation groups 1) +
            (Tensor.broadcast xs.2.1 (s.conv k dims stride padLow padHigh (fun _ =>
                1) dilation 1) (fun _ =>
                dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
              (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1)), ())
        backward := fun xs g =>
          (Tensor.convBackwardWeight xs.2.2.1 dims stride padLow padHigh dilation groups g.1,
            Tensor.castShape
                (Tensor.reduce .sum g.1
                (Shape.otherAxes (s.conv k dims stride padLow padHigh (fun _ =>
                1) dilation 1) [dims.output stride padLow padHigh (fun _ =>
                1) dilation 1 ⟨1, by omega⟩]) false)
              (Shape.reduce_conv_otherAxes dims stride padLow padHigh (fun _ => 1) dilation 1),
            Tensor.convBackwardInput xs.1 dims stride padLow padHigh dilation groups g.1, ())
        jvp? := some fun xs dxs =>
          let bias (b : Tensor α [k.extent (dims.kernel ⟨0, by omega⟩)]) :=
            Tensor.broadcast b (s.conv k dims stride padLow padHigh (fun _ => 1) dilation 1)
              (fun _ => dims.output stride padLow padHigh (fun _ => 1) dilation 1 ⟨1, by omega⟩)
              (dims.broadcastDims_conv stride padLow padHigh (fun _ => 1) dilation 1)
          ((Tensor.conv dxs.2.2.1 xs.1 dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
              bias (Tensor.full _ (0 : α))) +
            (Tensor.conv xs.2.2.1 dxs.1 dims stride padLow padHigh (fun _ => 1) dilation groups 1 +
              bias dxs.2.1), ()) } :=
  rfl

/-- The rule of `Op.pool`. -/
theorem rule?_pool [Op.Capabilities α] {s : Shape} (op : PoolOp) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    (Op.pool op dims kernel stride padLow padHigh dilation ceilMode).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.pool op xs.1 dims kernel stride padLow padHigh dilation ceilMode, ())
        backward := fun xs g =>
          (Tensor.poolBackward op xs.1 dims kernel stride padLow padHigh dilation ceilMode g.1,
              ())
        jvp? := some fun xs dxs =>
          (Tensor.poolJVP op xs.1 dxs.1 dims kernel stride padLow padHigh dilation ceilMode,
            ()) } :=
  rfl

/-- The rule of `Op.smoothMaxPool`. -/
theorem rule?_smoothMaxPool [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    (Op.smoothMaxPool dims kernel stride padLow padHigh dilation ceilMode).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.smoothMaxPool xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode, ())
        backward := fun xs g =>
          (Tensor.smoothMaxPoolBackward xs.2.1.item xs.1 dims kernel stride padLow padHigh dilation
            ceilMode g.1, (), ())
        validate := fun xs =>
          if xs.2.1.item == 0 || !(xs.2.1.item - xs.2.1.item == 0) then
            .error "autograd: smooth_max_pool requires finite nonzero beta"
          else .ok ()
        jvp? := some fun xs dxs =>
          (Tensor.smoothMaxPoolJVP xs.2.1.item xs.1 dxs.1 dims kernel stride padLow padHigh
            dilation ceilMode, ()) } :=
  rfl

/-- The rule of `Op.convTranspose`. -/
theorem rule?_convTranspose [Op.Capabilities α] {s k : Shape} {r : Nat}
    (dims : Shape.ConvDims s k r)
    (stride padLow padHigh outputPadding dilation : Fin r → Nat) (groups : Nat) :
    (Op.convTranspose dims stride padLow padHigh outputPadding dilation groups).rule?
      (α := α) = some
      { forward := fun xs =>
          ((Tensor.convTranspose xs.2.2.1 xs.1 dims stride padLow padHigh outputPadding
              dilation groups) +
            (Tensor.broadcast xs.2.1
                (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)
                (fun _ =>
                dims.transposeOutput stride padLow padHigh outputPadding dilation groups ⟨1,
                by omega⟩)
              (dims.broadcastDims_convTranspose stride padLow padHigh outputPadding dilation
                  groups)), ())
        backward := fun xs g =>
          (Tensor.convTransposeBackwardWeight xs.2.2.1 dims stride padLow padHigh outputPadding
              dilation groups g.1,
            Tensor.castShape
                (Tensor.reduce .sum g.1
                (Shape.otherAxes
                (s.convTranspose k dims stride padLow padHigh outputPadding dilation
                groups) [dims.transposeOutput stride padLow padHigh outputPadding dilation groups
                ⟨1, by omega⟩]) false)
              (Shape.reduce_convTranspose_otherAxes dims stride padLow padHigh outputPadding
                  dilation
                groups),
            Tensor.convTransposeBackwardInput xs.1 dims stride padLow padHigh outputPadding dilation
              groups g.1, ())
        jvp? := some fun xs dxs =>
          let bias (b : Tensor α [groups * k.extent (dims.kernel ⟨0, by omega⟩)]) :=
            Tensor.broadcast b
              (s.convTranspose k dims stride padLow padHigh outputPadding dilation groups)
              (fun _ =>
                dims.transposeOutput stride padLow padHigh outputPadding dilation groups ⟨1,
                by omega⟩)
              (dims.broadcastDims_convTranspose stride padLow padHigh outputPadding dilation groups)
          (((Tensor.convTranspose dxs.2.2.1 xs.1 dims stride padLow padHigh outputPadding
                dilation groups) +
              bias (Tensor.full _ (0 : α))) +
            ((Tensor.convTranspose xs.2.2.1 dxs.1 dims stride padLow padHigh outputPadding
                dilation groups) +
              bias dxs.2.1), ()) } :=
  rfl

end Op

end TorchLean
