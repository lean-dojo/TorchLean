/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the attention operators

Each lemma here states one arm of `Op.rule?` for `attention`, `attentionFromScores` or
`multiheadAttention`. `Op.multiheadAttention_forward_dim` and `Op.multiheadAttention_backward_dim`
reduce `multiheadAttention` along a leading axis to its slices.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.attentionFromScores`. -/
theorem rule?_attentionFromScores [Op.Capabilities α] {batch : Shape} {rows keys width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) :
    (Op.attentionFromScores (batch := batch) (width := width) mask).rule? (α := α) = some
      { forward := fun xs =>
          (Attention.values (Attention.weights xs.1 mask) xs.2.1
            (some xs.2.2.1), ())
        backward := fun xs g =>
          let weights := Attention.weights xs.1 mask
          let (ds, dv, dm) := Attention.valuesBackward weights xs.2.1 xs.2.2.1 g.1
          (ds, dv, dm, ())
        prepare := fun xs flags =>
          let weights := Attention.weights xs.1 mask
          ⟨(Attention.values weights xs.2.1 (some xs.2.2.1), ()), fun g =>
            let (ds, dv, dm) := Attention.valuesBackward weights xs.2.1 xs.2.2.1 g.1
            Tangents.select _ flags (ds, dv, dm, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Attention.valuesJVP (Attention.weights xs.1 mask) xs.2.1
            xs.2.2.1 dxs.1 dxs.2.1 dxs.2.2.1, ()) } :=
  rfl

/-- The rule of `Op.multiheadAttention`. -/
theorem rule?_multiheadAttention [Op.Capabilities α] {batch : Shape}
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n])) :
    (Op.multiheadAttention (batch := batch) (numHeads := numHeads) (dModel := dModel)
        (headDim := headDim) h1 mask).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.mapLeading batch (fun x => MultiHeadAttention.forward (hQ := h1) (hK := h1)
            (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩) (query := x) (keyValue := x)
            (mask := mask))
            xs.2.2.2.2.1, ())
        backward := fun xs g =>
          let mha : MultiHeadAttention α numHeads dModel headDim :=
            ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩
          let gradients := MultiHeadAttention.backwardLeading h1 mha mask batch xs.2.2.2.2.1 g.1
          let parameters := MultiHeadAttention.sum mha gradients.2
          (parameters.queryWeight, parameters.keyWeight, parameters.valueWeight,
            parameters.outputWeight, gradients.1, ())
        order := [4, 0, 1, 2, 3]
        order_perm := (by decide : ([4, 0, 1, 2, 3] : List (Fin 5)).Perm (List.finRange 5))
        jvp? := some fun xs dxs =>
          (Tensor.zipEach batch [n, dModel] (fun (x dx : Tensor α [n, dModel]) =>
            MultiHeadAttention.jvp (h1 := h1) (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩)
              (dmha := ⟨dxs.1, dxs.2.1, dxs.2.2.1, dxs.2.2.2.1⟩) (x := x) (dx := dx)
              (mask := mask)) xs.2.2.2.2.1 dxs.2.2.2.2.1, ()) } :=
  rfl

/-- With no leading axes `Op.multiheadAttention` is the attention of one sequence: the value
`MultiHeadAttention.forward` with the input as query and key/value, and the pullback
`MultiHeadAttention.backward`. -/
theorem rule?_multiheadAttention_single [Op.Capabilities α] {n numHeads dModel headDim : Nat}
    (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n])) :
    (Op.multiheadAttention (batch := []) (numHeads := numHeads) (dModel := dModel)
        (headDim := headDim) h1 mask).rule? (α := α) = some
      { forward := fun xs =>
          (MultiHeadAttention.forward (hQ := h1) (hK := h1)
            (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩) (query := xs.2.2.2.2.1)
            (keyValue := xs.2.2.2.2.1) (mask := mask), ())
        backward := fun xs g =>
          let gradients := MultiHeadAttention.backward (h1 := h1)
            (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩) (x := xs.2.2.2.2.1) (mask := mask)
            (gradOutput := g.1)
          (gradients.parameters.queryWeight, gradients.parameters.keyWeight,
            gradients.parameters.valueWeight, gradients.parameters.outputWeight, gradients.input,
                ())
        order := [4, 0, 1, 2, 3]
        order_perm := (by decide : ([4, 0, 1, 2, 3] : List (Fin 5)).Perm (List.finRange 5))
        jvp? := some fun xs dxs =>
          (MultiHeadAttention.jvp (h1 := h1) (mha := ⟨xs.1, xs.2.1, xs.2.2.1, xs.2.2.2.1⟩)
            (dmha := ⟨dxs.1, dxs.2.1, dxs.2.2.1, dxs.2.2.2.1⟩) (x := xs.2.2.2.2.1)
            (dx := dxs.2.2.2.2.1) (mask := mask), ()) } :=
  rfl

/-- Along a leading axis the value of `Op.multiheadAttention` at index `i` is its value on the
`i`-th slice of the input. -/
theorem multiheadAttention_forward_dim [Op.Capabilities α] {m : Nat} {rest : Shape}
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (wq wk wv : Tensor α [dModel, numHeads * headDim]) (wo : Tensor α [numHeads * headDim, dModel])
    (x : Tensor α ((Shape.dim m rest) ++ [n, dModel])) :
    (((Op.multiheadAttention (batch := .dim m rest) (numHeads := numHeads) (dModel := dModel)
        (headDim := headDim) h1 mask).rule? (α := α)).get rfl).forward (wq, wk, wv, wo, x, ()) =
      (Tensor.dim fun i =>
        ((((Op.multiheadAttention (batch := rest) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := α)).get rfl).forward
            (wq, wk, wv, wo, Tensor.unstack x i, ())).1, ()) :=
  rfl

/-- Along a leading axis the input's gradient of `Op.multiheadAttention` at index `i` is its input
gradient on the `i`-th slices of the input and of the cotangent. The projection gradients add those
of the sequences in row-major order (`rule?_multiheadAttention`), one entry per sequence
(`MultiHeadAttention.backwardLeading_scalar`, `MultiHeadAttention.backwardLeading_dim`). -/
theorem multiheadAttention_backward_dim [Op.Capabilities α] {m : Nat} {rest : Shape}
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) (mask : Option (Tensor Bool [n, n]))
    (wq wk wv : Tensor α [dModel, numHeads * headDim]) (wo : Tensor α [numHeads * headDim, dModel])
    (x g : Tensor α ((Shape.dim m rest) ++ [n, dModel])) :
    ((((Op.multiheadAttention (batch := .dim m rest) (numHeads := numHeads) (dModel := dModel)
        (headDim := headDim) h1 mask).rule? (α := α)).get rfl).backward (wq, wk, wv, wo, x, ())
        (g, ())).2.2.2.2.1 =
      Tensor.dim fun i =>
        ((((Op.multiheadAttention (batch := rest) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim) h1 mask).rule? (α := α)).get rfl).backward
            (wq, wk, wv, wo, Tensor.unstack x i, ()) (Tensor.unstack g i, ())).2.2.2.2.1 := by
  show (MultiHeadAttention.backwardLeading h1 ⟨wq, wk, wv, wo⟩ mask (.dim m rest) x g).1 = _
  rw [MultiHeadAttention.backwardLeading_dim]
  rfl

/-- The rule of `Op.attention`. -/
theorem rule?_attention [Op.Capabilities α] {batch : Shape} {rows keys depth width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) (schedule : Softmax.Schedule) :
    (Op.attention (batch := batch) (depth := depth) (width := width) mask schedule).rule? (α :=
        α) = some
      { forward := fun xs =>
          (Tensor.attention xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none schedule, ())
        backward := fun xs g =>
          let gradients :=
              Tensor.attentionBackward xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none g.1
          (gradients.1, gradients.2.1, gradients.2.2.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.attentionJVP xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item mask none dxs.1 dxs.2.1
            dxs.2.2.1 none, ()) } :=
  rfl

end Op

end TorchLean
