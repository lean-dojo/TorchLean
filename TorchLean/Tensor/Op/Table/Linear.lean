/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the linear algebra operators

Each lemma here states one arm of `Op.rule?` for `linear`, `matmul`, `scaledMatmul`, `contract` or a
real Fourier transform (`rfft`, `irfft`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.linear` at the empty leading shape: the vector rule. -/
theorem rule?_linear [Op.Capabilities α] {inDim outDim : Nat} :
    (Op.linear (lead := []) (inDim := inDim) (outDim := outDim)).rule? (α := α) = some
      { forward := fun xs => (Linear.forward { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ())
        backward := fun xs g =>
          (Linear.weightsDeriv xs.2.2.1 g.1, g.1, Linear.inputDeriv xs.1 g.1, ())
        prepare := fun xs flags =>
          ⟨(Linear.forward { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ()), fun g =>
          (if flags.1 then some (Linear.weightsDeriv xs.2.2.1 g.1) else none,
            if flags.2.1 then some g.1 else none,
            if flags.2.2.1 then some (Linear.inputDeriv xs.1 g.1) else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Linear.forward { weights := dxs.1, bias := dxs.2.1 } xs.2.2.1 +
            Linear.forward { weights := xs.1, bias := Tensor.full _ (0 : α) } dxs.2.2.1, ()) } :=
  rfl

/-- `Op.linear` has a rule at every leading shape. -/
theorem rule?_linear_isSome [Op.Capabilities α] {lead : Shape} {inDim outDim : Nat} :
    ((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := α)).isSome := by
  cases lead <;> rfl

/-- At every leading shape the forward value of `Op.linear` is `Linear.forwardLeading`. -/
theorem rule?_linear_forward [Op.Capabilities α] {lead : Shape} {inDim outDim : Nat}
    (h : ((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := α)).isSome)
    (W : Tensor α [outDim, inDim]) (b : Tensor α [outDim]) (x : Tensor α (lead ++ [inDim])) :
    (((Op.linear (lead := lead) (inDim := inDim) (outDim := outDim)).rule? (α := α)).get h).forward
        (W, b, x, ()) = (Linear.forwardLeading lead { weights := W, bias := b } x, ()) := by
  cases lead <;> rfl

/-- The rule of `Op.linear` at a leading shape of positive rank. -/
theorem rule?_linear_leading [Op.Capabilities α] {n : Nat} {rest : Shape} {inDim outDim : Nat} :
    (Op.linear (lead := .dim n rest) (inDim := inDim) (outDim := outDim)).rule? (α := α) = some
      { forward := fun xs =>
          (Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ())
        backward := fun xs g =>
          (Linear.weightsDerivLeading (.dim n rest) xs.2.2.1 g.1,
            Linear.biasDerivLeading (.dim n rest) g.1,
            Linear.inputDerivLeading (.dim n rest) xs.1 g.1, ())
        prepare := fun xs flags =>
          ⟨(Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := xs.2.1 } xs.2.2.1, ()),
            fun g =>
            (if flags.1 then some (Linear.weightsDerivLeading (.dim n rest) xs.2.2.1 g.1)
              else none,
              if flags.2.1 then some (Linear.biasDerivLeading (.dim n rest) g.1) else none,
              if flags.2.2.1 then
                some (Linear.inputDerivLeading (.dim n rest) xs.1 g.1)
              else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Linear.forwardLeading (.dim n rest) { weights := dxs.1, bias := dxs.2.1 } xs.2.2.1 +
            Linear.forwardLeading (.dim n rest) { weights := xs.1, bias := Tensor.full _ (0 : α) }
              dxs.2.2.1, ()) } :=
  rfl

/-- The rule of `Op.matmul`. -/
theorem rule?_matmul [Op.Capabilities α] {batchA batchB batch : Shape} {m n p : Nat} (lv rv : Bool)
    (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    (Op.matmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.reshape (Tensor.batchMatmul broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)))
            _ (Shape.size_matmulOut hl hr _).symm, ())
        backward := fun xs g =>
          let gradients := Tensor.matmulBackward broadcastA broadcastB
            (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
            (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
            (Tensor.reshape g.1 _ (Shape.size_matmulOut hl hr _))
          (Tensor.reshape gradients.1 _ (Shape.size_matmulLeft hl _ _).symm,
            Tensor.reshape gradients.2 _ (Shape.size_matmulRight hr _ _).symm, ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (Tensor.batchMatmul broadcastA broadcastB
                (Tensor.reshape dxs.1 _ (Shape.size_matmulLeft hl _ _))
                (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)) +
              Tensor.batchMatmul broadcastA broadcastB
                (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
                (Tensor.reshape dxs.2.1 _ (Shape.size_matmulRight hr _ _)))
            _ (Shape.size_matmulOut hl hr _).symm, ()) } :=
  rfl

/-- The rule of `Op.matmul` with matrix operands. -/
theorem rule?_matmul_matrix [Op.Capabilities α] {batchA batchB batch : Shape} {m n p : Nat}
    (hl : false = true → m = 1) (hr : false = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    (Op.matmul (m := m) (n := n) (p := p) false false hl hr broadcastA broadcastB).rule? (α := α) =
      some
      { forward := fun xs => (Tensor.batchMatmul broadcastA broadcastB xs.1 xs.2.1, ())
        backward := fun xs g =>
          let gradients := Tensor.matmulBackward broadcastA broadcastB xs.1 xs.2.1 g.1
          (gradients.1, gradients.2, ())
        jvp? := some fun xs dxs =>
          (Tensor.batchMatmul broadcastA broadcastB dxs.1 xs.2.1 +
            Tensor.batchMatmul broadcastA broadcastB xs.1 dxs.2.1, ()) } :=
  rfl

/-- The rule of `Op.scaledMatmul`. -/
theorem rule?_scaledMatmul [Op.Capabilities α] {batchA batchB batch : Shape} {m n p : Nat}
    (lv rv : Bool) (hl : lv = true → m = 1) (hr : rv = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    (Op.scaledMatmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB).rule? (α := α) =
      some
      { forward := fun xs =>
          (Tensor.reshape (Attention.scaledMatmul broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _)) xs.2.2.1.item)
            _ (Shape.size_matmulOut hl hr _).symm, ())
        backward := fun xs g =>
          let gradients := Attention.scaledMatmulBackward broadcastA broadcastB
            (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
            (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
            (Tensor.reshape g.1 _ (Shape.size_matmulOut hl hr _)) xs.2.2.1.item
          (Tensor.reshape gradients.1 _ (Shape.size_matmulLeft hl _ _).symm,
            Tensor.reshape gradients.2 _ (Shape.size_matmulRight hr _ _).symm, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.reshape (Attention.scaledMatmulJVP broadcastA broadcastB
              (Tensor.reshape xs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape dxs.1 _ (Shape.size_matmulLeft hl _ _))
              (Tensor.reshape xs.2.1 _ (Shape.size_matmulRight hr _ _))
              (Tensor.reshape dxs.2.1 _ (Shape.size_matmulRight hr _ _)) xs.2.2.1.item)
            _ (Shape.size_matmulOut hl hr _).symm, ()) } :=
  rfl

/-- The rule of `Op.scaledMatmul` with matrix operands. -/
theorem rule?_scaledMatmul_matrix [Op.Capabilities α] {batchA batchB batch : Shape} {m n p : Nat}
    (hl : false = true → m = 1) (hr : false = true → p = 1)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    (Op.scaledMatmul (m := m) (n := n) (p := p) false false hl hr broadcastA
        broadcastB).rule? (α := α) = some
      { forward := fun xs =>
          (Attention.scaledMatmul broadcastA broadcastB xs.1 xs.2.1 xs.2.2.1.item, ())
        backward := fun xs g =>
          let gradients :=
            Attention.scaledMatmulBackward broadcastA broadcastB xs.1 xs.2.1 g.1 xs.2.2.1.item
          (gradients.1, gradients.2, (), ())
        jvp? := some fun xs dxs =>
          (Attention.scaledMatmulJVP broadcastA broadcastB xs.1 dxs.1 xs.2.1 dxs.2.1
            xs.2.2.1.item, ()) } :=
  rfl

/-- The rule of `Op.contract`. -/
theorem rule?_contract [Op.Capabilities α] {s t : Shape}
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.ContractValid s t dims batchDims) :
    (Op.contract dims batchDims h).rule? (α := α) = some
      { forward := fun xs => (Tensor.contract xs.1 xs.2.1 dims batchDims _ h rfl, ())
        backward := fun xs g =>
          (Tensor.contractBackwardLeft dims batchDims h g.1 xs.2.1,
            Tensor.contractBackwardRight dims batchDims h xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.contract dxs.1 xs.2.1 dims batchDims _ h rfl +
            Tensor.contract xs.1 dxs.2.1 dims batchDims _ h rfl, ()) } :=
  rfl

/-- The rule of `Op.rfft`. -/
theorem rule?_rfft [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n) :
    (Op.rfft dim n hn).rule? (α := α) = some
      { forward := fun xs => (Tensor.rfft dim n xs.1, ())
        backward := fun _ g => (Tensor.rfftBackward dim n g.1, ())
        jvp? := some fun _ dxs => (Tensor.rfft dim n dxs.1, ()) } :=
  rfl

/-- The rule of `Op.irfft`. -/
theorem rule?_irfft [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (n : Nat) (hn : 0 < n) :
    (Op.irfft dim n hn).rule? (α := α) = some
      { forward := fun xs => (Tensor.irfft dim n xs.1, ())
        backward := fun _ g => (Tensor.irfftBackward dim n g.1, ())
        jvp? := some fun _ dxs => (Tensor.irfft dim n dxs.1, ()) } :=
  rfl

end Op

end TorchLean
