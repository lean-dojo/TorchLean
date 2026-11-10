/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer
public import TorchLean.NN.Functional

/-!
# TorchLean NN: Operator Layers

`op o` is the parameter-free layer of one registered operator `o` with one value input, and
`Activation.Kind.op` names the operator of each activation kind. `square` multiplies its input by
itself and `sum` adds every entry. `dropout` is inverted dropout through `Op.dropout` in training
mode and the identity in evaluation mode.
-/

@[expose] public section

namespace TorchLean
namespace nn.Layer

open TorchLean.Tensor
open Ops

/--
The parameter-free layer of the registered operator `o` on one value input. Its forward pass is
one `Ops.apply o` call, which returns the value output of `o` and drops any index or Boolean
output after it. `validateConfig` runs `Op.check` on the attributes of `o`, and whether `o` runs
is up to the interpreter. The hypothesis `o.need = .none` supplies the empty capability record at
every scalar type.

PyTorch analogues: the parameter-free modules such as `torch.nn.ReLU`, `torch.nn.Softmax(dim)` and
`torch.nn.Flatten`, written `op .relu`, `op (.softmax d)` and `op (.reshape h)`.
-/
def op {σ τ : Shape} {rest : List Port} (o : Op [.value σ] (.value τ :: rest))
    (h : o.need = .none := by rfl) : Layer σ τ :=
  { kind := (Op.encode ⟨_, _, o⟩).tag
    stateShapes := []
    initState := .nil
    validateConfig := o.check
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      letI : Op.Supported α o :=
        { caps := Op.Need.Supplied.empty.caps
          available := by rw [h]; trivial }
      Ops.apply (m := m) (α := α) o (x, ()) }

/--
Pointwise square `x ↦ x^2` (no parameters).

PyTorch analogy: `torch.square(x)` / `x.square()`.
-/
def square {s : Shape} : Layer s s :=
  { kind := "Square"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => functional.square (m := m) (α := α) (s := s) x
  }

/--
Sum-reduce all elements of the input to a scalar (no parameters).

PyTorch analogy: `x.sum()`.
-/
def sum {s : Shape} : Layer s Shape.scalar :=
  { kind := "Sum"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (Ref.cast (Shape.reduce_finRange s) <$>
              Ops.apply (m := m) (α := α)
                (.reduce (s := s) .sum (List.finRange _) false
                  (fun h => by rcases h with h | h <;> cases h)) (x, ()) : m _)
  }

/--
Dropout layer controlled by `Mode`, with the drop probability `p` of its configuration, which
`validateConfig` requires to be finite and in `[0, 1]`.

- In `Mode.train`, the layer applies `nn.functional.dropout`, which runs `Op.dropout` on a uniform
  sample drawn under `seed`. The two ends draw no random key: at `p = 0` the layer returns its
  input, and at `p = 1` it applies `Op.dropout` to a zero sample, which makes every entry `+0`.
- In `Mode.eval`, it is the identity.

PyTorch analogy: `torch.nn.Dropout(p)` / `torch.nn.functional.dropout(x, p, training=...)`.
-/
def dropout {s : Shape} (p : Float) (seed : Nat := 0) : Layer s s :=
  { kind := s!"Dropout(p={p})"
    stateShapes := []
    initState := .nil
    validateConfig :=
      if functional.inUnitInterval p then
        pure ()
      else
        .error s!"Dropout: probability must be finite and in [0, 1], got {p}"
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          (show m (Ref (m := m) (α := α) s) from
            if mode == .eval || p == 0.0 then
              pure x
            else if p == 1.0 then do
              let sample ← Ops.const (m := m) (α := α) (Tensor.zeros s)
              Ops.apply (m := m) (α := α) (.dropout (s := s))
                (x, sample, .data
                  (Ops.dataConst (m := m) (α := α) (Tensor.scalar 1)), ())
            else
              functional.dropout (m := m) (α := α) (s := s) x p seed)
  }

end nn.Layer
end TorchLean

namespace TorchLean.Activation.Kind

/-- The registered operator of an activation kind, with the tanh approximation for GELU. -/
def op {s : Shape} : Activation.Kind → Op [.value s] [.value s]
  | .relu => .relu
  | .geluTanh => .gelu .tanh
  | .silu => .silu
  | .tanh => .tanh
  | .sigmoid => .sigmoid

/-- No activation operator reads a scalar capability. -/
theorem need_op (k : Activation.Kind) {s : Shape} : (k.op (s := s)).need = .none := by
  cases k <;> rfl

/-- The empty capability record is an `Op.Supported` witness for every activation operator. -/
instance {α : Type} (k : Activation.Kind) {s : Shape} :
    Op.Supported α (k.op (s := s)) :=
  { caps := Op.Need.Supplied.empty.caps
    available := by rw [k.need_op]; trivial }

end TorchLean.Activation.Kind
