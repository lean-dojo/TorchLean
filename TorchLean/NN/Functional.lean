/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program

@[expose] public section


namespace TorchLean
namespace nn.functional

open TorchLean.Tensor
open Ops

/-!
# Functional Core

Small functional helpers built from the primitive `Ops` API: elementwise
helpers, broadcasting, embedding lookup, reductions and dropout. Eager and typed graph execution
share them, so each stays close to the primitive operation it wraps, and `nn.functional` exports
them.
-/

/-! ## Elementwise helpers -/

/--
Elementwise square: $x\mapsto x^2$, recorded as `.mul` of `x` with itself.

PyTorch analogue: `torch.square`.
-/
def square {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  -- Not `.square`: eager CUDA execution has no LibTorch kernel for it (`LibTorch.Kernel.ofOp`), and
  -- the verifier recorder (`TorchLean.IR.record`) refuses it as outside its fragment.
  Ops.apply (m := m) (α := α) (.mul (s := s)) (x, x, ())

/-! ## Elementwise transcendentals for scientific forward models

Scientific forward models often combine affine terms with `exp`, `log`, `sin` or `cos`. The affine
helpers below are exported through `nn.functional`, and the transcendentals are the registered
operators `Op.exp`, `Op.log`, `Op.sin` and `Op.cos`, applied with `Ops.apply`. Each has a
registered backward rule, so a forward equation written once as a pure function can be
differentiated with reverse-mode `jacrev` and `grad`.

PyTorch analogues: `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, and `c·x` / `c·x + k` via
`torch.mul`/`torch.add` against scalars. -/

/-!
`Op.log` checks no input values (`Op.validate_eq_ok`). The eager CPU tape (`Tape.apply`), the IR
evaluator (`IR.Graph.eval`) and the typed graph nodes (`TypedGraph.GraphM.apply_forward`)
accept every input and compute its forward map `Tensor.log` (`Op.rule?_log`). At `Float` and
`Float32` that map is `Float.log` and `Float32.log`: `-∞` at `+0` and at `-0`, NaN below zero and
at NaN, and `+∞` at `+∞`, the values of IEEE 754-2019 9.2 and 6.2. The CUDA tape runs LibTorch's
`at::log` (`LibTorch.Buffer.log`) with no check, and the CUDA math library specifies the same
values for it. Over `ℝ` the map is `Real.log`, with `Real.log 0 = 0` and
`Real.log |x| = Real.log x` (`Real.log_zero`, `Real.log_abs`), so `Real.exp (Real.log x) = x`
needs `0 < x` (`Real.exp_log`). Use `Op.safeLog`, which computes `log (softplus x + ε)` for a
frozen scalar input `ε`, when a model needs an epsilon-protected log-like operation.
-/

/-- Add a constant scalar $c$ to every element: $x\mapsto x+c$. The constant is recorded with
`Ops.const` at scalar shape and broadcast to `s`. -/
def shift {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) := do
  let cs ← Ops.const (m := m) (α := α) (s := Shape.scalar)
    (Tensor.scalar c)
  let cb ← Ops.apply (m := m) (α := α)
      (.broadcast (s := Shape.scalar) (t := s) (Shape.scalar_canBroadcastTo s).dims
        (Shape.scalar_canBroadcastTo s).broadcastDims) (cs, ())
  Ops.apply (m := m) (α := α) (.add (s := s)) (x, cb, ())

/-- Scalar affine map $x\mapsto xc+k$: `Op.scale` by `c`, then `shift` by `k`. -/
def affine {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (c k : α) : m (Ref (m := m) (α := α) s) := do
  let sx ← Ops.apply (m := m) (α := α) (.scale (s := s))
      (x, .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar c)), ())
  shift (m := m) (α := α) (s := s) sx k

/-! ## Detach

Models stop gradients with the registered operator, `Ops.apply .detach (x, ())`, as
`VQVAE.Autograd.straightThrough` does. -/


/-! ## Broadcasting helpers -/

/-- Broadcast `x` to `t`, returning `x` itself with no broadcast node when `s = t`. -/
def broadcastUnlessSame {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s t : Shape} [Shape.BroadcastTo s t]
    (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) t) :=
  if h : s = t then
    pure (h ▸ x)
  else
    Ops.apply (m := m) (α := α)
        (.broadcast (s := s) (t := t) (Shape.BroadcastTo.proof).dims
          (Shape.BroadcastTo.proof).broadcastDims) (x, ())

/--
Broadcasting add: compute `x + y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.add` (broadcasting semantics).
-/
def addB {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : Ref (m := m) (α := α) s₁) (y : Ref (m := m) (α := α) s₂) :
    m (Ref (m := m) (α := α) t) := do
  let xb ← broadcastUnlessSame (m := m) (α := α) (s := s₁) (t := t) x
  let yb ← broadcastUnlessSame (m := m) (α := α) (s := s₂) (t := t) y
  Ops.apply (m := m) (α := α) (.add (s := t)) (xb, yb, ())

/--
Broadcasting multiply: compute `x * y` after broadcasting both inputs to the target shape `t`.

PyTorch analogue: `torch.mul` (broadcasting semantics).
-/
def mulB {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s₁ s₂ t : Shape} [Shape.BroadcastTo s₁ t] [Shape.BroadcastTo s₂ t]
    (x : Ref (m := m) (α := α) s₁) (y : Ref (m := m) (α := α) s₂) :
    m (Ref (m := m) (α := α) t) := do
  let xb ← broadcastUnlessSame (m := m) (α := α) (s := s₁) (t := t) x
  let yb ← broadcastUnlessSame (m := m) (α := α) (s := s₂) (t := t) y
  Ops.apply (m := m) (α := α) (.mul (s := t)) (xb, yb, ())

/-! ## Indexing helpers -/

/--
Embedding lookup for a tensor of bounded token ids of any shape `s`.

The gather primitive reads a flat vector of indices, so the ids are flattened, their rows gathered
from the table, and the result reshaped to the axes of `s` with the embedding width appended. The
element type `Fin vocabularySize` makes an out-of-range token unrepresentable.

PyTorch analogue: `torch.nn.functional.embedding(ids, weight)`.
-/
def embedding {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {vocabularySize embeddingWidth : Nat} {s : Shape}
    (weight : Ref (m := m) (α := α) [vocabularySize, embeddingWidth])
    (indices : Ops.DataRef
      (m := m) (α := α) (Fin vocabularySize) s) :
    m (Ref (m := m) (α := α) (s ++ [embeddingWidth])) := do
  -- The rows at the flattened ids: a gather along axis 0 of the table.
  let flatIds := Ops.mapData (m := m) (α := α)
    (fun x => Tensor.map (fun tuple => tuple ⟨0, Nat.one_pos⟩)
      (Tensor.selectIndices (s := [vocabularySize, embeddingWidth]) ⟨0, Shape.AxisInBounds.proof⟩
        (Tensor.reshape (source := s) x [s.size] (by simp)))) indices
  let gathered : Ref (m := m) (α := α) [s.size, embeddingWidth] ←
    Ref.cast (m := m) (α := α)
        (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
      Ops.apply (m := m) (α := α)
        (.gather (Shape.selectDims [vocabularySize, embeddingWidth] ⟨0, Shape.AxisInBounds.proof⟩)
          (Shape.selectBatch [vocabularySize, embeddingWidth] ⟨0, Shape.AxisInBounds.proof⟩ s.size)
          (.select [vocabularySize, embeddingWidth] ⟨0, Shape.AxisInBounds.proof⟩ s.size))
        (weight, flatIds, ())
  Ops.apply (m := m) (α := α)
      (.reshape (s := [s.size, embeddingWidth]) (t := s ++ [embeddingWidth])
        (by simp)) (gathered, ())

/-! ## Reductions -/

/-- Mean over all elements: `Op.mean` over every axis, the sum divided once by the element count.

An empty tensor divides by one and returns its zero sum. `torch.mean` returns NaN there.
-/
def mean {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
  {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ref.cast (Shape.reduce_finRange s) <$>
    Ops.apply (m := m) (α := α) (.mean (s := s) (List.finRange _) false)
      (x, ())

/-! ## Dropout -/

/--
Inverted dropout of `x` with drop probability `p` through `Op.dropout`. In training the sample `u`
comes from `Op.randUniform` with the interpreter's next key under `seed`, an entry is kept iff
`u ≥ p`, and a kept entry becomes `x / (1 - p)`. In evaluation the result is `x`. A finite `p`
enters as its exact value rounded once to the scalar, and a NaN or an infinity, which the layers
refuse, enters as `0 / 0`.

PyTorch analogue: `torch.nn.functional.dropout(x, p, training)`.
-/
def dropout {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape} (x : Ref (m := m) (α := α) s) (p : Float) (seed : Nat) (training : Bool := true) :
    m (Ref (m := m) (α := α) s) := do
  if !training then
    pure x
  else
    let probability : α :=
      match FloatLib.Floats.ExecFloat.Binary.toRat?
          (FloatLib.Floats.ExecFloat.Binary.ofFloat p : FloatLib.Floats.Binary64) with
      | some q => (q : α)
      | none => 0 / 0
    let sample ← Ops.applyRandom (m := m) (α := α)
        (.randUniform (s := s) 0) seed ()
    Ops.apply (m := m) (α := α) (.dropout (s := s))
      (x, sample, .data (Ops.dataConst (m := m) (α := α)
        (Tensor.scalar probability)), ())

/--
Whether `p` is finite and lies in `[0, 1]`, the check for dropout probabilities and for cumulative
diffusion coefficients.
-/
def inUnitInterval (p : Float) : Bool :=
  p.isFinite && 0.0 <= p && p <= 1.0

end nn.functional
end TorchLean
