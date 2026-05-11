/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra

/-!
# API Public

PyTorch-like facade over the TorchLean API.

Most user code should be able to `import NN` and then work with:

- `API.nn`     (model/layer builders)
- `API.optim`  (optimizer configs for training)
- `API.Adapters` (LoRA and other model adapters)
- `API.train`  (fit/predict helpers)
- `API.Data`   (datasets/loaders + CSV/NPY readers)
- `API.autograd` (grad/vjp/jacobian helpers)
- `API.rand` (deterministic RNG helpers)
- `API.text` (tokenizers and small text-model helpers)
- `API.ssl` (self-supervised sample/objective helpers)

Most of the executable runtime machinery lives under `API.TorchLean.*`; this module collects the
pieces into a smaller, PyTorch-shaped surface under `NN.API.*`.

### PyTorch References

This facade is inspired by the public shape of PyTorch:
- `torch.nn`: `https://pytorch.org/docs/stable/nn.html`
- `torch.nn.functional`: `https://pytorch.org/docs/stable/nn.functional.html`
- `torch.optim`: `https://pytorch.org/docs/stable/optim.html`
- `torch.utils.data`: `https://pytorch.org/docs/stable/data.html`

TorchLean differs in two important ways:
- tensor shapes are tracked in types (many "shape bugs" become type errors),
- some scalar dtypes are proof-only (see `NN.API.DType` for executable dtype selection).

### Recommended Import

This is the implementation module for the public facade. New user code should usually prefer
`import NN`; use `import NN.Entrypoint.API` when you want only the PyTorch-shaped facade.

Facade policy:
- `nn`, `functional`, `Loss`, `Norm`, `Autodiff`, `Optim`, `Data`, `tlist`, and `sample` are the
  intended public namespaces.
- low-level runtime composition helpers like `compAny` stay internal to `NN.API.Runtime`.
  Small correctness-first helpers such as `batchLayerDim0` are documented as internal and may move.
-/

@[expose] public section


namespace NN
namespace API

namespace nn

/-- Sequential model type (TorchLean `Seq`). This is the analogue of PyTorch `nn.Sequential`. -/
abbrev Sequential := TorchLean.NN.Seq

/-- Single-layer definition type (TorchLean `LayerDef`). This is the analogue of PyTorch
  `nn.Module`. -/
abbrev LayerDef := TorchLean.NN.LayerDef

/-!
Re-export common `Seq` helpers under `API.nn.*` so examples can stay on the public facade.

This intentionally mirrors the TorchLean names to keep the mapping obvious.
-/
export TorchLean.NN.Seq
  (paramShapes paramRequiresGrad initParams updateBuffers
   programWithMode program
   scalarModuleDefWithMode scalarModuleDef
   mseScalarModuleDefWithMode mseScalarModuleDef
   crossEntropyOneHotScalarModuleDefWithMode crossEntropyOneHotScalarModuleDef
   compileOutWithMode compileOut
   predict1WithMode predict1 eval1 eval1NoGrad eval1CompiledNoGrad predict1NoGrad)

/-- Lift a single layer definition into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : LayerDef σ τ) : Sequential σ τ :=
  TorchLean.Layers.of layer

/-!
All explicit-seed layer constructors live under `nn.pure.*`.

The top-level `nn.*` namespace is reserved for the *seeded builder* API that allocates
initialization seeds automatically (PyTorch-style ergonomics).
-/
namespace pure

/--
Linear layer on the last axis (prefix-shape preserving).

PyTorch analogue: `torch.nn.Linear`.
See `https://pytorch.org/docs/stable/generated/torch.nn.Linear.html`.

Unlike the lower-level TorchLean layer constructor (which is vector-only),
this public facade matches PyTorch’s convention:

- if `x` has shape `[..., inDim]`, `linear inDim outDim` returns a model of shape `[..., outDim]`.

The leading “prefix” dimensions are treated as a batch (they are flattened to `(numel(prefix),
  inDim)`,
the affine map is applied once, and the result is reshaped back).
-/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0)
    (pfx : Spec.Shape := Spec.Shape.scalar) :
    Sequential (pfx.appendDim inDim) (pfx.appendDim outDim) :=
  let WShape : Spec.Shape := NN.Tensor.Shape.Mat outDim inDim
  let bShape : Spec.Shape := NN.Tensor.Shape.Vec outDim
  let w0 : Spec.Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.xavierW
    (outDim := outDim) (inDim := inDim) (seed := seedW)
  let b0 : Spec.Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := .zeros) (seed := seedB)
  let batch : Nat := Spec.Shape.size pfx
  of
    { paramShapes := [WShape, bShape]
      initParams := TorchLean.tlist2 w0 b0
      paramRequiresGrad := [true, true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w b x =>
            let sIn : Spec.Shape := pfx.appendDim inDim
            let sOut : Spec.Shape := pfx.appendDim outDim
            ((do
              let x2D ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := NN.Tensor.Shape.Mat batch inDim)
                  x (by
                    -- size(sIn) = size(pfx) * inDim = batch * inDim = size(Mat batch inDim)
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])

              let wT ←
                TorchLean.transpose2d (m := m) (α := α)
                  (mDim := outDim) (nDim := inDim) w
              let y ← TorchLean.matmul (m := m) (α := α)
                (mDim := batch) (nDim := inDim) (pDim := outDim) x2D wT
              let y2D ←
                TorchLean.F.addB (m := m) (α := α) (t := NN.Tensor.Shape.Mat batch outDim) y b
              TorchLean.reshape (m := m) (α := α)
                (s₁ := NN.Tensor.Shape.Mat batch outDim)
                (s₂ := sOut)
                y2D (by
                  -- size(Mat batch outDim) = batch * outDim = size(pfx) * outDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (TorchLean.RefTy (m := m) (α := α) sOut))
    }

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:
`h_t = tanh(W [x_t; h_{t-1}] + b)`, with `h_{-1} = 0`.

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputSize, hiddenSize, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
-/
def rnn (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize) :=
  of (TorchLean.NN.rnn (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
GRU layer (time-major sequence, no batch axis).

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.GRU(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
-/
def gru (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize) :=
  of (TorchLean.NN.gru (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Trainable Mamba-style gated diagonal state-space layer.

The layer is time-major and single-batch, matching the simple `rnn`/`gru`/`lstm` constructors:
input `(seqLen × inputSize)`, output `(seqLen × hiddenSize)`.  It is unrolled with differentiable
TorchLean ops, so CPU and CUDA training use the same API.
-/
def mamba (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize) :=
  of (TorchLean.NN.mamba (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize)
    seedW seedB)

/--
LSTM layer (time-major sequence, no batch axis).

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.LSTM(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
-/
def lstm (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize) :=
  of (TorchLean.NN.lstm (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Embedding table initialization configuration (one-hot / token-distribution inputs).

This is the TorchLean-friendly analogue of `torch.nn.Embedding` in the common demo setting where
token ids are represented as one-hot vectors (or soft token distributions), so lookup is a matrix
multiplication rather than integer indexing.
-/
structure Embedding where
  /-- Seed for deterministic embedding-table initialization. -/
  seedW : Nat := 0
  /-- Initialization scheme for the embedding table. -/
  wInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Embedding layer for one-hot / token-distribution inputs (no bias).

Input shape: `[..., vocab]`
Output shape: `[..., embedDim]`

PyTorch analogue: conceptually `nn.Embedding(vocab, embedDim)` but applied to one-hot inputs.
-/
def embedding (vocab embedDim : Nat) (cfg : Embedding := {}) (pfx : Spec.Shape := Spec.Shape.scalar) :
    Sequential (pfx.appendDim vocab) (pfx.appendDim embedDim) :=
  let WShape : Spec.Shape := NN.Tensor.Shape.Mat vocab embedDim
  let w0 : Spec.Tensor Float WShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := WShape) (sch := cfg.wInit) (seed := cfg.seedW)
  let batch : Nat := Spec.Shape.size pfx
  of
    { paramShapes := [WShape]
      initParams := TorchLean.tlist1 w0
      paramRequiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn : Spec.Shape := pfx.appendDim vocab
            let sOut : Spec.Shape := pfx.appendDim embedDim
            ((do
              let x2D ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := NN.Tensor.Shape.Mat batch vocab)
                  x (by
                    -- size(sIn) = size(pfx) * vocab = batch * vocab
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
              let y ←
                TorchLean.matmul (m := m) (α := α)
                  (mDim := batch) (nDim := vocab) (pDim := embedDim)
                  x2D w
              TorchLean.reshape (m := m) (α := α)
                (s₁ := NN.Tensor.Shape.Mat batch embedDim)
                (s₂ := sOut)
                y (by
                  -- size(Mat batch embedDim) = batch * embedDim = size(pfx) * embedDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (TorchLean.RefTy (m := m) (α := α) sOut))
    }

/--
Learned positional embedding configuration.

This is a trainable parameter tensor of shape `(seqLen × embedDim)` that is broadcast across the
leading batch dimension and added to the input.
-/
structure LearnedPositionalEmbedding where
  /-- Seed for deterministic initialization. -/
  seedPos : Nat := 0
  /-- Initialization scheme for the positional embedding table. -/
  posInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Add learned positional embeddings to a batched `(batch × seqLen × embedDim)` tensor.

PyTorch analogue: `x + pos[:seqLen]` where `pos` is a parameter table.
-/
def learnedPositionalEmbedding {batch seqLen embedDim : Nat} (cfg : LearnedPositionalEmbedding := {}) :
    Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  let posShape : Spec.Shape := NN.Tensor.Shape.Mat seqLen embedDim
  let xShape : Spec.Shape := .dim batch posShape
  let pos0 : Spec.Tensor Float posShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := posShape) (sch := cfg.posInit) (seed := cfg.seedPos)
  of
    { paramShapes := [posShape]
      initParams := TorchLean.tlist1 pos0
      paramRequiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast `(seqLen × embedDim)` positional embeddings across the leading `batch` axis.
            (TorchLean.F.addB (m := m) (α := α) (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
    }

/--
Sinusoidal positional encoding configuration.

This is the classic (non-trainable) Transformer sinusoidal encoding, added to token embeddings.
`startPos` is an absolute-position offset (useful for KV-cache decoding).
-/
structure SinusoidalPositionalEncoding where
  /-- Absolute position offset for the first row of the encoding table. -/
  startPos : Nat := 0

/--
Add sinusoidal positional encodings to a batched `(batch × seqLen × embedDim)` tensor.

Implementation:
- precompute `PE : (seqLen × embedDim)` at initialization time (stored as a non-trainable buffer),
- broadcast it across the leading `batch` axis and add to the input.
-/
def sinusoidalPositionalEncoding {batch seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  let peShape : Spec.Shape := NN.Tensor.Shape.Mat seqLen embedDim
  let xShape : Spec.Shape := .dim batch peShape
  let pe0 : Spec.Tensor Float peShape :=
    Spec.sinusoidalPositionalEncodingSpec (α := Float) seqLen embedDim cfg.startPos
  of
    { paramShapes := [peShape]
      initParams := TorchLean.tlist1 pe0
      paramRequiresGrad := [false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (seqLen × embedDim)` across the leading `batch` axis.
            (TorchLean.F.addB (m := m) (α := α) (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Rotary positional embedding (RoPE) configuration.

`startPos` is an absolute-position offset (useful for KV-cache decoding).
-/
structure RoPE where
  /-- Absolute position offset for the first row of RoPE angles. -/
  startPos : Nat := 0

/--
Apply RoPE to a batched multi-head tensor `(batch × numHeads × seqLen × headDim)`.

This matches the standard identity:

`rope(x) = x * cos + rotatePairs(x) * sin`

where `cos`/`sin` depend only on `(pos, dim)` and broadcast across `(batch, numHeads)`.

Notes:
- This layer is *differentiable* (gradients flow through the rotation), but it has no trainable
  parameters; the precomputed `cos`/`sin` tables are stored as non-trainable buffers.
- The pure spec version is in `NN.Spec.Layers.PositionalEncoding` (`Spec.rope_apply_heads_spec`).
-/
def rope {batch numHeads seqLen headDim : Nat} (cfg : RoPE := {}) :
    Sequential
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim)))
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim))) :=
  let xShape : Spec.Shape := .dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim))
  let csShape : Spec.Shape := NN.Tensor.Shape.Mat seqLen headDim

  -- Precompute cos/sin tables (as Float buffers). These depend only on `(seqLen, headDim, startPos)`.
  let cos0 : Spec.Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeCosLastdimSpec (α := Float) (cfg.startPos + pos.val) headDim)
  let sin0 : Spec.Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeSinLastdimSpec (α := Float) (cfg.startPos + pos.val) headDim)

  -- Column permutation indices implementing pairwise swap `(0↔1, 2↔3, ...)`.
  -- When `headDim` is odd, the last index is left unchanged.
  let permIdx : Spec.Tensor Nat (.dim headDim .scalar) :=
    Spec.Tensor.dim (fun (j : Fin headDim) =>
      let idx := j.val
      let out : Nat :=
        if idx % 2 = 0 then
          if idx + 1 < headDim then idx + 1 else idx
        else
          idx - 1
      Spec.Tensor.scalar out)

  of
    { paramShapes := [csShape, csShape]
      initParams := TorchLean.tlist2 cos0 sin0
      paramRequiresGrad := [false, false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate last-dim pairs by a fixed 2D permutation/sign pattern.
            let rowsFold : Nat := batch * numHeads * seqLen
            let flatShape : Spec.Shape := NN.Tensor.Shape.Mat rowsFold headDim

            let x2d ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := xShape) (s₂ := flatShape)
                x (by
                  -- size(xShape) = batch * numHeads * seqLen * headDim = rowsFold * headDim = size(flatShape)
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size,
                    Nat.mul_left_comm, Nat.mul_comm])

            let xT ←
              TorchLean.transpose2d (m := m) (α := α)
                (mDim := rowsFold) (nDim := headDim) x2d

            let xPerm ←
              TorchLean.gatherRowsNat (m := m) (α := α)
                (rows := headDim) (cols := rowsFold) (k := headDim)
                xT permIdx

            let xBack ←
              TorchLean.transpose2d (m := m) (α := α)
                (mDim := headDim) (nDim := rowsFold) xPerm

            -- Sign pattern for `rotatePairs`: even outputs get a negation (except the final unpaired entry).
            let signT : Spec.Tensor α (.dim headDim .scalar) :=
              Spec.Tensor.dim (fun (j : Fin headDim) =>
                let idx := j.val
                let v : α :=
                  if idx % 2 = 0 ∧ idx + 1 < headDim then (-1 : α) else (1 : α)
                Spec.Tensor.scalar v)
            let sign ← TorchLean.const (m := m) (α := α) (s := .dim headDim .scalar) signT

            let xRot2d ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := .dim headDim .scalar) (t := flatShape)
                xBack sign

            let xRot ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := flatShape) (s₂ := xShape)
                xRot2d (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size,
                    Nat.mul_left_comm, Nat.mul_comm])

            -- Apply the RoPE formula with broadcasting of `cos/sin : (seqLen × headDim)`.
            let xCos ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                x cos
            let rotSin ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                xRot sin
            TorchLean.add (m := m) (α := α) (s := xShape) xCos rotSin
            ) : m (TorchLean.RefTy (m := m) (α := α) xShape))
    }

/-- Elementwise ReLU. PyTorch analogue: `torch.nn.ReLU` / `torch.nn.functional.relu`. -/
def relu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.relu (s := s)
/-- Elementwise SiLU/Swish. PyTorch analogue: `torch.nn.SiLU` / `torch.nn.functional.silu`. -/
def silu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.silu (s := s)
/-- Elementwise GELU. PyTorch analogue: `torch.nn.GELU` / `torch.nn.functional.gelu`. -/
def gelu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.gelu (s := s)
/-- Elementwise sigmoid. PyTorch analogue: `torch.nn.Sigmoid` / `torch.nn.functional.sigmoid`. -/
def sigmoid {s : Spec.Shape} : Sequential s s := TorchLean.Layers.sigmoid (s := s)
/-- Elementwise tanh. PyTorch analogue: `torch.nn.Tanh` / `torch.nn.functional.tanh`. -/
def tanh {s : Spec.Shape} : Sequential s s := TorchLean.Layers.tanh (s := s)
/-- Softmax. PyTorch analogue: `torch.nn.Softmax` / `torch.nn.functional.softmax`. -/
def softmax {s : Spec.Shape} : Sequential s s := TorchLean.Layers.softmax (s := s)
/-- Reduce-sum to a scalar. PyTorch analogue: `torch.sum`. -/
def sum {s : Spec.Shape} : Sequential s Spec.Shape.scalar := TorchLean.Layers.sum (s := s)
/-- Flatten any tensor into a 1D vector of length `size s`. PyTorch analogue: `torch.flatten`. -/
def flatten {s : Spec.Shape} : Sequential s (.dim (Spec.Shape.size s) .scalar) :=
  TorchLean.Layers.flatten (s := s)

/--
Flatten a batched tensor `N × σ` into a matrix `N × (size σ)`.

PyTorch analogue: `torch.flatten(x, start_dim=1)`.
-/
def flattenBatch {n : Nat} {s : Spec.Shape} :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s)) :=
  of
    { paramShapes := []
      initParams := .nil
      paramRequiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim n s)
              (s₂ := NN.Tensor.Shape.Mat n (Spec.Shape.size s))
              x (by simp [Spec.Shape.size])
    }

/--
Flatten a batched tensor starting at dimension 1 (keep dim0).

Synonym for `flattenBatch`, matching PyTorch’s `start_dim=1` wording.
-/
def flattenStart1 {n : Nat} {s : Spec.Shape} :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s)) :=
  flattenBatch (n := n) (s := s)
/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  TorchLean.Layers.dropout (s := s) p seed
/--
Convenience block: `Flatten -> Linear`.

This is common for "image to classifier head" demos.
-/
def flattenLinear {s : Spec.Shape} (outDim : Nat) (seedW seedB : Nat := 0) :
    Sequential s (NN.Tensor.Shape.Vec outDim) :=
  TorchLean.Layers.flattenLinear (s := s) outDim seedW seedB

/-!
`nn.functional` mirrors `torch.nn.functional`: pure, stateless building blocks.

In TorchLean these are defined as derived ops over the small primitive `Ops` surface, so the same
code works on both the eager backend and the compiled backend.
-/
namespace functional

/-!
PyTorch references:
- `torch.nn.functional`: `https://pytorch.org/docs/stable/nn.functional.html`
-/

export TorchLean.F
  (square checkpoint
   detach stopGrad
   addB mulB
   embedding mean
   dropoutSeeded)

end functional

/-!
## Batch Lifting

`batchDim0 n model` wraps a *single-example* model `σ → τ` into a batched model
`(dim n σ) → (dim n τ)` by running the underlying model once per batch element.

This is a correctness-first helper used to expose PyTorch-like `N×...` APIs even when a primitive
only exists for the unbatched shape.
-/

/--
Lift a single-example `LayerDef σ τ` to operate on a dimension-0 batch.

This is a correctness-first helper: it runs the underlying layer independently on each batch
element. Prefer a primitive batched layer when one exists.
-/
def batchLayerDim0 (n : Nat) {σ τ : Spec.Shape} (l : LayerDef σ τ) :
    LayerDef (.dim n σ) (.dim n τ) :=
  let inSize : Nat := Spec.Shape.size σ
  let outSize : Nat := Spec.Shape.size τ
  { paramShapes := l.paramShapes
    initParams := l.initParams
    paramRequiresGrad := l.paramRequiresGrad
    updateBuffers := none
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := l.paramShapes ++ [.dim n σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) (.dim n τ)))
          (fun args => do
            let (ps, xBatch) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := l.paramShapes) (τ := .dim n σ) args
            let xMat ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim n σ) (s₂ := .dim n (.dim inSize .scalar))
                xBatch (by simp [Spec.Shape.size, inSize])
            let zeros : Spec.Tensor α (.dim n (.dim outSize .scalar)) :=
              _root_.Spec.Tensor.dim (fun _ =>
                _root_.Spec.Tensor.dim (fun _ =>
                  _root_.Spec.Tensor.scalar (0 : α)))
            let out0 ← TorchLean.const (m := m) (α := α) (s := .dim n (.dim outSize .scalar)) zeros
            let outMat ← (List.finRange n).foldlM (init := out0) (fun acc i => do
              let xRow ← TorchLean.gatherRow (m := m) (α := α) (rows := n) (cols := inSize) xMat i
              let xSample ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := .dim inSize .scalar) (s₂ := σ)
                  xRow (by simp [Spec.Shape.size, inSize])
              let ySample ←
                _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                  (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                  (ss := l.paramShapes ++ [σ])
                  (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                  (l.forward mode (α := α) (m := m))
                  (_root_.Runtime.Autograd.Torch.RefList.append ps (.cons xSample .nil))
              let yRow ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := τ) (s₂ := .dim outSize .scalar)
                  ySample (by simp [Spec.Shape.size, outSize])
              TorchLean.scatterAddRow (m := m) (α := α) (rows := n) (cols := outSize) acc yRow i)
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim n (.dim outSize .scalar)) (s₂ := .dim n τ)
              outMat (by simp [Spec.Shape.size, outSize]))
  }

/-- Lift a sequential model to act pointwise on a leading dim0 batch axis. -/
def batchDim0 (n : Nat) {σ τ : Spec.Shape} : Sequential σ τ → Sequential (.dim n σ) (.dim n τ)
  | .id s => .id (.dim n s)
  | .cons l rest => .cons (batchLayerDim0 n l) (batchDim0 n rest)

/-!
Note: some low-level TorchLean layers (notably conv/pool/norm) have Nat-side well-formedness
proof arguments (e.g. `kH ≠ 0`).

The public path is *record-based specs* that hide those proofs via typeclasses like `NeZero`,
so examples can stay PyTorch-like without relying on positional macros.
-/

/--
Named-field Conv2d configuration (CHW layout).

This is the public, PyTorch-like entry point for convolution in TorchLean.
PyTorch analogue: `torch.nn.Conv2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html`.
-/
structure Conv2d where
  /-- Output channels. -/
  outC : Nat
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 1
  /-- Zero-padding (shared for height/width). -/
  padding : Nat := 0
  /-- Seed for deterministic kernel initialization. -/
  seedK : Nat := 0
  /-- Seed for deterministic bias initialization. -/
  seedB : Nat := 0
  /-- Initialization scheme for the kernel weights. -/
  kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1

@[inherit_doc Conv2d]
abbrev Conv := Conv2d

/--
2D convolution over a CHW tensor, using explicit well-formedness proofs.
-/
def conv2dCHWWith {inC inH inW : Nat} (cfg : Conv2d)
    (hInC : inC ≠ 0) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.conv2d inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding inH inW
    (hInC := hInC) (hKH := hKH) (hKW := hKW)
    (seedK := cfg.seedK) (seedB := cfg.seedB) (kInit := cfg.kInit)

/--
2D convolution over a CHW tensor, with a PyTorch-like named-field spec.

This hides the Nat-side proof arguments via the `NeZero` typeclass.
-/
def conv2dCHW {inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) ) :=
  conv2dCHWWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _) (NeZero.ne _)

/-- 2D convolution over a batched image tensor (shape `N×C×H×W`, like PyTorch). -/
def conv2d {n inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (conv2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

@[inherit_doc conv2dCHWWith]
def convCHWWith := @conv2dCHWWith

@[inherit_doc conv2dCHW]
def convCHW := @conv2dCHW

/--
Convolution over batched CHW images, using the PyTorch-style `Conv2d` config record.

Shorthand for `conv2d`.
-/
def conv {n inC inH inW : Nat} (cfg : Conv) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :=
  conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
MaxPool2d configuration for CHW inputs.

PyTorch analogue: `torch.nn.MaxPool2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html`.
-/
structure MaxPool2d where
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 2

@[inherit_doc MaxPool2d]
abbrev MaxPool := MaxPool2d

/-- MaxPool2d with explicit nonzero kernel proofs. -/
def maxPool2dWith {inC inH inW : Nat} (cfg : MaxPool2d) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.maxPool2d cfg.kH cfg.kW inH inW inC cfg.stride (hKH := hKH) (hKW := hKW)

/-- MaxPool2d over CHW inputs using `NeZero` to hide nonzero kernel proofs. -/
def maxPool2dCHW {inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  maxPool2dWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _)

/-- MaxPool2d using `NeZero` to hide nonzero kernel proofs. -/
def maxPool2d {n inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (maxPool2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

/-- Shorthand for `maxPool2dWith` (PyTorch-style). -/
def maxPoolWith := @maxPool2dWith

/-- Shorthand for `maxPool2dCHW` (PyTorch-style). -/
def maxPoolCHW := @maxPool2dCHW

/--
Max pooling over batched CHW images, using the PyTorch-style `MaxPool2d` config record.

Shorthand for `maxPool2d`.
-/
def maxPool {n inC inH inW : Nat} (cfg : MaxPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
AvgPool2d configuration for CHW inputs.

PyTorch analogue: `torch.nn.AvgPool2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html`.
-/
structure AvgPool2d where
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 2

@[inherit_doc AvgPool2d]
abbrev AvgPool := AvgPool2d

/-- AvgPool2d with explicit nonzero kernel proofs. -/
def avgPool2dWith {inC inH inW : Nat} (cfg : AvgPool2d) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.avgPool2d cfg.kH cfg.kW inH inW inC cfg.stride (hKH := hKH) (hKW := hKW)

/-- AvgPool2d over CHW inputs using `NeZero` to hide nonzero kernel proofs. -/
def avgPool2dCHW {inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  avgPool2dWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _)

/-- AvgPool2d over batched NCHW inputs (shape `N×C×H×W`, like PyTorch). -/
def avgPool2d {n inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (avgPool2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

/-- Shorthand for `avgPool2dWith` (PyTorch-style). -/
def avgPoolWith := @avgPool2dWith

/-- Shorthand for `avgPool2dCHW` (PyTorch-style). -/
def avgPoolCHW := @avgPool2dCHW

/--
Average pooling over batched CHW images, using the PyTorch-style `AvgPool2d` config record.

Shorthand for `avgPool2d`.
-/
def avgPool {n inC inH inW : Nat} (cfg : AvgPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
Global average pooling over a CHW tensor.

PyTorch analogue: `torch.nn.AdaptiveAvgPool2d((1, 1))` followed by flattening.
-/
def globalAvgPoolCHW := TorchLean.Layers.globalAvgPoolCHW

/-- Global average pooling over an NCHW tensor (preserves the batch dimension). -/
def globalAvgPoolNCHW := TorchLean.Layers.globalAvgPoolNCHW

/--
LayerNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

PyTorch analogue: `torch.nn.LayerNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html`.
-/
structure LayerNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This matches the common Transformer usage: normalize each token’s `embedDim`-vector independently,
with learnable scale/shift parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied to a tensor of shape
`(batch, seqLen, embedDim)`.

Most users should call `nn.layerNorm`, which uses `NeZero` to discharge the positivity proofs.
-/
def layerNormWith {batch seqLen embedDim : Nat} (cfg : LayerNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  TorchLean.Layers.layerNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (hSeq := hSeq) (hEmbed := hEmbed)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes each `embedDim`-vector (per batch element, per sequence position), and applies
learned affine parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` on a tensor shaped `(batch, seqLen, embedDim)`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def layerNorm {batch seqLen embedDim : Nat} (cfg : LayerNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  layerNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/--
RMSNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

This is a common alternative to LayerNorm in modern transformer architectures.
-/
structure RMSNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This is like LayerNorm but without mean subtraction: we scale by the root-mean-square over the
`embedDim` axis, and apply a learned scale `gamma`.

PyTorch analogue: many libraries provide an `RMSNorm(embedDim)` module; conceptually it is applied
to tensors shaped `(batch, seqLen, embedDim)`.

Most users should call `nn.rmsNorm`, which uses `NeZero` to discharge the positivity proofs.
-/
def rmsNormWith {batch seqLen embedDim : Nat} (cfg : RMSNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  TorchLean.Layers.rmsNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (hSeq := hSeq) (hEmbed := hEmbed)
    (seedGamma := cfg.seedGamma)

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes by the root-mean-square over the `embedDim` axis (per batch element, per position),
then applies a learned scale `gamma`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def rmsNorm {batch seqLen embedDim : Nat} (cfg : RMSNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  rmsNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/--
BatchNorm2d configuration (learned scale/shift).

PyTorch analogue: `torch.nn.BatchNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html`.
-/
structure BatchNorm2d where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/-- BatchNorm2d over NCHW inputs (train/eval is handled by `Seq` mode). -/
def batchNorm2dNCHWWith {n c h w : Nat} (cfg : BatchNorm2d)
    (hN : n > 0) (hC : c > 0) (hH : h > 0) (hW : w > 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.batchNorm2dNCHW (n := n) (c := c) (h := h) (w := w)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
BatchNorm2d over NCHW inputs, using `NeZero` to hide the positivity proofs.

PyTorch analogue: `torch.nn.BatchNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html`.
-/
def batchNorm2d {n c h w : Nat} (cfg : BatchNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  batchNorm2dNCHWWith (n := n) (c := c) (h := h) (w := w) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := n)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := c)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := h)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := w)))

/--
InstanceNorm2d configuration (learned scale/shift).

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
structure InstanceNorm2d where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
InstanceNorm2d over NCHW inputs, using explicit positivity proofs.

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
def instanceNorm2dWith {n c h w : Nat} (cfg : InstanceNorm2d)
    (hN : n > 0) (hC : c > 0) (hH : h > 0) (hW : w > 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.instanceNorm2dNCHW (n := n) (c := c) (h := h) (w := w)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
InstanceNorm2d over NCHW inputs, using `NeZero` to hide the positivity proofs.

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
def instanceNorm2d {n c h w : Nat} (cfg : InstanceNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  instanceNorm2dWith (n := n) (c := c) (h := h) (w := w) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := n)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := c)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := h)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := w)))

/--
GroupNorm over NCHW inputs.

PyTorch analogue: `torch.nn.GroupNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html`.
-/
def groupNorm2dNCHW (n c h w groups : Nat) {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    {hG : groups > 0} (hGE : c ≥ groups) (hDiv : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.groupNorm2dNCHW (n := n) (c := c) (h := h) (w := w) (groups := groups)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW) (hG := hG)
    (hGE := hGE) (hDiv := hDiv) (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Multi-head self-attention configuration.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
-/
structure MultiheadAttention where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Base seed for deterministic parameter initialization. -/
  seedW : Nat := 0

/--
Multi-head self-attention with an explicit nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiheadAttentionWith {batch n dModel : Nat} (cfg : MultiheadAttention) (hN : n ≠ 0)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel))
      :=
  TorchLean.Layers.attention (batch := batch) (n := n) (dModel := dModel)
    (numHeads := cfg.numHeads) (headDim := cfg.headDim)
    (hN := hN) (seedW := cfg.seedW) (mask := mask)

/--
Multi-head self-attention using `NeZero` to hide the nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiheadAttention {batch n dModel : Nat} (cfg : MultiheadAttention) [NeZero n]
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel))
      :=
  multiheadAttentionWith (batch := batch) (n := n) (dModel := dModel) cfg (NeZero.ne (n := n))
    (mask := mask)

namespace blocks

/--
Small set of activation choices for block builders.

PyTorch analogues:
- `relu`    <-> `torch.nn.ReLU`
- `gelu`    <-> `torch.nn.GELU`
- `silu`    <-> `torch.nn.SiLU`
- `tanh`    <-> `torch.nn.Tanh`
- `sigmoid` <-> `torch.nn.Sigmoid`
-/
inductive Activation where
  | relu
  | gelu
  | silu
  | tanh
  | sigmoid
deriving Repr, DecidableEq

/-- Interpret an `Activation` as a TorchLean layer. -/
def activation {s : Spec.Shape} : Activation → Sequential s s
  | .relu => relu (s := s)
  | .gelu => gelu (s := s)
  | .silu => silu (s := s)
  | .tanh => tanh (s := s)
  | .sigmoid => sigmoid (s := s)

/--
MLP (multi-layer perceptron) configuration.

This is a lightweight builder that produces a sequential stack of linear layers with activations
and optional dropout.

PyTorch analogue: a hand-written `nn.Sequential(Linear(...), ReLU(), ..., Linear(...))`.
-/
structure MLP where
  /-- Hidden layer widths (each entry creates a `Linear -> Activation` stage). -/
  hidden : List Nat := []
  /-- Activation used after each hidden linear layer. -/
  activation : Activation := .relu
  /-- Optional dropout probability after each activation. -/
  dropout? : Option Float := none
  /-- Base seed used to deterministically initialize all linear layers (and dropout if present). -/
  seedBase : Nat := 0

/--
Internal recursion for `mlp`.

This builds the sequential stack stage-by-stage, threading a seed so each linear (and optional
dropout) layer gets a deterministic initialization key.
-/
def mlpGo (act : Activation) (dropout? : Option Float) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) → (seed : Nat) →
      Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim)
  | inDim, [], outDim, seed =>
      linear inDim outDim seed (seed + 1)
  | inDim, h :: hs, outDim, seed =>
      let lin : Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec h) :=
        linear inDim h seed (seed + 1)
      let seed' := seed + 2
      let actLayer : Sequential (NN.Tensor.Shape.Vec h) (NN.Tensor.Shape.Vec h) :=
        activation (s := NN.Tensor.Shape.Vec h) act
      let mid : Sequential (NN.Tensor.Shape.Vec h) (NN.Tensor.Shape.Vec h) × Nat :=
        match dropout? with
        | none => (actLayer, seed')
        | some p =>
            ((seq! actLayer, dropout (s := NN.Tensor.Shape.Vec h) p (seed := seed')), seed' + 1)
      let rest :=
        mlpGo act dropout? h hs outDim mid.snd
      seq! lin, mid.fst, rest

/--
Build an MLP as a sequential stack of linear layers and activations.

This is a small "PyTorch-shaped" helper: a typical call looks like:
`API.nn.blocks.mlp 784 10 { hidden := [128, 128], activation := .relu }`.
-/
def mlp (inDim outDim : Nat) (cfg : MLP := {}) :
    Sequential (NN.Tensor.Shape.Vec inDim) (NN.Tensor.Shape.Vec outDim) :=
  mlpGo cfg.activation cfg.dropout? inDim cfg.hidden outDim cfg.seedBase

/--
Conv2d + activation (+ optional dropout) block configuration (CHW layout).

This compact helper is used by vision examples before moving to larger curated blocks.
-/
structure Conv2dAct where
  /-- Conv hyperparameters and seeds. -/
  conv : Conv2d
  /-- Activation applied after the convolution. -/
  activation : Activation := .relu
  /-- Optional dropout probability after the activation. -/
  dropout? : Option Float := none
  /-- Seed for dropout RNG (only used when `dropout?` is present). -/
  seedDropout : Nat := 0

/-- `Conv2d -> Activation -> (optional Dropout)` over CHW inputs. -/
def conv2dAct {inC inH inW : Nat} (cfg : Conv2dAct) [NeZero inC] [NeZero cfg.conv.kH] [NeZero
  cfg.conv.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let core := seq! (conv2dCHW (inC := inC) (inH := inH) (inW := inW) cfg.conv), activation
    cfg.activation
  match cfg.dropout? with
  | none => core
  | some p => seq! core, dropout p (seed := cfg.seedDropout)

/-!
## Vision blocks

These are small, *named-field* building blocks intended for public examples:

- reduce seed/proof noise at call sites,
- keep composition explicit (still `seq!` stacking),
- provide canonical blocks users expect from PyTorch codebases.

They are intentionally conservative: the goal is readability and stable typing, not maximum
  coverage.
-/

/--
Configuration for a common vision block:
`Conv2d -> BatchNorm2d -> Activation -> (optional Dropout)`.

This is used by `conv2dNormActCHW` (single-image CHW) and `conv2dNormAct` (batched NCHW).
We keep deterministic seed allocation explicit via `seedBase` so examples stay reproducible.
-/
structure Conv2dNormAct where
  /-- Conv hyperparameters (seeds inside this record are ignored; use `seedBase`). -/
  conv : Conv2d
  /-- Activation after normalization. -/
  activation : Activation := .relu
  /-- Optional dropout applied after the activation. -/
  dropout? : Option Float := none
  /-- Base seed for deterministic init (derived seeds are allocated in a fixed order). -/
  seedBase : Nat := 0

/--
`Conv2d -> BatchNorm -> Activation -> (optional Dropout)`, over a single CHW image (no batch axis).

Seed allocation (relative to `seedBase`):

- `seedBase + 0,1`: conv kernel / bias
- `seedBase + 2..5`: BN gamma / beta / running-mean / running-var
- `seedBase + 6`: dropout
-/
def conv2dNormActCHW {inC inH inW : Nat} (cfg : Conv2dNormAct)
    [NeZero inC] [NeZero cfg.conv.kH] [NeZero cfg.conv.kW] [NeZero cfg.conv.outC] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let conv : Conv2d :=
    { cfg.conv with seedK := cfg.seedBase, seedB := cfg.seedBase + 1 }
  let outH : Nat := (inH + 2 * conv.padding - conv.kH) / conv.stride + 1
  let outW : Nat := (inW + 2 * conv.padding - conv.kW) / conv.stride + 1
  have hOutH : outH > 0 := by
    -- `outH = _ + 1`
    simp [outH]
  have hOutW : outW > 0 := by
    simp [outW]
  let bn : Sequential (NN.Tensor.Shape.Image conv.outC outH outW) (NN.Tensor.Shape.Image conv.outC
    outH outW) :=
    TorchLean.Layers.batchNormCHW conv.outC outH outW
      (hC := Nat.pos_of_ne_zero (NeZero.ne (n := conv.outC)))
      (hH := hOutH) (hW := hOutW)
      (seedGamma := cfg.seedBase + 2)
      (seedBeta := cfg.seedBase + 3)
      (seedMean := cfg.seedBase + 4)
      (seedVar := cfg.seedBase + 5)
  let act : Sequential (NN.Tensor.Shape.Image conv.outC outH outW) (NN.Tensor.Shape.Image conv.outC
    outH outW) :=
    activation (s := NN.Tensor.Shape.Image conv.outC outH outW) cfg.activation
  let core := seq! (conv2dCHW (inC := inC) (inH := inH) (inW := inW) conv), bn, act
  match cfg.dropout? with
  | none => core
  | some p =>
      let s : Spec.Shape := NN.Tensor.Shape.Image conv.outC outH outW
      seq! core, dropout (s := s) p (seed := cfg.seedBase + 6)

/--
Configuration for `conv2dNormActPool*`: a `Conv2dNormAct` block followed by max-pooling.

This matches the common “conv-bn-act-pool” pattern used in small CNNs.
-/
structure Conv2dNormActPool where
  /-- Conv/BN/activation/dropout block configuration. -/
  block : Conv2dNormAct
  /-- Pooling hyperparameters (defaults to `2×2` stride-2 max pool). -/
  pool : MaxPool2d := { kH := 2, kW := 2, stride := 2 }

/-- `conv2dNormActCHW` followed by `MaxPool2dCHW`. -/
def conv2dNormActPoolCHW {inC inH inW : Nat} (cfg : Conv2dNormActPool)
    [NeZero inC]
    [NeZero cfg.block.conv.kH] [NeZero cfg.block.conv.kW] [NeZero cfg.block.conv.outC]
    [NeZero cfg.pool.kH] [NeZero cfg.pool.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.block.conv.outC
        ((((inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride + 1) -
          cfg.pool.kH) / cfg.pool.stride + 1)
        ((((inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride + 1) -
          cfg.pool.kW) / cfg.pool.stride + 1)) :=
  let core := conv2dNormActCHW (inC := inC) (inH := inH) (inW := inW) cfg.block
  -- Pool input dims are the conv output dims.
  let outH : Nat := (inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride +
    1
  let outW : Nat := (inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride +
    1
  let pool : Sequential (NN.Tensor.Shape.Image cfg.block.conv.outC outH outW)
      (NN.Tensor.Shape.Image cfg.block.conv.outC ((outH - cfg.pool.kH) / cfg.pool.stride + 1) ((outW
        - cfg.pool.kW) / cfg.pool.stride + 1)) :=
    maxPool2dCHW (inC := cfg.block.conv.outC) (inH := outH) (inW := outW) cfg.pool
  seq! core, pool

/--
`Conv2d -> BatchNorm2d -> Activation -> (optional Dropout)`, over batched image tensors (`N×C×H×W`).

This is the public PyTorch-like path: examples should build CNNs directly over batched images.
-/
def conv2dNormAct {n inC inH inW : Nat} (cfg : Conv2dNormAct)
    [NeZero n] [NeZero inC] [NeZero cfg.conv.kH] [NeZero cfg.conv.kW] [NeZero cfg.conv.outC] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.conv.outC
        ((inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1)
        ((inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1)) :=
  let conv : Conv2d :=
    { cfg.conv with seedK := cfg.seedBase, seedB := cfg.seedBase + 1 }
  let outH : Nat := (inH + 2 * conv.padding - conv.kH) / conv.stride + 1
  let outW : Nat := (inW + 2 * conv.padding - conv.kW) / conv.stride + 1
  have hOutH : outH > 0 := by simp [outH]
  have hOutW : outW > 0 := by simp [outW]
  let bn : Sequential (NN.Tensor.Shape.Images n conv.outC outH outW) (NN.Tensor.Shape.Images n
    conv.outC outH outW) :=
    TorchLean.Layers.batchNorm2dNCHW n conv.outC outH outW
      (hN := Nat.pos_of_ne_zero (NeZero.ne (n := n)))
      (hC := Nat.pos_of_ne_zero (NeZero.ne (n := conv.outC)))
      (hH := hOutH) (hW := hOutW)
      (seedGamma := cfg.seedBase + 2)
      (seedBeta := cfg.seedBase + 3)
      (seedMean := cfg.seedBase + 4)
      (seedVar := cfg.seedBase + 5)
  let act : Sequential (NN.Tensor.Shape.Images n conv.outC outH outW) (NN.Tensor.Shape.Images n
    conv.outC outH outW) :=
    activation (s := NN.Tensor.Shape.Images n conv.outC outH outW) cfg.activation
  let core := seq! (conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) conv), bn, act
  match cfg.dropout? with
  | none => core
  | some p =>
      let s : Spec.Shape := NN.Tensor.Shape.Images n conv.outC outH outW
      seq! core, dropout (s := s) p (seed := cfg.seedBase + 6)

/-- `conv2dNormAct` followed by `MaxPool2d`, over batched image tensors. -/
def conv2dNormActPool {n inC inH inW : Nat} (cfg : Conv2dNormActPool)
    [NeZero n]
    [NeZero inC]
    [NeZero cfg.block.conv.kH] [NeZero cfg.block.conv.kW] [NeZero cfg.block.conv.outC]
    [NeZero cfg.pool.kH] [NeZero cfg.pool.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.block.conv.outC
        ((((inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride + 1) -
          cfg.pool.kH) / cfg.pool.stride + 1)
        ((((inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride + 1) -
          cfg.pool.kW) / cfg.pool.stride + 1)) :=
  let core := conv2dNormAct (n := n) (inC := inC) (inH := inH) (inW := inW) cfg.block
  -- Pool input dims are the conv output dims.
  let outH : Nat := (inH + 2 * cfg.block.conv.padding - cfg.block.conv.kH) / cfg.block.conv.stride +
    1
  let outW : Nat := (inW + 2 * cfg.block.conv.padding - cfg.block.conv.kW) / cfg.block.conv.stride +
    1
  let pool : Sequential (NN.Tensor.Shape.Images n cfg.block.conv.outC outH outW)
      (NN.Tensor.Shape.Images n cfg.block.conv.outC ((outH - cfg.pool.kH) / cfg.pool.stride + 1)
        ((outW - cfg.pool.kW) / cfg.pool.stride + 1)) :=
    maxPool2d (n := n) (inC := cfg.block.conv.outC) (inH := outH) (inW := outW) cfg.pool
  seq! core, pool

/--
Residual/skip-connection wrapper as a single `LayerDef`.

Given `inner : Seq s s`, this builds a layer that computes `x |-> inner(x) + x`.

PyTorch analogue: `x + f(x)` blocks used throughout ResNets and Transformers.
-/
def residualLayer {s : Spec.Shape} (inner : Sequential s s) : LayerDef s s :=
  let ps := TorchLean.NN.Seq.paramShapes inner
  { paramShapes := ps
    initParams := TorchLean.NN.Seq.initParams inner
    paramRequiresGrad := TorchLean.NN.Seq.paramRequiresGrad inner
    updateBuffers := some (fun mode {α} _ _ ps x =>
      TorchLean.NN.Seq.updateBuffers (α := α) (model := inner) mode ps x)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := ps ++ [s])
          (β := m (TorchLean.RefTy (m := m) (α := α) s))
          (fun args => do
            let (_psRefs, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps) (τ := s) args
            let y ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps ++ [s])
                (β := m (TorchLean.RefTy (m := m) (α := α) s))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := inner) (α := α))
                args
            TorchLean.add (m := m) (α := α) (s := s) y xRef)
  }

/-- Lift `residualLayer` into a sequential model. -/
def residual {s : Spec.Shape} (inner : Sequential s s) : Sequential s s :=
  nn.of (residualLayer inner)

/-!
## Branching (skip connections)

`Seq` is linear, but we sometimes want a PyTorch-like `x |-> f(x) + g(x)` block.

We expose this as a single `LayerDef` whose parameter list is `params(f) ++ params(g)` and whose
forward pass runs both programs and adds their outputs.
-/

/--
Combine two sequential branches into a single layer that adds their outputs.

The resulting layer runs both `f` and `g` on the same input `x` and returns `f(x) + g(x)`.
Parameters are concatenated as `params(f) ++ params(g)`.
-/
def addBranchesLayer {σ τ : Spec.Shape} (f g : Sequential σ τ) : LayerDef σ τ :=
  let psF := TorchLean.NN.Seq.paramShapes f
  let psG := TorchLean.NN.Seq.paramShapes g
  { paramShapes := psF ++ psG
    initParams :=
      tlist.append (α := Float) (ss₁ := psF) (ss₂ := psG)
        (TorchLean.NN.Seq.initParams f) (TorchLean.NN.Seq.initParams g)
    paramRequiresGrad := TorchLean.NN.Seq.paramRequiresGrad f ++ TorchLean.NN.Seq.paramRequiresGrad
      g
    updateBuffers := some (fun mode {α} _ _ ps x => do
      let (psFv, psGv) := tlist.split (α := α) (ss₁ := psF) (ss₂ := psG) ps
      let psFv' ← TorchLean.NN.Seq.updateBuffers (α := α) (model := f) mode psFv x
      let psGv' ← TorchLean.NN.Seq.updateBuffers (α := α) (model := g) mode psGv x
      pure <| tlist.append (α := α) (ss₁ := psF) (ss₂ := psG) psFv' psGv'
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
          (ss := psF ++ psG ++ [σ])
          (β := m (TorchLean.RefTy (m := m) (α := α) τ))
          (fun args => do
            let (psAll, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitAppend1
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ psG) (τ := σ) args
            let (psFrefs, psGrefs) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss₁ := psF) (ss₂ := psG) psAll
            let yF ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ [σ])
                (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := f) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psFrefs (.cons xRef .nil))
            let yG ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psG ++ [σ])
                (β := m (TorchLean.RefTy (m := m) (α := α) τ))
                (TorchLean.NN.Seq.programWithMode (mode := mode) (model := g) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psGrefs (.cons xRef .nil))
            TorchLean.add (m := m) (α := α) (s := τ) yF yG)
  }

/--
Combine two models with the same input/output shapes by summing their outputs.

This is a typed “residual add” helper: `addBranches f g` represents the model `x ↦ f(x) + g(x)`,
and its parameter list is the concatenation of the two branches’ parameter lists.
-/
def addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.of (addBranchesLayer f g)

/-!
## ResNet BasicBlock

We provide a *typed* and *composable* ResNet-18 style BasicBlock over CHW tensors.

Key idea: we use a small canonical stride-2 formula `down2` (matching `GraphSpec/Models/resnet18`)
so projection shortcuts typecheck cleanly without leaking Nat arithmetic at call sites.
-/

/--
Canonical stride-2 spatial downsampling formula used by ResNet blocks.

`down2 h = (h - 1) / 2 + 1 = ceil(h / 2)`.

This matches the output-size formula for common stride-2 layers used in ResNet downsampling
(e.g. `3×3` conv with padding `1`, or `1×1` conv with padding `0`).
-/
abbrev down2 (h : Nat) : Nat := (h - 1) / 2 + 1

/-- `down2` is always positive (used to discharge `NeZero` goals). -/
lemma down2_pos (h : Nat) : down2 h > 0 := by
  simp [down2]

/--
Shape arithmetic helper: `3×3` conv with stride `1` and padding `1` preserves a positive spatial
  size.

This matches the standard conv output formula used by `conv2dCHW`.
-/
lemma conv3_same_out_eq {h : Nat} (hh : h > 0) : ((h + 2 * 1 - 3) / 1 + 1) = h := by
  cases h with
  | zero => cases (Nat.lt_irrefl 0 hh)
  | succ _n => simp

/--
Shape arithmetic helper: `1×1` conv with stride `1` and padding `0` preserves a positive spatial
  size.
-/
lemma conv1_same_out_eq {h : Nat} (hh : h > 0) : ((h + 2 * 0 - 1) / 1 + 1) = h := by
  cases h with
  | zero => cases (Nat.lt_irrefl 0 hh)
  | succ _n => simp

/--
ResNet helper: `3×3` convolution with padding `1`, stride `1` (shape-preserving), over CHW images.
-/
def conv3x3Same {inC outC h w : Nat}
    [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC h w) := by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 1, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : h - 1 + 1 = h := by
    cases h with
    | zero => cases (Nat.lt_irrefl 0 hh)
    | succ _n => simp
  have hW : w - 1 + 1 = w := by
    cases w with
    | zero => cases (Nat.lt_irrefl 0 hw)
    | succ _n => simp
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 1 - 3) / 1 + 1) ((w + 2 * 1 - 3) / 1 + 1) =
        NN.Tensor.Shape.Image outC h w := by
    simpa [NN.Tensor.Shape.Image] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/--
ResNet helper: `3×3` convolution with padding `1`, stride `2` (spatial downsampling via `down2`),
  over CHW images.
-/
def conv3x3Down {inC outC h w : Nat} [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC (down2 h) (down2 w)) :=
      by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 2, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.CHW outC ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        NN.Tensor.Shape.CHW outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.CHW, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet helper: `1×1` convolution with stride `1` (shape-preserving), over CHW images. -/
def conv1x1Same {inC outC h w : Nat}
    [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC h w) := by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 1, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : h - 1 + 1 = h := by
    cases h with
    | zero => cases (Nat.lt_irrefl 0 hh)
    | succ _n => simp
  have hW : w - 1 + 1 = w := by
    cases w with
    | zero => cases (Nat.lt_irrefl 0 hw)
    | succ _n => simp
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 0 - 1) / 1 + 1) ((w + 2 * 0 - 1) / 1 + 1) =
        NN.Tensor.Shape.Image outC h w := by
    simpa [NN.Tensor.Shape.Image] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet helper: `1×1` convolution with stride `2` (spatial downsampling via `down2`), over CHW
  images. -/
def conv1x1Down {inC outC h w : Nat} [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC (down2 h) (down2 w)) :=
      by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 2, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 0 - 1) / 2 + 1) ((w + 2 * 0 - 1) / 2 + 1) =
        NN.Tensor.Shape.Image outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Image, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet helper: `3×3` convolution over batched images (`NCHW`-style), preserving spatial size. -/
def conv3x3SameImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC h w) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 1, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 1 - 3) / 1 + 1) = h := conv3_same_out_eq hh
  have hW : ((w + 2 * 1 - 3) / 1 + 1) = w := conv3_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 1 - 3) / 1 + 1) ((w + 2 * 1 - 3) / 1 + 1) =
        NN.Tensor.Shape.Images n outC h w := by
    simpa [NN.Tensor.Shape.Images] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet helper: `3×3` convolution over batched images (`NCHW`-style), downsampling via `down2`.
  -/
def conv3x3DownImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC (down2 h) (down2
      w)) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 2, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        NN.Tensor.Shape.Images n outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Images, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet helper: `1×1` convolution over batched images (`NCHW`-style), preserving spatial size. -/
def conv1x1SameImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC h w) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 1, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 0 - 1) / 1 + 1) = h := conv1_same_out_eq hh
  have hW : ((w + 2 * 0 - 1) / 1 + 1) = w := conv1_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 0 - 1) / 1 + 1) ((w + 2 * 0 - 1) / 1 + 1) =
        NN.Tensor.Shape.Images n outC h w := by
    simpa [NN.Tensor.Shape.Images] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet helper: `1×1` convolution over batched images (`NCHW`-style), downsampling via `down2`.
  -/
def conv1x1DownImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC (down2 h) (down2
      w)) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 2, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 0 - 1) / 2 + 1) ((w + 2 * 0 - 1) / 2 + 1) =
        NN.Tensor.Shape.Images n outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Images, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/--
ResNet-style "basic block" configuration (CHW layout).

PyTorch reference (conceptual):
`torchvision.models.resnet.BasicBlock` (see `https://pytorch.org/vision/stable/models/resnet.html`).
-/
structure ResNetBasicBlock where
  /-- Number of output channels produced by the block. -/
  outC : Nat
  /-- If true, use stride-2 downsampling + projection shortcut; otherwise preserve spatial dims. -/
  downsample : Bool := false
  /-- Activation used inside the block (and after the residual addition). -/
  activation : Activation := .relu
  /-- Base seed used to derive deterministic per-layer seeds inside the block. -/
  seedBase : Nat := 0

/--
ResNet-style "basic block" configuration (CHW layout).

This public building block follows the standard ResNet basic-block pattern:
`conv3x3 -> BN -> act -> conv3x3 -> BN` with a residual/skip connection.

PyTorch references (for the conceptual shape):
- Torchvision ResNet: `https://pytorch.org/vision/stable/models/resnet.html`
-/
def resnetBasicBlockCHW {inC h w : Nat} (cfg : ResNetBasicBlock)
    [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    Sequential
      (NN.Tensor.Shape.Image inC h w)
      (NN.Tensor.Shape.Image cfg.outC
        (if cfg.downsample then down2 h else h)
        (if cfg.downsample then down2 w else w)) := by
  classical
  -- Seed layout: conv/bn/conv/bn/(proj conv/bn) in a fixed order.
  let seedConv1K := cfg.seedBase + 0
  let seedConv1B := cfg.seedBase + 1
  let seedBN1G := cfg.seedBase + 2
  let seedBN1B := cfg.seedBase + 3
  let seedBN1M := cfg.seedBase + 4
  let seedBN1V := cfg.seedBase + 5
  let seedConv2K := cfg.seedBase + 6
  let seedConv2B := cfg.seedBase + 7
  let seedBN2G := cfg.seedBase + 8
  let seedBN2B := cfg.seedBase + 9
  let seedBN2M := cfg.seedBase + 10
  let seedBN2V := cfg.seedBase + 11
  let seedProjK := cfg.seedBase + 12
  let seedProjB := cfg.seedBase + 13
  let seedProjBNG := cfg.seedBase + 14
  let seedProjBNB := cfg.seedBase + 15
  let seedProjBNM := cfg.seedBase + 16
  let seedProjBNV := cfg.seedBase + 17

  if hDown : cfg.downsample = true then
    -- Stride-2 downsample.
    let h' := down2 h
    let w' := down2 w
    have hh' : h' > 0 := down2_pos h
    have hw' : w' > 0 := down2_pos w

    let conv1 : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w') :=
      conv3x3Down (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h' w') cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      conv3x3Same (inC := cfg.outC) (outC := cfg.outC) (h := h') (w := w')
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let projConv : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w')
      :=
      conv1x1Down (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedProjK) (seedB := seedProjB)
    let projBN : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC
      h' w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar :=
          seedProjBNV)
    let skip := seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w')
      :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC
      h' w') :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h' w') cfg.activation

    -- The `if` in the return type reduces via `hDown`.
    simpa [hDown] using (seq! summed, outAct)
  else
    -- Stride-1 (no spatial downsample).
    have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
    have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))

    let conv1 : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      conv3x3Same (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      TorchLean.Layers.batchNormCHW cfg.outC h w
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h w) cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      conv3x3Same (inC := cfg.outC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      TorchLean.Layers.batchNormCHW cfg.outC h w
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let skip : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      if hEq : cfg.outC = inC then
        -- Identity shortcut.
        have hShape : NN.Tensor.Shape.Image inC h w = NN.Tensor.Shape.Image cfg.outC h w := by
          simp [hEq]
        Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ)
          (_root_.Runtime.Autograd.TorchLean.NN.Seq.id (NN.Tensor.Shape.Image inC h w)) hShape
      else
        -- Projection shortcut (1x1 + BN).
        let projConv : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h
          w) :=
          conv1x1Same (inC := inC) (outC := cfg.outC) (h := h) (w := w)
            (seedK := seedProjK) (seedB := seedProjB)
        let projBN : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC
          h w) :=
          TorchLean.Layers.batchNormCHW cfg.outC h w
            (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
            (hH := hh) (hW := hw)
            (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar
              := seedProjBNV)
        seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h
      w) :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h w) cfg.activation

    have hDown' : cfg.downsample = false := by
      cases hds : cfg.downsample with
      | false => rfl
      | true => cases (hDown hds)
    simpa [hDown'] using (seq! summed, outAct)

/-- ResNet-18 style BasicBlock over batched image tensors (`N×C×H×W`). -/
def resnetBasicBlock {n inC h w : Nat} (cfg : ResNetBasicBlock)
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    Sequential
      (NN.Tensor.Shape.Images n inC h w)
      (NN.Tensor.Shape.Images n cfg.outC
        (if cfg.downsample then down2 h else h)
        (if cfg.downsample then down2 w else w)) := by
  classical
  have hn : n > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := n))
  -- Seed layout: conv/bn/conv/bn/(proj conv/bn) in a fixed order.
  let seedConv1K := cfg.seedBase + 0
  let seedConv1B := cfg.seedBase + 1
  let seedBN1G := cfg.seedBase + 2
  let seedBN1B := cfg.seedBase + 3
  let seedBN1M := cfg.seedBase + 4
  let seedBN1V := cfg.seedBase + 5
  let seedConv2K := cfg.seedBase + 6
  let seedConv2B := cfg.seedBase + 7
  let seedBN2G := cfg.seedBase + 8
  let seedBN2B := cfg.seedBase + 9
  let seedBN2M := cfg.seedBase + 10
  let seedBN2V := cfg.seedBase + 11
  let seedProjK := cfg.seedBase + 12
  let seedProjB := cfg.seedBase + 13
  let seedProjBNG := cfg.seedBase + 14
  let seedProjBNB := cfg.seedBase + 15
  let seedProjBNM := cfg.seedBase + 16
  let seedProjBNV := cfg.seedBase + 17

  if hDown : cfg.downsample = true then
    -- Stride-2 downsample.
    let h' := down2 h
    let w' := down2 w
    have hh' : h' > 0 := down2_pos h
    have hw' : w' > 0 := down2_pos w
    have hH' : h' ≠ 0 := Nat.ne_of_gt hh'
    have hW' : w' ≠ 0 := Nat.ne_of_gt hw'
    letI : NeZero h' := ⟨hH'⟩
    letI : NeZero w' := ⟨hW'⟩

    let conv1 : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h'
      w') :=
      conv3x3DownImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h' w') cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      conv3x3SameImages (n := n) (inC := cfg.outC) (outC := cfg.outC) (h := h') (w := w')
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let projConv : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC
      h' w') :=
      conv1x1DownImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedProjK) (seedB := seedProjB)
    let projBN : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar :=
          seedProjBNV)
    let skip := seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h'
      w') :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h' w') cfg.activation

    -- The `if` in the return type reduces via `hDown`.
    simpa [hDown] using (seq! summed, outAct)
  else
    -- Stride-1 (no spatial downsample).
    have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
    have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))

    let conv1 : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h
      w) :=
      conv3x3SameImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n cfg.outC
      h w) :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h w) cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      conv3x3SameImages (n := n) (inC := cfg.outC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n cfg.outC
      h w) :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let skip : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h w)
      :=
      if hEq : cfg.outC = inC then
        -- Identity shortcut.
        have hShape : NN.Tensor.Shape.Images n inC h w = NN.Tensor.Shape.Images n cfg.outC h w := by
          simp [hEq]
        Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ)
          (_root_.Runtime.Autograd.TorchLean.NN.Seq.id (NN.Tensor.Shape.Images n inC h w)) hShape
      else
        -- Projection shortcut (1x1 + BN).
        let projConv : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n
          cfg.outC h w) :=
          conv1x1SameImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
            (seedK := seedProjK) (seedB := seedProjB)
        let projBN : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
          cfg.outC h w) :=
          TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
            (hN := hn)
            (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
            (hH := hh) (hW := hw)
            (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar
              := seedProjBNV)
        seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h
      w) :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h w) cfg.activation

    have hDown' : cfg.downsample = false := by
      cases hds : cfg.downsample with
      | false => rfl
      | true => cases (hDown hds)
    simpa [hDown'] using (seq! summed, outAct)

/--
Config record for `transformerEncoderBlock`.

Separating the config as a structure makes it easier to write readable examples and keep seed
management deterministic.
-/
structure TransformerEncoderBlock where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Hidden dimension of the feed-forward network. -/
  ffnHidden : Nat
  /-- Activation used in the feed-forward network. -/
  activation : Activation := .gelu
  /-- Optional dropout probability for examples; `none` means no dropout. -/
  dropout? : Option Float := none
  /-- Base seed used to derive deterministic per-layer seeds inside the block. -/
  seedBase : Nat := 0

/--
Transformer encoder block configuration.

This follows the familiar pattern:
`(residual MHA) -> LayerNorm -> (residual FFN) -> LayerNorm`.

PyTorch analogue:
- `torch.nn.TransformerEncoderLayer`
  (`https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html`)
-/
def transformerEncoderBlockWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
  let seedAttn := cfg.seedBase
  let seedNorm1Gamma := cfg.seedBase + 1
  let seedNorm1Beta := cfg.seedBase + 2
  let seedFfnW1 := cfg.seedBase + 3
  let seedFfnB1 := cfg.seedBase + 4
  let seedFfnW2 := cfg.seedBase + 5
  let seedFfnB2 := cfg.seedBase + 6
  let seedNorm2Gamma := cfg.seedBase + 7
  let seedNorm2Beta := cfg.seedBase + 8
  let seedDrop1 := cfg.seedBase + 9
  let seedDrop2 := cfg.seedBase + 10

  let attn : Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
    multiheadAttentionWith (batch := batch) (n := n) (dModel := dModel)
      { numHeads := cfg.numHeads, headDim := cfg.headDim, seedW := seedAttn }
      (hN := NeZero.ne (n := n))
      (mask := mask)
  let attnInner :=
    match cfg.dropout? with
    | none => attn
    | some p =>
        seq! attn, dropout (s := .dim batch (NN.Tensor.Shape.Mat n dModel)) p (seed := seedDrop1)
  let norm1 : Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
    layerNorm (batch := batch) (seqLen := n) (embedDim := dModel)
      { seedGamma := seedNorm1Gamma, seedBeta := seedNorm1Beta }

  let ffn : Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
    seq!
      linear dModel cfg.ffnHidden seedFfnW1 seedFfnB1 (pfx := .dim batch (NN.Tensor.Shape.Vec n)),
      activation (s := .dim batch (NN.Tensor.Shape.Mat n cfg.ffnHidden)) cfg.activation,
      linear cfg.ffnHidden dModel seedFfnW2 seedFfnB2 (pfx := .dim batch (NN.Tensor.Shape.Vec n))
  let ffnInner :=
    match cfg.dropout? with
    | none => ffn
    | some p =>
        seq! ffn, dropout (s := .dim batch (NN.Tensor.Shape.Mat n dModel)) p (seed := seedDrop2)
  let norm2 : Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
    layerNorm (batch := batch) (seqLen := n) (embedDim := dModel)
      { seedGamma := seedNorm2Gamma, seedBeta := seedNorm2Beta }

  seq!
    residual attnInner,
    norm1,
    residual ffnInner,
    norm2

/--
Transformer encoder block.

This is `transformerEncoderBlockWithMask`; pass `mask := ...` to enable causal masking (or other
attention masks).
-/
def transformerEncoderBlock {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
  transformerEncoderBlockWithMask (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

/--
Config record for `transformerEncoderStack`.

This builds `layers` copies of `transformerEncoderBlock`, allocating seeds in a fixed stride.
-/
structure TransformerEncoderStack where
  /-- Layer stack. -/
  layers : Nat
  /-- Template config for each block (its `seedBase` is ignored; we allocate per-layer seeds). -/
  block : TransformerEncoderBlock
  /-- Base seed for the whole stack. -/
  seedBase : Nat := 0
  /-- Seed stride between consecutive blocks (must exceed the per-block seed footprint). -/
  seedStride : Nat := 100

/--
Internal recursion for `transformerEncoderStack`.

Builds `remaining` blocks starting at `layerIdx`, allocating each block's `seedBase` as
`seedBase + layerIdx * seedStride`.
-/
def transformerStackGoWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (template : TransformerEncoderBlock) (seedBase seedStride : Nat)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    (layerIdx : Nat) → (remaining : Nat) →
      Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel))
  | _layerIdx, 0 =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.id (.dim batch (NN.Tensor.Shape.Mat n dModel))
  | layerIdx, remaining + 1 =>
      let seed := seedBase + layerIdx * seedStride
      let blockCfg : TransformerEncoderBlock := { template with seedBase := seed }
      let here := transformerEncoderBlockWithMask (batch := batch) (n := n) (dModel := dModel) blockCfg
        (mask := mask)
      let rest :=
        transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
          template seedBase seedStride (mask := mask)
          (layerIdx + 1) remaining
      seq! here, rest

/--
Internal recursion for `transformerEncoderStack` (unmasked).

This is `transformerStackGoWithMask` with `mask := none`.
-/
def transformerStackGo {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (template : TransformerEncoderBlock) (seedBase seedStride : Nat) :
    (layerIdx : Nat) → (remaining : Nat) →
      Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
  transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
    template seedBase seedStride (mask := none)

/--
Stack `cfg.layers` copies of `blocks.transformerEncoderBlock`.

This is the TorchLean analogue of composing `torch.nn.TransformerEncoderLayer` into a
`torch.nn.TransformerEncoder` (modulo the fact that TorchLean uses `Seq` composition).
-/
def transformerEncoderStackWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
  transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
    cfg.block cfg.seedBase cfg.seedStride (mask := mask) 0 cfg.layers

/--
Stack `cfg.layers` copies of `blocks.transformerEncoderBlock`.

This is `transformerEncoderStackWithMask`; pass `mask := ...` to enable causal masking (or other
attention masks).
-/
def transformerEncoderStack {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel)) :=
  transformerEncoderStackWithMask (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

/--
Transformer encoder followed by a flatten+linear classification head.

PyTorch analogue (roughly): `nn.TransformerEncoder(...)` + pooling/flattening + `nn.Linear`.
-/
def transformerEncoderClassifier {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (classes : Nat) (cfg : TransformerEncoderStack) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Vec classes)) :=
  let enc := transformerEncoderStack (batch := batch) (n := n) (dModel := dModel) cfg
  let seedHeadW := cfg.seedBase + cfg.layers * cfg.seedStride
  let seedHeadB := seedHeadW + 1
  let flat : Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (NN.Tensor.Shape.Mat batch (Spec.Shape.size (NN.Tensor.Shape.Mat n dModel))) :=
    flattenBatch (n := batch) (s := NN.Tensor.Shape.Mat n dModel)
  let head : Sequential (NN.Tensor.Shape.Mat batch (Spec.Shape.size (NN.Tensor.Shape.Mat n dModel)))
      (.dim batch (NN.Tensor.Shape.Vec classes)) :=
    linear (Spec.Shape.size (NN.Tensor.Shape.Mat n dModel)) classes
      (seedW := seedHeadW) (seedB := seedHeadB)
      (pfx := NN.Tensor.Shape.Vec batch)
  seq! enc, flat, head

end blocks

namespace heads

/--
Classification head: `Flatten -> Linear`.

This is a small convenience wrapper around `nn.flattenLinear`.
-/
def classifier {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential s (NN.Tensor.Shape.Vec classes) :=
  flattenLinear (s := s) classes seedW seedB

/-- Regression head: `Flatten -> Linear` with `outDim` outputs. -/
def regressor {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential s (NN.Tensor.Shape.Vec outDim) :=
  flattenLinear (s := s) outDim seedW seedB

/--
`Flatten(start_dim=1) -> Linear` head for batched tensors.

Input:  `N × σ`
Output: `Mat N classes`
-/
def classifierBatch {n : Nat} {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n classes) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) classes (seedW := seedW) (seedB := seedB) (pfx :=
      NN.Tensor.Shape.Vec n)

/-- Batched regression head: `Flatten(start_dim=1) -> Linear(_, outDim)` producing `Mat N outDim`.
  -/
def regressorBatch {n : Nat} {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (NN.Tensor.Shape.Mat n outDim) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) outDim (seedW := seedW) (seedB := seedB) (pfx :=
      NN.Tensor.Shape.Vec n)

end heads

end pure

end nn

namespace optim

/-!
Optimizer configs for the high-level training helpers.

These mirror common PyTorch optimizers (by name and default hyperparameters), but they produce a
TorchLean trainer config rather than a mutable optimizer object.

PyTorch references:
- `torch.optim`: `https://pytorch.org/docs/stable/optim.html`
-/

@[inherit_doc TorchLean.Trainer.Optimizer]
abbrev Optimizer := TorchLean.Trainer.Optimizer

@[inherit_doc TorchLean.Trainer.sgd]
abbrev sgd := TorchLean.Trainer.sgd

@[inherit_doc TorchLean.Trainer.momentumSGD]
abbrev momentumSGD := TorchLean.Trainer.momentumSGD

@[inherit_doc TorchLean.Trainer.adam]
abbrev adam := TorchLean.Trainer.adam

@[inherit_doc TorchLean.Trainer.adamw]
abbrev adamw := TorchLean.Trainer.adamw

end optim

namespace loss

/-
Loss functions are re-exported from the TorchLean runtime.

PyTorch references:
- `torch.nn.functional` loss docs: `https://pytorch.org/docs/stable/nn.functional.html`
-/

@[inherit_doc TorchLean.Loss.Reduction]
abbrev Reduction := TorchLean.Loss.Reduction

export TorchLean.Loss
  (mse
   nllOneHot crossEntropyOneHot
   nllIndex nllNat crossEntropyIndex crossEntropyNat
   bceWithLogits bce)

end loss

namespace metrics

/- Small classification metrics helpers (argmax, one-hot correctness, etc.). -/
export TorchLean.Metrics (argmax? classOfOneHot? correctOneHot?)

end metrics

namespace train

/-!
High-level training helpers.

This namespace is designed for executable demos: it wires together
- a model (`nn.Sequential`)
- a loss (regression or classification)
- an optimizer config (`API.optim`)
- optional LR schedules

It stays intentionally lightweight: rather than hiding everything behind a large framework, it
exposes a small set of default building blocks so tutorials can focus on models and verification.

### PyTorch Mapping

These helpers correspond to the training loop code you would typically write around:
- `torch.optim.*`
- forward pass + loss
- `loss.backward()` + optimizer step
- batching via `torch.utils.data.DataLoader`
-/

@[inherit_doc TorchLean.Trainer.Task]
abbrev Task := TorchLean.Trainer.Task
@[inherit_doc TorchLean.Trainer.Runner]
abbrev Runner := TorchLean.Trainer.Runner
@[inherit_doc TorchLean.Trainer.Stepper]
abbrev Stepper := TorchLean.Trainer.Stepper

@[inherit_doc TorchLean.Trainer.FitConfig]
abbrev FitConfig := TorchLean.Trainer.FitConfig
@[inherit_doc TorchLean.Trainer.LoaderFitConfig]
abbrev LoaderFitConfig := TorchLean.Trainer.LoaderFitConfig
@[inherit_doc TorchLean.Trainer.FitReport]
abbrev FitReport := TorchLean.Trainer.FitReport

/-!
Most of `API.train.*` is just a public re-export of `TorchLean.Trainer.*`.

We use `export` (rather than rewriting 1-line forwarders) so this file stays small and avoids
duplicating implementation details at the facade layer.
-/

export TorchLean.Trainer
  (regression
   classificationOneHot
   steps epochs
   constantLR stepLR exponentialLR
   constantEpochLR stepEpochLR exponentialEpochLR
   instantiate instantiateWithOptions
   run
   params mode setMode trainMode evalMode isTraining
   backward
   predict predictBatch predictClass?
   accuracyOneHot)

/-!
## Metric Artifacts

The public training facade also exposes TorchLean's lightweight metric artifact format.  This is
the local equivalent of “log scalars during a run, then inspect them later”: write a JSON
`TrainLog`, view it with the training widgets, or adapt the JSON to an external tracker such as
Weights & Biases.
-/

export _root_.Runtime.Training
  (Series TrainLog Curve MetricHistory ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.MetricHistory (empty push toTrainLog)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)

/--
A runner bundled with the task that created it.

This is an ergonomic wrapper around `Runner α task`: it remembers the dependent `task`, so tutorial
code can call `tr.predict x`, `tr.fit cfg samples`, etc. without repeatedly writing
`(task := task)`.
-/
structure TaskRunner (σ τ : Spec.Shape) (α : Type)
    [Semantics.Scalar α] [DecidableEq Spec.Shape] where
  /-- The supervised task: model plus loss. -/
  task : Task σ τ
  /-- The instantiated runner for `task`. -/
  runner : Runner α task

namespace TaskRunner

/-- Bundle an existing runner with its task. -/
def ofRunner {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : TaskRunner σ τ α :=
  { task := task, runner := runner }

/-- Get the current model parameters from a bundled runner. -/
def params {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) :
    IO (TorchLean.TList α (TorchLean.Supervised.paramShapes tr.task)) :=
  train.params tr.runner

/-- Read the current mode (`.train` or `.eval`) from a bundled runner. -/
def mode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO TorchLean.NN.Mode :=
  train.mode tr.runner

/-- Set the mode (`.train` or `.eval`) on a bundled runner. -/
def setMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (value : TorchLean.NN.Mode) : IO Unit :=
  train.setMode tr.runner value

/-- Switch a bundled runner to training mode. -/
def trainMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Unit :=
  train.trainMode tr.runner

/-- Switch a bundled runner to evaluation mode. -/
def evalMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Unit :=
  train.evalMode tr.runner

/-- Check whether a bundled runner is in training mode. -/
def isTraining {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Bool :=
  train.isTraining tr.runner

/-- Predict on one input tensor using the bundled runner's active mode. -/
def predict {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) :=
  train.predict tr.runner x

/-- Predict on a list of inputs using the bundled runner's active mode. -/
def predictBatch {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  train.predictBatch tr.runner xs

/-- Mean loss over an entire dataset for a bundled runner. -/
def meanLossDataset {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (tr : TaskRunner σ τ α) (dataset : _root_.Runtime.Autograd.Train.Dataset
      (sample.Supervised α σ τ)) :
    IO α :=
  TorchLean.Trainer.meanLossDataset (task := tr.task) tr.runner dataset

/-- Fit a bundled runner on an explicit list of samples for a fixed number of steps. -/
def fit {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : FitConfig) (samples : List (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fit (task := tr.task) tr.runner cfg samples

/-- Fit a bundled runner on a `Dataset` for a fixed number of steps. -/
def fitDataset {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : FitConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fitDataset (task := tr.task) tr.runner cfg dataset

/-- Fit a bundled runner using a `DataLoader` for a fixed number of epochs. -/
def fitLoader {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :=
  TorchLean.Trainer.fitLoader (task := tr.task) tr.runner cfg loader

end TaskRunner

/--
CLI-oriented runner entry point that passes a bundled `TaskRunner` to the continuation.

This mirrors `train.run`, but removes the need to keep threading `(task := task)` after
instantiation.
-/
def runTask {σ τ : Spec.Shape} (task : Task σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [Runtime.Scalar α] → TaskRunner σ τ α → List String → IO Unit) :
    IO Unit :=
  run task args (fun {α} _ _ _ _ runner rest =>
    k (α := α) { task := task, runner := runner } rest)

/--
Count correct predictions in a one-hot labeled **batched** dataset.

This is the minibatch analogue of `accuracyOneHot`: the task already has a leading dim0 batch axis,
so we score each row of the batch independently and accumulate totals.

Returns `(correct, total)` where `total = batch * numBatches`.
-/
def accuracyOneHotBatched
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (samples : List (sample.Batch α batch σ (NN.Tensor.Shape.Vec classes)))
      :
    IO (Nat × Nat) := do
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for s in samples do
    let xBatch := sample.x s
    let yBatch := sample.y s
    let logitsBatch ← predict (task := task) runner xBatch
    for i in List.finRange batch do
      let logits := Spec.getAtSpec logitsBatch i
      let target := Spec.getAtSpec yBatch i
      if let some true := metrics.correctOneHot? logits target then
        correct := correct + 1
      total := total + 1
  pure (correct, total)

/-- Mean loss over an entire dataset (useful for quick before/after reports). -/
def meanLossDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task) (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ
      τ)) :
    IO α :=
  TorchLean.Trainer.meanLossDataset (task := task) runner dataset

/-- Fit on an explicit list of samples for a fixed number of steps. -/
def fit {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (samples : List (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fit (task := task) runner cfg samples

/-- Fit on a `Dataset` for a fixed number of steps. -/
def fitDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (dataset : _root_.Runtime.Autograd.Train.Dataset
      (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fitDataset (task := task) runner cfg dataset

/-- Fit using a `DataLoader` for a fixed number of epochs. -/
def fitLoader {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :=
  TorchLean.Trainer.fitLoader (task := task) runner cfg loader

/-- Callback event fired after each training step. -/
structure StepEvent (α : Type) where
  /-- Current epoch number. -/
  epoch : Nat
  /-- Global optimizer-step counter. -/
  step : Nat
  /-- Loss reported for this step. -/
  loss : α

/-- Callback event fired at the end of an epoch (how many steps ran). -/
structure EpochEvent where
  /-- Epoch number that just completed. -/
  epoch : Nat
  /-- Number of steps executed in the epoch. -/
  steps : Nat

/--
Hooks for instrumenting `fitLoaderBatched`-style training loops.

These are lightweight by design (IO callbacks). If you want richer logging, consider building a wrapper
in your own project that translates these events into structured JSON/metrics.
-/
structure Callbacks (α : Type) where
  /-- Called once before training starts. -/
  onTrainStart : IO Unit := pure ()
  /-- Called after each training step. -/
  onStep : StepEvent α → IO Unit := fun _ => pure ()
  /-- Called after each epoch. -/
  onEpochEnd : EpochEvent → IO Unit := fun _ => pure ()
  /-- Called once after training finishes. -/
  onTrainEnd : FitReport α → IO Unit := fun _ => pure ()

namespace Callbacks

/-- No-op callbacks. -/
def empty {α : Type} : Callbacks α := {}

/-- Combine two callback collections by running them in sequence. -/
def append {α : Type} (a b : Callbacks α) : Callbacks α :=
  { onTrainStart := do
      a.onTrainStart
      b.onTrainStart
    onStep := fun ev => do
      a.onStep ev
      b.onStep ev
    onEpochEnd := fun ev => do
      a.onEpochEnd ev
      b.onEpochEnd ev
    onTrainEnd := fun report => do
      a.onTrainEnd report
      b.onTrainEnd report
  }

/-- `∅` for callbacks: a no-op callback collection. -/
instance {α : Type} : EmptyCollection (Callbacks α) where
  emptyCollection := empty

/-- `Callbacks` form a monoid under sequential composition. -/
instance {α : Type} : Append (Callbacks α) where
  append := append

end Callbacks

/-- Build callbacks that run `action` once at the start of training. -/
def onTrainStart {α : Type} (action : IO Unit) : Callbacks α :=
  { onTrainStart := action }

/-- Build callbacks that observe every training step. -/
def onStep {α : Type} (f : StepEvent α → IO Unit) : Callbacks α :=
  { onStep := f }

/-- Build callbacks that run at the end of each epoch. -/
def onEpochEnd {α : Type} (f : EpochEvent → IO Unit) : Callbacks α :=
  { onEpochEnd := f }

/-- Build callbacks that run once at the end of training, with the final report. -/
def onTrainEnd {α : Type} (f : FitReport α → IO Unit) : Callbacks α :=
  { onTrainEnd := f }

/-- Callback helper: log the loss every `every` steps (if `every > 0`). -/
def logLossEvery {α : Type} [ToString α] (every : Nat := 1) : Callbacks α :=
  onStep (fun ev => do
    if every > 0 && ev.step % every = 0 then
      IO.println s!"step {ev.epoch}:{ev.step}: loss={ev.loss}")

/--
Run an action with the runner temporarily switched to `value` mode.

This is useful for "evaluate on a validation set during training" in callback-based loops.
-/
def withMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    {β : Type} (runner : Runner α task) (value : TorchLean.NN.Mode) (action : IO β) : IO β := do
  let prev ← mode runner
  setMode runner value
  try
    action
  finally
    setMode runner prev

/--
Mean loss for an already-instantiated scalar module over a typed minibatch loader.

This is the general streaming evaluation path used by the runtime examples.  It is deliberately
not CIFAR-specific: any supervised task whose loss module consumes
`[dim n σ, dim n τ]` can use the same loader.  The loader stores ordinary per-example samples
`(x : σ, y : τ)`; this helper asks `Data.epoch` for raw minibatches and calls
`Data.collateSupervised` to build one shape-typed batch at a time.

Two details are important for larger examples:

- We force `shuffle := false` for evaluation so before/after metrics are deterministic.
- We do not call `Data.BatchLoader.batchDataset`, because that would materialize every collated
  minibatch at once.  Streaming keeps the same API usable for image, sequence, and scientific ML
  examples where the batch tensors are much larger than small tabular datasets.
-/
def meanLossModuleLoader {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (loader : Data.BatchLoader α n σ τ) : IO α := do
  let evalLoader : Data.RawDataLoader (sample.Supervised α σ τ) :=
    { loader.raw with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match Data.epoch "train.meanLossModuleLoader" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"train.meanLossModuleLoader: {msg}"
  let mut total : α := 0
  let mut count : Nat := 0
  for rawBatch in rawBatches do
    let sample ← Common.orThrow "train.meanLossModuleLoader" <|
      Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
    let lossTensor ← TorchLean.Module.forward module sample
    let loss := Spec.Tensor.toScalar lossTensor
    total := total + loss
    count := count + 1
  if count = 0 then
    pure 0
  else
    pure (total / (count : α))

/--
Mean loss over a typed minibatch loader through a `train.Runner`.

This is the runner-facing wrapper around `meanLossModuleLoader`.  Use it when the example is built
around `train.run`, task modes, and the proof-facing trainer abstraction.  Use
`meanLossModuleLoader` directly when the example has already instantiated a runtime
`TorchLean.Module.ScalarModule`, which is the common fast path for CUDA demos.
-/
def meanLossBatchLoader {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task) (loader : Data.BatchLoader α n σ τ) : IO α :=
  meanLossModuleLoader runner.module loader

/-- One-hot accuracy over a typed minibatch loader without materializing all collated batches. -/
def accuracyOneHotBatchLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ (NN.Tensor.Shape.Vec classes)) :
    IO (Nat × Nat) := do
  let evalLoader : Data.RawDataLoader (sample.Supervised α σ (NN.Tensor.Shape.Vec classes)) :=
    { loader.raw with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match Data.epoch "train.accuracyOneHotBatchLoader" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"train.accuracyOneHotBatchLoader: {msg}"
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for rawBatch in rawBatches do
    let sample ← Common.orThrow "train.accuracyOneHotBatchLoader" <|
      Data.collateSupervised (α := α) (σ := σ) (τ := NN.Tensor.Shape.Vec classes) batch rawBatch
    let (c, t) ← accuracyOneHotBatched (task := task) runner [sample]
    correct := correct + c
    total := total + t
  pure (correct, total)

/--
Train a runtime scalar module from a typed minibatch loader.

This is the shared "real epoch loop" for model examples that instantiate a module directly with
`TorchLean.Module.instantiateWithOptions`, including CUDA runs.  It mirrors the PyTorch structure:

1. create an optimizer state for the module parameters;
2. for each epoch, ask the general `Data.batchLoader` for shuffled raw batches;
3. collate each raw batch into a shape-typed `(xBatch, yBatch)` sample;
4. report the scalar loss through callbacks;
5. run `forward/backward/optimizer.step` through `TorchLean.Module.stepWith`.

The function is polymorphic in the input shape `σ`, target shape `τ`, batch size `n`, scalar type
`α`, parameter shapes, and optimizer.  It is not an image-specific helper.  CNN, ResNet, ViT, MLP,
sequence, operator-learning, and future model demos should all be able to use this path whenever
their supervised loss module has input shapes `[dim n σ, dim n τ]`.
-/
def fitModuleLoaderWith {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (epochs : Nat)
    (loader : Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  let before ← meanLossModuleLoader module loader
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptim module optimizer
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:epochs] do
    let (rawNext, rawBatches) ←
      match Data.epoch "train.fitModuleLoaderWith" dl.raw with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.fitModuleLoaderWith: {msg}"
    dl := { raw := rawNext }
    for rawBatch in rawBatches do
      let sample ← Common.orThrow "train.fitModuleLoaderWith" <|
        Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let lossTensor ← TorchLean.Module.forward module sample
      let loss := Spec.Tensor.toScalar lossTensor
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      optState ← TorchLean.Module.stepWith module optimizer optState sample
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  let after ← meanLossModuleLoader module dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Train from a runner-backed loader with explicit callbacks instead of inline printing in example
code.

This is the proof/trainer-facing public escape hatch for PyTorch-style custom loops:
- keep the optimizer/scheduler logic in the library,
- inject logging, evaluation, and probe reporting through callbacks.

This path keeps the `Runner` abstraction, including task modes and scheduler support.  For
CUDA-heavy tutorials that already have a `TorchLean.Module.ScalarModule`, prefer
`fitModuleLoaderWith`; both paths consume the same general `API.Data.batchLoader`.
-/
def fitLoaderWith {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  evalMode runner
  let before ← meanLossBatchLoader (task := task) runner loader
  callbacks.onTrainStart

  trainMode runner
  let loop ← TorchLean.Trainer.stepper (task := task) runner cfg.optimizer (scheduler :=
    cfg.scheduler)
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:cfg.epochs] do
    let (rawNext, rawBatches) ←
      match Data.epoch "train.fitLoaderWith" dl.raw with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.fitLoaderWith: {msg}"
    dl := { raw := rawNext }
    for rawBatch in rawBatches do
      let sample ← Common.orThrow "train.fitLoaderWith" <|
        Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let loss ← TorchLean.Trainer.step (task := task) loop sample
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  evalMode runner
  let after ← meanLossBatchLoader (task := task) runner dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Public minibatch training path.

`data.batchLoader` produces a typed `BatchLoader` (with a type-level batch size `n`), and this
helper bridges from an untyped runtime loader into the typed training loop.
-/
def fitLoaderBatched {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) := do
  if loader.batchSize != n then
    throw <| IO.userError
      s!"train.fitLoaderBatched: expected loader.batchSize={n}, got {loader.batchSize}"
  if !loader.dropLast then
    throw <| IO.userError "train.fitLoaderBatched: expected loader.dropLast=true"

  -- Prefer the typed-loader implementation: it is the canonical “minibatch” training loop.
  let typed : Data.BatchLoader α n σ τ := { raw := loader }
  let callbacks :=
    if cfg.logEvery > 0 then
      logLossEvery (α := α) cfg.logEvery
    else
      Callbacks.empty
  let (report, typed') ← fitLoaderWith (runner := runner) (cfg := cfg) (loader := typed) (callbacks
    := callbacks)
  pure (report, typed'.raw)

/--
Create a `Stepper` loop for a runner and optimizer (optionally with an LR scheduler).

This corresponds to the “inner training loop” state in typical PyTorch code:
an optimizer state plus (optional) schedule state, ready to step on a batch.
-/
def stepper {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (optimizer : optim.Optimizer)
    (scheduler : Option TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) :=
  TorchLean.Trainer.stepper (task := task) runner optimizer scheduler

/-- Run one optimization step on a single supervised sample (one batch). -/
def step {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : sample.Supervised α σ τ) : IO α :=
  TorchLean.Trainer.step (task := task) loop sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (sample.Supervised α σ τ)) : IO (List α) :=
  TorchLean.Trainer.epoch (task := task) loop samples

namespace Report

/-!
### Small Reporting Helpers (IO)

These helpers keep tutorial code readable by factoring out common "print a loss/accuracy table"
patterns. They do not affect semantics: they only call the underlying `train.*` functions and
print human-facing summaries.
-/

/-- Print a titled list of probe lines. -/
def reportProbes {β : Type} (title : String) (probes : List β) (lineOf : β → IO String) : IO Unit :=
  do
  IO.println title
  for p in probes do
    IO.println (← lineOf p)

/-- Convenience: mean loss on a dataset, printed with a label. -/
def reportMeanLoss
    {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  IO.println s!"mean_loss({label}) = {loss}"

/-- Convenience: mean loss on a typed minibatch loader, streamed batch by batch. -/
def reportMeanLossLoader
    {σ τ : Spec.Shape} {batch : Nat} {task : Task (.dim batch σ) (.dim batch τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← meanLossBatchLoader (task := task) runner loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Convenience: mean loss on a typed minibatch loader for an already-instantiated runtime module.

Use this in direct CUDA/runtime examples to avoid building a `Runner` only for logging.  The data
path is still the same public loader path: `Data.batchLoader` plus `Data.collateSupervised`.
-/
def reportMeanLossModuleLoader
    {σ τ : Spec.Shape} {batch : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim batch σ, Spec.Shape.dim
      batch τ])
    (loader : Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← meanLossModuleLoader module loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Report predicted classes on a list of named probes.

Each probe entry is `(name, x, expectedClass)`.
If `includeLogits := true`, also prints the raw model outputs.
-/
def reportClassProbes
    {σ : Spec.Shape} {classes : Nat} {task : Task σ (.dim classes .scalar)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  reportProbes title probes (fun (name, x, expected) => do
    let logits ← predict (task := task) runner x
    let pred? := metrics.argmax? logits
    let predStr :=
      match pred? with
      | some k => toString k.val
      | none => "none"
    let logitsStr :=
      if includeLogits then
        s!" logits={Spec.pretty logits}"
      else
        ""
    pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/--
Report predicted classes on a list of named probes, for a **batched** model.

This expects probes of the *unbatched* input shape `σ` and replicates each probe across the batch
axis, then reports the prediction for row 0.
-/
def reportClassProbesBatchedFromSingle
    {σ : Spec.Shape} {classes batch : Nat} {task : Task (.dim batch σ) (.dim batch
      (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  reportProbes title probes (fun (name, xSingle, expected) => do
    let xBatch : Spec.Tensor α (.dim batch σ) :=
      Spec.Tensor.dim (fun _ => xSingle)
    let logitsBatch ← predict (task := task) runner xBatch
    -- If `batch = 0`, there is no row to display. That case is not meaningful for training anyway.
    match List.finRange batch with
    | [] =>
        pure s!"  {name}: batch=0 (no prediction)"
    | i0 :: _ =>
        let logits0 := Spec.getAtSpec logitsBatch i0
        let pred? := metrics.argmax? logits0
        let predStr :=
          match pred? with
          | some k => toString k.val
          | none => "none"
        let logitsStr :=
          if includeLogits then
            s!" logits={Spec.pretty logits0}"
          else
            ""
        pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/-- Convenience: mean loss + one-hot accuracy on a dataset, printed with a label. -/
def reportLossAccuracyOneHot
    {σ : Spec.Shape} {classes : Nat} {task : Task σ (.dim classes .scalar)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ (.dim classes .scalar)))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  let (correct, total) ← accuracyOneHot (task := task) runner dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Batched variant of `reportLossAccuracyOneHot`. -/
def reportLossAccuracyOneHotBatched
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Batch α batch σ (NN.Tensor.Shape.Vec
      classes)))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  let (correct, total) ← accuracyOneHotBatched (task := task) runner dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Loader variant of `reportLossAccuracyOneHotBatched`, streaming through minibatches. -/
def reportLossAccuracyOneHotLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ (NN.Tensor.Shape.Vec classes))
    (label : String) : IO Unit := do
  let loss ← meanLossBatchLoader (task := task) runner loader
  let (correct, total) ← accuracyOneHotBatchLoader (task := task) runner loader
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

end Report

end train

namespace nn

/-!
## Model Builders and Seeding

TorchLean keeps initialization randomness explicit so examples are reproducible.

- `nn.*` is the default *seeded builder* API: layer constructors allocate initialization seeds
  via `nn.M` (a deterministic seed stream).
- `nn.pure.*` contains the explicit-seed constructors (proof/reproducibility-friendly).

Typical patterns:

1. Explicit seeds (best for proofs / reproducibility-sensitive code):
   - build with `nn.pure.linear ... (seedW := ...) (seedB := ...)` etc
   - compose with `seq! ...` / `>>>`

2. Script-style “manual seed once”:
   - `nn.manualSeed seed`
   - `let seed ← nn.nextSeed`
   - `let model := nn.run seed <| nn.sequential![ ... ]`

Note: `nn.Sequential` lives in `Type 2`, so it cannot be returned directly from `IO`. We keep
model building pure by drawing a base seed in `IO` and then calling `nn.run`.
-/

/--
PyTorch-like global seeding convenience for seeded model builders.

This sets the global seed stream used by `nn.runGlobal` / `nn.nextSeed`.
-/
def manualSeed (seed : Nat) : IO Unit :=
  rand.manualSeed seed

/-
`nn.pure.*` holds the explicit-seed constructors (proof/reproducibility-friendly).

For user ergonomics, we re-export the *config records* and the pure helper namespaces
(`functional`, `blocks`, `heads`) at `nn.*`, while keeping the explicit layer constructors under
`nn.pure.*`.
-/

@[inherit_doc pure.Embedding]
abbrev Embedding := pure.Embedding

@[inherit_doc pure.LearnedPositionalEmbedding]
abbrev LearnedPositionalEmbedding := pure.LearnedPositionalEmbedding

@[inherit_doc pure.SinusoidalPositionalEncoding]
abbrev SinusoidalPositionalEncoding := pure.SinusoidalPositionalEncoding

@[inherit_doc pure.RoPE]
abbrev RoPE := pure.RoPE

@[inherit_doc pure.Conv2d]
abbrev Conv2d := pure.Conv2d

@[inherit_doc Conv2d]
abbrev Conv := Conv2d

@[inherit_doc pure.MaxPool2d]
abbrev MaxPool2d := pure.MaxPool2d

@[inherit_doc MaxPool2d]
abbrev MaxPool := MaxPool2d

@[inherit_doc pure.AvgPool2d]
abbrev AvgPool2d := pure.AvgPool2d

@[inherit_doc AvgPool2d]
abbrev AvgPool := AvgPool2d

@[inherit_doc pure.LayerNorm]
abbrev LayerNorm := pure.LayerNorm

@[inherit_doc pure.RMSNorm]
abbrev RMSNorm := pure.RMSNorm

@[inherit_doc pure.BatchNorm2d]
abbrev BatchNorm2d := pure.BatchNorm2d

@[inherit_doc pure.InstanceNorm2d]
abbrev InstanceNorm2d := pure.InstanceNorm2d

@[inherit_doc pure.MultiheadAttention]
abbrev MultiheadAttention := pure.MultiheadAttention

@[inherit_doc pure.globalAvgPoolCHW]
def globalAvgPoolCHW := pure.globalAvgPoolCHW

@[inherit_doc pure.globalAvgPoolNCHW]
def globalAvgPoolNCHW := pure.globalAvgPoolNCHW

namespace functional
export pure.functional
  (square checkpoint
   detach stopGrad
   addB mulB
   embedding mean
   dropoutSeeded)
end functional

namespace blocks
export pure.blocks
  (Activation activation
   MLP mlp
   Conv2dAct conv2dAct
   Conv2dNormAct conv2dNormActCHW conv2dNormAct
   Conv2dNormActPool conv2dNormActPoolCHW conv2dNormActPool
   down2 down2_pos
   conv3x3Same conv3x3Down conv1x1Same conv1x1Down
   conv3x3SameImages conv3x3DownImages conv1x1SameImages conv1x1DownImages
   ResNetBasicBlock resnetBasicBlock
   TransformerEncoderBlock transformerEncoderBlockWithMask transformerEncoderBlock
   TransformerEncoderStack transformerEncoderStackWithMask transformerEncoderStack
   transformerEncoderClassifier
   residual residualLayer
   addBranches addBranchesLayer)
end blocks

namespace heads
export pure.heads (classifier regressor classifierBatch regressorBatch)
end heads

/-!
## Seeded Builders (Default `nn.*`)

For end-user code, the default `nn.*` layer constructors allocate initialization seeds
automatically via `nn.M` (a deterministic seed-stream builder).

Use `nn.pure.*` when you want to pass explicit seeds (proof-friendly / fully reproducible).
-/

open Spec

/-- Seeded builder monad: a state monad over `API.rand.SeedStream`. -/
abbrev M := rand.SeedM

/-- Run a seeded builder starting from a base seed. -/
def run {α : Type 2} (seed : Nat) (x : M α) : α :=
  (x (rand.SeedStream.init seed)).1

/-- Lift a pure value into the seeded builder (consumes no seeds). -/
def lift {α : Type 2} (x : α) : M α :=
  pure x

/-- Consume one fresh seed and pass it to `k`. -/
def withSeed {α : Type 2} (k : Nat → α) : M α :=
  fun st =>
    let (seed, st') := rand.SeedStream.next st
    (k seed, st')

/-- Consume two fresh seeds and pass them to `k` (in order). -/
def withSeeds2 {α : Type 2} (k : Nat → Nat → α) : M α :=
  fun st =>
    let (a, st') := rand.SeedStream.next st
    let (b, st'') := rand.SeedStream.next st'
    (k a b, st'')

@[inherit_doc pure.relu]
def relu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.relu (s := s))

@[inherit_doc pure.silu]
def silu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.silu (s := s))

@[inherit_doc pure.gelu]
def gelu {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.gelu (s := s))

@[inherit_doc pure.sigmoid]
def sigmoid {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.sigmoid (s := s))

@[inherit_doc pure.tanh]
def tanh {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.tanh (s := s))

@[inherit_doc pure.softmax]
def softmax {s : Spec.Shape} : M (Sequential s s) :=
  lift (pure.softmax (s := s))

@[inherit_doc pure.sum]
def sum {s : Spec.Shape} : M (Sequential s Spec.Shape.scalar) :=
  lift (pure.sum (s := s))

@[inherit_doc pure.flatten]
def flatten {s : Spec.Shape} : M (Sequential s (.dim (Spec.Shape.size s) .scalar)) :=
  lift (pure.flatten (s := s))

@[inherit_doc pure.flattenBatch]
def flattenBatch {n : Nat} {s : Spec.Shape} :
    M (Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s))) :=
  lift (pure.flattenBatch (n := n) (s := s))

@[inherit_doc pure.flattenStart1]
def flattenStart1 {n : Nat} {s : Spec.Shape} :
    M (Sequential (.dim n s) (NN.Tensor.Shape.Mat n (Spec.Shape.size s))) :=
  lift (pure.flattenStart1 (n := n) (s := s))

@[inherit_doc pure.maxPool2d]
def maxPool2d {n inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1))) :=
  lift (pure.maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg)

/--
Max pooling over batched CHW images, allocating any required initialization seeds automatically.

Shorthand for `maxPool2d`.
-/
def maxPool {n inC inH inW : Nat} (cfg : MaxPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.avgPool2d]
def avgPool2d {n inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1))) :=
  lift (pure.avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg)

/--
Average pooling over batched CHW images, allocating any required initialization seeds automatically.

Shorthand for `avgPool2d`.
-/
def avgPool {n inC inH inW : Nat} (cfg : AvgPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.linear]
def linear (inDim outDim : Nat) (pfx : Spec.Shape := Spec.Shape.scalar) :
    M (Sequential (pfx.appendDim inDim) (pfx.appendDim outDim)) :=
  withSeeds2 (fun seedW seedB =>
    pure.linear inDim outDim seedW seedB (pfx := pfx))

/--
Vector-only linear layer alias.

This is shorthand for `nn.linear inDim outDim` at scalar prefix shape, so examples do not need to
mention `pfx := Spec.Shape.scalar`.
-/
def linearV (inDim outDim : Nat) : M (Sequential (NN.Tensor.Shape.Vec inDim)
    (NN.Tensor.Shape.Vec outDim)) :=
  linear inDim outDim

@[inherit_doc pure.rnn]
def rnn (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.rnn seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.gru]
def gru (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.gru seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.mamba]
def mamba (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.mamba seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.lstm]
def lstm (seqLen inputSize hiddenSize : Nat) :
    M (Sequential
      (NN.Tensor.Shape.Mat seqLen inputSize)
      (NN.Tensor.Shape.Mat seqLen hiddenSize)) :=
  withSeeds2 (fun seedW seedB =>
    pure.lstm seqLen inputSize hiddenSize seedW seedB)

@[inherit_doc pure.conv2d]
def conv2d {n inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1))) :=
  withSeeds2 (fun seedK seedB =>
    let cfg' : Conv2d := { cfg with seedK := seedK, seedB := seedB }
    pure.conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg')

/--
Convolution over batched CHW images, allocating initialization seeds automatically.

Shorthand for `conv2d`.
-/
def conv {n inC inH inW : Nat} (cfg : Conv) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :=
  conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

@[inherit_doc pure.batchNorm2d]
def batchNorm2d {n c h w : Nat} (cfg : BatchNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    M (Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w)) :=
  withSeeds2 (fun seedGamma seedBeta =>
    let cfg' : BatchNorm2d := { cfg with seedGamma := seedGamma, seedBeta := seedBeta }
    pure.batchNorm2d (n := n) (c := c) (h := h) (w := w) cfg')

/--
BatchNorm over batched CHW images, allocating initialization seeds automatically.

Shorthand for `batchNorm2d`.
-/
def batchNorm {n c h w : Nat} (cfg : BatchNorm2d := {}) [NeZero n] [NeZero c] [NeZero h] [NeZero w] :=
  batchNorm2d (n := n) (c := c) (h := h) (w := w) cfg

@[inherit_doc pure.embedding]
def embedding (vocab embedDim : Nat) (cfg : Embedding := {}) {pfx : Spec.Shape} :
    M (Sequential (pfx.appendDim vocab) (pfx.appendDim embedDim)) :=
  withSeed (fun seedW =>
    pure.embedding vocab embedDim { cfg with seedW := seedW } (pfx := pfx))

@[inherit_doc pure.sinusoidalPositionalEncoding]
def sinusoidalPositionalEncoding {batch seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  lift <|
    pure.sinusoidalPositionalEncoding (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg

@[inherit_doc pure.rope]
def rope {batch numHeads seqLen headDim : Nat} (cfg : RoPE := {}) :
    M (Sequential
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim)))
      (.dim batch (.dim numHeads (NN.Tensor.Shape.Mat seqLen headDim)))) :=
  lift <|
    pure.rope (batch := batch) (numHeads := numHeads) (seqLen := seqLen) (headDim := headDim) cfg

@[inherit_doc pure.learnedPositionalEmbedding]
def learnedPositionalEmbedding {batch seqLen embedDim : Nat} (cfg : LearnedPositionalEmbedding := {}) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  withSeed (fun seedPos =>
    pure.learnedPositionalEmbedding (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { cfg with seedPos := seedPos })

@[inherit_doc pure.layerNorm]
def layerNorm {batch seqLen embedDim : Nat} [NeZero seqLen] [NeZero embedDim] :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))) :=
  withSeeds2 (fun seedGamma seedBeta =>
    pure.layerNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
      { seedGamma := seedGamma, seedBeta := seedBeta })

@[inherit_doc pure.multiheadAttention]
def multiheadAttention {batch n dModel : Nat} [NeZero n]
    (cfg : MultiheadAttention)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedW =>
    pure.multiheadAttention (batch := batch) (n := n) (dModel := dModel)
      { cfg with seedW := seedW } (mask := mask))

@[inherit_doc blocks.transformerEncoderBlock]
def transformerEncoderBlock {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderBlock := { cfg with seedBase := seedBase }
    blocks.transformerEncoderBlock (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

@[inherit_doc blocks.transformerEncoderStack]
def transformerEncoderStack {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : blocks.TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    M (Sequential
      (.dim batch (NN.Tensor.Shape.Mat n dModel))
      (.dim batch (NN.Tensor.Shape.Mat n dModel))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.TransformerEncoderStack := { cfg with seedBase := seedBase }
    blocks.transformerEncoderStack (batch := batch) (n := n) (dModel := dModel) cfg'
      (mask := mask))

@[inherit_doc blocks.resnetBasicBlock]
def resnetBasicBlock {n inC h w : Nat} (cfg : blocks.ResNetBasicBlock)
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    M (Sequential
      (NN.Tensor.Shape.Images n inC h w)
      (NN.Tensor.Shape.Images n cfg.outC
        (if cfg.downsample then blocks.down2 h else h)
        (if cfg.downsample then blocks.down2 w else w))) :=
  withSeed (fun seedBase =>
    let cfg' : blocks.ResNetBasicBlock := { cfg with seedBase := seedBase }
    blocks.resnetBasicBlock (n := n) (inC := inC) (h := h) (w := w) cfg')

@[inherit_doc pure.dropout]
def dropout {s : Spec.Shape} (p : Float) : M (Sequential s s) :=
  withSeed (fun seed =>
    pure.dropout (s := s) p (seed := seed))

/--
Run a seeded builder using the global seed stream set by `nn.manualSeed` (results in `Type`).

Note: model values like `nn.Sequential` live in `Type 2`, so they cannot be returned from `IO`.
For models, use `nn.run` with an explicit base seed (obtained from `nn.nextSeed`).
-/
def runGlobal {α : Type} (x : M α) : IO α :=
  rand.runGlobal x

/-- Draw a fresh base seed from the global seed stream set by `nn.manualSeed`. -/
def nextSeed : IO Nat :=
  rand.nextSeedGlobal

/-- Draw `n` fresh base seeds from the global seed stream. -/
def nextSeeds (n : Nat) : IO (List Nat) :=
  rand.nextSeedsGlobal n

/-!
### Naming Convenience

`nn.run` / `nn.nextSeed` are the core primitives, but in user code it is often clearer to read:
- “build a model from this seed” (`nn.build`)
- “draw a fresh init seed” (`nn.freshSeed`)
- “build a model using the next global init seed” (`nn.withModel`)
-/

/-- Alias for `nn.run` (PyTorch-style wording: build/init a model from a base seed). -/
abbrev build {α : Type 2} (seed : Nat) (x : M α) : α :=
  run seed x

/-- Alias for `nn.nextSeed` (draw a fresh base seed from the global seed stream). -/
abbrev freshSeed : IO Nat :=
  nextSeed

/--
Build a model using the next global seed, then run a continuation.

Why this exists: `nn.Sequential` lives in `Type 2`, so we can't directly return a model from `IO`.
This helper keeps model construction pure while letting executable code avoid repeating the
`nextSeed/run` pattern.
-/
def withModel {σ τ : Spec.Shape} {β : Type}
    (mk : M (Sequential σ τ)) (k : Sequential σ τ → IO β) : IO β := do
  let seed ← nextSeed
  let model := run seed mk
  k model

end nn

namespace autograd

/-!
Autograd helpers (grad/vjp/jacobian) over TorchLean programs.

This namespace is conceptually similar to PyTorch autograd + functorch/`torch.func`:
- gradients of losses w.r.t. parameters and inputs
- VJPs and Jacobians for analysis and verification tooling

PyTorch references:
- Autograd: `https://pytorch.org/docs/stable/autograd.html`
- `torch.func` (jacfwd/jacrev, etc.): `https://pytorch.org/docs/stable/func.html`
-/

namespace model

/-
Model-shaped autograd: a TorchLean `NN.Seq` plus an `OutputLoss` over its output.

This is the common "training" use case.
-/

@[inherit_doc TorchLean.Autodiff.Model.Params]
abbrev Params {σ τ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (α : Type) :=
  TorchLean.Autodiff.Model.Params model α

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss]
abbrev OutputLoss (τ υ : Spec.Shape) :=
  TorchLean.Autodiff.Model.OutputLoss τ υ

@[inherit_doc TorchLean.Autodiff.Model.linearParams]
abbrev linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : Spec.Tensor α (NN.Tensor.Shape.Mat outDim inDim))
    (b : Spec.Tensor α (NN.Tensor.Shape.Vec outDim)) :
    Params (TorchLean.Layers.linear inDim outDim seedW seedB) α :=
  TorchLean.Autodiff.Model.linearParams
    (α := α) (inDim := inDim) (outDim := outDim) (seedW := seedW) (seedB := seedB) w b

namespace OutputLoss

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.mse]
abbrev mse {τ : Spec.Shape} (reduction : TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  TorchLean.Autodiff.Model.OutputLoss.mse (τ := τ) (reduction := reduction)

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.crossEntropyOneHot]
abbrev crossEntropyOneHot {τ : Spec.Shape} (reduction : TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  TorchLean.Autodiff.Model.OutputLoss.crossEntropyOneHot (τ := τ) (reduction := reduction)

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.detach]
abbrev detach {τ υ : Spec.Shape} (loss : model.OutputLoss τ υ) :
    model.OutputLoss τ υ :=
  TorchLean.Autodiff.Model.OutputLoss.detach loss

end OutputLoss

/--
Gradient of a model-loss w.r.t. the model parameters.

This is the common training use case (PyTorch analogue: `loss.backward()` followed by parameter
  updates).
-/
def gradParams {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.gradParams (α := α) model loss params x target

/-- Gradient of the loss w.r.t. the inputs (`x` and `target`). -/
def gradInputs {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (TorchLean.TList α [σ, υ]) :=
  TorchLean.Autodiff.Model.gradInputs (α := α) model loss params x target

/-- Convenience: gradient of the loss w.r.t. `x`. -/
def gradX {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α σ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) params x target
  pure (tlist.get0 gxs)

/-- Convenience: gradient of the loss w.r.t. the `target` argument. -/
def gradTarget {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α υ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) params x target
  pure (tlist.get1 gxs)

/--
Forward+backward result for a scalar loss built from a model output.

PyTorch comparison: this is the "compute loss + backward" payload, but with shapes tracked.
-/
structure ValueAndGrads {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (α : Type) where
  /-- Value at the current point. -/
  value : Spec.Tensor α Spec.Shape.scalar
  /-- Gradients w.r.t. parameters. -/
  dparams : TorchLean.Autodiff.Model.Params model α
  /-- Gradient w.r.t. input. -/
  dx : Spec.Tensor α σ
  /-- Gradient w.r.t. target. -/
  dtarget : Spec.Tensor α υ

/--
Run `loss(model(params, x), target)` and compute gradients w.r.t:

- model parameters,
- `x`,
- `target`.

This hides the `CompiledScalar`/argument-pack boilerplate for the common "one sample" case.
-/
def valueAndGrads {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (ValueAndGrads (model := model) (α := α) (σ := σ) (υ := υ)) := do
  let paramShapes := TorchLean.NN.Seq.paramShapes model
  let c ←
    TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes) (inputShapes := [σ, υ])
      (TorchLean.Autodiff.Model.lossProgram (model := model) loss)

  let args : TorchLean.TList α (paramShapes ++ [σ, υ]) :=
    tlist.append (ss₁ := paramShapes) (ss₂ := [σ, υ]) params (tlist.mk2 x target)

  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.forward (α := α) (Γ := paramShapes ++ [σ, υ]) c
      args

  let gAll : TorchLean.TList α (paramShapes ++ [σ, υ]) :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := paramShapes ++ [σ, υ]) c
      args

  let (dps, dxys) :=
    tlist.split (α := α) (ss₁ := paramShapes) (ss₂ := [σ, υ]) gAll

  pure
    { value := value
      dparams := dps
      dx := tlist.get0 dxys
      dtarget := tlist.get1 dxys }

/-- Return just `(loss_value, grad_params)`. -/
def valueAndGradParams {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × TorchLean.Autodiff.Model.Params model α) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dparams)

/-- `valueAndGradParams`, but convert the 0-dim loss tensor to a scalar `α`. -/
def valueAndGradParamsScalar {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (α × TorchLean.Autodiff.Model.Params model α) := do
  let (valueT, dps) ← valueAndGradParams (model := model) (loss := loss) (α := α) params x target
  pure (Spec.Tensor.toScalar valueT, dps)

/-- Return `(loss_value, grad_x)`. -/
def valueAndGradX {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dx)

/-- Return `(loss_value, grad_target)`. -/
def valueAndGradTarget {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α υ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dtarget)

/--
Vector-Jacobian product (VJP) w.r.t. model parameters.

This is the "grad of outputs back into parameters" primitive. It is useful for custom losses or
analysis tooling when you already have a seed tensor `seedOut : τ`.
-/
def vjpParams {σ τ : Spec.Shape} (model : TorchLean.NN.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.vjpParams (α := α) model params x seedOut

/--
VJP w.r.t. the model input.

This returns a one-element `TList` to match the general "inputs list" API shape.
For the common case, use `vjpInput` to get the tensor directly.
-/
def vjpInputs {σ τ : Spec.Shape} (model : TorchLean.NN.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (TorchLean.TList α [σ]) :=
  TorchLean.Autodiff.Model.vjpInputs (α := α) model params x seedOut

/-- Convenience wrapper: unwrap `vjpInputs` to return just `dx`. -/
def vjpInput {σ τ : Spec.Shape} (model : TorchLean.NN.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let dxs ← vjpInputs (model := model) (α := α) params x seedOut
  pure (tlist.unpack1 dxs)

/--
Reverse-mode Jacobian (`jacrev`) of the model output w.r.t. parameters.

Returns an array of parameter-structured gradients: one entry per output coordinate.
This mirrors the usual "jacrev returns a stack of per-output gradients" shape.
-/
def jacrevParams {σ τ : Spec.Shape} (model : TorchLean.NN.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) :
    IO (Array (TorchLean.Autodiff.Model.Params model α)) :=
  TorchLean.Autodiff.Model.jacrevParams (α := α) model params x

/--
Jacobian-vector product (JVP) of a scalar loss w.r.t. parameters.

This is the directional derivative in the direction `vparams`.
Conceptually: `d/dt loss(params + t*vparams, x, target) | t = 0`.
-/
def jvpParams {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : TorchLean.Autodiff.Model.Params model α) :
    IO α :=
  TorchLean.Autodiff.Model.jvpParams (α := α) model loss params x target vparams

/--
Hessian-vector product (HVP) of a scalar loss w.r.t. parameters.

Returns a parameter-structured tensor list of the same shape as `params`.
-/
def hvpParams {σ τ υ : Spec.Shape} (model : TorchLean.NN.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : TorchLean.Autodiff.Model.Params model α) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.hvpParams (α := α) model loss params x target vparams
end model

namespace fn1

/-
Function-1 autograd: treat a pure function `f : Tensor σ -> Tensor τ` as the object of
differentiation (no parameters).
-/

/-!
In PyTorch terms, this is the "functorch" style: differentiate plain functions, not modules.
-/

@[inherit_doc TorchLean.Autodiff.Function1.Fn]
abbrev Fn (σ τ : Spec.Shape) :=
  TorchLean.Autodiff.Function1.Fn σ τ

/-- Forward-mode Jacobian (`jacfwd`) for a pure tensor function. -/
def jacfwd {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α τ)) :=
  TorchLean.Autodiff.Function1.jacfwd (α := α) f x

/-- Hessian for a scalar-valued function. -/
def hessian {σ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) :=
  TorchLean.Autodiff.Function1.hessian (α := α) f x

/-- Vector-Jacobian product (VJP) for a pure function. -/
def vjp {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let params : TorchLean.TList α ([] : List Spec.Shape) := .nil
  let gxs ←
    TorchLean.Autodiff.vjpOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (TorchLean.Autodiff.Function1.program (σ := σ) (τ := τ) f)
      params (tlist.mk1 x) seedOut
  pure (tlist.unpack1 gxs)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns the Jacobian rows as an array of `doutput/dinput` tensors.
-/
def jacrev {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) := do
  let params : TorchLean.TList α ([] : List Spec.Shape) := .nil
  let rows ←
    TorchLean.Autodiff.jacrevOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (TorchLean.Autodiff.Function1.program (σ := σ) (τ := τ) f)
      params (tlist.mk1 x)
  pure <| rows.map tlist.unpack1

/-- Gradient of a scalar-valued function w.r.t. its input. -/
def grad {σ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α σ) := do
  let params : TorchLean.TList α ([] : List Spec.Shape) := .nil
  let gxs ←
    TorchLean.Autodiff.gradInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (TorchLean.Autodiff.Function1.program (σ := σ) (τ := Spec.Shape.scalar) f)
      params (tlist.mk1 x)
  pure (tlist.unpack1 gxs)

/-- Return `(value, grad)` for a scalar-valued function at `x`. -/
def valueAndGrad {σ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let c ←
    TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (TorchLean.Autodiff.Function1.program (σ := σ) (τ := Spec.Shape.scalar) f)
  let args : TorchLean.TList α [σ] := tlist.mk1 x
  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.forward (α := α) (Γ := [σ]) c args
  let gAll : TorchLean.TList α [σ] :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := [σ]) c args
  pure (value, tlist.unpack1 gAll)

/-- `valueAndGrad`, but convert the 0-dim value tensor to a scalar `α`. -/
def valueAndGradScalar {σ : Spec.Shape} (f : TorchLean.Autodiff.Function1.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (α × Spec.Tensor α σ) := do
  let (valueT, g) ← valueAndGrad (f := f) (α := α) x
  pure (Spec.Tensor.toScalar valueT, g)
end fn1

end autograd

end API
end NN
