/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Types
public import NN.Runtime.Autograd.Torch.TypedGraphSession.Neural
public import NN.Runtime.Autograd.Torch.TypedGraphSession.ShapeIndex

/-!
# Session Shape and Index Operations

This file contains the session-level operations that preserve or rearrange tensor shape: activation
helpers, reshapes, indexing, gathers, broadcasts, and reductions. Each operation dispatches through
the same eager/typed graph session boundary as the lower-level tensor ops.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Session

/--
Rectified Linear Unit (ReLU) activation.

This is the pointwise nonlinearity $\operatorname{ReLU}(x)=\max(x,0)$, recorded as part of the
session’s autograd
graph.

PyTorch analogy: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} [TorchLean.Storage α] (s : Session α)
  [Mul α] [Add α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.relu (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.relu (α := α) sess (sh := sh) x

/--
Sigmoid (logistic) activation, applied pointwise.

PyTorch analogy: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.sigmoid (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.sigmoid (α := α) sess (sh := sh) x

/--
Hyperbolic tangent activation, applied pointwise.

PyTorch analogy: `torch.tanh(x)`.
-/
def tanh {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.tanh (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.tanh (α := α) sess (sh := sh) x

/-- Softmax along an explicitly selected tensor dimension. -/
def softmax {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
  (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.softmax (α := α) sess (sh := sh) axis x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.softmax
        (α := α) sess (sh := sh) axis x

/-- Stable log-softmax along an explicitly selected tensor dimension. -/
def logSoftmax {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
  (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.logSoftmax (α := α) sess (sh := sh) axis x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.logSoftmax
        (α := α) sess (sh := sh) axis x

/--
Softplus activation, applied pointwise:
$\operatorname{softplus}(x)=\log(1+\exp x)$.

PyTorch analogy: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.softplus (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.softplus (α := α) sess (sh := sh) x

/--
Elementwise exponential.

PyTorch analogy: `torch.exp(x)`.
-/
def exp {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.exp (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.exp (α := α) sess (sh := sh) x

/-- Elementwise sine of angles in radians, using the session's eager or typed graph execution. -/
def sin {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
    {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.sin (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.sin (α := α) sess (sh := sh) x

/-- Elementwise cosine, with gradients multiplied by `-sin(x)` on either execution path. -/
def cos {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
    {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
    IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.cos (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.cos (α := α) sess (sh := sh) x

/--
Elementwise natural logarithm.

PyTorch analogy: `torch.log(x)`.

If you need a total (always-defined) "log-like" surrogate without positivity side conditions, see
`safeLog`.
-/
def log {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.log (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.log (α := α) sess (sh := sh) x

/--
Elementwise safe-log surrogate:
$\operatorname{safeLog}(x;\varepsilon)=\log(\operatorname{softplus}(x)+\varepsilon)$.

We use this when we want something log-like but would rather not carry side conditions about inputs
being strictly positive.

PyTorch analogy: `torch.log(torch.nn.functional.softplus(x) + eps)`.
-/
def safeLog {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (x : Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Context.defaultEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.state with
  | .eager sess => EagerSession.safeLog (α := α) sess (sh := sh) x (ε := ε)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.safeLog (α := α) sess (sh := sh) x (ε := ε)

/--
Sum-reduce all elements of a tensor to a scalar.

PyTorch analogy: `x.sum()` (with no `dim` argument).
-/
def sum {α : Type} [TorchLean.Storage α] (s : Session α) [Context α] {sh : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.state with
  | .eager sess =>
      EagerSession.sum (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.sum (α := α) sess (sh := sh) x

/--
Flatten a tensor to a 1D vector of length `Spec.Shape.size sh`.

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {α : Type} [TorchLean.Storage α] (s : Session α)
    [Inhabited α] [Zero α] {sh : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α [Spec.Shape.size sh]) := do
  match s.state with
  | .eager sess => EagerSession.flatten (α := α) sess (sh := sh) x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.flatten (α := α) sess (sh := sh) x

/--
Reshape a tensor without changing the number of elements.

The proof `h : Spec.Shape.size sh1 = Spec.Shape.size sh2` plays the role of PyTorch’s runtime check
performed by `reshape`/`view`.

PyTorch analogy: `x.reshape(new_shape)` (when the element count matches).
-/
def reshape {α : Type} [TorchLean.Storage α] (s : Session α)
    [Inhabited α] [Zero α] {sh1 sh2 : Shape}
  (x : Runtime.Autograd.Torch.TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) :
  IO (Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.state with
  | .eager sess => EagerSession.reshape (α := α) sess (sh1 := sh1) (sh2 := sh2) x h
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.reshape (α := α) sess
        (sh1 := sh1) (sh2 := sh2) x h

/--
Generic "swap adjacent axes" view operation.

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} [TorchLean.Storage α] (s : Session α)
    [Context α] {sh : Shape}
  (depth : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  match s.state with
  | .eager sess => EagerSession.swapAdjacentAtDepth (α := α) sess (sh := sh) depth x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.swapAdjacentAtDepth (α := α) sess
        (sh := sh) depth x

/-- Broadcast a tensor to a larger shape (dispatches by execution mode). -/
def broadcastTo {α : Type} [TorchLean.Storage α] (s : Session α)
    [Inhabited α] [Add α] [Zero α]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.state with
  | .eager sess => EagerSession.broadcastTo (α := α) sess (sh1 := sh1) (sh2 := sh2) cb x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.broadcastTo (α := α) sess (sh1 := sh1)
        (sh2 := sh2) cb x

/-- Reduce-sum along an axis (dispatches by execution mode). -/
def reduceSum {α : Type} [TorchLean.Storage α] (s : Session α)
    [Add α] [Zero α] [Inhabited α]
  {sh : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.state with
  | .eager sess => EagerSession.reduceSum (α := α) sess (sh := sh) axis x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.reduceSum (α := α) sess (sh := sh) axis x

/-- Reduce-mean along an axis (dispatches by execution mode). -/
def reduceMean {α : Type} [TorchLean.Storage α] (s : Session α)
    [Context α]
  {sh : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.state with
  | .eager sess => EagerSession.reduceMean (α := α) sess (sh := sh) axis x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.reduceMean (α := α) sess (sh := sh) axis x

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} [TorchLean.Storage α] (s : Session α) [Zero α]
    {shape : Shape} (axis : Nat) (x : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (index : Fin (Shape.axisSize shape axis)) :
    IO (Runtime.Autograd.Torch.TensorRef α (shape.eraseAxis axis)) := do
  match s.state with
  | .eager session => EagerSession.select (α := α) session axis x index
  | .typedGraph session =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.select
        (α := α) session axis x index

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} [TorchLean.Storage α] (s : Session α)
    [Add α] [Zero α]
    {shape : Shape} (axis count : Nat) (x : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count)) := do
  match s.state with
  | .eager session => EagerSession.indexSelect (α := α) session axis count x indices
  | .typedGraph session =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.indexSelect
        (α := α) session axis count x indices

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} [TorchLean.Storage α] (s : Session α)
    [Add α] [Zero α]
    {shape : Shape} (axis count : Nat) (base : Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (source : Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count))
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (Runtime.Autograd.Torch.TensorRef α shape) := do
  match s.state with
  | .eager session => EagerSession.scatterAdd (α := α) session axis count base source indices
  | .typedGraph session =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.scatterAdd
        (α := α) session axis count base source indices

end Session

end Model
end Autograd
end Runtime
