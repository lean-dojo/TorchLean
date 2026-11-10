/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Layers.Pooling
public import TorchLean.Tensor.Conversion
public import TorchLean.Tensor.Numeric.InverseTrig
public import TorchLean.NN.Curried
public import TorchLean.Tensor.Op.Args
public import TorchLean.Tensor.Op.Need

/-!
# Backend Operation Interface

`Ops` supplies the tensor primitives for models shared by eager and typed graph execution.
`Ref` and `DataRef` are the reference types of the interpreter. `apply` runs or records a
registered operator (`TorchLean.Op`) with a value output from one argument per input port
(`TorchLean.PortArgs`), at a scalar that has the capability the operator needs
(`TorchLean.Op.Supported`). That capability is all its type asks for, so an interpreter can
refuse an operator it does not implement: the typed graph and the eager CPU tape one without a rule
in `Op.rule?` (`.custom`), the eager CUDA tape one without a LibTorch entry for its attributes (such
as `.square`), and the verifier recorder (`TorchLean.IR.record`) one outside its
fragment or whose lowering reads an optional capability.

`applyRandom` is `apply` with the interpreter's next random key, `applyNative?` offers a native
implementation where an interpreter has one, and `atomic` runs several recordings as one tape
transaction.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/--
Tensor operations shared by eager and typed graph execution.

A model is polymorphic over this class. Eager instances execute and record a tape, and typed graph
instances build reusable graph data. Each must preserve the corresponding operator's forward and
VJP semantics. The pure instance at `Id` (Ops/Pure.lean) is a total denotation, not a
checked interpreter. `apply` takes any registered operator and its port arguments. The
eager instance on its CPU tape and the typed-graph instance run its rule `Op.rule?` with the call
site's capability record `supported.caps` (the eager CUDA tape runs the operator's LibTorch entry).
The verifier recorder lowers only operators of its fragment whose lowering reads no optional
capability.
-/
class Ops (m : Type → Type) (α : Type) [Storage α] [Context α] where
  /-- A differentiable tensor handle, indexed by its shape. -/
  Ref : Shape → Type
  /-- Backend representation of non-differentiable tensor data. -/
  DataRef : (β : Type) → [Storage β] → Shape → Type
  /-- Lift fixed non-differentiable data into the backend representation. -/
  dataConst : {β : Type} → [Storage β] → {s : Shape} → (x : Tensor β s) → DataRef β s
  /-- Apply a pure transformation to non-differentiable data. -/
  mapData : {β γ : Type} → [Storage β] → [Storage γ] → {s₁ s₂ : Shape} →
      (f : Tensor β s₁ → Tensor γ s₂) → (x : DataRef β s₁) → DataRef γ s₂
  /--
  Observe a stateful layer's actual forward input and update its persistent buffers.

  Pure interpreters may omit this hook. Runtime interpreters keep the recorded state values for
  differentiation and write updated values only to their persistent, non-trainable storage.
  -/
  updateBuffers? : Option ({ss : List Shape} → {s : Shape} → Pack Ref ss → Ref s →
    (TensorPack α ss → Tensor α s → IO (TensorPack α ss)) → m Unit) := none
  /-- Record a fixed tensor value. -/
  const : {s : Shape} → (t : Tensor α s) → m (Ref s)
  /-- Inspect a scalar for eager control flow. The comparison is not differentiated, and the tape
  records only the selected branch. Static graph builders reject this dynamic observation. -/
  observe : Ref Shape.scalar → m α
  /-- Run or record the registered operator `op` on `args` and return its value output. An index or
  Boolean output after it is dropped. The eager instance on its CPU tape and the typed-graph
  instance run `Op.rule?` with the call site's capability record `supported.caps`. The eager CUDA
  tape runs the operator's LibTorch entry. The verifier recorder does not read that record: it
  lowers only operators of its fragment (`TorchLean.IR.record`) whose lowering reads no
  optional capability, and refuses the others, `.square`, `.rfft`, `.irfft` and `.selectiveScan`
  among them. -/
  apply {ins rest : List Port} {s : Shape} (op : Op ins (.value s :: rest))
      [supported : Op.Supported α op] (args : PortArgs α Ref DataRef ins) : m (Ref s)
  /-- `apply` on the random operator `op` with its key replaced (`Op.withKey?`) by the next key of
  the interpreter's stream under `seed`. An operator without a key is refused. The pure instance
  uses the first key under `seed` on every call and runs a keyless operator as `apply` does. -/
  applyRandom {ins rest : List Port} {s : Shape} (op : Op ins (.value s :: rest))
      [supported : Op.Supported α op] (seed : Nat) (args : PortArgs α Ref DataRef ins) : m (Ref s)
  /-- A native implementation of operators where the interpreter has one. When it declines `op`, it
  returns `none` without recording anything and the caller runs its composite. A `some` result has
  the value and gradients of `apply` on `op`. -/
  applyNative? : Option ({ins rest : List Port} → {s : Shape} → (op : Op ins (.value s :: rest)) →
      [Op.Supported α op] → PortArgs α Ref DataRef ins → m (Option (Ref s))) := none
  /-- Run `act` as one tape transaction. When `act` succeeds, every node it recorded stays on the
  tape: a reshape, a convolution and a reshape stay three nodes. When `act` fails, the eager
  instance restores only its tape, as `Session.atomic` does: the random counter, backend
  selections and the other session state keep what `act` changed. The default is `act` itself, for
  interpreters whose failed recording keeps no state. -/
  atomic {β : Type} (act : m β) : m β := act

variable {m : Type → Type} {α : Type} [Storage α] [Context α]
    [Ops (m := m) (α := α)]

/-- The reference `x` at an equal shape. -/
def Ops.Ref.cast {s t : Shape} (h : s = t) (x : Ref (m := m) (α := α) s) :
    Ref (m := m) (α := α) t :=
  h ▸ x

end TorchLean
