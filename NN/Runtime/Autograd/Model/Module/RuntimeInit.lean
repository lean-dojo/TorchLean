/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Initialization
public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Runtime.Autograd.Model.Program -- shake: keep

/-!
# Runtime Initialization

Casting helpers and shape-indexed initialization plans for executable modules. Runtime
initializers can materialize parameter storage on the host or directly in CUDA buffers.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

/-! ## Small helpers -/

namespace Module

/--
Cast a Float tensor to a backend scalar type `α` by mapping a scalar cast function.

This is mainly used to turn ordinary Float tensor literals into
`Float`/`ExecFloat.Binary 8 23`/etc.
-/
def castTensor {α : Type} [TorchLean.Storage α]
    (cast : Float → α) {s : Shape} (t : Tensor Float s) : Tensor α s :=
  TorchLean.Tensor.map cast t

/-- List-shaped `castTensor` for TorchLean's `TorchLean.TensorPack` parameter bundles. -/
def castPack {α : Type} [TorchLean.Storage α] (cast : Float → α) :
    {ss : List Shape} → TorchLean.TensorPack Float ss → TorchLean.TensorPack α ss
  | .nil, .nil => .nil
  | .cons _s ss, .cons x xs => .cons (castTensor cast x) (castPack (cast := cast) (ss := ss) xs)

/-! ## Runtime Float Initializers -/

namespace RuntimeInit

/--
Runtime initializer for a Float parameter.

The usual `ObjectiveDef.initState` path stores initializers as typed Lean tensors. That is the
right representation when the initial value itself is part of the Lean object being inspected.
For large Float runs, it is better to allocate runtime storage from a compact initialization scheme
and synchronize the host tensor only when parameters are explicitly read back.

The design mirrors the storage-first APIs used by mainstream runtimes:

- PyTorch exposes in-place initializers such as `torch.nn.init.uniform_`,
  `torch.nn.init.xavier_uniform_`, and `torch.nn.init.kaiming_uniform_` for already-allocated
  tensors: `https://pytorch.org/docs/stable/nn.init.html`.
- PyTorch's meta-device / `to_empty` path separates "module structure exists" from "real storage is
  materialized", after which users explicitly initialize parameters:
  `https://docs.pytorch.org/docs/main/meta.html`.

TorchLean keeps the semantic parameter type (`Tensor Float s`) available, but this runtime path lets
CPU/CUDA execution initialize real storage directly.
-/
inductive FloatInit where
  /-- Fill with zeros. PyTorch analogue: `torch.nn.init.zeros_`. -/
  | zeros
  /-- Fill with ones. PyTorch analogue: `torch.nn.init.ones_`. -/
  | ones
  /-- Uniform distribution over `[lo, hi)`, using TorchLean's deterministic runtime RNG. -/
  | uniform (lo hi : Float) (seed : Nat := 0)
  /-- Normal distribution with explicit mean and standard deviation. -/
  | normal (mean std : Float) (seed : Nat := 0)
  /-- Xavier/Glorot uniform with explicit fan-in and fan-out. -/
  | xavierUniform (fanIn fanOut : Nat) (seed : Nat := 0)
  /-- Kaiming/He uniform with explicit fan-in. -/
  | kaimingUniform (fanIn : Nat) (seed : Nat := 0)
  /-- Exact row-major payload. Used for imported checkpoints or generated tensors. -/
  | flat (values : FloatArray)

/-- Translate a proof-visible initializer scheme into its storage-first runtime form. -/
def FloatInit.ofScheme (scheme : Torch.Init.Scheme) (seed : Nat := 0) : FloatInit :=
  match scheme with
  | .zeros => .zeros
  | .ones => .ones
  | .uniform lo hi => .uniform lo hi seed
  | .normal mean std => .normal mean std seed
  | .xavierUniform fanIn fanOut => .xavierUniform fanIn fanOut seed
  | .kaimingUniform fanIn => .kaimingUniform fanIn seed

/-- Validate one runtime initializer before it allocates or mutates parameter storage. -/
def FloatInit.validate (expectedSize : Nat) : FloatInit → Except String Unit
  | .zeros
  | .ones =>
      pure ()
  | .uniform lo hi _ =>
      (Torch.Init.Scheme.uniform lo hi).validate
  | .normal mean std _ =>
      (Torch.Init.Scheme.normal mean std).validate
  | .xavierUniform fanIn fanOut _ =>
      (Torch.Init.Scheme.xavierUniform fanIn fanOut).validate
  | .kaimingUniform fanIn _ =>
      (Torch.Init.Scheme.kaimingUniform fanIn).validate
  | .flat values =>
      if values.size = expectedSize then
        pure ()
      else
        throw s!"torch.runtimeInit: flat initializer length mismatch \
          (expected {expectedSize}, got {values.size})"

/--
A shape-indexed initialization plan.

This is the typed runtime-initialization API for modules with a known parameter shape list.  It is
the initialization analogue of `TorchLean.TensorPack`: the type says there is exactly one
initializer for each parameter shape, in the same order. That removes the runtime failure mode
where a plain list is one element too short or too long.

The initializers themselves are runtime schemes rather than proof objects.  Proofs still concern
the ordinary `Tensor Float s` parameter value; this plan only controls how the executable
Float runtime materializes those tensors on CPU or CUDA.
-/
inductive Plan : List Shape → Type where
  /-- No parameters, no initializers. -/
  | nil : Plan []
  /-- Initializer for the head parameter, followed by the plan for the remaining parameters. -/
  | cons {s : Shape} {ss : List Shape} (init : FloatInit) (rest : Plan ss) : Plan (s :: ss)

namespace Plan

/-- Concatenate two shape-indexed initializer plans. -/
def append : {ss₁ ss₂ : List Shape} → Plan ss₁ → Plan ss₂ → Plan (ss₁ ++ ss₂)
  | .nil, _, .nil, ys => ys
  | .cons _ _, _, .cons x xs, ys => .cons x (append xs ys)

/-- Forget the shape index when interoperating with runtime-sized callers. -/
def toArray : {ss : List Shape} → Plan ss → Array FloatInit
  | .nil, .nil => #[]
  | .cons _ _, .cons init rest => #[init] ++ toArray rest

/--
The type index is not decorative: forgetting a `Plan ss` to an array produces exactly `ss.length`
initializers.  This checked fact lets the runtime API avoid the usual
"initializer sequence does not match parameter list" class of bugs once a plan has been built.
-/
theorem size_toArray : {ss : List Shape} → (plan : Plan ss) → plan.toArray.size = ss.length
  | .nil, .nil => rfl
  | .cons _ _, .cons _ rest => by
      simp [toArray, size_toArray rest, Nat.add_comm]

/-- Validate every initializer against its parameter shape before applying the plan. -/
def validate : {ss : List Shape} → Plan ss → Except String Unit
  | .nil, .nil => pure ()
  | .cons shape shapes, .cons init rest => do
      init.validate (Shape.size shape)
      validate (ss := shapes) rest

/-- Recover a shape-indexed plan from a runtime-sized initializer array. -/
def ofArray? (ss : List Shape) (inits : Array FloatInit) : Except String (Plan ss) :=
  let rec go : (remaining : List Shape) → Nat → Except String (Plan remaining)
    | .nil, index =>
        if index = inits.size then
          .ok .nil
        else
          .error "torch.runtimeInit: initializer array longer than parameter list"
    | .cons _ remaining, index =>
        match inits[index]? with
        | none => .error "torch.runtimeInit: initializer array shorter than parameter list"
        | some init => do
            let restPlan ← go remaining (index + 1)
            pure (.cons init restPlan)
  go ss 0

end Plan

/-- Product of a list of dimensions, used for convolutional receptive-field sizes. -/
def dimProduct (xs : List Nat) : Nat :=
  xs.foldl (fun acc x => acc * x) 1

/--
Infer `(fanIn, fanOut)` from a parameter shape using the common linear/conv convention.

For a matrix shaped `[out, in]`, this returns `(in, out)`. For convolution-like weights shaped
`[outChannels, inChannels, k1, ..., kd]`, it returns:

$$
\begin{aligned}
\operatorname{fanIn}
  &=\operatorname{inChannels}\,k_1\cdots k_d,\\
\operatorname{fanOut}
  &=\operatorname{outChannels}\,k_1\cdots k_d.
\end{aligned}
$$

This is the same fan convention documented by PyTorch's Xavier/Kaiming initialization utilities.
-/
def fanInOut? (s : Shape) : Option (Nat × Nat) :=
  match Shape.toList s with
  | .cons outDim (.cons inDim spatial) =>
      let receptive := dimProduct spatial
      some (inDim * receptive, outDim * receptive)
  | _ => none

/-- Build a Xavier initializer by deriving fan-in/fan-out from a Linear/Conv-style weight shape. -/
def xavierUniformForShape (s : Shape) (seed : Nat := 0) : Except String FloatInit :=
  match fanInOut? s with
  | some (fanIn, fanOut) => .ok (.xavierUniform fanIn fanOut seed)
  | none =>
      .error s!"torch.runtimeInit: Xavier initialization expects at least 2 dimensions, got \
        {Shape.pretty s}"

/-- Build a Kaiming initializer by deriving fan-in from a Linear/Conv-style weight shape. -/
def kaimingUniformForShape (s : Shape) (seed : Nat := 0) : Except String FloatInit :=
  match fanInOut? s with
  | some (fanIn, _fanOut) => .ok (.kaimingUniform fanIn seed)
  | none =>
      .error s!"torch.runtimeInit: Kaiming initialization expects at least 2 dimensions, got \
        {Shape.pretty s}"

/--
Deterministic unit sample shared with the pure tensor initializer.

Calling the canonical sampler here keeps CPU storage-first initialization equal to the semantic
tensor initializer. The CUDA path uses the same SplitMix64 key/index sequence, materialized as
float32 device values.
-/
def unitAt (seed idx : Nat) : Float :=
  let key := Random.keyOf seed 0
  let denom : Nat := (2 : Nat) ^ 32
  Random.sampleUnit (α := Float) (Random.sampleNat key idx denom) denom

/-- Scalar value generated by a `FloatInit` at a row-major flat index. A flat payload that is
shorter than the index is an error, not a panic. -/
def sampleAt : FloatInit → Nat → Except String Float
  | .zeros, _ => .ok 0.0
  | .ones, _ => .ok 1.0
  | .uniform lo hi seed, idx => .ok (lo + unitAt seed idx * (hi - lo))
  | .normal mean std seed, idx =>
      .ok (Torch.Init.sampleAt (.normal mean std) seed idx)
  | .xavierUniform fanIn fanOut seed, idx =>
      let limit := Torch.Init.xavierUniformLimit fanIn fanOut
      .ok ((-limit) + unitAt seed idx * (2.0 * limit))
  | .kaimingUniform fanIn seed, idx =>
      let limit := Torch.Init.kaimingUniformLimit fanIn
      .ok ((-limit) + unitAt seed idx * (2.0 * limit))
  | .flat values, idx =>
      match values[idx]? with
      | some value => .ok value
      | none =>
          .error s!"torch.runtimeInit: flat initializer index {idx} is outside its \
            {values.size} values"

/--
Materialize an initializer as a host `FloatArray`.

CPU execution uses this path directly. CUDA uses it only when the initializer already is an exact
flat payload; analytic initializers such as uniform/Xavier/Kaiming are created on the runtime side.
-/
def floatArrayOf (n : Nat) (init : FloatInit) : IO FloatArray := do
  match init with
  | .flat values =>
      if values.size = n then
        pure values
      else
        throw <| IO.userError
          s!"torch.runtimeInit: flat initializer length mismatch (expected {n}, got {values.size})"
  | _ =>
      let mut out : Array Float := Array.mkEmpty n
      for i in [0:n] do
        match sampleAt init i with
        | .ok value => out := out.push value
        | .error message => throw <| IO.userError message
      pure (FloatArray.mk out)

/-- Checked conversion to the current CUDA buffer API's `UInt32` element count. -/
def natToU32Checked (ctx : String) (n : Nat) : IO UInt32 := do
  let u := UInt32.ofNat n
  if u.toNat = n then
    pure u
  else
    throw <| IO.userError s!"{ctx}: tensor too large for CUDA buffer API ({n} elements)"

/--
Allocate a CUDA buffer filled with `U(lo, hi)`.

The implementation keeps all element generation on the runtime side: first create a CUDA uniform
buffer in `[0,1)`, then perform `lo + (hi-lo) * u` with CUDA buffer ops.
-/
def cudaUniformBuffer (n : Nat) (lo hi : Float) (seed : Nat) :
    IO Runtime.Autograd.Cuda.Buffer := do
  let n32 ← natToU32Checked "torch.runtimeInit" n
  let key := Spec.Random.keyOf seed 0
  let u := Runtime.Autograd.Cuda.Buffer.randUniform n32 key
  let shift := Runtime.Autograd.Cuda.Buffer.full n32 lo
  let out := Runtime.Autograd.Cuda.Buffer.axpy shift u (hi - lo)
  pure <| Runtime.Autograd.Cuda.Buffer.releaseThen u <|
    Runtime.Autograd.Cuda.Buffer.releaseThen shift out

/--
Allocate a CUDA buffer for a `FloatInit`.

For analytic schemes (`zeros`, `ones`, `uniform`, `xavierUniform`, `kaimingUniform`), this avoids
building a large nested Lean tensor. For `.flat`, the caller already supplied the exact payload, so
we upload that payload directly.
-/
def cudaBufferOf (n : Nat) (init : FloatInit) : IO Runtime.Autograd.Cuda.Buffer := do
  let n32 ← natToU32Checked "torch.runtimeInit" n
  match init with
  | .zeros => pure <| Runtime.Autograd.Cuda.Buffer.zeros n32
  | .ones => pure <| Runtime.Autograd.Cuda.Buffer.full n32 1.0
  | .uniform lo hi seed =>
      cudaUniformBuffer n lo hi seed
  | .normal mean std seed =>
      let key := Spec.Random.keyOf seed 0
      pure <| Runtime.Autograd.Cuda.Buffer.randNormal n32 mean std key
  | .xavierUniform fanIn fanOut seed =>
      let limit := Torch.Init.xavierUniformLimit fanIn fanOut
      cudaUniformBuffer n (-limit) limit seed
  | .kaimingUniform fanIn seed =>
      let limit := Torch.Init.kaimingUniformLimit fanIn
      cudaUniformBuffer n (-limit) limit seed
  | .flat values =>
      if values.size = n then
        Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO values
      else
        throw <| IO.userError
          s!"torch.runtimeInit: flat initializer length mismatch (expected {n}, got {values.size})"

/-- Materialize a runtime initializer as a normal host tensor. Used for CPU execution. -/
def hostTensorOf {α : Type} [TorchLean.Storage α]
    (cast : Float → α) {s : Shape} (init : FloatInit) :
    IO (Tensor α s) := do
  let values ← floatArrayOf (Spec.Shape.size s) init
  -- `floatArrayOf` returns exactly `Shape.size s` values or throws.
  let tensor := Runtime.Autograd.Cuda.Convert.Internal.unflattenFloat (s := s) values 0
  pure <| TorchLean.Tensor.map cast tensor

/--
Host slots for a parameter list before runtime initialization installs the real values.

CUDA runtime initialization immediately replaces these with CUDA mirrors and marks the host values
stale. These entries still give the existing `Param` type a valid host slot for later explicit
readback.
-/
def zeroPack {α : Type} [TorchLean.Storage α]
    (zero : α) : {ss : List Shape} → TorchLean.TensorPack α ss
  | .nil => .nil
  | .cons s ss => .cons (Tensor.full s zero) (zeroPack zero (ss := ss))

/-- Apply a plan after the public entrypoint has validated every initializer. -/
def Internal.applyPlanUnchecked {α : Type} [TorchLean.Storage α] [Torch.TensorTransfer α]
    (cast : Float → α) (options : Torch.Config) :
    {ss : List Shape} → Torch.ParamList α ss → Plan ss → IO Unit
  | .nil, .nil, .nil => pure ()
  | .cons s ss, .cons p ps, .cons init rest => do
      if options.usesCuda then
        let buf ← cudaBufferOf (Spec.Shape.size s) init
        Runtime.Autograd.Torch.Internal.setParamCudaValue (α := α) (sh := s) p
          { s := s, buf := buf }
      else
        let t ← hostTensorOf cast (s := s) init
        Runtime.Autograd.Torch.Internal.setParamHostValue (α := α) (sh := s) p t
      Internal.applyPlanUnchecked (α := α) cast (options := options) (ss := ss) ps rest

/--
Apply a shape-indexed initialization plan to an already-created parameter list.

The shape list appears on both sides of the type:

```lean
Torch.ParamList α ss → RuntimeInit.Plan ss → IO Unit
```

The complete plan is validated before any parameter is mutated, so an invalid later initializer
cannot leave the module partially initialized.
-/
def applyPlan {α : Type} [TorchLean.Storage α] [Torch.TensorTransfer α]
    (cast : Float → α) (options : Torch.Config) {ss : List Shape}
    (parameters : Torch.ParamList α ss) (plan : Plan ss) : IO Unit := do
  match plan.validate with
  | .error message => throw <| IO.userError message
  | .ok () =>
      Internal.applyPlanUnchecked (α := α) cast options parameters plan

end RuntimeInit

end Module

end Model
end Autograd
end Runtime
