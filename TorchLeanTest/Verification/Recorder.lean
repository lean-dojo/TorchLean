/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Kernel.Function
meta import Lean.Elab.Command

/-!
# The recorder's checked equality and its instance check

`record_function%`, which `Function.run` uses with `grad := true`, records a scalar or tensor
function on the `Ops` interface and adds a kernel-checked theorem: the recorded program, run by
the pure interpreter `TorchLean.instOpsId`, computes the source function. The theorem is about
that interpreter, never an `Ops Id` instance in scope. When a scalar function does not record at
its input's shape, for example one that branches on its input, `Function.run` records it at shape
`[]` and lifts it with `autograd.Recording.elementwise`, and no theorem covers the lift.

`#recorded` checks the left side of a recording's theorem and prints the theorem. The programs
below record what the source computes: a map, a sum through a definition, a function named like
`Tensor.relu`, activations, a recurrence with a variable count and a `Mul` instance equal to the
context's. A local `Mul` that subtracts is an error wherever an operation uses it: `x * x` on a
`Float` tensor, `matmul` on `Float32`, and `x * x` on `Float32` through `Function.run` with
gradients, an elaboration error for `grad := true`, also under a local `Ops Id` instance that
subtracts too, and an IO error for `grad` known only at run time. So is a recording context whose
multiplication subtracts.
-/

open TorchLean

namespace TorchLeanTest.Recorder

open Lean Elab Command Meta in
/-- `#recorded r` finds the theorem that recording the program of the `Recording` named `r` added,
checks that its left side is `r.program` run by the pure interpreter `TorchLean.instOpsId`, and
prints the theorem with that left side. -/
elab "#recorded " id:ident : command => liftTermElabM do
  let decl ← realizeGlobalConstNoOverloadWithInfo id
  let theorems := (← getEnv).constants.map₂.toList.filterMap fun (name, info) =>
    let recorded := match name with
      | .str _ last => last.startsWith "_recordedArithmetic"
      | _ => false
    if decl.isPrefixOf name && recorded && info matches .thmInfo _ then some info else none
  let [info] := theorems |
    throwError "expected one recording theorem for {decl}, found {theorems.length}"
  forallTelescope info.type fun xs body => do
    let some (_, lhs, rhs) := body.eq? | throwError "the recording theorem is not an equation"
    let recording := mkAppN (← mkConstWithLevelParams decl) xs.pop
    let idM := mkConst ``Id [Level.zero]
    let idMonad := mkConst ``Id.instMonad [Level.zero]
    let type ← whnf (← inferType recording)
    let context ← mkAppM ``autograd.Recording.context #[recording]
    let idOps := mkAppN (mkConst ``TorchLean.instOpsId)
      #[type.getAppArgs[0]!, type.getAppArgs[1]!, context]
    let program ← mkAppOptM ``autograd.Recording.program
      #[none, none, none, none, some recording, some idM, some idMonad, some idOps, some xs.back!]
    let same ← withTheReader Core.Context ({ · with maxHeartbeats := 200000 })
      (isDefEq lhs program)
    unless same do
      throwError "the left side is not the recorded program:{indentExpr lhs}"
    logInfo m!"{← mkForallFVars xs (← mkEq program rhs)}"

def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def refine (depth : UInt64) (x : Float32) : Nat → Float32
  | 0 => x
  | n + 1 => power (refine depth x n) depth.toNat

def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (Tensor.square x)

/-! ## Programs that record -/

def quadratic : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x + 1) }

/--
info: ∀ (t : Tensor Float32 []), quadratic.program t = Tensor.map (fun x ↦ x * x + 1) t
-/
#guard_msgs in
#recorded quadratic

def energyTensor : autograd.Recording Float32 [3] [] :=
  { context := inferInstance, program := record_function% energy }

/--
info: ∀ (t : Tensor Float32 [3]), energyTensor.program t = Tensor.scalar (energy t)
-/
#guard_msgs in
#recorded energyTensor

namespace Shadow
/-- A function named like `Tensor.relu` that negates. -/
def Tensor.relu {s : Shape} (x : TorchLean.Tensor Float32 s) : TorchLean.Tensor Float32 s := -x
end Shadow

def shadowed : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Shadow.Tensor.relu x) }

/--
info: ∀ (t : Tensor Float32 [3]), shadowed.program t = Shadow.Tensor.relu t
-/
#guard_msgs in
#recorded shadowed

def activations : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] =>
      Tensor.relu x - Tensor.sigmoid x / Tensor.tanh (-x)) }

section
variable (steps : Nat)

def refineSteps : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => refine 2 x steps) }
end

section
local instance : Mul Float32 := ⟨Float32.mul⟩

def sameMul : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x) }
end

/-! ## Instances that compute another operation -/

section
local instance : Mul Float := ⟨fun a b => a - b⟩

/--
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float [2] [2] :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float [2] => x * x) }
end

section
local instance : Mul Float32 := ⟨fun a b => a - b⟩

/--
error: recording: the operation
  x.matmul x
does not compute what `Op.matmul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [2, 2] [2, 2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [2, 2] => x.matmul x) }

/-- Multiplication defined as subtraction. -/
def subtractSquare (x : Float32) : Float32 := x * x

#guard subtractSquare 3 == 0

/--
error: could not synthesize default value for parameter 'execution' using tactics
---
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : IO (autograd.Result Float32 [1] [1]) :=
  subtractSquare.run (Tensor.full [1] 3) (grad := true)

/-- With `grad` known only at run time, the failed recording is an IO error of the gradient
branch. -/
def subtractAt (grad : Bool) := subtractSquare.run (Tensor.full [1] (3 : Float32)) (grad := grad)

/--
error: recording: the operation
  x * x
does not compute what `TorchLean.Op.mul` computes on `Float32` in the recording's context. An
operation is recorded only when its instance agrees with that context's.
-/
#guard_msgs in
#eval do discard <| subtractAt true

/-- A context whose multiplication subtracts. -/
@[instance_reducible] def subtracting : Context Float32 :=
  { (inferInstance : Context Float32) with mul := fun a b => a - b }

section
/-- A pure interpreter that applies each operator in `subtracting`, so that the source and this
interpreter agree on `subtractSquare`. The recorder does not use it. -/
local instance (priority := high) : Ops Id Float32 where
  Ref := fun s => Tensor Float32 s
  DataRef := fun β _ s => Tensor β s
  dataConst := fun x => x
  mapData := fun f x => f x
  const := fun t => pure t
  observe := fun x => pure (Tensor.item x)
  apply := fun {_ _ _} op [supported : Op.Supported Float32 op] args =>
    pure (@Ops.pureApply Float32 _ subtracting _ _ _ op supported.caps args)
  applyRandom := fun {_ _ _} op [supported : Op.Supported Float32 op] _ args =>
    pure (@Ops.pureApply Float32 _ subtracting _ _ _ op supported.caps args)

/--
error: could not synthesize default value for parameter 'execution' using tactics
---
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : IO (autograd.Result Float32 [1] [1]) :=
  subtractSquare.run (Tensor.full [1] 3) (grad := true)
end
end

/--
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := { (inferInstance : Context Float32) with mul := fun a b => a - b },
    program := record_function% (fun x : Float32 => x * x) }

end TorchLeanTest.Recorder
