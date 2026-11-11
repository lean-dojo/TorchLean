/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Kernel.Function
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Literal
meta import Lean.Elab.Command

/-!
# The recorder's checked equality and its instance check

`record_function%`, which `Function.run` uses with `grad := true`, records a scalar or tensor
function on the `Ops` interface and adds a kernel-checked theorem: the recorded program, run by
the pure interpreter `Id`, computes the source function. This module checks, at build time:

- Programs that record, in `Float32`, `Float`, `ℝ` and the configured format `Binary 15 112`:
  scalar arithmetic, branches on the input, recurrences with a literal and a variable count and
  through each other, shared `let` bindings, constants, tensor arithmetic, `relu`, `sigmoid`,
  `tanh`, `square`, `sum`, `Tensor.scalar` and `matmul`, a shape-polymorphic program, and through
  `Function.run` a function bound by `let`. `#recorded` shows the generated theorem of some of them,
  with the program on the left and the source function on the right.
- Instances that compute something other than the recorded operator are errors: a local `Add`,
  `Sub`, `Mul`, `Div`, `Neg`, `HMul`, `Max`, `Zero` or `BEq` on the scalar, a local `Mul` on a
  tensor type, a recording context with another multiplication, sigmoid or tanh, and an `Add` that
  also adds zero. Among them is the reviewer's example, multiplication defined as subtraction,
  through `Function.run` with `grad := true`, also with a local `Ops Id` instance whose
  multiplication subtracts too: the theorem is about `TorchLean.instOpsId`, never an instance in
  scope.
- What the source computes is what is recorded: an instance equal to the context's, a name that
  shadows a recognized operation, a local notation, a pointwise-multiplication instance that the
  source does not use, and a comparison instance in a branch.
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

/-- `WideBinary` is the configured binary format with 15 exponent and 112 fraction bits. -/
abbrev WideBinary := FloatLib.Floats.ExecFloat.Binary 15 112

def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

def evolve {α : Type} [One α] [Add α] [Mul α] (x : α) : Nat → α
  | 0 => x
  | n + 1 =>
      let y := evolve x n
      y * y + 1

def refine (depth : UInt64) (x : Float32) : Nat → Float32
  | 0 => x
  | n + 1 => power (refine depth x n) depth.toNat

def polynomial (x : Tensor Float32 [3]) : Tensor Float32 [3] :=
  let y := x * x
  y * y + Tensor.full [3] 1

def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (x * x)

def weights : Tensor Float32 [3, 2] := [[1, 2], [3, 4], [5, 6]]

def project (x : Tensor Float32 [2, 3]) : Tensor Float32 [2, 2] := x.matmul weights

def update {s : Shape} (x : Tensor Float32 s) : Nat → Tensor Float32 s
  | 0 => x
  | n + 1 => let y := update x n; y * y + Tensor.full s 1

/-! ## Programs that record -/

def quadratic : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x + 1) }

/--
info: ∀ (t : Tensor Float32 []), quadratic.program t = Tensor.map (fun x ↦ x * x + 1) t
-/
#guard_msgs in
#recorded quadratic

def quotient : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => -(x / (x + 1))) }

def guarded : autograd.Recording Float32 [] [] :=
  { context := inferInstance,
    program := record_function% (fun (x : Float32) => if x == 0 then x else -(x * x / x)) }

/--
info: ∀ (t : Tensor Float32 []), guarded.program t =
Tensor.map (fun x ↦ if (x == 0) = true then x else -(x * x / x)) t
-/
#guard_msgs in
#recorded guarded

def positiveSquare : autograd.Recording Float32 [] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Float32 => if x < 0 then 0 else x * x) }

def fifth : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => power x 5) }

section
variable (steps : Nat)

def powerSteps : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => power x steps) }

def evolveSteps : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => evolve x steps) }

/--
info: ∀ (steps : ℕ) (t : Tensor Float32 []), (evolveSteps steps).program t =
Tensor.map (fun x ↦ evolve x steps) t
-/
#guard_msgs in
#recorded evolveSteps

def refineSteps : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => refine 2 x steps) }

def updateSteps : autograd.Recording Float32 [2] [2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [2] => update x steps) }
end

def shared : autograd.Recording Float32 [] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Float32 => let y := x * x; y * y) }

def sharedBranch : autograd.Recording Float32 [] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Float32 => let y := x * x; if y == 0 then x else y) }

def one : autograd.Recording Float [] [] :=
  { context := inferInstance, program := record_function% (fun (_ : Float) => (1 : Float)) }

def polynomialTensor : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance, program := record_function% polynomial }

/--
info: ∀ (t : Tensor Float32 [3]), polynomialTensor.program t = polynomial t
-/
#guard_msgs in
#recorded polynomialTensor

def energyTensor : autograd.Recording Float32 [3] [] :=
  { context := inferInstance, program := record_function% energy }

/--
info: ∀ (t : Tensor Float32 [3]), energyTensor.program t = Tensor.scalar (energy t)
-/
#guard_msgs in
#recorded energyTensor

def projectTensor : autograd.Recording Float32 [2, 3] [2, 2] :=
  { context := inferInstance, program := record_function% project }

/--
info: ∀ (t : Tensor Float32 [2, 3]), projectTensor.program t = project t
-/
#guard_msgs in
#recorded projectTensor

def sums : autograd.Recording Float32 [3] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] =>
      Tensor.scalar (Tensor.sum x * Tensor.sum x)) }

def activations : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] =>
      Tensor.relu x - Tensor.sigmoid x / Tensor.tanh (-x)) }

def squares : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.square x) }

def gram : autograd.Recording Float32 [2, 2] [2, 2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [2, 2] => x.matmul x) }

def anyShape {s : Shape} : autograd.Recording Float32 s s :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float32 s => x * x + x) }

def binary64 : autograd.Recording Float [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float => x * x) }

def binary64Tensor : autograd.Recording Float [2] [2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float [2] => x * x) }

noncomputable def real : autograd.Recording ℝ [] [] :=
  { context := inferInstance, program := record_function% (fun x : ℝ => x * x + 1) }

noncomputable def realTensor : autograd.Recording ℝ [2] [2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor ℝ [2] => x * x - x / x) }

def wideQuotient : autograd.Recording WideBinary [] [] :=
  { context := inferInstance,
    program := record_function% (fun x : WideBinary => -(x * x / (x + 1))) }

/--
info: ∀ (t : Tensor WideBinary []), wideQuotient.program t =
Tensor.map (fun x ↦ -(x * x / (x + 1))) t
-/
#guard_msgs in
#recorded wideQuotient

def wideGuarded : autograd.Recording WideBinary [] [] :=
  { context := inferInstance,
    program := record_function% (fun (x : WideBinary) => if x == 0 then x else -(x * x / x)) }

def wideConstant : autograd.Recording WideBinary [] [] :=
  { context := inferInstance,
    program := record_function%
      (fun (_ : WideBinary) => (Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat) : WideBinary)) }

def wideTensor : autograd.Recording WideBinary [2] [2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor WideBinary [2] => x * x) }

def wideGram : autograd.Recording WideBinary [2, 2] [2, 2] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor WideBinary [2, 2] => x.matmul x) }

def wideSum : autograd.Recording WideBinary [3] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor WideBinary [3] => Tensor.sum x) }

def wideEnergy : autograd.Recording WideBinary [3] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor WideBinary [3] => Tensor.sum (x * x)) }

/-- A function bound by `let` inside a program records through its body. -/
def cubeGradient : IO (Tensor Float32 [2]) := do
  let cube := fun (x : Float32) => x * x * x
  let result ← cube.run (Tensor.full [2] 2) (grad := true)
  result.backward

def wideSquare (x : WideBinary) : WideBinary := x * x

/-- A configured-format function records through `Function.run`. -/
def wideSquareGradient : IO (Tensor WideBinary [1]) := do
  let result ← wideSquare.run (Tensor.full [1] 1) (grad := true)
  result.backward

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
example : autograd.Recording Float [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float => x * x) }

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
local instance : Add Float32 := ⟨fun a b => a * b⟩

/--
error: recording: the operation
  x + x
does not compute what `Op.add` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x + x) }

/--
error: recording: the operation
  x.sum
does not compute what `Op.reduce` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.sum x) }
end

section
local instance : Neg Float32 := ⟨fun a => a⟩

/--
error: recording: the operation
  -x
does not compute what `Op.neg` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => -x) }

/--
error: recording: the operation
  -x
does not compute what `Op.neg` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float32 [3] => -x) }
end

section
local instance : Div Float32 := ⟨fun a b => a + b⟩
local instance : Sub Float32 := ⟨fun a b => a + b⟩

/--
error: recording: the operation
  x / x
does not compute what `Op.div` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x / x) }

/--
error: recording: the operation
  x - x
does not compute what `Op.sub` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float32 [3] => x - x) }
end

section
local instance : Mul Float32 := ⟨fun a b => a - b⟩

/--
error: recording: the operation
  x.square
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.square x) }

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

/-- The reviewer's example: multiplication defined as subtraction. -/
def subtractSquare (x : Float32) : Float32 := x * x

#guard subtractSquare 3 == 0

/-- Without a gradient, `run` evaluates the function itself and records nothing. -/
def subtractForward : IO (Tensor Float32 [1]) := subtractSquare.run (Tensor.full [1] 3)

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

/-- A context whose multiplication subtracts. -/
@[instance_reducible] def subtracting : Context Float32 :=
  { (inferInstance : Context Float32) with mul := fun a b => a - b }

section
/-- A pure interpreter that applies each operator in `subtracting`, so that the source and this
interpreter agree on the reviewer's example. The recorder does not use it. -/
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
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x) }

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

section
local instance : Max Float32 := ⟨fun a _ => a⟩

/--
error: recording: the operation
  x.relu
does not compute what `Op.relu` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.relu x) }
end

section
local instance : Zero Float32 := ⟨1⟩

/--
error: recording: the operation
  x.sum
does not compute what `Op.reduce` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.sum x) }

/--
error: recording: the operation
  x.relu
does not compute what `Op.relu` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.relu x) }
end

section
local instance : BEq Float32 := ⟨fun _ _ => true⟩

/--
error: recording: the operation
  x.relu
does not compute what `Op.relu` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := inferInstance,
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.relu x) }
end

section
local instance : Mul (Tensor Float32 [2]) := ⟨fun a b => a - b⟩

/--
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [2] [2] :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float32 [2] => x * x) }
end

section
local instance : HMul Float32 Float32 Float32 := ⟨fun a b => a - b⟩

/--
error: recording: the operation
  x * x
does not compute what `Op.mul` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x) }
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

/--
error: recording: the operation
  x.sigmoid
does not compute what `Op.sigmoid` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := { (inferInstance : Context Float32) with sigmoid := fun x => x },
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.sigmoid x) }

/--
error: recording: the operation
  x.tanh
does not compute what `Op.tanh` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [3] [3] :=
  { context := { (inferInstance : Context Float32) with tanh := fun x => x },
    program := record_function% (fun x : Tensor Float32 [3] => Tensor.tanh x) }

section
local instance : Add Float32 := ⟨fun a b => a + b + 0⟩

/--
error: recording: the operation
  x + x
does not compute what `Op.add` computes on `Float32` in the recording's context. An operation is
recorded only when its instance agrees with that context's.
-/
#guard_msgs in
example : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x + x) }
end

/-! ## What the source computes is recorded -/

section
local instance : Mul Float32 := ⟨Float32.mul⟩

def sameMul : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x) }
end

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

section
local macro_rules | `($a * $b) => `($a - $b)

def localNotation : autograd.Recording Float32 [] [] :=
  { context := inferInstance, program := record_function% (fun x : Float32 => x * x) }
end

section
local instance : Tensor.Pointwise .mul Float32 Float32 Float32 where
  scalar a b := a - b
  apply x y := Tensor.zipWith (fun a b => a - b) x y
  apply_at _ _ _ := by simp

def pointwise : autograd.Recording Float32 [2] [2] :=
  { context := inferInstance, program := record_function% (fun x : Tensor Float32 [2] => x * x) }
end

section
local instance : BEq Float32 := ⟨fun _ _ => true⟩

def alwaysEqual : autograd.Recording Float32 [] [] :=
  { context := inferInstance,
    program := record_function% (fun x : Float32 => if x == 0 then x else -x) }
end

end TorchLeanTest.Recorder
