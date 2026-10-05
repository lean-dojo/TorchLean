/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda.Correctness

/-!
# Certified custom-operation programs

A program retains the Lean function supplied to the frontend and a proof that its typed expression
has the same result, including failed reads. The certificate concerns Lean semantics. Native CUDA
emission, compilation and execution have the separate trust boundary described in `Runtime`.
-/

@[expose] public section

namespace NN.Kernel

/-- Binary32 scalar operations in the same order as ordinary Lean arithmetic. -/
def Arithmetic.binary32 : Arithmetic Float32 where
  binary
    | .add, x, y => x + y
    | .sub, x, y => x - y
    | .mul, x, y => x * y
    | .div, x, y => x / y
  neg := fun x => -x
  compare
    | .eq, x, y => x == y
    | .lt, x, y => x < y
    | .le, x, y => x ≤ y

/-- Binary64 scalar operations in the same order as ordinary Lean arithmetic. -/
def Arithmetic.binary64 : Arithmetic Float where
  binary
    | .add, x, y => x + y
    | .sub, x, y => x - y
    | .mul, x, y => x * y
    | .div, x, y => x / y
  neg := fun x => -x
  compare
    | .eq, x, y => x == y
    | .lt, x, y => x < y
    | .le, x, y => x ≤ y

/-- Bind the single free variable to the row-major output index. -/
def Env.output {α : Type} (index : UInt64) : Env α [.index] :=
  ⟨fun v => match v with | .zero => index | .succ v => nomatch v⟩

/-- Evaluate one output element using a supplied arithmetic implementation and input reader. -/
def evaluate {α : Type} (arithmetic : Arithmetic α) (expr : Expr α [.index] .scalar)
    (read : Reader α) (index : UInt64) : Except Error α :=
  expr.eval arithmetic read (Env.output index)

/-- A frontend result with its original Lean function and checked semantic correspondence. -/
structure Program (α : Type) where
  format : Cuda.Format
  bits : α → UInt64
  arithmetic : Arithmetic α
  reference : Reader α → UInt64 → Except Error α
  expression : Expr α [.index] .scalar
  correct : ∀ read index, evaluate arithmetic expression read index = reference read index

/-- An elementwise program whose reference calculation is the supplied scalar function. -/
structure Elementwise {α : Type} (f : α → α) where
  program : Program α
  reference_eq : ∀ read index, program.reference read index = do
    let x ← read 0 index
    pure (f x)

/-- The scoped target before CUDA name allocation preserves the original source function. -/
theorem Program.lower_correct {α : Type} (program : Program α) (read : Reader α)
    (index : UInt64) :
    (Target.lower program.expression).eval program.arithmetic read (Env.output index) =
      program.reference read index := by
  rw [Target.eval_lower]
  exact program.correct read index

/-- The named statements consumed by CUDA rendering preserve the original Lean function.

Only the output index must be initialized. Other local values may be arbitrary; the allocation
proof ensures that none can affect the result. Read failures are retained, and scalar arithmetic
follows the supplied operation order. Rendering, NVRTC and GPU execution are separate boundaries.
-/
theorem Program.named_correct {α : Type} (program : Program α) (read : Reader α)
    (index : UInt64) (locals : Cuda.Locals α) (hIndex : locals.get .index = index) :
    let ((statement, result), _) :=
      (Cuda.Internal.lower Cuda.Internal.Names.output (Target.lower program.expression)).run 0
    (statement.eval program.arithmetic read locals).run.map (·.map (·.get result)) =
      some (program.reference read index) := by
  have hNames : ∀ {t : Ty} (v : Var [.index] t),
      (Cuda.Internal.Names.output.get v).Before 0 := by
    intro t v
    cases v with
    | zero => trivial
    | succ v => nomatch v
  have hScope : ∀ {t : Ty} (v : Var [.index] t),
      locals.get (Cuda.Internal.Names.output.get v) = (Env.output index).get v := by
    intro t v
    cases v with
    | zero => exact hIndex
    | succ v => nomatch v
  have h := Cuda.eval_lower (Target.lower program.expression) Cuda.Internal.Names.output 0
    locals (Env.output index) hNames hScope program.arithmetic read
  rw [program.lower_correct] at h
  exact h

end NN.Kernel
