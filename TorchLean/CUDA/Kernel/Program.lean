/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Emit.Correctness
public import TorchLean.CUDA.Kernel.Scalar

/-!
# Certified custom-operation programs

A `Program` keeps the Lean function supplied to the frontend together with a proof that its typed
expression gives the same result at every output index, failed reads included. `Elementwise` and
`Zip` tie that reference to a unary or binary scalar function. `Program.lower_correct` and
`Program.named_correct` carry the certificate through scoping and through CUDA name allocation.

The certificate is about Lean semantics. `CUDA.denotes_emit` relates it to the rendered native
source, and NVRTC compilation and device execution sit behind the foreign-code boundary of
`TorchLean.CUDA.Kernel.Launch`.
-/

@[expose] public section

namespace TorchLean.Kernel

/-- Bind the single free variable to the row-major output index. -/
def Env.output {α : Type} (index : UInt64) : Env α [.index] :=
  ⟨fun v => match v with | .zero => index | .succ v => nomatch v⟩

/-- The output environment binds its only variable to the output index. -/
@[simp] theorem Env.get_output {α : Type} (index : UInt64) :
    (Env.output (α := α) index).get Var.zero = index := rfl

/-- Evaluate one output element using a supplied arithmetic implementation and input reader. -/
def evaluate {α : Type} (arithmetic : Arithmetic α) (expr : Expr α [.index] .scalar)
    (read : Reader α) (index : UInt64) : Except Error α :=
  expr.eval arithmetic read (Env.output index)

/-- A frontend result: the typed `expression`, the original Lean function as `reference`, and a
proof that both give the same result for every reader and output index. -/
structure Program (α : Type) where
  /-- The scalar type that the device computes with. -/
  scalar : Scalar α
  /-- The arithmetic of `expression` and of its certificate. -/
  arithmetic : Arithmetic α
  /-- The Lean function given to the frontend. -/
  reference : Reader α → UInt64 → Except Error α
  /-- The typed expression of one output entry. -/
  expression : Expr α [.index] .scalar
  /-- `expression` gives `reference` for every reader and output index. -/
  correct : ∀ read index, evaluate arithmetic expression read index = reference read index
  /-- The source certificate uses the arithmetic selected for device execution. -/
  native : arithmetic = scalar.arithmetic

/-- A program whose reference reads input `0` at the output index and applies `f`. -/
structure Elementwise {α : Type} (f : α → α) where
  /-- The certified program. -/
  program : Program α
  /-- The reference reads input `0` at the output index and applies `f`. -/
  reference_eq : ∀ read index, program.reference read index = do
    let x ← read 0 index
    pure (f x)

/-- A program whose reference reads inputs `0` and `1`, in that order, and applies `f`. -/
structure Zip {α : Type} (f : α → α → α) where
  /-- The certified program. -/
  program : Program α
  /-- The reference reads inputs `0` and `1` at the output index, in order, and applies `f`. -/
  reference_eq : ∀ read index, program.reference read index = do
    let x ← read 0 index
    let y ← read 1 index
    pure (f x y)

/-- The scoped target before CUDA name allocation preserves the original source function. -/
theorem Program.lower_correct {α : Type} (program : Program α) (read : Reader α)
    (index : UInt64) :
    (Target.lower program.expression).eval program.arithmetic read (Env.output index) =
      program.reference read index := by
  rw [Target.eval_lower]
  exact program.correct read index

/-- The named statements consumed by CUDA rendering preserve the original Lean function.

Only the output index has to be initialized (`hIndex`). Every other local may hold any value: the
allocation proof shows that none of them affects the result. Read failures carry through, and the
scalar arithmetic follows the supplied operation order. Rendering, NVRTC and GPU execution are
separate boundaries.
-/
theorem Program.named_correct {α : Type} (program : Program α) (read : Reader α)
    (index : UInt64) (locals : CUDA.Locals α) (hIndex : locals.get .index = index) :
    let ((statement, result), _) :=
      (CUDA.lower CUDA.Names.output (Target.lower program.expression)).run 0
    (statement.eval program.arithmetic read locals).run.map (·.map (·.get result)) =
      some (program.reference read index) := by
  have hNames : ∀ {t : Ty} (v : Var [.index] t),
      (CUDA.Names.output.get v).Before 0 := by
    intro t v
    cases v with
    | zero => trivial
    | succ v => nomatch v
  have hScope : ∀ {t : Ty} (v : Var [.index] t),
      locals.get (CUDA.Names.output.get v) = (Env.output index).get v := by
    intro t v
    cases v with
    | zero => exact hIndex
    | succ v => nomatch v
  have h := CUDA.eval_lower (Target.lower program.expression) CUDA.Names.output 0
    locals (Env.output index) hNames hScope program.arithmetic read
  rw [program.lower_correct] at h
  exact h

end TorchLean.Kernel
