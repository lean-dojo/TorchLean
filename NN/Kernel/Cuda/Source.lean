/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda.Statements
public import NN.Kernel.Program

/-!
# Semantics of the generated CUDA entrypoint

The signature grammar checks the numbered input pointer/length pairs and the output, error and
count arguments. Entrypoint semantics promote the thread coordinates before multiplication, skip
out-of-range threads and write only after a successful body. The body uses the structured text
semantics from `Cuda.Statements`. This covers the emitter's grammar, not arbitrary CUDA syntax.
Physical buffers, launch coordinates, atomic error recording and NVIDIA compilation remain
external contracts.
-/

@[expose] public section

namespace NN.Kernel.Cuda

/-- The input arguments occupy consecutive numbered pointer/length pairs, followed by the three
fixed output, error and count arguments. Scalar pointer types must decode at the selected format.
-/
inductive Arguments.Denotes (format : Format) : Nat → Nat → List String → Prop
  | nil {first : Nat} {typeText : String}
      (ht : Internal.type? format typeText = some .scalar) :
      Denotes format first 0
        [s!"{typeText}* output", "unsigned long long* error", "unsigned long long count"]
  | cons {first count : Nat} {typeText : String} {rest : List String}
      (ht : Internal.type? format typeText = some .scalar)
      (tail : Denotes format (first + 1) count rest) :
      Denotes format first (count + 1)
        (s!"const {typeText}* input{first}, unsigned long long size{first}" :: rest)

private theorem arguments_range (format : Format) (count first : Nat) :
    Arguments.Denotes format first count
      ((List.range' first count).map (fun i =>
        s!"const {Internal.Format.name format}* input{i}, unsigned long long size{i}") ++
        [s!"{Internal.Format.name format}* output", "unsigned long long* error",
          "unsigned long long count"]) := by
  have ht : Internal.type? format (Internal.Format.name format) = some .scalar := by
    cases format <;> rfl
  induction count generalizing first with
  | zero => simpa using Arguments.Denotes.nil (first := first) ht
  | succ count ih =>
      simpa [List.range'_succ] using Arguments.Denotes.cons ht (ih (first + 1))

/-- The signature emitted by `source` contains exactly the declared number of native inputs. -/
theorem Arguments.denotes_range (format : Format) (inputs : Nat) :
    Arguments.Denotes format 0 inputs
      ((List.range inputs).map (fun i =>
        s!"const {Internal.Format.name format}* input{i}, unsigned long long size{i}") ++
        [s!"{Internal.Format.name format}* output", "unsigned long long* error",
          "unsigned long long count"]) := by
  simpa [List.range_eq_range'] using arguments_range format inputs 0

/-- Execution of one thread. `none` means undefined arithmetic; `.ok none` means the output guard
skipped the thread. An error prevents the final output write. Coordinates are supplied externally.
-/
def Source.eval {α : Type} (body : Locals α → ExceptT Error Option (Locals α))
    (result : Name .scalar) (block blockSize thread : UInt32) (count : UInt64)
    (locals : Locals α) : Option (Except Error (Option α)) :=
  let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
  if index ≥ count then some (.ok none)
  else (body (locals.set .index index)).run.map (·.map fun after => some (after.get result))

/-- The complete entrypoint text, with a decoded signature, body and final result name.

The index formula has unsigned 64-bit semantics: the block coordinate is promoted before the
product. This relation does not certify the values supplied by a physical CUDA launch.
-/
inductive Source.Denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (inputs : Nat) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value) :
    String → (UInt32 → UInt32 → UInt32 → UInt64 → Locals format.Value →
      Option (Except Error (Option format.Value))) → Prop
  | kernel {arguments : List String} {body resultText : String}
      {run : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (result : Name .scalar)
      (hArguments : Arguments.Denotes format 0 inputs arguments)
      (hBody : Statement.Denotes format arithmetic inputs size load body run)
      (hResult : Name.parse .scalar resultText = some result) :
      Denotes format arithmetic inputs size load
        (s!"extern \"C\" __global__ void torchlean_kernel({", ".intercalate arguments}) \{\n" ++
        "unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n" ++
        "if (index >= count) return;\n" ++ body ++ s!"output[index] = {resultText};\n}\n")
        (Source.eval run result)

/-- Successfully generated native source preserves the Lean expression at every supplied thread
coordinate, including read failures and the output guard. Arithmetic and buffer contents remain
explicit contracts; compilation, physical memory and device execution are not proved here.
-/
theorem source_denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value)
    (hRead : ∀ operand, operand < inputs → ∀ index,
      read operand index = if index ≥ size operand then
        .error (.bounds operand index (size operand).toNat) else .ok (load operand index))
    (expr : Expr format.Value [.index] .scalar) {generated : Source}
    (hSource : source format (fun x => (Literal.ofValue format x).bits) inputs expr =
      .ok generated) :
    Source.Denotes format arithmetic inputs size load generated.text
      (fun block blockSize thread count _ =>
        let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
        if index ≥ count then some (.ok none)
        else some ((expr.eval arithmetic read (Env.output index)).map some)) := by
  have hNames : ∀ {t : Ty} (v : Var [.index] t),
      (Internal.Names.output.get v).Before 0 := by
    intro t v
    cases v with
    | zero => trivial
    | succ v => nomatch v
  have hBody := @Statement.denotes_lower format arithmetic read inputs size load hRead
    [.index] .scalar (Target.lower expr) Internal.Names.output 0 hNames
  have hEval := fun (locals : Locals format.Value) (index : UInt64) =>
    eval_lower (Target.lower expr) Internal.Names.output 0 (locals.set .index index)
      (Env.output index) hNames (by
        intro t v
        cases v with
        | zero => exact Locals.get_set _ _ _
        | succ v => nomatch v) arithmetic read
  unfold source at hSource
  generalize hAllocated : (Internal.lower Internal.Names.output (Target.lower expr)).run 0 =
    allocated at hSource hBody hEval
  rcases allocated with ⟨⟨statement, result⟩, next⟩
  dsimp only at hBody hEval hSource
  cases hRender : Internal.render format (fun x => (Literal.ofValue format x).bits)
      inputs statement with
  | error message => simp [hRender, bind, Except.bind] at hSource
  | ok body =>
      simp [hRender, bind, Except.bind] at hSource
      dsimp [pure, Except.instMonad, Except.pure] at hSource
      injection hSource with hSource
      subst generated
      have hDenotes := Source.Denotes.kernel result (Arguments.denotes_range format inputs)
        (hBody hRender) (Name.parse_render result)
      have hRun : Source.eval (statement.eval arithmetic read) result =
          (fun block blockSize thread count (_ : Locals format.Value) =>
            let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
            if index ≥ count then some (.ok none)
            else some ((expr.eval arithmetic read (Env.output index)).map some)) := by
        funext block blockSize thread count locals
        dsimp only [Source.eval]
        split
        · rfl
        · have h := hEval locals (block.toUInt64 * blockSize.toUInt64 + thread.toUInt64)
          rw [Target.eval_lower] at h
          have hMaps : ∀ value : Option (Except Error (Locals format.Value)),
              value.map (·.map fun after => some (after.get result)) =
                (value.map (·.map (·.get result))).map (·.map some) := by
            intro value
            cases value with
            | none => rfl
            | some value => cases value <;> rfl
          rw [hMaps, h]
          rfl
      rw [hRun] at hDenotes
      exact hDenotes

end NN.Kernel.Cuda
