/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Emit.Binary
public import TorchLean.CUDA.Kernel.Emit.Correctness
public import TorchLean.CUDA.Kernel.Emit.Source

/-!
# Row kernels

A row kernel computes an expression once for every row of a table of columns. Thread `index`
reads variable `p` of its row from column `p` at `index`, then runs the expression with the
checked input reads of `emit` and writes the result, a scalar or an index, to `output[index]`.
The variables of the expression are the columns, so a fold step over the slots of a tree level is
one row kernel. A thread that starts after a failed read was recorded returns at once, so a
launch that follows a failed one does no work and reads no result that a failed row left
unwritten.

`emitRow` lowers the expression with the allocation of `emit`, the columns taking the first
temporaries. `RowSource.Denotes` gives the text its meaning: the preamble binds the columns,
and the body has the semantics of `Statement.Denotes`. `denotes_emitRow` proves that native row
kernels denote the expression at the row of each thread, read failures and both guards included.
`emitRowBinary` renders the same statements for a configured binary format through the dialect
of `emitBinary`. Compilation, the column buffers and device execution are outside the theorems,
as for `emit`.
-/

@[expose] public section

namespace TorchLean.CUDA

open FloatLib.Floats.Formats.BinaryInterchange
open Kernel

/-- The names of the variables of `Γ` when they are read from columns: variable `p` is the
temporary `first + p`. -/
def Names.columns : (Γ : List Ty) → Nat → Names Γ
  | [], _ => ⟨fun v => nomatch v⟩
  | _ :: Γ, first => (Names.columns Γ (first + 1)).push (.temporary first)

/-- The column names are allocated before `first + Γ.length`. -/
theorem Names.columns_before : ∀ {Γ : List Ty} {first : Nat} {t : Ty} (v : Var Γ t),
    ((Names.columns Γ first).get v).Before (first + Γ.length)
  | _ :: _, first, _, .zero => by
      change first < first + (_ + 1)
      omega
  | _ :: Γ, first, _, .succ v => by
      have h := Names.columns_before (Γ := Γ) (first := first + 1) v
      change ((Names.columns Γ (first + 1)).get v).Before (first + (Γ.length + 1))
      rwa [show first + (Γ.length + 1) = first + 1 + Γ.length by omega]

/-- The locals after reading a row from the columns: the temporary `first + p` holds variable `p`
of `env`. -/
def Locals.columns {α : Type} : (Γ : List Ty) → Nat → Env α Γ → Locals α → Locals α
  | [], _, _, locals => locals
  | _ :: Γ, first, env, locals =>
      Locals.columns Γ (first + 1) (env.pull Var.succ)
        (locals.set (.temporary first) (env.get .zero))

/-- Reading the columns leaves every local allocated before `first` unchanged. -/
theorem Locals.get_columns_of_before {α : Type} {s : Ty} :
    ∀ (Γ : List Ty) (first : Nat) (env : Env α Γ) (locals : Locals α) (other : Name s),
      other.Before first → (Locals.columns Γ first env locals).get other = locals.get other
  | [], _, _, _, _, _ => rfl
  | _ :: Γ, first, env, locals, other, h => by
      rw [Locals.columns, get_columns_of_before Γ (first + 1) _ _ other
        (Name.before_mono other (Nat.le_succ first) h),
        Locals.get_set_before locals first _ other h]

/-- After reading the columns, the name of each variable holds its value. -/
theorem Locals.get_columns {α : Type} : ∀ {Γ : List Ty} (first : Nat) (env : Env α Γ)
    (locals : Locals α) {t : Ty} (v : Var Γ t),
    (Locals.columns Γ first env locals).get ((Names.columns Γ first).get v) = env.get v
  | s :: Γ, first, env, locals, _, .zero => by
      change (Locals.columns Γ (first + 1) _ _).get (.temporary first) = env.get .zero
      rw [get_columns_of_before Γ (first + 1) _ _ (Name.temporary (t := s) first)
        (Nat.lt_succ_self first), Locals.get_set]
  | _ :: _, first, env, _, _, .succ v => get_columns (first + 1) (env.pull Var.succ) _ v

/-- The declarations that read the variables of `Γ` at the thread index: variable `p` goes into
the temporary `first + p` from `column{first + p}`. -/
def preamble (typeName : Ty → String) : List Ty → Nat → String
  | [], _ => ""
  | t :: Γ, first =>
      s!"{typeName t} {(Name.temporary (t := t) first).render} = column{first}[index];\n" ++
        preamble typeName Γ (first + 1)

/-- The column arguments of a row kernel, one read-only pointer per variable of `Γ`. -/
def columnArguments (typeName : Ty → String) : List Ty → Nat → List String
  | [], _ => []
  | t :: Γ, first => s!"const {typeName t}* column{first}" :: columnArguments typeName Γ (first + 1)

/-- The guard that ends a thread which starts after a failed read was recorded. -/
def errorGuard : String := "if (*(volatile unsigned long long*)error != 0ULL) return;\n"

/-- The signature, guards, column reads and output write of a row kernel with results of type
`t`. The inputs and the fixed arguments follow the columns as in `entrypoint`. -/
def rowEntrypoint (typeName : Ty → String) (Γ : List Ty) (inputs : Nat) (body : String)
    {t : Ty} (result : Name t) : String :=
  let arguments := columnArguments typeName Γ 0 ++
    (((List.range inputs).map fun i =>
      s!"const {typeName .scalar}* input{i}, unsigned long long size{i}") ++
    [s!"{typeName t}* output", "unsigned long long* error", "unsigned long long count"])
  s!"extern \"C\" __global__ void torchlean_kernel({", ".intercalate arguments}) \{\n" ++
    "unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    "if (index >= count) return;\n" ++ errorGuard ++ preamble typeName Γ 0 ++ body ++
    s!"output[index] = {result.render};\n}\n"

/-- Emit the row kernel of `expr` with exact bit-pattern literals: thread `index` reads variable
`p` from column `p` at `index`, computes `expr` with checked input reads and writes the result to
`output[index]`. Callers check the error record before they use the output. -/
def emitRow {α : Type} (format : Format) (bits : α → UInt64) (inputs : Nat) {Γ : List Ty}
    {t : Ty} (expr : Expr α Γ t) : Except String Source := do
  let ((statement, result), _) := (lower (Names.columns Γ 0) (Target.lower expr)).run Γ.length
  let body ← Statement.render format bits inputs statement
  let text := rowEntrypoint (Dialect.native format bits).typeName Γ inputs body result
  return { format, inputs, text }

/-- Emit the row kernel of `expr` for a configured binary format, with the literals, arithmetic
and template of `emitBinary`. -/
@[no_expose] def emitRowBinary {α : Type} (format : FloatFormat) (encode : α → Nat)
    (inputs : Nat) {Γ : List Ty} {t : Ty} (expr : Expr α Γ t) : Except String String := do
  let prelude ← binaryPrelude format
  let dialect := Dialect.configured format encode
  let ((statement, result), _) := (lower (Names.columns Γ 0) (Target.lower expr)).run Γ.length
  let body ← Statement.renderWith dialect inputs statement
  return prelude ++ rowEntrypoint dialect.typeName Γ inputs body result

/-- The inputs of a row kernel occupy numbered pointer and length pairs, as in
`Arguments.Denotes`, followed by an output pointer of type `t` and the error and count arguments.
-/
inductive RowArguments.Denotes (format : Format) (t : Ty) : Nat → Nat → List String → Prop
  | nil {first : Nat} {typeText : String} (ht : Format.type? format typeText = some t) :
      RowArguments.Denotes format t first 0
        [s!"{typeText}* output", "unsigned long long* error", "unsigned long long count"]
  | cons {first count : Nat} {typeText : String} {rest : List String}
      (ht : Format.type? format typeText = some .scalar)
      (tail : RowArguments.Denotes format t (first + 1) count rest) :
      RowArguments.Denotes format t first (count + 1)
        (s!"const {typeText}* input{first}, unsigned long long size{first}" :: rest)

/-- The inputs from number `first` on, then the output, error and count arguments, rendered with
type names that `Format.type?` decodes, have the grammar of `RowArguments.Denotes`. -/
theorem RowArguments.denotes_range (format : Format) {typeName : Ty → String}
    (hType : ∀ t, Format.type? format (typeName t) = some t) (t : Ty) (count first : Nat) :
    RowArguments.Denotes format t first count
      ((List.range' first count).map (fun i =>
        s!"const {typeName .scalar}* input{i}, unsigned long long size{i}") ++
        [s!"{typeName t}* output", "unsigned long long* error", "unsigned long long count"]) := by
  induction count generalizing first with
  | zero => simpa using RowArguments.Denotes.nil (first := first) (hType t)
  | succ count ih =>
      simpa [List.range'_succ] using RowArguments.Denotes.cons (hType .scalar) (ih (first + 1))

/-- The column arguments read variable `p` of `Γ` from a pointer `column{first + p}` of its type,
and the inputs and the fixed arguments of `RowArguments.Denotes` follow. -/
inductive Columns.Denotes (format : Format) (t : Ty) : List Ty → Nat → Nat → List String → Prop
  | nil {first inputs : Nat} {arguments : List String}
      (h : RowArguments.Denotes format t 0 inputs arguments) :
      Columns.Denotes format t [] first inputs arguments
  | cons {s : Ty} {Γ : List Ty} {first inputs : Nat} {typeText : String} {rest : List String}
      (ht : Format.type? format typeText = some s)
      (tail : Columns.Denotes format t Γ (first + 1) inputs rest) :
      Columns.Denotes format t (s :: Γ) first inputs
        (s!"const {typeText}* column{first}" :: rest)

/-- The preamble reads variable `p` of the row into the temporary `first + p` from
`column{first + p}` at the thread index. Each declared type must decode to the variable's type. -/
inductive Preamble.Denotes {α : Type} (format : Format) :
    (Γ : List Ty) → Nat → String → (Env α Γ → Locals α → Locals α) → Prop
  | nil {first : Nat} : Preamble.Denotes format [] first "" fun _ locals => locals
  | cons {t : Ty} {Γ : List Ty} {first : Nat} {typeText nameText rest : String}
      {run : Env α Γ → Locals α → Locals α}
      (ht : Format.type? format typeText = some t)
      (hn : Name.parse t nameText = some (.temporary first))
      (tail : Preamble.Denotes format Γ (first + 1) rest run) :
      Preamble.Denotes format (t :: Γ) first
        (s!"{typeText} {nameText} = column{first}[index];\n" ++ rest)
        fun env locals => run (env.pull Var.succ) (locals.set (.temporary first) (env.get .zero))

/-- One thread of a row kernel with results of type `t`, when the error record was already set
at its start (`recorded`) or not. A thread past the rows or after a recorded failure writes
nothing (`.ok none`), `none` means undefined arithmetic, and an error prevents the output write.
-/
def RowSource.eval {α : Type} {t : Ty} (body : Locals α → ExceptT Error Option (Locals α))
    (result : Name t) (recorded : Bool) (block blockSize thread : UInt32) (count : UInt64)
    (locals : Locals α) : Option (Except Error (Option (t.Value α))) :=
  let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
  if index ≥ count then some (.ok none)
  else if recorded then some (.ok none)
  else (body (locals.set .index index)).run.map (·.map fun after => some (after.get result))

/-- The complete text of a row kernel with decoded column arguments, preamble, body and result.
Column `p` holds variable `p` of the row `column index` at each in-range thread `index`. -/
inductive RowSource.Denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (inputs : Nat) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value) {Γ : List Ty}
    {t : Ty} (column : UInt64 → Env format.Value Γ) :
    String → (Bool → UInt32 → UInt32 → UInt32 → UInt64 → Locals format.Value →
      Option (Except Error (Option (t.Value format.Value)))) → Prop
  | kernel {arguments : List String} {preambleText body resultText : String}
      {bind : Env format.Value Γ → Locals format.Value → Locals format.Value}
      {run : Locals format.Value → ExceptT Error Option (Locals format.Value)}
      (result : Name t)
      (hArguments : Columns.Denotes format t Γ 0 inputs arguments)
      (hPreamble : Preamble.Denotes format Γ 0 preambleText bind)
      (hBody : Statement.Denotes format arithmetic inputs size load body run)
      (hResult : Name.parse t resultText = some result) :
      RowSource.Denotes format arithmetic inputs size load column
        (s!"extern \"C\" __global__ void torchlean_kernel({", ".intercalate arguments}) \{\n" ++
        "unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n" ++
        "if (index >= count) return;\n" ++ errorGuard ++ preambleText ++ body ++
        s!"output[index] = {resultText};\n}\n")
        (RowSource.eval (fun locals => run (bind (column (locals.get .index)) locals)) result)

/-- The column arguments, followed by the inputs and the fixed arguments with an output of type
`t`, rendered with type names that `Format.type?` decodes, denote the columns of `Γ`. -/
theorem Columns.denotes_arguments (format : Format) {typeName : Ty → String}
    (hType : ∀ t, Format.type? format (typeName t) = some t) (t : Ty) (inputs : Nat) :
    ∀ (Γ : List Ty) (first : Nat), Columns.Denotes format t Γ first inputs
      (columnArguments typeName Γ first ++
        ((List.range inputs).map (fun i =>
          s!"const {typeName .scalar}* input{i}, unsigned long long size{i}") ++
          [s!"{typeName t}* output", "unsigned long long* error", "unsigned long long count"]))
  | [], _ => .nil (by
      simpa [columnArguments, List.range_eq_range'] using
        RowArguments.denotes_range format hType t inputs 0)
  | s :: Γ, first =>
      .cons (hType s) (denotes_arguments format hType t inputs Γ (first + 1))

/-- The preamble, rendered with type names that `Format.type?` decodes, reads the columns into
their names. -/
theorem Preamble.denotes_preamble {α : Type} (format : Format) {typeName : Ty → String}
    (hType : ∀ t, Format.type? format (typeName t) = some t) :
    ∀ (Γ : List Ty) (first : Nat),
      Preamble.Denotes (α := α) format Γ first (preamble typeName Γ first)
        (Locals.columns Γ first)
  | [], _ => .nil
  | t :: Γ, first => .cons (hType t) (Name.parse_render _)
      (denotes_preamble format hType Γ (first + 1))

/-- A native row kernel that `emitRow` generates denotes the expression at the row of every
thread, read failures and both guards included.

The statement holds for any `arithmetic`, for the rows `column index` that the column buffers
hold, and for input buffers given by `load`, with `hRead` tying each read to a bounds-checked
load. Compilation, physical memory and device execution are outside it. -/
theorem denotes_emitRow (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value)
    (hRead : ∀ operand, operand < inputs → ∀ index,
      read operand index = if index ≥ size operand then
        .error (.bounds operand index (size operand).toNat) else .ok (load operand index))
    {Γ : List Ty} {t : Ty} (column : UInt64 → Env format.Value Γ) (expr : Expr format.Value Γ t)
    {generated : Source}
    (hSource : emitRow format (fun x => (Literal.ofValue format x).bits) inputs expr =
      .ok generated) :
    RowSource.Denotes format arithmetic inputs size load column generated.text
      (fun recorded block blockSize thread count _ =>
        let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
        if index ≥ count then some (.ok none)
        else if recorded then some (.ok none)
        else some ((expr.eval arithmetic read (column index)).map some)) := by
  have hNames : ∀ {t : Ty} (v : Var Γ t), ((Names.columns Γ 0).get v).Before Γ.length := by
    intro t v
    simpa using Names.columns_before (first := 0) v
  have hBody := @Statement.denotes_lower format arithmetic read inputs size load hRead
    Γ t (Target.lower expr) (Names.columns Γ 0) Γ.length hNames
  have hEval := fun (locals : Locals format.Value) (index : UInt64) =>
    eval_lower (Target.lower expr) (Names.columns Γ 0) Γ.length
      (Locals.columns Γ 0 (column index) (locals.set .index index)) (column index) hNames
      (fun v => Locals.get_columns 0 _ _ v) arithmetic read
  unfold emitRow at hSource
  generalize hAllocated : (lower (Names.columns Γ 0) (Target.lower expr)).run Γ.length =
    allocated at hSource hBody hEval
  rcases allocated with ⟨⟨statement, result⟩, next⟩
  dsimp only at hBody hEval hSource
  cases hRender : Statement.render format (fun x => (Literal.ofValue format x).bits)
      inputs statement with
  | error message => simp [hRender, bind, Except.bind] at hSource
  | ok body =>
      simp [hRender, bind, Except.bind] at hSource
      dsimp [pure, Except.instMonad, Except.pure] at hSource
      injection hSource with hSource
      subst generated
      have hType := Format.type_typeName format (fun x => (Literal.ofValue format x).bits)
      have hDenotes := RowSource.Denotes.kernel (column := column) result
        (Columns.denotes_arguments format hType t inputs Γ 0)
        (Preamble.denotes_preamble format hType Γ 0) (hBody hRender) (Name.parse_render result)
      have hRun : RowSource.eval (fun locals => statement.eval arithmetic read
          (Locals.columns Γ 0 (column (locals.get .index)) locals)) result =
          (fun recorded block blockSize thread count (_ : Locals format.Value) =>
            let index := block.toUInt64 * blockSize.toUInt64 + thread.toUInt64
            if index ≥ count then some (.ok none)
            else if recorded then some (.ok none)
            else some ((expr.eval arithmetic read (column index)).map some)) := by
        funext recorded block blockSize thread count locals
        dsimp only [RowSource.eval]
        split
        · rfl
        split
        · rfl
        · have h := hEval locals (block.toUInt64 * blockSize.toUInt64 + thread.toUInt64)
          rw [Target.eval_lower] at h
          rw [Locals.get_set]
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

end TorchLean.CUDA
