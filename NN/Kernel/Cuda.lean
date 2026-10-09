/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Target
public import Mathlib.Logic.Function.Basic
import FloatLib.Floats.Formats.IEEE754.Native.Representation
import Std.Data.String.ToNat

/-!
# CUDA emission for custom operations

Each output element is computed independently. Inputs are immutable and only the current thread's
output is written. Temporaries sequence arithmetic; branches and folds become structured CUDA
control flow. Native compilation must use `--fmad=false` and `--ftz=false`, without fast math.

Scoped target blocks are allocated into named statements before rendering. `Target.eval_lower`
certifies the scoped lowering; the named evaluator models mutable storage and control flow.
Checked literal words retain the supplied bits; native-value round trips reuse FloatLib. The
named allocation is certified in `Cuda.Correctness`. CUDA syntax interpretation, NVIDIA's
compilation and device execution require separate evidence; the lowering theorems alone do not
establish correctness of rendered text.

See NVIDIA's NVRTC manual and CUDA Math API for round-to-nearest arithmetic intrinsics:
https://docs.nvidia.com/cuda/nvrtc/index.html
https://docs.nvidia.com/cuda/cuda-math-api/index.html
-/

@[expose] public section

namespace NN.Kernel.Cuda

/-- Native Lean scalar formats; configured binary values use their complete-word emitter. -/
inductive Format where
  | binary32 | binary64
  deriving Repr, DecidableEq

/-- The native Lean scalar corresponding to a CUDA format. -/
abbrev Format.Value : Format → Type
  | .binary32 => Float32
  | .binary64 => Float

/-- Literal interchange words carry their width, rather than relying on a renderer-side cast. -/
inductive Literal : Format → Type where
  | binary32 (bits : UInt32) : Literal .binary32
  | binary64 (bits : UInt64) : Literal .binary64

/-- Retain the literal word in the common unsigned ABI without changing its bits. -/
def Literal.bits {format : Format} : Literal format → UInt64
  | .binary32 bits => bits.toUInt64
  | .binary64 bits => bits

/-- Interpret a word in Lean's native float model, which canonicalizes NaN payloads. -/
def Literal.value {format : Format} : Literal format → format.Value
  | .binary32 bits => Float32.ofBits bits
  | .binary64 bits => Float.ofBits bits

/-- Reject an oversized binary32 encoding before narrowing it to the format's word type. -/
def Literal.ofBits (format : Format) (bits : UInt64) : Except String (Literal format) :=
  match format with
  | .binary32 =>
      if bits.toNat < 2 ^ 32 then .ok (.binary32 bits.toUInt32)
      else .error "kernel: binary32 literal has more than 32 bits"
  | .binary64 => .ok (.binary64 bits)

/-- Obtain the exact interchange word of a native value. -/
def Literal.ofValue : (format : Format) → format.Value → Literal format
  | .binary32, value => .binary32 value.toBits
  | .binary64, value => .binary64 value.toBits

/-- Every accepted encoding retains all the supplied bits. -/
theorem Literal.bits_ofBits {format : Format} {bits : UInt64} {literal : Literal format}
    (h : Literal.ofBits format bits = .ok literal) : literal.bits = bits := by
  cases format with
  | binary32 =>
      by_cases hWidth : bits.toNat < 2 ^ 32
      · simp only [ofBits, hWidth, ↓reduceIte, Except.ok.injEq] at h
        cases h
        apply UInt64.toNat.inj
        simp only [Literal.bits, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32,
          Nat.mod_eq_of_lt hWidth]
      · simp only [ofBits, hWidth, ↓reduceIte] at h
        cases h
  | binary64 => cases h; rfl

/-- Reconstructing the native value from its encoded word is exact in Lean's float model. -/
theorem Literal.value_ofValue (format : Format) (value : format.Value) :
    (Literal.ofValue format value).value = value := by
  cases format with
  | binary32 => exact FloatLib.Floats.ExecFloat.Binary.toFloat32_ofFloat32 value
  | binary64 => exact FloatLib.Floats.ExecFloat.Binary.toFloat_ofFloat value

/-- Native-value encodings always pass the word-width check. -/
theorem Literal.ofBits_ofValue (format : Format) (value : format.Value) :
    Literal.ofBits format (Literal.ofValue format value).bits =
      .ok (Literal.ofValue format value) := by
  cases format with
  | binary32 =>
      simp only [ofBits, ofValue, bits, UInt32.toNat_toUInt64,
        UInt32.toNat_lt, ↓reduceIte, UInt32.toUInt32_toUInt64]
  | binary64 => rfl

/-- Render a checked word as a CUDA bit reinterpretation, never a decimal float approximation. -/
def Literal.render {format : Format} : Literal format → String
  | .binary32 bits => s!"__uint_as_float({bits.toNat}U)"
  | .binary64 bits => s!"__longlong_as_double((long long){bits.toNat}ULL)"

namespace Internal

/-- Scalar spellings shared by hardware and software arithmetic emitters. -/
structure Dialect (α : Type) where
  scalar : String
  literal : α → Except String String
  binary : ScalarOp → String

/-- Remove an exact prefix and suffix, rejecting mismatched emitted syntax. -/
def unframe (first last text : String) : Option String := do
  let chars := text.toList
  if chars.take first.toList.length != first.toList then none else do
    let rest := (chars.drop first.toList.length).reverse
    if rest.take last.toList.length != last.toList.reverse then none else
      pure (String.ofList (rest.drop last.toList.length).reverse)

theorem unframe_append (first last text : String) :
    unframe first last (first ++ text ++ last) = some text := by
  simp [unframe, String.toList_append, List.append_assoc, List.reverse_append]

/-- A successful parse consumes the entire framed text, not just a matching fragment. -/
theorem unframe_eq_some {first last text middle : String}
    (h : unframe first last text = some middle) : text = first ++ middle ++ last := by
  simp only [unframe] at h
  split at h
  · contradiction
  · next hp =>
    split at h
    · contradiction
    · next hs =>
      simp only [Option.pure_def, Option.some.injEq] at h
      have hp' : text.toList.take first.toList.length = first.toList := by simpa using hp
      have hs' : (text.toList.drop first.toList.length).reverse.take last.toList.length =
          last.toList.reverse := by simpa using hs
      have hdrop : text.toList.drop first.toList.length =
          ((text.toList.drop first.toList.length).reverse.drop last.toList.length).reverse ++
            last.toList := by
        apply List.reverse_injective
        simp only [List.reverse_append, List.reverse_reverse]
        rw [← hs']
        exact (List.take_append_drop last.toList.length
          (text.toList.drop first.toList.length).reverse).symm
      subst middle
      apply String.toList_injective
      simp only [String.toList_append, String.toList_ofList]
      calc
        text.toList = text.toList.take first.toList.length ++
            text.toList.drop first.toList.length := (List.take_append_drop _ _).symm
        _ = _ := by
          rw [hp']
          simpa only [List.append_assoc] using congrArg (first.toList ++ ·) hdrop

end Internal

/-- Decode the emitted bit-reinterpretation syntax, rejecting invalid or oversized words.

This recognizes literal text, not arbitrary CUDA expressions. The word is checked before any
narrowing conversion, so an out-of-range decimal integer cannot silently wrap. Only canonical
decimal spelling is accepted; leading zeros would give the integer an octal meaning in CUDA.
-/
def Literal.parse : (format : Format) → String → Option (Literal format)
  | .binary32, text => do
      let digits ← Internal.unframe "__uint_as_float(" "U)" text
      let value ← digits.toNat?
      if digits != Nat.repr value then none else
      if value < 2 ^ 32 then some (.binary32 value.toUInt32) else none
  | .binary64, text => do
      let digits ← Internal.unframe "__longlong_as_double((long long)" "ULL)" text
      let value ← digits.toNat?
      if digits != Nat.repr value then none else
      if value < 2 ^ 64 then some (.binary64 value.toUInt64) else none

/-- The renderer's actual text decodes to the original interchange word in either format. -/
theorem Literal.parse_render {format : Format} (literal : Literal format) :
    Literal.parse format literal.render = some literal := by
  cases literal with
  | binary32 bits =>
      have hWidth : bits.toNat < 4294967296 := UInt32.toNat_lt bits
      simp only [Literal.render, Literal.parse]
      change (Internal.unframe "__uint_as_float(" "U)"
        ("__uint_as_float(" ++ Nat.repr bits.toNat ++ "U)") >>= _) = _
      rw [Internal.unframe_append]
      simp [Nat.toNat?_repr, hWidth]
  | binary64 bits =>
      have hWidth : bits.toNat < 18446744073709551616 := UInt64.toNat_lt bits
      simp only [Literal.render, Literal.parse]
      change (Internal.unframe "__longlong_as_double((long long)" "ULL)"
        ("__longlong_as_double((long long)" ++ Nat.repr bits.toNat ++ "ULL)") >>= _) = _
      rw [Internal.unframe_append]
      simp [Nat.toNat?_repr, hWidth]

/-- Literal text preserves a native value in Lean's float model, including signed zeros.

As with `value_ofValue`, this concerns the model's NaN value, not native NaN payload preservation.
-/
theorem Literal.value_parse_render (format : Format) (value : format.Value) :
    (Literal.parse format (Literal.ofValue format value).render).map Literal.value =
      some value := by
  rw [Literal.parse_render]
  simp [Literal.value_ofValue]

/-- Unsigned CUDA expressions over already-evaluated operands.

Raw division and remainder have no defined result at a zero denominator. The lowering inserts a
lazy conditional instead of relying on Lean's total unsigned operations to model that case.
-/
inductive Unsigned (ν : Type) where
  | literal (value : UInt64)
  | var (name : ν)
  | binary (op : IndexOp) (x y : Unsigned ν)
  | ifZero (test yes no : Unsigned ν)

/-- Interpret unsigned expressions; `none` represents an undefined raw division or remainder. -/
def Unsigned.eval {ν : Type} (values : ν → UInt64) : Unsigned ν → Option UInt64
  | .literal value => some value
  | .var name => some (values name)
  | .binary op x y => do
      let x ← x.eval values
      let y ← y.eval values
      match op with
      | .div | .mod => if y == 0 then none else some (op.eval x y)
      | .add | .sub | .mul => some (op.eval x y)
  | .ifZero test yes no => do
      let test ← test.eval values
      if test == 0 then yes.eval values else no.eval values

/-- Lower an index operation on named values, guarding native division and remainder. -/
def Unsigned.lower {ν : Type} (op : IndexOp) (x y : ν) : Unsigned ν :=
  let x := Unsigned.var x
  let y := Unsigned.var y
  match op with
  | .div => .ifZero y (.literal 0) (.binary .div x y)
  | .mod => .ifZero y x (.binary .mod x y)
  | .add | .sub | .mul => .binary op x y

/-- Guarded native operations are defined and retain Lean's unsigned result for every operand. -/
theorem Unsigned.eval_lower {ν : Type} (op : IndexOp) (x y : ν) (values : ν → UInt64) :
    (Unsigned.lower op x y).eval values = some (op.eval (values x) (values y)) := by
  cases op <;> simp [lower, eval, IndexOp.eval]
  all_goals by_cases h : values y = 0 <;> simp [h]

/-- Render the same lazy guard represented by the unsigned expression semantics. -/
def Unsigned.render {ν : Type} {m : Type → Type} [Monad m]
    (names : ν → m String) : Unsigned ν → m String
  | .literal value => pure s!"{value.toNat}ULL"
  | .var name => names name
  | .binary op x y => do
      let symbol := match op with
        | .add => "+" | .sub => "-" | .mul => "*" | .div => "/" | .mod => "%"
      return s!"({← x.render names} {symbol} {← y.render names})"
  | .ifZero test yes no => do
      return s!"({← test.render names} == 0ULL ? {← yes.render names} : {← no.render names})"

/-- Atomic operands with a typed address space for local variables. -/
inductive Atom (α : Type) (ν : Ty → Type) : Ty → Type where
  | scalar (value : α) : Atom α ν .scalar
  | index (value : UInt64) : Atom α ν .index
  | predicate (value : Bool) : Atom α ν .predicate
  | var {t : Ty} (name : ν t) : Atom α ν t

/-- Read an atomic operand from a typed local-variable valuation. -/
def Atom.eval {α : Type} {ν : Ty → Type} {t : Ty}
    (values : {s : Ty} → ν s → s.Value α) : Atom α ν t → t.Value α
  | .scalar x => x
  | .index i => i
  | .predicate p => p
  | .var name => values name

/-- Replace a target variable address with its assigned typed name. -/
def Atom.lower {α : Type} {Γ : List Ty} {ν : Ty → Type} {t : Ty}
    (names : {s : Ty} → Var Γ s → ν s) : Target.Atom α Γ t → Atom α ν t
  | .scalar x => .scalar x
  | .index i => .index i
  | .predicate p => .predicate p
  | .var v => .var (names v)

/-- Assigning names preserves atomic values when the named locals represent the source scope. -/
theorem Atom.eval_lower {α : Type} {Γ : List Ty} {ν : Ty → Type} {t : Ty}
    (names : {s : Ty} → Var Γ s → ν s) (values : {s : Ty} → ν s → s.Value α)
    (env : Env α Γ) (h : ∀ {s : Ty} (v : Var Γ s), values (names v) = env.get v)
    (atom : Target.Atom α Γ t) : (Atom.lower names atom).eval values = atom.eval env := by
  cases atom <;> simp [lower, eval, Target.Atom.eval, h]

/-- Typed CUDA primitives, including the explicit guards required by unsigned arithmetic.

Scalar operations use the supplied arithmetic contract. A read remains checked by the reader;
its CUDA implementation needs a bounds-check statement, not just a right-hand-side expression.
-/
inductive Primitive (α : Type) (ν : Ty → Type) : Ty → Type where
  | copy {t : Ty} (atom : Atom α ν t) : Primitive α ν t
  | load (operand : Nat) (index : Atom α ν .index) : Primitive α ν .scalar
  | binary (op : ScalarOp) (x y : Atom α ν .scalar) : Primitive α ν .scalar
  | neg (x : Atom α ν .scalar) : Primitive α ν .scalar
  | unsigned (value : Unsigned (Atom α ν .index)) : Primitive α ν .index
  | compare (op : Compare) (x y : Atom α ν .scalar) : Primitive α ν .predicate
  | indexCompare (op : Compare) (x y : Atom α ν .index) : Primitive α ν .predicate

/-- Interpret a primitive; `none` means undefined unsigned arithmetic, not a failed input read. -/
def Primitive.eval {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    {ν : Ty → Type} {t : Ty} (values : {s : Ty} → ν s → s.Value α) :
    Primitive α ν t → Option (Except Error (t.Value α))
  | .copy atom => some (.ok (atom.eval values))
  | .load operand i => some (read operand (i.eval values))
  | .binary op x y => some (.ok (arithmetic.binary op (x.eval values) (y.eval values)))
  | .neg x => some (.ok (arithmetic.neg (x.eval values)))
  | .unsigned value => (value.eval (fun x => x.eval values)).map Except.ok
  | .compare op x y => some (.ok (arithmetic.compare op (x.eval values) (y.eval values)))
  | .indexCompare op x y => some (.ok (op.index (x.eval values) (y.eval values)))

/-- Lower a target primitive, inserting native unsigned guards before rendering. -/
def Primitive.lower {α : Type} {Γ : List Ty} {ν : Ty → Type} {t : Ty}
    (names : {s : Ty} → Var Γ s → ν s) : Target.Primitive α Γ t → Primitive α ν t
  | .load operand i => .load operand (Atom.lower names i)
  | .binary op x y => .binary op (Atom.lower names x) (Atom.lower names y)
  | .neg x => .neg (Atom.lower names x)
  | .indexBinary op x y => .unsigned (Unsigned.lower op (Atom.lower names x) (Atom.lower names y))
  | .compare op x y => .compare op (Atom.lower names x) (Atom.lower names y)
  | .indexCompare op x y => .indexCompare op (Atom.lower names x) (Atom.lower names y)

/-- Lowering introduces no undefined operation and preserves both values and input errors.

The arithmetic contract is unchanged: this does not identify an arbitrary contract with NVIDIA's
intrinsics, or prove that rendering and native compilation implement the interpreted primitive.
-/
theorem Primitive.eval_lower {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    {Γ : List Ty} {ν : Ty → Type} {t : Ty}
    (names : {s : Ty} → Var Γ s → ν s) (values : {s : Ty} → ν s → s.Value α)
    (env : Env α Γ) (h : ∀ {s : Ty} (v : Var Γ s), values (names v) = env.get v)
    (operation : Target.Primitive α Γ t) :
    (Primitive.lower names operation).eval arithmetic read values =
      some (operation.eval arithmetic read env) := by
  cases operation <;> simp [lower, eval, Target.Primitive.eval, Unsigned.eval_lower,
    Atom.eval_lower names values env h]

/-- The output index and generated temporaries form the only local-variable name space. -/
inductive Name : Ty → Type where
  | index : Name .index
  | temporary {t : Ty} (number : Nat) : Name t
  deriving DecidableEq

/-- Render a local name without admitting arbitrary identifier text. -/
def Name.render {t : Ty} : Name t → String
  | .index => "index"
  | .temporary number => s!"v{number}"

/-- Decode generated local names at their declared type, without accepting arbitrary identifiers.

The output index is available only at index type. Temporary numbers use canonical decimal spelling;
names with leading zeros, trailing source text or missing allocation numbers are rejected.
-/
def Name.parse (t : Ty) (text : String) : Option (Name t) :=
  match Internal.unframe "v" "" text with
  | some digits => do
      let number ← digits.toNat?
      if digits = Nat.repr number then some (.temporary number) else none
  | none =>
      if text = "index" then
        match t with
        | .index => some .index
        | .scalar | .predicate => none
      else none

/-- Decoding an emitted local name recovers the same typed address and allocation number. -/
theorem Name.parse_render {t : Ty} (name : Name t) :
    Name.parse t name.render = some name := by
  cases name with
  | index => simp [Name.parse, Name.render, Internal.unframe]
  | temporary number =>
      have hframe : Internal.unframe "v" "" ("v" ++ Nat.repr number) =
          some (Nat.repr number) := by
        simpa using Internal.unframe_append "v" "" (Nat.repr number)
      simp [Name.render, Name.parse, toString, hframe, Nat.toNat?_repr]

/-- Accepted identifier text is exactly the canonical spelling of the decoded name. -/
theorem Name.render_parse {t : Ty} {text : String} {name : Name t}
    (h : Name.parse t text = some name) : name.render = text := by
  simp only [Name.parse] at h
  split at h
  · next digits hframe =>
    cases hnumber : digits.toNat? with
    | none => simp [hnumber] at h
    | some number =>
      by_cases hdigits : digits = Nat.repr number
      · simp [hdigits, Nat.toNat?_repr] at h
        subst name
        simpa [Name.render, toString, hdigits] using (Internal.unframe_eq_some hframe).symm
      · simp [hnumber, hdigits] at h
  · by_cases htext : text = "index"
    · cases t with
      | index =>
        simp only [htext, ↓reduceIte, Option.some.injEq] at h
        cases h
        exact htext.symm
      | scalar => simp [htext] at h
      | predicate => simp [htext] at h
    · simp [htext] at h

/-- Distinct allocation numbers have distinct CUDA spellings, even across scalar types. -/
theorem Name.temporary_render_inj {s t : Ty} {m n : Nat} :
    (Name.temporary (t := s) m).render = (Name.temporary (t := t) n).render ↔ m = n := by
  change "v" ++ Nat.repr m = "v" ++ Nat.repr n ↔ m = n
  rw [String.append_right_inj, Nat.repr_inj]

/-- Generated temporaries cannot shadow the output index. -/
theorem Name.index_ne_temporary {t : Ty} (n : Nat) :
    Name.index.render ≠ (Name.temporary (t := t) n).render := by
  intro h
  have hFirst := congrArg (fun text : String => text.toList.head?) h
  simp [render, toString, String.toList_append] at hFirst

/-- A local is the output index or was allocated before the next temporary number. -/
def Name.Before {t : Ty} (next : Nat) : Name t → Prop
  | .index => True
  | .temporary number => number < next

/-- A typed valuation of named locals; values outside the current scope are unconstrained. -/
abbrev Locals (α : Type) := (address : (t : Ty) × Name t) → address.1.Value α

/-- Read a named local at its declared type. -/
def Locals.get {α : Type} (locals : Locals α) {t : Ty} (name : Name t) : t.Value α :=
  locals ⟨t, name⟩

/-- Replace one local value using mathlib's dependent function update. -/
def Locals.set {α : Type} (locals : Locals α) {t : Ty} (name : Name t)
    (value : t.Value α) : Locals α := Function.update locals ⟨t, name⟩ value

/-- A write returns the assigned value when read at the same typed address. -/
theorem Locals.get_set {α : Type} (locals : Locals α) {t : Ty} (name : Name t)
    (value : t.Value α) : (locals.set name value).get name = value :=
  Function.update_self _ _ _

/-- A write cannot change a local with a different CUDA spelling, regardless of its type. -/
theorem Locals.get_set_of_ne {α : Type} (locals : Locals α) {s t : Ty}
    (name : Name t) (value : t.Value α) (other : Name s)
    (h : other.render ≠ name.render) : (locals.set name value).get other = locals.get other := by
  apply Function.update_of_ne
  intro hAddress
  exact h (congrArg (fun address : (t : Ty) × Name t => address.2.render) hAddress)

/-- Allocating the next temporary leaves every earlier local unchanged. -/
theorem Locals.get_set_before {α : Type} (locals : Locals α) {s t : Ty}
    (next : Nat) (value : t.Value α) (other : Name s) (h : other.Before next) :
    (locals.set (.temporary next) value).get other = locals.get other := by
  apply get_set_of_ne
  cases other with
  | index => exact Name.index_ne_temporary next
  | temporary number =>
      intro hName
      exact (Nat.ne_of_lt h) (Name.temporary_render_inj.mp hName)

/-- One named assignment, including the checked read of a primitive's inputs. -/
structure Assignment (α : Type) (t : Ty) where
  name : Name t
  operation : Primitive α Name t

/-- Execute the assignment in its local valuation. Failed reads never supply a stored value. -/
def Assignment.eval {α : Type} {t : Ty} (assignment : Assignment α t)
    (arithmetic : Arithmetic α) (read : Reader α) (locals : Locals α) :
    Option (Except Error (Locals α)) :=
  (assignment.operation.eval arithmetic read locals.get).map fun result =>
    result.map fun value => locals.set assignment.name value

/-- A lowered primitive writes exactly its source result and retains the source read errors. -/
theorem Assignment.eval_lower {α : Type} {Γ : List Ty} {t : Ty}
    (names : {s : Ty} → Var Γ s → Name s) (locals : Locals α) (env : Env α Γ)
    (h : ∀ {s : Ty} (v : Var Γ s), locals.get (names v) = env.get v)
    (name : Name t) (operation : Target.Primitive α Γ t)
    (arithmetic : Arithmetic α) (read : Reader α) :
    (Assignment.mk name (Primitive.lower names operation)).eval arithmetic read locals =
      some ((operation.eval arithmetic read env).map fun value => locals.set name value) := by
  simp [eval, Primitive.eval_lower arithmetic read names locals.get env h]

/-- Assigning a fresh local preserves every previously represented source variable. -/
theorem Assignment.eval_preserves_scope {α : Type} {Γ : List Ty} {t : Ty}
    (assignment : Assignment α t) (names : {s : Ty} → Var Γ s → Name s)
    (locals after : Locals α) (env : Env α Γ)
    (hScope : ∀ {s : Ty} (v : Var Γ s), locals.get (names v) = env.get v)
    (hFresh : ∀ {s : Ty} (v : Var Γ s), (names v).render ≠ assignment.name.render)
    (arithmetic : Arithmetic α) (read : Reader α)
    (hResult : assignment.eval arithmetic read locals = some (.ok after)) :
    ∀ {s : Ty} (v : Var Γ s), after.get (names v) = env.get v := by
  cases hEval : assignment.operation.eval arithmetic read locals.get with
  | none => simp [eval, hEval] at hResult
  | some result =>
      cases result with
      | error error => simp [eval, hEval, Except.map] at hResult
      | ok value =>
          simp only [eval, hEval, Option.map_some, Except.map, Option.some.injEq,
            Except.ok.injEq] at hResult
          subst after
          intro s v
          rw [Locals.get_set_of_ne _ _ _ _ (hFresh v)]
          exact hScope v

/-- Named statements used by both the CUDA renderer and the mutable-local evaluator.

Branch results are copied into their destination only after the selected arm finishes. Loop bodies
read the current counter and accumulator, then copy their result back into the accumulator.
-/
inductive Statement (α : Type) where
  | skip
  | assign {t : Ty} (assignment : Assignment α t)
  | seq (first second : Statement α)
  | branch {t : Ty} (name : Name t) (condition : Atom α Name .predicate)
      (yes : Statement α) (yesResult : Name t) (no : Statement α) (noResult : Name t)
  | loop (acc : Name .scalar) (counter : Name .index) (count : Atom α Name .index)
      (initial : Atom α Name .scalar) (body : Statement α) (next : Name .scalar)

namespace Statement

namespace Internal

/-- Execute at most the remaining bounded steps, stopping on the unsigned guard or an error. -/
def steps {α : Type} (body : Locals α → ExceptT Error Option (Locals α))
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar) (count : UInt64) :
    Nat → UInt64 → Locals α → ExceptT Error Option (Locals α)
  | 0, _, locals => pure locals
  | n + 1, i, locals => do
      if i < count then
        let after ← body (locals.set counter i)
        steps body acc counter next count n (i + 1) (after.set acc (after.get next))
      else pure locals

/-- A defined body gives exactly the remaining source iterations, including their errors. -/
theorem steps_eq_iterate {α : Type}
    (body : Locals α → ExceptT Error Option (Locals α))
    (step : Locals α → Except Error (Locals α))
    (hBody : ∀ locals, (body locals).run = some (step locals))
    (acc : Name .scalar) (counter : Name .index) (next : Name .scalar) (count : UInt64)
    (n : Nat) (i : UInt64) (locals : Locals α)
    (hBound : n + i.toNat ≤ count.toNat) :
    (steps body acc counter next count n i locals).run =
      some (iterate (fun i locals =>
        (step (locals.set counter i)).map fun after => after.set acc (after.get next))
        n i locals) := by
  induction n generalizing i locals with
  | zero => rfl
  | succ n ih =>
      have h : i < count := by
        simp only [UInt64.lt_iff_toNat_lt]
        omega
      have hNext : n + (i + 1).toNat ≤ count.toNat := by
        rw [Target.Internal.increment_toNat h]
        omega
      simp only [steps, h, ↓reduceIte, iterate]
      change (body (locals.set counter i) >>= _).run = _
      rw [show body (locals.set counter i) = ExceptT.mk (some (step (locals.set counter i)))
        from hBody _]
      cases step (locals.set counter i) with
      | error error => rfl
      | ok after =>
          simp only [Except.map]
          change (steps body acc counter next count n (i + 1)
            (after.set acc (after.get next))).run = _
          exact ih (i + 1) (after.set acc (after.get next)) hNext

end Internal

/-- Evaluate the same named assignments and control flow that the renderer consumes.

`none` denotes undefined raw unsigned arithmetic; checked input failures remain `Except.error`.
The loop's fuel is its unsigned bound, so it cannot truncate an increasing counter from zero.
-/
def eval {α : Type} (arithmetic : Arithmetic α) (read : Reader α) :
    Statement α → Locals α → ExceptT Error Option (Locals α)
  | .skip, locals => pure locals
  | .assign assignment, locals => ExceptT.mk (assignment.eval arithmetic read locals)
  | .seq first second, locals => do
      let after ← first.eval arithmetic read locals
      second.eval arithmetic read after
  | .branch name condition yes y no n, locals => do
      if condition.eval locals.get then
        let after ← yes.eval arithmetic read locals
        pure (after.set name (after.get y))
      else
        let after ← no.eval arithmetic read locals
        pure (after.set name (after.get n))
  | .loop acc counter count initial body next, locals =>
      let count := count.eval locals.get
      Internal.steps (body.eval arithmetic read) acc counter next count count.toNat 0
        (locals.set acc (initial.eval locals.get))

/-- A named loop agrees with the unsigned guarded loop when its body has defined semantics.

The step may fail with a checked error. The hypothesis excludes only undefined raw arithmetic,
not failed input reads; each successful step copies its result into the mutable accumulator.
-/
theorem eval_loop_eq {α : Type} (arithmetic : Arithmetic α) (read : Reader α)
    (acc : Name .scalar) (counter : Name .index) (count : Atom α Name .index)
    (initial : Atom α Name .scalar) (body : Statement α) (next : Name .scalar)
    (step : Locals α → Except Error (Locals α))
    (hBody : ∀ locals, (body.eval arithmetic read locals).run = some (step locals))
    (locals : Locals α) :
    ((Statement.loop acc counter count initial body next).eval arithmetic read locals).run =
      some (Target.loop (fun i locals =>
        (step (locals.set counter i)).map fun after => after.set acc (after.get next))
        (count.eval locals.get) 0 (locals.set acc (initial.eval locals.get))) := by
  rw [Target.loop_eq_iterate]
  simpa [eval] using Internal.steps_eq_iterate (body.eval arithmetic read) step hBody
    acc counter next (count.eval locals.get) (count.eval locals.get).toNat 0
    (locals.set acc (initial.eval locals.get)) (by simp)

end Statement

/-- Generated CUDA source and its checked input signature. Compilation fixes the FP policy. -/
structure Source where
  format : Format
  inputs : Nat
  text : String
  deriving Repr

namespace Internal

def Format.name : Format → String
  | .binary32 => "float"
  | .binary64 => "double"

def typeName (format : Format) : Ty → String
  | .scalar => Format.name format
  | .index => "unsigned long long"
  | .predicate => "bool"

def Format.binary (format : Format) (op : ScalarOp) : String :=
  let start := match format with | .binary32 => "__f" | .binary64 => "__d"
  let name := match op with | .add => "add" | .sub => "sub" | .mul => "mul" | .div => "div"
  start ++ name ++ "_rn"

def comparison : Compare → String
  | .eq => "=="
  | .lt => "<"
  | .le => "<="

structure Names (Γ : List Ty) where
  get : {t : Ty} → Var Γ t → Name t

def Names.push {Γ : List Ty} {t : Ty} (name : Name t)
    (names : Names Γ) : Names (t :: Γ) :=
  ⟨fun v => match v with | .zero => name | .succ v => names.get v⟩

/-- Bind the sole free output coordinate to the kernel entry point's index local. -/
def Names.output : Names [.index] := ⟨fun v => match v with
  | .zero => .index
  | .succ v => nomatch v⟩

abbrev Allocate := StateM Nat

def fresh (t : Ty) : Allocate (Name t) := do
  let n ← get
  set (n + 1)
  return .temporary n

@[reducible] def atomWith {α : Type} (dialect : Dialect α)
    {t : Ty} : Atom α Name t → Except String String
  | .scalar value => dialect.literal value
  | .index value => pure s!"{value.toNat}ULL"
  | .predicate value => pure (if value then "true" else "false")
  | .var name => pure name.render

abbrev native {α : Type} (format : Format) (bits : α → UInt64) : Dialect α :=
  ⟨Format.name format, fun value => (Literal.ofBits format (bits value)).map Literal.render,
    Format.binary format⟩

abbrev atom {α : Type} (format : Format) (bits : α → UInt64)
    {t : Ty} (value : Atom α Name t) : Except String String :=
  atomWith (native format bits) value

/-- Emit a primitive expression. Loads require a separate bounds-check statement. -/
@[reducible] def expressionWith {α : Type} (dialect : Dialect α)
    {t : Ty} : Primitive α Name t → Except String String
  | .copy value => atomWith dialect value
  | .load _ _ => .error "kernel: an input read requires a checked statement"
  | .binary op x y => do
      let x ← atomWith dialect x
      let y ← atomWith dialect y
      return s!"{dialect.binary op}({x}, {y})"
  | .neg x => do
      let x ← atomWith dialect x
      return s!"(-{x})"
  | .unsigned value =>
      value.render (atomWith dialect)
  | .compare op x y => do
      let x ← atomWith dialect x
      let y ← atomWith dialect y
      return s!"({x} {comparison op} {y})"
  | .indexCompare op x y => do
      let x ← atomWith dialect x
      let y ← atomWith dialect y
      return s!"({x} {comparison op} {y})"

abbrev expression {α : Type} (format : Format) (bits : α → UInt64)
    {t : Ty} (value : Primitive α Name t) : Except String String :=
  expressionWith (native format bits) value

@[reducible] def Dialect.typeName {α : Type} (dialect : Dialect α) : Ty → String
  | .scalar => dialect.scalar
  | .index => "unsigned long long"
  | .predicate => "bool"

theorem typeName_native {α : Type} (format : Format) (bits : α → UInt64) (t : Ty) :
    (native format bits).typeName t = typeName format t := by cases t <;> rfl

@[reducible] def assignmentWith {α : Type} (dialect : Dialect α) (inputs : Nat)
    {t : Ty} (statement : Assignment α t) : Except String String := do
  let declaration (rhs : String) : String :=
    s!"{dialect.typeName t} {statement.name.render} = {rhs};\n"
  match t, statement.operation with
  | .scalar, .load operand index => do
      if operand ≥ inputs then throw s!"kernel: input {operand} is not declared"
      let i ← atomWith dialect index
      let check := s!"if ({i} >= size{operand}) \{\n" ++
        s!"if (atomicCAS(error, 0ULL, 1ULL) == 0ULL) \{ " ++
        s!"error[1] = {operand}ULL; error[2] = {i}; error[3] = size{operand}; }\n" ++
        "return;\n}\n"
      return check ++ declaration s!"input{operand}[{i}]"
  | _, operation => return declaration (← expressionWith dialect operation)

abbrev assignment {α : Type} (format : Format) (bits : α → UInt64) (inputs : Nat)
    {t : Ty} (value : Assignment α t) : Except String String :=
  assignmentWith (native format bits) inputs value

def lower {α : Type} {Γ : List Ty} {t : Ty} (names : Names Γ) :
    Target.Block α Γ t → Allocate (Statement α × Name t)
  | .result value => do
      let value := Atom.lower names.get value
      match value with
      | .var name => pure (.skip, name)
      | _ =>
          let name ← fresh t
          return (.assign ⟨name, .copy value⟩, name)
  | .compute operation => do
      let name ← fresh t
      return (.assign ⟨name, Primitive.lower names.get operation⟩, name)
  | .branch p yes no => do
      let name ← fresh t
      let (yc, y) ← lower names yes
      let (nc, n) ← lower names no
      return (.branch name (Atom.lower names.get p) yc y nc n, name)
  | .bind value body => do
      let (vc, v) ← lower names value
      let (bc, b) ← lower (names.push v) body
      return (.seq vc bc, b)
  | .loop count initial body => do
      let acc ← fresh .scalar
      let i ← fresh .index
      let (bc, next) ← lower
        (Names.push (t := .scalar) acc (Names.push (t := .index) i names)) body
      return (.loop acc i (Atom.lower names.get count) (Atom.lower names.get initial) bc next, acc)

@[reducible] def renderWith {α : Type} (dialect : Dialect α) (inputs : Nat) :
    Statement α → Except String String
  | .skip => pure ""
  | .assign value => assignmentWith dialect inputs value
  | .seq first second => do
      return (← renderWith dialect inputs first) ++ (← renderWith dialect inputs second)
  | .branch (t := t) name condition yes y no n => do
      let p ← atomWith dialect condition
      let yc ← renderWith dialect inputs yes
      let nc ← renderWith dialect inputs no
      return s!"{dialect.typeName t} {name.render};\nif ({p}) \{\n" ++ yc ++
        s!"{name.render} = {y.render};\n} else \{\n" ++ nc ++
        s!"{name.render} = {n.render};\n}\n"
  | .loop acc i count initial body next => do
      let count ← atomWith dialect count
      let initial ← atomWith dialect initial
      let bc ← renderWith dialect inputs body
      return s!"{dialect.scalar} {acc.render} = {initial};\n" ++
        s!"for (unsigned long long {i.render} = 0ULL; {i.render} < {count}; " ++
        s!"++{i.render}) \{\n" ++ bc ++ s!"{acc.render} = {next.render};\n}\n"

abbrev render {α : Type} (format : Format) (bits : α → UInt64) (inputs : Nat)
    (statement : Statement α) : Except String String :=
  renderWith (native format bits) inputs statement

/-- Shared signature, output guard and write for native and configured scalar arithmetic. -/
def entrypoint (scalar : String) (inputs : Nat) (body : String) (result : Name .scalar) :
    String :=
  let arguments := (List.range inputs).map fun i =>
    s!"const {scalar}* input{i}, unsigned long long size{i}"
  let arguments := String.intercalate ", "
    (arguments ++ [s!"{scalar}* output", "unsigned long long* error", "unsigned long long count"])
  s!"extern \"C\" __global__ void torchlean_kernel({arguments}) \{\n" ++
    "unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n" ++
    "if (index >= count) return;\n" ++ body ++ s!"output[index] = {result.render};\n}\n"

end Internal

open Internal

/-- Emit one independent output calculation with exact bit-pattern scalar literals.

The expression's sole free variable is its row-major output index. Each input has its own pointer
and length argument. A four-word error record reports an observed invalid read; callers must check
it after execution before exposing the output. No caller-provided text enters the source.
-/
def source {α : Type} (format : Format) (bits : α → UInt64) (inputs : Nat)
    (expr : Expr α [.index] .scalar) : Except String Source := do
  let ((statement, result), _) := (lower Names.output (Target.lower expr)).run 0
  let body ← render format bits inputs statement
  let text := entrypoint (Internal.Format.name format) inputs body result
  return { format, inputs, text }

end NN.Kernel.Cuda
