/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda
import Std.Data.String.ToNat

/-!
# CUDA operand decoding and checked assignment semantics

Atomic operands decode at their declared type. Scalar constants reuse checked bit-pattern literals,
unsigned constants retain their width and decimal spelling, and variable names retain their typed
addresses. The expression grammar also assigns semantics to emitted scalar intrinsics, comparisons,
unsigned arithmetic and lazy guards. Declaration and checked-load text retain their local writes
and bounds errors under an explicit buffer contract. `Cuda.Statements` composes these results for
sequences, branches and unsigned loops; `Cuda.Source` covers the emitted entrypoint. This is not a
general CUDA parser. Atomic error recording, physical memory and NVIDIA compilation
remain external boundaries.
-/

@[expose] public section

namespace NN.Kernel.Cuda

/-- Decode an atomic operand in the emitter's grammar, rejecting malformed or oversized constants.

Scalar values use the selected native float model, including its canonicalized NaN value. Index
constants require the `ULL` suffix and canonical decimal spelling; casts never wrap an oversized
input. Variables are decoded at their declared type using `Name.parse`.
-/
def Atom.parse (format : Format) : (t : Ty) → String → Option (Atom format.Value Name t)
  | t, text =>
    match Name.parse t text with
    | some name => some (.var name)
    | none =>
      match t with
      | .scalar => (Literal.parse format text).map fun literal => .scalar literal.value
      | .index => do
          let digits ← Internal.unframe "" "ULL" text
          let number ← digits.toNat?
          if digits = Nat.repr number && number < 2 ^ 64 then
            some (.index number.toUInt64)
          else none
      | .predicate =>
          if text = "true" then some (.predicate true)
          else if text = "false" then some (.predicate false) else none

private theorem name_parse_unsigned (number : UInt64) :
    Name.parse .index (Nat.repr number.toNat ++ "ULL") = none := by
  cases hparse : Name.parse .index (Nat.repr number.toNat ++ "ULL") with
  | none => rfl
  | some name =>
    have htext := Name.render_parse hparse
    have hdigit : ∀ c ∈ (Nat.repr number.toNat).toList, c.isDigit = true := by
      intro c hc
      exact Nat.isDigit_of_mem_toDigits (b := 10) (n := number.toNat) (by decide) (by decide)
        (by simpa only [Nat.toList_repr] using hc)
    cases name with
    | index =>
      have hmem : 'i' ∈ (Nat.repr number.toNat).toList := by
        have : 'i' ∈ (Nat.repr number.toNat ++ "ULL").toList := by
          rw [← htext]
          simp [Name.render]
        simpa using this
      have := hdigit 'i' hmem
      contradiction
    | temporary n =>
      have hmem : 'v' ∈ (Nat.repr number.toNat).toList := by
        have : 'v' ∈ (Nat.repr number.toNat ++ "ULL").toList := by
          rw [← htext]
          simp [Name.render, toString, String.toList_append]
        simpa using this
      have := hdigit 'v' hmem
      contradiction

/-- Every emitted native operand decodes to the same value or typed variable address.

The scalar encoding is the native interchange word. This preserves the native Lean float model,
not arbitrary hardware NaN payloads, and does not assume correctness of NVIDIA's intrinsics.
-/
theorem Atom.parse_render (format : Format) {t : Ty} (value : Atom format.Value Name t) :
    (Internal.atom format (fun x => (Literal.ofValue format x).bits) value).toOption.bind
      (Atom.parse format t) = some value := by
  cases value with
  | var name =>
      change Atom.parse format t name.render = some (.var name)
      simp [Atom.parse, Name.parse_render]
  | index number =>
      have hwidth : number.toNat < 18446744073709551616 := UInt64.toNat_lt number
      have hframe : Internal.unframe "" "ULL" (Nat.repr number.toNat ++ "ULL") =
          some (Nat.repr number.toNat) := by
        simpa using Internal.unframe_append "" "ULL" (Nat.repr number.toNat)
      change Atom.parse format .index (Nat.repr number.toNat ++ "ULL") = some (.index number)
      simp [Atom.parse, name_parse_unsigned, hframe, Nat.toNat?_repr, hwidth]
  | predicate value =>
      change Atom.parse format .predicate (if value then "true" else "false") =
        some (.predicate value)
      cases value <;> simp [Atom.parse, Name.parse, Internal.unframe]
  | scalar value =>
      simp only [Internal.atom]
      rw [Literal.ofBits_ofValue]
      change Atom.parse format .scalar (Literal.ofValue format value).render = some (.scalar value)
      have hname : Name.parse .scalar (Literal.ofValue format value).render = none := by
        cases hparse : Name.parse .scalar (Literal.ofValue format value).render with
        | none => rfl
        | some name =>
          have htext := Name.render_parse hparse
          cases name with
          | temporary number =>
            have hfirst := congrArg (fun text : String => text.toList.head?) htext
            cases format <;>
              simp [Name.render, Literal.ofValue, Literal.render, toString,
                String.toList_append] at hfirst
      simp [Atom.parse, hname, Literal.parse_render, Literal.value_ofValue]

namespace Unsigned

/-- Semantics of the emitted unsigned-expression grammar.

The leaves retain canonical names and unsigned literal spelling. Arithmetic is unsigned; raw
division and remainder by zero are undefined, and the conditional selects only one result.
This relation describes the CUDA expression fragment; it makes no assumption about NVIDIA's
implementation.
-/
inductive Denotes (values : Name .index → UInt64) : String → Option UInt64 → Prop where
  | literal (value : UInt64) : Denotes values s!"{value.toNat}ULL" (some value)
  | var (name : Name .index) : Denotes values name.render (some (values name))
  | add {x y : String} {a b : Option UInt64}
      (hx : Denotes values x a) (hy : Denotes values y b) :
      Denotes values s!"({x} + {y})" (a.bind fun a => b.map (a + ·))
  | sub {x y : String} {a b : Option UInt64}
      (hx : Denotes values x a) (hy : Denotes values y b) :
      Denotes values s!"({x} - {y})" (a.bind fun a => b.map (a - ·))
  | mul {x y : String} {a b : Option UInt64}
      (hx : Denotes values x a) (hy : Denotes values y b) :
      Denotes values s!"({x} * {y})" (a.bind fun a => b.map (a * ·))
  | div {x y : String} {a b : Option UInt64}
      (hx : Denotes values x a) (hy : Denotes values y b) :
      Denotes values s!"({x} / {y})"
        (a.bind fun a => b.bind fun b => if b == 0 then none else some (a / b))
  | mod {x y : String} {a b : Option UInt64}
      (hx : Denotes values x a) (hy : Denotes values y b) :
      Denotes values s!"({x} % {y})"
        (a.bind fun a => b.bind fun b => if b == 0 then none else some (a % b))
  | ifZero {test yes no : String} {t a b : Option UInt64}
      (ht : Denotes values test t)
      (ha : Denotes values yes a) (hb : Denotes values no b) :
      Denotes values s!"({test} == 0ULL ? {yes} : {no})"
        (t.bind fun t => if t == 0 then a else b)

/-- The renderer's actual unsigned text has the result of the typed expression, including
undefined raw division and lazy zero-denominator guards. -/
theorem denotes_render (format : Format)
    (values : {t : Ty} → Name t → t.Value format.Value)
    (expression : Unsigned (Atom format.Value Name .index)) {text : String}
    (h : expression.render (Internal.atom format (fun x => (Literal.ofValue format x).bits)) =
      Except.ok text) :
    Denotes (fun name => values name) text
      (expression.eval (fun value => value.eval values)) := by
  induction expression generalizing text with
  | literal value =>
      change Except.ok s!"{value.toNat}ULL" = Except.ok text at h
      cases h
      exact Denotes.literal value
  | var value =>
      change Internal.atom format (fun x => (Literal.ofValue format x).bits) value = .ok text at h
      cases value with
      | index value =>
          change Except.ok s!"{value.toNat}ULL" = Except.ok text at h
          cases h
          exact Denotes.literal value
      | var name =>
          change Except.ok name.render = Except.ok text at h
          cases h
          exact Denotes.var name
  | binary op x y ihx ihy =>
      cases hx : x.render (Internal.atom format (fun x => (Literal.ofValue format x).bits)) with
      | error error =>
          simp only [Unsigned.render, hx] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok xt =>
        cases hy : y.render (Internal.atom format (fun x => (Literal.ofValue format x).bits)) with
        | error error =>
            simp only [Unsigned.render, hx, hy] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok yt =>
          have hdx := ihx hx
          have hdy := ihy hy
          simp only [Unsigned.render, hx, hy] at h
          dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
            Except.pure] at h
          injection h with htext
          subst text
          cases op
          · simpa [Unsigned.eval, IndexOp.eval, Option.map_eq_bind, Function.comp_def,
              toString, String.append_assoc] using Denotes.add hdx hdy
          · simpa [Unsigned.eval, IndexOp.eval, Option.map_eq_bind, Function.comp_def,
              toString, String.append_assoc] using Denotes.sub hdx hdy
          · simpa [Unsigned.eval, IndexOp.eval, Option.map_eq_bind, Function.comp_def,
              toString, String.append_assoc] using Denotes.mul hdx hdy
          · simpa [Unsigned.eval, IndexOp.eval, toString, String.append_assoc]
              using Denotes.div hdx hdy
          · simpa [Unsigned.eval, IndexOp.eval, toString, String.append_assoc]
              using Denotes.mod hdx hdy
  | ifZero test yes no iht ihy ihn =>
      cases ht : test.render (Internal.atom format (fun x => (Literal.ofValue format x).bits)) with
      | error error =>
          simp only [Unsigned.render, ht] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok tt =>
        cases hy : yes.render (Internal.atom format (fun x => (Literal.ofValue format x).bits)) with
        | error error =>
            simp only [Unsigned.render, ht, hy] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok yt =>
          cases hn : no.render
              (Internal.atom format (fun x => (Literal.ofValue format x).bits)) with
          | error error =>
              simp only [Unsigned.render, ht, hy, hn] at h
              change Except.error error = Except.ok text at h
              contradiction
          | ok nt =>
            simp only [Unsigned.render, ht, hy, hn] at h
            dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
              Except.pure] at h
            injection h with htext
            subst text
            exact Denotes.ifZero (iht ht) (ihy hy) (ihn hn)

/-- Rendering a lowered unsigned operation retains Lean's total result, including division and
remainder at zero. The raw undefined operations occur only in the unselected conditional arm. -/
theorem denotes_lower (format : Format)
    (values : {t : Ty} → Name t → t.Value format.Value) (op : IndexOp)
    (x y : Atom format.Value Name .index) {text : String}
    (h : (Unsigned.lower op x y).render
      (Internal.atom format (fun x => (Literal.ofValue format x).bits)) = Except.ok text) :
    Denotes (fun name => values name) text (some (op.eval (x.eval values) (y.eval values))) := by
  simpa only [Unsigned.eval_lower] using denotes_render format values (Unsigned.lower op x y) h

end Unsigned

namespace Internal

/-- Decode the round-to-nearest intrinsic names used by the native scalar formats. -/
def scalarOp? : Format → String → Option ScalarOp
  | .binary32, "__fadd_rn" => some .add
  | .binary32, "__fsub_rn" => some .sub
  | .binary32, "__fmul_rn" => some .mul
  | .binary32, "__fdiv_rn" => some .div
  | .binary64, "__dadd_rn" => some .add
  | .binary64, "__dsub_rn" => some .sub
  | .binary64, "__dmul_rn" => some .mul
  | .binary64, "__ddiv_rn" => some .div
  | _, _ => none

/-- Decode comparison tokens without accepting assignments or reversed inequalities. -/
def compare? : String → Option Compare
  | "==" => some .eq
  | "<" => some .lt
  | "<=" => some .le
  | _ => none

theorem scalarOp_binary (format : Format) (op : ScalarOp) :
    scalarOp? format (Format.binary format op) = some op := by
  cases format <;> cases op <;> rfl

theorem compare_comparison (op : Compare) : compare? (comparison op) = some op := by
  cases op <;> rfl

end Internal

/-- Semantics of the emitted primitive-expression fragment.

Operands are decoded at their declared types. Intrinsic names and comparison symbols are decoded
independently of the emitter. The arithmetic argument specifies what each scalar intrinsic must
implement; the relation does not certify NVIDIA's implementation. Checked loads belong to the
statement grammar because an unchecked pointer expression cannot preserve read errors.
-/
inductive Primitive.Denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (locals : Locals format.Value) :
    (t : Ty) → String → Option (Except Error (t.Value format.Value)) → Prop where
  | copy {t : Ty} {text : String} (value : Atom format.Value Name t)
      (h : Atom.parse format t text = some value) :
      Primitive.Denotes format arithmetic locals t text (some (.ok (value.eval locals.get)))
  | binary {name xtext ytext : String} (op : ScalarOp)
      (x y : Atom format.Value Name .scalar)
      (hn : Internal.scalarOp? format name = some op)
      (hx : Atom.parse format .scalar xtext = some x)
      (hy : Atom.parse format .scalar ytext = some y) :
      Primitive.Denotes format arithmetic locals .scalar s!"{name}({xtext}, {ytext})"
        (some (.ok (arithmetic.binary op (x.eval locals.get) (y.eval locals.get))))
  | neg {text : String} (x : Atom format.Value Name .scalar)
      (hx : Atom.parse format .scalar text = some x) :
      Primitive.Denotes format arithmetic locals .scalar s!"(-{text})"
        (some (.ok (arithmetic.neg (x.eval locals.get))))
  | unsigned {text : String} {result : Option UInt64}
      (h : Unsigned.Denotes (fun name => locals.get name) text result) :
      Primitive.Denotes format arithmetic locals .index text (result.map Except.ok)
  | compare {symbol xtext ytext : String} (op : Compare)
      (x y : Atom format.Value Name .scalar)
      (hs : Internal.compare? symbol = some op)
      (hx : Atom.parse format .scalar xtext = some x)
      (hy : Atom.parse format .scalar ytext = some y) :
      Primitive.Denotes format arithmetic locals .predicate s!"({xtext} {symbol} {ytext})"
        (some (.ok (arithmetic.compare op (x.eval locals.get) (y.eval locals.get))))
  | indexCompare {symbol xtext ytext : String} (op : Compare)
      (x y : Atom format.Value Name .index)
      (hs : Internal.compare? symbol = some op)
      (hx : Atom.parse format .index xtext = some x)
      (hy : Atom.parse format .index ytext = some y) :
      Primitive.Denotes format arithmetic locals .predicate s!"({xtext} {symbol} {ytext})"
        (some (.ok (op.index (x.eval locals.get) (y.eval locals.get))))

private theorem atom_parse_of_render (format : Format) {t : Ty}
    (value : Atom format.Value Name t) {text : String}
    (h : Internal.atom format (fun x => (Literal.ofValue format x).bits) value = .ok text) :
    Atom.parse format t text = some value := by
  have hp := Atom.parse_render format value
  simpa only [h, Except.toOption, Option.bind_some] using hp

/-- Every successfully rendered primitive expression has its typed result in the text grammar.

This retains the supplied scalar contract and evaluation order; it does not replace that contract
with an assumption that NVIDIA implements it. A load cannot enter this theorem without its check,
because the expression renderer rejects it.
-/
theorem Primitive.denotes_expression (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (locals : Locals format.Value) {t : Ty}
    (operation : Primitive format.Value Name t) {text : String}
    (h : Internal.expression format (fun x => (Literal.ofValue format x).bits) operation =
      Except.ok text) :
    Primitive.Denotes format arithmetic locals t text
      (operation.eval arithmetic read locals.get) := by
  cases operation with
  | copy value =>
      exact Primitive.Denotes.copy value (atom_parse_of_render format value h)
  | load operand index =>
      simp [Internal.expression] at h
  | binary op x y =>
      cases hx : Internal.atom format (fun x => (Literal.ofValue format x).bits) x with
      | error error =>
          simp only [Internal.expression, hx] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok xt =>
        cases hy : Internal.atom format (fun x => (Literal.ofValue format x).bits) y with
        | error error =>
            simp only [Internal.expression, hx, hy] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok yt =>
          simp only [Internal.expression, hx, hy] at h
          dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
            Except.pure] at h
          injection h with htext
          subst text
          exact Primitive.Denotes.binary op x y (Internal.scalarOp_binary format op)
            (atom_parse_of_render format x hx) (atom_parse_of_render format y hy)
  | neg x =>
      cases hx : Internal.atom format (fun x => (Literal.ofValue format x).bits) x with
      | error error =>
          simp only [Internal.expression, hx] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok xt =>
          simp only [Internal.expression, hx] at h
          change Except.ok s!"(-{xt})" = Except.ok text at h
          injection h with htext
          subst text
          exact Primitive.Denotes.neg x (atom_parse_of_render format x hx)
  | unsigned expression =>
      exact Primitive.Denotes.unsigned (Unsigned.denotes_render format locals.get expression h)
  | compare op x y =>
      cases hx : Internal.atom format (fun x => (Literal.ofValue format x).bits) x with
      | error error =>
          simp only [Internal.expression, hx] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok xt =>
        cases hy : Internal.atom format (fun x => (Literal.ofValue format x).bits) y with
        | error error =>
            simp only [Internal.expression, hx, hy] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok yt =>
          simp only [Internal.expression, hx, hy] at h
          dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
            Except.pure] at h
          injection h with htext
          subst text
          exact Primitive.Denotes.compare op x y (Internal.compare_comparison op)
            (atom_parse_of_render format x hx) (atom_parse_of_render format y hy)
  | indexCompare op x y =>
      cases hx : Internal.atom format (fun x => (Literal.ofValue format x).bits) x with
      | error error =>
          simp only [Internal.expression, hx] at h
          change Except.error error = Except.ok text at h
          contradiction
      | ok xt =>
        cases hy : Internal.atom format (fun x => (Literal.ofValue format x).bits) y with
        | error error =>
            simp only [Internal.expression, hx, hy] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok yt =>
          simp only [Internal.expression, hx, hy] at h
          dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
            Except.pure] at h
          injection h with htext
          subst text
          exact Primitive.Denotes.indexCompare op x y (Internal.compare_comparison op)
            (atom_parse_of_render format x hx) (atom_parse_of_render format y hy)

/-- The emitted expression of a lowered target primitive retains its source result under a
valuation representing the source scope. Loads are excluded by the renderer, not by an axiom. -/
theorem Primitive.denotes_lower (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) {Γ : List Ty} {t : Ty}
    (names : {s : Ty} → Var Γ s → Name s) (locals : Locals format.Value)
    (env : Env format.Value Γ)
    (hScope : ∀ {s : Ty} (v : Var Γ s), locals.get (names v) = env.get v)
    (operation : Target.Primitive format.Value Γ t) {text : String}
    (h : Internal.expression format (fun x => (Literal.ofValue format x).bits)
      (Primitive.lower names operation) = Except.ok text) :
    Primitive.Denotes format arithmetic locals t text
      (some (operation.eval arithmetic read env)) := by
  simpa only [Primitive.eval_lower arithmetic read names locals.get env hScope] using
    Primitive.denotes_expression format arithmetic read locals (Primitive.lower names operation) h

namespace Internal

/-- Decode declaration types without accepting a different scalar width or signed indices. -/
def type? : Format → String → Option Ty
  | .binary32, "float" => some .scalar
  | .binary64, "double" => some .scalar
  | _, "unsigned long long" => some .index
  | _, "bool" => some .predicate
  | _, _ => none

theorem type_typeName (format : Format) (t : Ty) :
    type? format (typeName format t) = some t := by
  cases format <;> cases t <;> rfl

end Internal

/-- Semantics of declaration and checked-load statement text for one output thread.

The memory functions describe declared native buffers: their unsigned lengths and the scalar at
an in-bounds address. A failed check returns before any load or local write. The error-record text
is retained, but the relation does not verify atomic operations, concurrent error selection or the
physical buffer implementation. Those remain native-runtime boundaries.
-/
inductive Assignment.Denotes (format : Format) (arithmetic : Arithmetic format.Value)
    (inputs : Nat) (size : Nat → UInt64) (load : Nat → UInt64 → format.Value)
    (locals : Locals format.Value) :
    String → Option (Except Error (Locals format.Value)) → Prop where
  | declaration {t : Ty} {typeText nameText rhs : String}
      {result : Option (Except Error (t.Value format.Value))} (name : Name t)
      (ht : Internal.type? format typeText = some t)
      (hn : Name.parse t nameText = some name)
      (hr : Primitive.Denotes format arithmetic locals t rhs result) :
      Assignment.Denotes format arithmetic inputs size load locals
        s!"{typeText} {nameText} = {rhs};\n"
        (result.map fun result => result.map fun value => locals.set name value)
  | checkedLoad {typeText nameText indexText : String} (operand : Nat)
      (name : Name .scalar) (index : Atom format.Value Name .index)
      (hOperand : operand < inputs)
      (ht : Internal.type? format typeText = some .scalar)
      (hn : Name.parse .scalar nameText = some name)
      (hi : Atom.parse format .index indexText = some index) :
      Assignment.Denotes format arithmetic inputs size load locals
        (s!"if ({indexText} >= size{operand}) \{\n" ++
          s!"if (atomicCAS(error, 0ULL, 1ULL) == 0ULL) \{ " ++
          s!"error[1] = {operand}ULL; error[2] = {indexText}; error[3] = size{operand}; }\n" ++
          "return;\n}\n" ++
          (s!"{typeText} {nameText} = " ++ s!"input{operand}[{indexText}]" ++ ";\n"))
        (some ((show Except Error format.Value from if index.eval locals.get ≥ size operand then
            .error (.bounds operand (index.eval locals.get) (size operand).toNat)
          else .ok (load operand (index.eval locals.get))).map fun value => locals.set name value))

namespace Internal

private theorem denotes_declaration (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value) (locals : Locals format.Value) {t : Ty}
    (name : Name t) (operation : Primitive format.Value Name t) {text : String}
    (hForm : assignment format (fun x => (Literal.ofValue format x).bits) inputs ⟨name, operation⟩ =
      (expression format (fun x => (Literal.ofValue format x).bits) operation).map
        (fun rhs => s!"{typeName format t} {name.render} = {rhs};\n"))
    (h : assignment format (fun x => (Literal.ofValue format x).bits) inputs ⟨name, operation⟩ =
      Except.ok text) :
    Assignment.Denotes format arithmetic inputs size load locals text
      ((Assignment.mk name operation).eval arithmetic read locals) := by
  rw [hForm] at h
  cases he : expression format (fun x => (Literal.ofValue format x).bits) operation with
  | error error => simp [he, Except.map] at h
  | ok rhs =>
      simp only [he, Except.map, Except.ok.injEq] at h
      subst text
      exact Assignment.Denotes.declaration name (type_typeName format t) (Name.parse_render name)
        (Primitive.denotes_expression format arithmetic read locals operation he)

end Internal

/-- Emitted assignment text agrees with the named evaluator when the reader represents the
declared buffers. The reader contract identifies both successful reads and exact bounds errors.
Undeclared operands are rejected by source generation, before a native pointer can be used. -/
theorem Assignment.denotes_render (format : Format) (arithmetic : Arithmetic format.Value)
    (read : Reader format.Value) (inputs : Nat) (size : Nat → UInt64)
    (load : Nat → UInt64 → format.Value)
    (hRead : ∀ operand, operand < inputs → ∀ index,
      read operand index = if index ≥ size operand then
        .error (.bounds operand index (size operand).toNat) else .ok (load operand index))
    (locals : Locals format.Value) {t : Ty} (statement : Assignment format.Value t)
    {text : String}
    (h : Internal.assignment format (fun x => (Literal.ofValue format x).bits) inputs statement =
      Except.ok text) :
    Assignment.Denotes format arithmetic inputs size load locals text
      (statement.eval arithmetic read locals) := by
  rcases statement with ⟨name, operation⟩
  cases operation with
  | load operand index =>
      by_cases hOperand : operand ≥ inputs
      · simp only [Internal.assignment, hOperand, ↓reduceIte] at h
        change Except.error s!"kernel: input {operand} is not declared" = Except.ok text at h
        contradiction
      · have hDeclared : operand < inputs := Nat.lt_of_not_ge hOperand
        cases hi : Internal.atom format (fun x => (Literal.ofValue format x).bits) index with
        | error error =>
            simp only [Internal.assignment, hOperand, ↓reduceIte, hi] at h
            change Except.error error = Except.ok text at h
            contradiction
        | ok it =>
            simp only [Internal.assignment, hOperand, ↓reduceIte, hi] at h
            dsimp only [bind, Functor.map, Except.instMonad, Except.bind, Except.map,
              Except.pure] at h
            injection h with htext
            subst text
            simpa [Assignment.eval, Primitive.eval, hRead operand hDeclared,
              toString, String.append_assoc] using
              Assignment.Denotes.checkedLoad (format := format) (arithmetic := arithmetic)
                (size := size) (load := load) (locals := locals) operand name index hDeclared
                (Internal.type_typeName format .scalar) (Name.parse_render name)
                (atom_parse_of_render format index hi)
  | copy value =>
      apply Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.copy value) _ h
      cases t <;> rfl
  | binary op x y =>
      exact Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.binary op x y) rfl h
  | neg x =>
      exact Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.neg x) rfl h
  | unsigned value =>
      exact Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.unsigned value) rfl h
  | compare op x y =>
      exact Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.compare op x y) rfl h
  | indexCompare op x y =>
      exact Internal.denotes_declaration format arithmetic read inputs size load locals
        name (.indexCompare op x y) rfl h

end NN.Kernel.Cuda
