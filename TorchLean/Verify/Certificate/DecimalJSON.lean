/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import TorchLean.Tensor.ConfiguredBinary
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Runtime
public import FloatLib.Numerics.Exact.DecimalText.Runtime
import all Std.Internal.Parsec.Basic
import all Std.Internal.Parsec.String
import all Lean.Data.Json.Parser

/-!
# JSON with exact signed decimals

A JSON reader and writer whose numbers are exact decimals that keep the sign of zero. Lean's
`JsonNumber` has an integer mantissa, so it reads `-0` and `-0.0` as `0`, and it expands a positive
exponent into the mantissa. Here the tokens `-0`, `-0.0` and `-0e3` are the negative zero, and an
exponent stays an integer.

The number tokens are those of RFC 8259 section 6, and nothing but white space may follow the
value. An object with a repeated key is refused, a restriction of the certificate format that RFC
8259 does not make. Keys are compared after decoding, so `"x"` and `"\u0078"` are the same key.
Strings are decoded as Lean's JSON string parser decodes them, which reads an escaped lone
surrogate as U+FFFD.

The reader is total. Its string and digit loops are copies of the loops of `Lean.Json.Parser` by
well-founded recursion on the position, and its nested calls run on a fuel of
`2 * s.utf8ByteSize + 2`. `TorchLean.Verify.Certificate.DecimalJSONFuel` proves that this fuel
suffices: every fuel of at least `2 * s.utf8ByteSize + 1` gives the result of `parse s`
(`run_value_eq_parse`, from `value_fuel_congr`). `TorchLean.Verify.Certificate.DecimalJSONRoundTrip`
proves that `parse` reads `Value.render v` back as `v` for every `v` with `Value.Canonical v`
(`parse_render`).

A number is FloatLib's exact `Decimal`. `Model.convertDecimalText` rounds one once into a binary
format in a given direction, as the conversion of IEEE 754-2019 section 5.12 does. `Decimal.toRat?`
gives the rational it denotes, refusing a nonzero decimal whose exponent exceeds `2 ^ 11 + Nat.log2`
of its significand in absolute value. `Decimal.ofBinary32?` writes a finite binary32 value as its
exact decimal, so writing and rounding back to nearest return the same binary32 value, `-0`
included.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics.DecimalText (Decimal)

namespace TorchLean.Certificate.DecimalJson

/-- A JSON value whose numbers are exact signed decimals and whose objects keep their fields in
order. -/
inductive Value where
  /-- The JSON `null`. -/
  | null
  /-- A JSON Boolean. -/
  | bool (b : Bool)
  /-- A JSON number, as an exact signed decimal. -/
  | num (d : Decimal)
  /-- A JSON string. -/
  | str (s : String)
  /-- A JSON array. -/
  | arr (elems : Array Value)
  /-- A JSON object, with its fields in order. -/
  | obj (fields : Array (String × Value))
  deriving Inhabited

namespace Decimal

/-- Whether `a * 10 ^ e ≤ b * 10 ^ f`, comparing the orders of magnitude first, so that the powers
formed are bounded by the digit counts of `a` and `b`. -/
def magnitudeLe (a : Nat) (e : Int) (b : Nat) (f : Int) : Bool :=
  if a = 0 then true
  else if b = 0 then false
  else
    let o := ((Nat.repr a).length : Int) + e
    let p := ((Nat.repr b).length : Int) + f
    if o ≠ p then decide (o < p)
    else decide (a * 10 ^ (e - min e f).toNat ≤ b * 10 ^ (f - min e f).toNat)

/-- Whether `x ≤ y` as real numbers, where `-0` and `0` are equal. -/
def le (x y : Decimal) : Bool :=
  match x.negative && x.significand != 0, y.negative && y.significand != 0 with
  | false, false => magnitudeLe x.significand x.exponent y.significand y.exponent
  | true, true => magnitudeLe y.significand y.exponent x.significand x.exponent
  | true, false => true
  | false, true => false

/-- The exact decimal of a finite binary32 value, with the sign of zero, and `none` for an infinity
or NaN, which JSON cannot write. -/
def ofBinary32? (x : ExecFloat.Binary 8 23) : Option Decimal := do
  let q ← ExecFloat.Binary.toRat? x
  let k := q.den.log2
  pure { negative := ExecFloat.Binary.signBit x, significand := q.num.natAbs * 5 ^ k,
         exponent := -k }

/-- The JSON token of a decimal: `-0.0` for the negative zero, which reads as negative even in
readers that take `-0` for the integer `0`, `0` for the zero, and otherwise `Decimal.format`.
`Decimal.format` writes the signed digits followed by `e<exponent>` unless the exponent is `0`. -/
def render (d : Decimal) : String :=
  if d.significand = 0 then (if d.negative then "-0.0" else "0") else d.format

/-- The rational that `d` denotes, or `none` for a nonzero `d` whose exponent exceeds
`2 ^ 11 + Nat.log2 d.significand` in absolute value. The bound keeps the power of ten formed linear
in the length of the token: the twelve characters `1e-999999999` would otherwise ask for a power of
ten with a billion digits. -/
def toRat? (d : Decimal) : Option Rat :=
  if d.significand = 0 then some 0
  else if d.exponent.natAbs ≤ 2 ^ 11 + d.significand.log2 then some d.toRat else none

/-- `toRat? d` is `some q` exactly when `d` is zero or its exponent is within the bound and `q` is
the rational `d` denotes. -/
theorem toRat?_eq_some_iff (d : Decimal) (q : Rat) :
    toRat? d = some q ↔
      (d.significand = 0 ∨ d.exponent.natAbs ≤ 2 ^ 11 + d.significand.log2) ∧ d.toRat = q := by
  unfold toRat?
  by_cases h0 : d.significand = 0
  · simp [h0, Decimal.toRat]
  · simp only [h0, ↓reduceIte, false_or]
    split <;> simp_all

end Decimal

section Parser

open Std.Internal.Parsec Std.Internal.Parsec.String

/-- Whether the next character satisfies `p`, and `false` at the end of the input. -/
def Parser.nextIs (p : Char → Bool) : Parser Bool := do
  pure (((← peek?).map p).getD false)

/-- The parser `p` never moves back: when it succeeds, at most as many bytes of the input are left
as before. -/
def Forward {α : Type} (p : Parser α) : Prop :=
  ∀ ⦃it it' : Sigma String.Pos⦄ ⦃a : α⦄, p it = .success it' a →
    it'.2.remainingBytes ≤ it.2.remainingBytes

/-- At a position other than the end of `s`, the next position has fewer bytes of `s` left. -/
theorem remainingBytes_next_lt {s : String} {p : s.Pos} (h : p ≠ s.endPos) :
    (p.next h).remainingBytes < p.remainingBytes :=
  (String.Pos.lt_iff_remainingBytes_lt _ _).1 String.Pos.lt_next

/-- An input with a next character is not at its end. -/
theorem not_isAtEnd_of_hasNext {it : Sigma String.Pos} (h : Input.hasNext it = true) :
    ¬it.2.IsAtEnd :=
  of_decide_eq_true h

/-- `pure a` never moves back. -/
theorem forward_pure {α : Type} (a : α) : Forward (pure a : Parser α) := by
  intro it it' b h
  cases h
  exact Nat.le_refl _

/-- `fail msg` never moves back, since it never succeeds. -/
theorem forward_fail {α : Type} (msg : String) : Forward (fail msg : Parser α) := by
  intro it it' b h
  cases h

/-- `p >>= f` never moves back when `p` and every `f a` never move back. -/
theorem Forward.bind {α β : Type} {p : Parser α} {f : α → Parser β} (hp : Forward p)
    (hf : ∀ a, Forward (f a)) : Forward (p >>= f) := by
  intro it it' b h
  change Std.Internal.Parsec.bind p f it = _ at h
  unfold Std.Internal.Parsec.bind at h
  split at h
  next rem a hpa => exact Nat.le_trans (hf a h) (hp hpa)
  next => cases h

/-- When `any` succeeds, fewer bytes of the input are left. -/
theorem any_lt {it it' : Sigma String.Pos} {c : Char}
    (h : (any : Parser Char) it = .success it' c) :
    it'.2.remainingBytes < it.2.remainingBytes := by
  unfold any at h
  split at h
  next hn =>
    cases h
    exact remainingBytes_next_lt (not_isAtEnd_of_hasNext hn)
  next => cases h

/-- `any` never moves back. -/
theorem forward_any : Forward (any : Parser Char) :=
  fun _ _ _ h => Nat.le_of_lt (any_lt h)

/-- `attempt p <|> q` never moves back when `p` and `q` never move back. -/
theorem Forward.attempt_orElse {α : Type} {p q : Parser α} (hp : Forward p) (hq : Forward q) :
    Forward (attempt p <|> q) := by
  intro it it' a h
  change Std.Internal.Parsec.tryCatch (attempt p) pure (fun _ => q) it = _ at h
  unfold Std.Internal.Parsec.tryCatch attempt at h
  split at h
  next rem b hpa =>
    split at hpa
    next rem' c hpc =>
      cases hpa
      cases h
      exact hp hpc
    next => cases hpa
  next rem err hpe =>
    split at hpe
    next => cases hpe
    next =>
      cases hpe
      split at h
      · exact hq h
      · cases h

/-- `Lean.Json.Parser.hexChar` never moves back. -/
theorem forward_hexChar : Forward Lean.Json.Parser.hexChar := by
  unfold Lean.Json.Parser.hexChar
  repeat' first
    | with_reducible exact forward_pure _
    | with_reducible exact forward_fail _
    | with_reducible exact forward_any
    | with_reducible refine Forward.bind ?_ fun _ => ?_
    | with_reducible refine iteInduction (motive := Forward) (fun _ => ?_) fun _ => ?_

/-- `Lean.Json.Parser.finishSurrogatePair low` never moves back. -/
theorem forward_finishSurrogatePair (low : UInt16) :
    Forward (Lean.Json.Parser.finishSurrogatePair low) := by
  unfold Lean.Json.Parser.finishSurrogatePair
  repeat' first
    | with_reducible exact forward_pure _
    | with_reducible exact forward_fail _
    | with_reducible exact forward_any
    | with_reducible exact forward_hexChar
    | with_reducible refine Forward.bind ?_ fun _ => ?_
    | with_reducible refine iteInduction (motive := Forward) (fun _ => ?_) fun _ => ?_
    | split
    | dsimp only

/-- `Lean.Json.Parser.escapedChar` never moves back. -/
theorem forward_escapedChar : Forward Lean.Json.Parser.escapedChar := by
  unfold Lean.Json.Parser.escapedChar
  repeat' first
    | with_reducible exact forward_pure _
    | with_reducible exact forward_fail _
    | with_reducible exact forward_any
    | with_reducible exact forward_hexChar
    | with_reducible apply Forward.attempt_orElse (forward_finishSurrogatePair _)
    | with_reducible refine Forward.bind ?_ fun _ => ?_
    | with_reducible refine iteInduction (motive := Forward) (fun _ => ?_) fun _ => ?_
    | split
    | dsimp only

/-- The characters of a JSON string after its opening quote and before its closing quote, with
their escapes decoded, appended to `acc`, and the position after the closing quote: the loop
`Lean.Json.Parser.strCore`, with the same accepted characters, results and errors, by well-founded
recursion on the position. -/
def strCore (acc : String) : Parser String := fun it =>
  if h : it.2.IsAtEnd then .error it .eof
  else
    let c := it.2.get h
    if c == '"' then .success ⟨it.1, it.2.next h⟩ acc
    else
      let it' : Sigma String.Pos := ⟨it.1, it.2.next h⟩
      if c == '\\' then
        match _he : Lean.Json.Parser.escapedChar it' with
        | .success it'' e => strCore (acc.push e) it''
        | .error it'' err => .error it'' err
      else if 0x0020 <= c.val && c.val <= 0x10ffff then strCore (acc.push c) it'
      else .error it' (.other "unexpected character in string")
termination_by it => it.2.remainingBytes
decreasing_by
  · exact Nat.lt_of_le_of_lt (forward_escapedChar _he) (remainingBytes_next_lt h)
  · exact remainingBytes_next_lt h

/-- The decimal digits at the position read after the digits of `acc`, and the position after
them: the loop `Lean.Json.Parser.natCore`, with the same results, by well-founded recursion on the
position. -/
def natCore (acc : Nat) : Parser Nat := fun it =>
  if h : it.2.IsAtEnd then .success it acc
  else
    let c := it.2.get h
    if '0' <= c && c <= '9' then
      natCore (10 * acc + (c.val - '0'.val).toNat) ⟨it.1, it.2.next h⟩
    else .success it acc
termination_by it => it.2.remainingBytes
decreasing_by exact remainingBytes_next_lt h

/-- The decimal digits at the position read after the digits of `acc`, with `digits` counting
them on from its value: the loop `Lean.Json.Parser.natCoreNumDigits`, with the same results, by
well-founded recursion on the position. -/
def natCoreNumDigits (acc digits : Nat) : Parser (Nat × Nat) := fun it =>
  if h : it.2.IsAtEnd then .success it (acc, digits)
  else
    let c := it.2.get h
    if '0' <= c && c <= '9' then
      natCoreNumDigits (10 * acc + (c.val - '0'.val).toNat) (digits + 1) ⟨it.1, it.2.next h⟩
    else .success it (acc, digits)
termination_by it => it.2.remainingBytes
decreasing_by exact remainingBytes_next_lt h

/-- A JSON number token (RFC 8259 section 6) as its exact decimal, the exponent not expanded. The
three digit runs are read as `Lean.Json.Parser.nat`, `Lean.Json.Parser.natNumDigits` and
`Lean.Json.Parser.natMaybeZero` read them, with the loops `natCore` and `natCoreNumDigits`. -/
def Parser.decimal : Parser Decimal := do
  let negative ← nextIs (· == '-')
  if negative then skip
  let whole ← do
    let c ← peek!
    if c == '0' then
      skip
      pure 0
    else
      Lean.Json.Parser.lookahead (fun c => '1' <= c && c <= '9') "1-9"
      natCore 0
  let mut frac := 0
  let mut fracDigits := 0
  if (← nextIs (· == '.')) then
    skip
    Lean.Json.Parser.lookahead (fun c => '0' <= c && c <= '9') "digit"
    (frac, fracDigits) ← natCoreNumDigits 0 0
  let mut exponent : Int := 0
  if (← nextIs (fun c => c == 'e' || c == 'E')) then
    skip
    let expNegative ← nextIs (· == '-')
    if expNegative || (← nextIs (· == '+')) then skip
    Lean.Json.Parser.lookahead (fun c => '0' <= c && c <= '9') "0-9"
    let e ← natCore 0
    exponent := if expNegative then -(e : Int) else e
  pure { negative, significand := whole * 10 ^ fracDigits + frac,
         exponent := exponent - fracDigits }

mutual

/-- The elements of an array after its `[`, failing when the calls nest deeper than the fuel. Two
fuels of at least `2 * it.2.remainingBytes + 2` give the same result at `it`
(`arrayItems_fuel_congr` in `TorchLean.Verify.Certificate.DecimalJSONFuel`). -/
def Parser.arrayItems : Nat → Array Value → Parser (Array Value)
  | 0, _ => fail "unexpected input"
  | fuel + 1, acc => do
    let v ← value fuel
    let acc := acc.push v
    let c ← any
    if c == ']' then
      ws
      pure acc
    else if c == ',' then
      ws
      arrayItems fuel acc
    else fail "unexpected character in array"

/-- The fields of an object after its `{`, failing at a key equal after decoding to an earlier one
and when the calls nest deeper than the fuel. Two fuels of at least `2 * it.2.remainingBytes + 2`
give the same result at `it` (`objectFields_fuel_congr` in
`TorchLean.Verify.Certificate.DecimalJSONFuel`). -/
def Parser.objectFields : Nat → Array (String × Value) → Parser (Array (String × Value))
  | 0, _ => fail "unexpected input"
  | fuel + 1, acc => do
    Lean.Json.Parser.lookahead (fun c => c == '"') "\""
    skip
    let k ← strCore ""
    ws
    Lean.Json.Parser.lookahead (fun c => c == ':') ":"
    skip
    ws
    if acc.any (·.1 == k) then fail s!"repeated key {Lean.Json.renderString k}"
    let v ← value fuel
    let acc := acc.push (k, v)
    let c ← any
    if c == '}' then
      ws
      pure acc
    else if c == ',' then
      ws
      objectFields fuel acc
    else fail "unexpected character in object"

/-- One JSON value and the white space after it, failing when the calls nest deeper than the
fuel. Two fuels of at least `2 * it.2.remainingBytes + 1` give the same result at `it`
(`value_fuel_congr` in `TorchLean.Verify.Certificate.DecimalJSONFuel`). -/
def Parser.value : Nat → Parser Value
  | 0 => fail "unexpected input"
  | fuel + 1 => do
    let c ← peek!
    if c == '[' then
      skip
      ws
      if (← nextIs (· == ']')) then
        skip
        ws
        pure (.arr #[])
      else
        let xs ← arrayItems fuel #[]
        pure (.arr xs)
    else if c == '{' then
      skip
      ws
      if (← nextIs (· == '}')) then
        skip
        ws
        pure (.obj #[])
      else
        let fields ← objectFields fuel #[]
        pure (.obj fields)
    else if c == '"' then
      skip
      let s ← strCore ""
      ws
      pure (.str s)
    else if c == 'f' then
      skipString "false"
      ws
      pure (.bool false)
    else if c == 't' then
      skipString "true"
      ws
      pure (.bool true)
    else if c == 'n' then
      skipString "null"
      ws
      pure .null
    else if c == '-' || c.isDigit then
      let d ← decimal
      ws
      pure (.num d)
    else fail "unexpected input"

end

end Parser

/-- Parse a JSON text with fuel `2 * s.utf8ByteSize + 2`. Every fuel of at least
`2 * s.utf8ByteSize + 1` gives the same result (`run_value_eq_parse` in
`TorchLean.Verify.Certificate.DecimalJSONFuel`), and `parse` reads `Value.render v` back as `v` for
every `v` with `Value.Canonical v` (`parse_render` in
`TorchLean.Verify.Certificate.DecimalJSONRoundTrip`). -/
def parse (s : String) : Except String Value :=
  Std.Internal.Parsec.String.Parser.run
    (do Std.Internal.Parsec.String.ws; let v ← Parser.value (2 * s.utf8ByteSize + 2)
        Std.Internal.Parsec.eof; pure v) s

namespace Value

/-- Compact JSON text of a value, with each number written by `Decimal.render`. -/
def render : Value → String
  | .null => "null"
  | .bool b => toString b
  | .num d => Decimal.render d
  | .str s => Lean.Json.renderString s
  | .arr xs => "[" ++ ",".intercalate (xs.map render).toList ++ "]"
  | .obj fields =>
      "{" ++ ",".intercalate (fields.map fun field =>
        Lean.Json.renderString field.1 ++ ":" ++ render field.2).toList ++ "}"
decreasing_by
  all_goals simp_wf
  · have := Array.sizeOf_lt_of_mem ‹_ ∈ xs›
    omega
  · have := Array.sizeOf_lt_of_mem ‹field ∈ fields›
    cases field
    simp only [Prod.mk.sizeOf_spec] at this ⊢
    omega

/-- The field `k` of `v`, with `ctx` naming `v` in the error for a value that is not an object or
has no field `k`. -/
def field (v : Value) (k ctx : String) : Except String Value :=
  match v with
  | .obj fields =>
      match fields.find? (·.1 == k) with
      | some (_, x) => .ok x
      | none => .error s!"{ctx}: missing field {k}"
  | _ => .error s!"{ctx}: expected an object"

/-- The field `k` of `v` if present, with `ctx` naming `v` in the error for a value that is not an
object. -/
def optionalField? (v : Value) (k ctx : String) : Except String (Option Value) :=
  match v with
  | .obj fields => .ok ((fields.find? (·.1 == k)).map (·.2))
  | _ => .error s!"{ctx}: expected an object"

/-- The elements of an array, with `ctx` naming `v` in the error. -/
def array (v : Value) (ctx : String) : Except String (Array Value) :=
  match v with
  | .arr xs => .ok xs
  | _ => .error s!"{ctx}: expected an array"

/-- The string of a string value, with `ctx` naming `v` in the error. -/
def string (v : Value) (ctx : String) : Except String String :=
  match v with
  | .str s => .ok s
  | _ => .error s!"{ctx}: expected a string"

/-- `string` returns `s` exactly for the string value `s`. -/
theorem string_eq_ok_iff (v : Value) (ctx s : String) : v.string ctx = .ok s ↔ v = .str s := by
  cases v <;> simp [string, eq_comm]

/-- The decimal of a number, or of a string that `parse` reads as one number. -/
def decimal? : Value → Option Decimal
  | .num d => some d
  | .str s =>
      match parse s with
      | .ok (.num d) => some d
      | _ => none
  | _ => none

/-- `decimal?` returns `d` exactly for the number `d` and for a string that `parse` reads as
the number `d`. -/
theorem decimal?_eq_some_iff (v : Value) (d : Decimal) :
    v.decimal? = some d ↔ v = .num d ∨ ∃ s, v = .str s ∧ parse s = .ok (.num d) := by
  cases v with
  | num e => simp [decimal?, eq_comm]
  | str s =>
      simp only [decimal?, reduceCtorEq, false_or, Value.str.injEq, exists_eq_left']
      split <;> simp_all
  | _ => simp [decimal?]

/-- An integer: a number whose decimal exponent, the exponent of the token less its count of
fraction digits, is from `0` to `40`, which bounds the power formed, or a string of decimal digits
after an optional `-`. So `1e1` and `1.0e1` are `10`, and the integral values `1.0` and `10e-1` are
not accepted. -/
def int? : Value → Option Int
  | .num d =>
      if 0 ≤ d.exponent ∧ d.exponent ≤ 40 then
        let n := d.significand * 10 ^ d.exponent.toNat
        some (if d.negative then -(n : Int) else n)
      else none
  | .str s => if s.all fun c => c.isDigit || c == '-' then s.toInt? else none
  | _ => none

/-- A natural number: a nonnegative `int?` number or a string of decimal digits, with `ctx` naming
`v` in the error. -/
def nat (v : Value) (ctx : String) : Except String Nat :=
  let n? := match v with
    | .str s => if s.all Char.isDigit then s.toNat? else none
    | _ => match int? v with
      | some (.ofNat n) => some n
      | _ => none
  match n? with
  | some n => .ok n
  | none => .error s!"{ctx}: expected a natural number"

/-- The decimals of a JSON array of exactly `dim` numbers. -/
def decimals? (dim : Nat) (v : Value) : Option (Fin dim → Decimal) := do
  let .arr xs := v | none
  let ds ← xs.mapM fun
    | .num d => some d
    | _ => none
  if h : ds.size = dim then
    pure fun i => ds[i.val]'(by rw [h]; exact i.isLt)
  else
    none

end Value

/-- Read the JSON file `path`, whose top-level value must be an object. -/
def readObjectFile (path : String) : IO Value := do
  let v ← IO.ofExcept <| (parse (← IO.FS.readFile path)).mapError fun e => s!"{path}: {e}"
  match v with
  | .obj _ => pure v
  | _ => throw <| IO.userError s!"{path}: expected a top-level object"

end TorchLean.Certificate.DecimalJson
