/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Floats.IEEEExec.Exec32
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Float32

Float32 viewer widget (executable IEEE-754 backend).

Commands:
- `#float32_view x` renders an `IEEE32Exec` value as bits + fields + basic classification flags.
- `#float32_round_view x` shows how a Lean `Float` (binary64) rounds to `IEEE32Exec` (binary32).

These widgets are meant for debugging/teaching, not for proof scripts.

## Main definitions

- `float32Html`: inspect class/fields/bits for one `IEEE32Exec` value.
- `float32RoundHtml`: show `Float64 -> Float32` rounding behavior.
- `float32CompareHtml`: side-by-side bit-level comparison.
- `#float32_view`, `#float32_round_view`, `#float32_compare_view`: command entry points.

## Implementation notes

- Explicit sign/exp/frac bit pills are easier to read than one long bit
  string when debugging rounding and special values.
- We include both classification badges and raw field values: in practice users want both the
  high-level class and exact bit-level evidence.
- We keep this widget purely informational; no arithmetic semantics are changed here.

## References

- [IEEE 754 floating-point standard overview](https://en.wikipedia.org/wiki/IEEE_754)
- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

float32, ieee754, rounding, bits, proofwidgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open TorchLean.Floats.IEEE754
open UI

namespace Float32Internal

def u32Hex (u : UInt32) : String :=
  -- `UInt32`'s `repr` is a compact debugging view (hex-like when printed in Lean).
  reprStr u

def u64Hex (u : UInt64) : String :=
  reprStr u

/-- Render exactly `width` low-order bits of `n` as a binary string. -/
def bitsFixed (width : Nat) (n : Nat) : String :=
  let rec go (i : Nat) (acc : List Char) : List Char :=
    match i with
    | 0 => acc
    | i + 1 =>
        let j := i
        let c := if Nat.testBit n j then '1' else '0'
        go i (c :: acc)
  String.ofList (go width []).reverse

def bitPill (label bits : String) (bg : String) : ProofWidgets.Html :=
  let styleObj : Lean.Json :=
    Lean.Json.mkObj [
      ("display", Lean.Json.str "inline-flex"),
      ("gap", Lean.Json.str "6px"),
      ("align-items", Lean.Json.str "center"),
      ("padding", Lean.Json.str "4px 8px"),
      ("border-radius", Lean.Json.str "10px"),
      ("border", Lean.Json.str "1px solid var(--vscode-panel-border, #e0e0e0)"),
      ("background", Lean.Json.str bg),
      ("font-size", Lean.Json.str "12px"),
      ("line-height", Lean.Json.str "18px")
    ]
  ;
  <span style={styleObj}>
    <span style={json% {"opacity": 0.85}}>{.text label}</span>
    {monospace bits}
  </span>

/-- Classify an IEEE32 value into normal/subnormal/zero/inf/nan variants. -/
def classify (x : IEEE32Exec) : String :=
  if IEEE32Exec.isNaN x then
    if IEEE32Exec.isSNaN x then "sNaN" else "qNaN"
  else if IEEE32Exec.isInf x then
    if IEEE32Exec.signBit x then "-Inf" else "+Inf"
  else if IEEE32Exec.isZero x then
    if IEEE32Exec.signBit x then "-0" else "+0"
  else if IEEE32Exec.expField x == 0 then
    "subnormal"
  else
    "normal"

def dyadicString (d : IEEE32Exec.Dyadic) : String :=
  let sign := if d.sign then "-" else "+"
  s!"{sign}{d.mant} * 2^{d.exp}"

end Float32Internal

open Float32Internal

/-- Render an executable float32 (`IEEE32Exec`) as HTML. -/
def float32Html (x : IEEE32Exec) : ProofWidgets.Html :=
  let b := IEEE32Exec.toBits x
  let s := IEEE32Exec.signBit x
  let e := IEEE32Exec.expField x
  let f := IEEE32Exec.fracField x
  let cls := classify x
  let f64 := IEEE32Exec.toFloat x
  let dyadic? := IEEE32Exec.toDyadic? x
  let signBits := if s then "1" else "0"
  let expBits := bitsFixed 8 e.toNat
  let fracBits := bitsFixed 23 f.toNat;
  <div style={json% {
    "display": "block",
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "IEEE32Exec"} {pill s!"class={cls}"} {pill s!"bits={u32Hex b}"} {pill
        s!"asFloat={toString f64}"}
    </div>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {bitPill "sign" signBits "rgba(255, 80, 80, 0.18)"}
      {bitPill "exp" expBits "rgba(80, 160, 255, 0.18)"}
      {bitPill "frac" fracBits "rgba(0, 200, 120, 0.14)"}
    </div>
    <div style={json% {"display": "flex", "gap": "6px", "flex-wrap": "wrap"}}>
      {flagBadge "NaN" (IEEE32Exec.isNaN x)} {flagBadge "Inf" (IEEE32Exec.isInf x)}
      {flagBadge "Zero" (IEEE32Exec.isZero x)} {flagBadge "Finite" (IEEE32Exec.isFinite x)}
      {flagBadge "qNaN" (IEEE32Exec.isQNaN x)} {flagBadge "sNaN" (IEEE32Exec.isSNaN x)}
    </div>
    <details style={json% {"margin-top": "10px"}} «open»={false}>
      <summary>{.text "Raw fields"}</summary>
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "6px",
        "margin-top": "8px"}}>
        <div><b>toBits:</b> {monospace (reprStr (IEEE32Exec.toBits x))}</div>
        <div><b>signBit:</b> {monospace (reprStr (IEEE32Exec.signBit x))}</div>
        <div><b>expField:</b> {monospace (reprStr (IEEE32Exec.expField x))}</div>
        <div><b>fracField:</b> {monospace (reprStr (IEEE32Exec.fracField x))}</div>
        <div><b>quietBit:</b> {monospace (reprStr ((IEEE32Exec.toBits x &&& IEEE32Exec.quietBit) !=
          0))}</div>
      </div>
    </details>
    <details style={json% {"margin-top": "10px"}} «open»={false}>
      <summary>{.text "Exact value (dyadic) when finite"}</summary>
      <div style={json% {"margin-top": "8px"}}>
        {match dyadic? with
          | none => <span style={json% {"opacity": 0.75}}>{.text "(none: NaN/Inf)"}</span>
          | some d => monospace (dyadicString d)}
      </div>
    </details>
  </div>

/--
Element renderer for `IEEE32Exec` used by `#tensor_view`.

Renders the float value, with a tooltip that includes a small classification and the raw bit
pattern.
-/
instance : TensorElemView IEEE32Exec :=
  ⟨fun x =>
    let b := IEEE32Exec.toBits x
    let cls := Float32Internal.classify x
    let v := IEEE32Exec.toFloat x
    let title := s!"class={cls}\nbits={Float32Internal.u32Hex b}\nasFloat={toString v}";
    <span title={title}>{monospace (toString v)}</span>⟩

namespace Float32Internal

/-- Compare two `IEEE32Exec` values at the bit level and render the results as HTML. -/
def float32CompareHtml (x y : IEEE32Exec) : ProofWidgets.Html :=
  let bx := IEEE32Exec.toBits x
  let byBits := IEEE32Exec.toBits y
  let diff := bx ^^^ byBits
  let same := decide (bx = byBits);
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "IEEE32Exec compare"} {pill s!"sameBits={same}"} {pill s!"xor={u32Hex diff}"}
    </div>
    <div style={json% {"display": "grid", "grid-template-columns": "1fr 1fr", "gap": "10px"}}>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "x"}</div>
        {float32Html x}
      </div>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "y"}</div>
        {float32Html y}
      </div>
    </div>
  </div>

/-- Show how a Lean `Float` (binary64) rounds to an executable float32 (`IEEE32Exec`, binary32). -/
def float32RoundHtml (x : Float) : ProofWidgets.Html :=
  let b64 : UInt64 := x.toBits
  let x32 : IEEE32Exec := IEEE32Exec.ofFloat x
  let y : Float := IEEE32Exec.toFloat x32
  let err : Float := y - x;
  <span style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <span style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "Float -> IEEE32Exec"}
      {pill s!"input(Float64)={toString x}"}
      {pill s!"inputBits={u64Hex b64}"}
      {pill s!"rounded(Float32)={toString y}"}
      {pill s!"roundingError={toString err}"}
    </span>
    {float32Html x32}
  </span>

end Float32Internal

/-!
## Commands
-/

syntax (name := float32ViewCmd) "#float32_view " term : command

macro "#float32_view " x:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (float32Html $x))

syntax (name := float32RoundViewCmd) "#float32_round_view " term : command

macro "#float32_round_view " x:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (Float32Internal.float32RoundHtml $x))

syntax (name := float32CompareViewCmd) "#float32_compare_view " term ", " term : command

macro "#float32_compare_view " x:term ", " y:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (Float32Internal.float32CompareHtml $x $y))

end NN.Widgets
