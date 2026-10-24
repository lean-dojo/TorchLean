/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public meta import NN.Widgets.Core.Tensor
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured -- shake: keep
public import NN.Widgets.Core.Tensor
import NN.Spec.Core.Tensor
import ProofWidgets.Component.HtmlDisplay

/-!
# Float32

Float32 viewer widget (executable IEEE-754 backend).

Commands:
- `#float32_view x` renders an `ExecFloat.Binary 8 23` value as bits + fields + basic classification
flags.
- `#float32_round_view x` shows how a Lean `Float` (binary64) rounds to `ExecFloat.Binary 8 23`
(binary32).

These widgets are meant for debugging/teaching, not for proof scripts.

## Main definitions

- `Float32.html`: inspect class/fields/bits for one `ExecFloat.Binary 8 23` value.
- `Float32.roundHtml`: show `Float64 -> Float32` rounding behavior.
- `Float32.compareHtml`: side-by-side bit-level comparison.
- `#float32_view`, `#float32_round_view`, `#float32_compare_view`: command entry points.
-/

public meta section

open scoped ProofWidgets.Jsx

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Widgets

open UI

namespace Float32

/-- Render a bit vector in hexadecimal, preserving the width encoded by its type. -/
def hex {width : Nat} (bits : BitVec width) : String :=
  "0x" ++ bits.toHex

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

/-- One labelled, colour-coded run of bits: the sign, exponent and fraction fields each get one.

Colours are given as `rgba` overlays rather than solid fills so the widget stays readable against
both light and dark editor themes. -/
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
def classify (x : ExecFloat.Binary 8 23) : String :=
  if ExecFloat.Binary.isNaN x then
    if ExecFloat.Binary.isSignalingNaN x then "sNaN" else "qNaN"
  else if ExecFloat.Binary.isInfinite x then
    if ExecFloat.Binary.signBit x then "-Inf" else "+Inf"
  else if ExecFloat.Binary.isZero x then
    if ExecFloat.Binary.signBit x then "-0" else "+0"
  else if (ExecFloat.Binary.toModel x).expField == 0 then
    "subnormal"
  else
    "normal"

/-- Render a dyadic rational as `±mantissa * 2^exponent`.

The significand and exponent come from FloatLib's decoded dyadic value; the sign is kept explicit
even for zero. -/
def dyadicString (d : FloatLib.Numerics.Dyadic) : String :=
  let sign := if d.negative then "-" else "+"
  s!"{sign}{d.significand} * 2^{d.exponent}"

/-- Render an executable float32 (`ExecFloat.Binary 8 23`) as HTML. -/
def html (x : ExecFloat.Binary 8 23) : ProofWidgets.Html :=
  let b := ExecFloat.Binary.toBits32 x
  let s := ExecFloat.Binary.signBit x
  let e := (ExecFloat.Binary.toModel x).expField
  let f := (ExecFloat.Binary.toModel x).fracField
  let cls := classify x
  let f64 := ExecFloat.Binary.toFloat x
  let dyadic? := (ExecFloat.Binary.toModel x).toDyadic?
  let signBits := if s then "1" else "0"
  let expBits := bitsFixed 8 e
  let fracBits := bitsFixed 23 f;
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
      {pill "FloatLib binary32"} {pill s!"class={cls}"} {pill s!"bits={hex b.toBitVec}"} {pill
        s!"display(binary64)={toString f64}"}
    </div>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {bitPill "sign" signBits "rgba(255, 80, 80, 0.18)"}
      {bitPill "exp" expBits "rgba(80, 160, 255, 0.18)"}
      {bitPill "frac" fracBits "rgba(0, 200, 120, 0.14)"}
    </div>
    <div style={json% {"display": "flex", "gap": "6px", "flex-wrap": "wrap"}}>
      {flagBadge "NaN" (ExecFloat.Binary.isNaN x)} {flagBadge "Inf" (ExecFloat.Binary.isInfinite x)}
      {flagBadge "Zero" (ExecFloat.Binary.isZero x)} {flagBadge "Finite" (ExecFloat.Binary.isFinite
        x)}
      {flagBadge "qNaN" (ExecFloat.Binary.isQuietNaN x)} {flagBadge "sNaN"
        (ExecFloat.Binary.isSignalingNaN x)}
    </div>
    <details style={json% {"margin-top": "10px"}} «open»={false}>
      <summary>{.text "Raw fields"}</summary>
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "6px",
        "margin-top": "8px"}}>
        <div><b>toBits:</b> {monospace (reprStr (ExecFloat.Binary.toBits32 x))}</div>
        <div><b>signBit:</b> {monospace (reprStr (ExecFloat.Binary.signBit x))}</div>
        <div><b>expField:</b> {monospace (reprStr ((ExecFloat.Binary.toModel x).expField))}</div>
        <div><b>fracField:</b> {monospace (reprStr ((ExecFloat.Binary.toModel x).fracField))}</div>
        <div><b>quietBit:</b> {monospace (reprStr ((ExecFloat.Binary.toBits32 x &&& 0x00400000) !=
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
Element renderer for `ExecFloat.Binary 8 23` used by `#tensor_view`.

Renders the float value, with a tooltip that includes a small classification and the raw bit
pattern.
-/
instance : TensorElemView (ExecFloat.Binary 8 23) :=
  ⟨fun x =>
    let b := ExecFloat.Binary.toBits32 x
    let cls := classify x
    let v := ExecFloat.Binary.toFloat x
    let title := s!"class={cls}\nbits={hex b.toBitVec}\ndisplay(binary64)={toString v}";
    <span title={title}>{monospace (toString v)}</span>⟩


/-- Compare two `ExecFloat.Binary 8 23` values at the bit level and render the results as HTML. -/
def compareHtml (x y : ExecFloat.Binary 8 23) : ProofWidgets.Html :=
  let bx := ExecFloat.Binary.toBits32 x
  let byBits := ExecFloat.Binary.toBits32 y
  let diff := bx ^^^ byBits
  let same := decide (bx = byBits);
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "binary32 compare"} {pill s!"sameBits={same}"} {pill s!"xor={hex diff.toBitVec}"}
    </div>
    <div style={json% {"display": "grid",
      "grid-template-columns": "repeat(auto-fit, minmax(min(100%, 340px), 1fr))",
      "gap": "10px"}}>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "x"}</div>
        {html x}
      </div>
      <div>
        <div style={json% {"margin-bottom": "6px"}}>{pill "y"}</div>
        {html y}
      </div>
    </div>
  </div>

/-- Show how a Lean `Float` (binary64) rounds to an executable float32 (`ExecFloat.Binary 8 23`,
binary32).

The result is widened to native binary64 for its decimal display. The displayed difference is
computed by native binary64 subtraction; it is not an exact-real error certificate. -/
def roundHtml (x : Float) : ProofWidgets.Html :=
  let b64 : UInt64 := x.toBits
  let x32 : ExecFloat.Binary 8 23 := ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary64 FloatFormat.binary32
      (ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat x : FloatLib.Floats.Binary64))
  let y := ExecFloat.Binary.toFloat x32
  let err : Float := y - x;
  <span style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <span style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "native binary64 → FloatLib binary32"}
      {pill s!"input(native binary64)={toString x}"}
      {pill s!"inputBits={hex b64.toBitVec}"}
      {pill s!"rounded(widened to binary64)={toString y}"}
      {pill s!"difference(binary64)={toString err}"}
    </span>
    {html x32}
  </span>

end Float32

/-!
## Commands
-/

syntax (name := float32ViewCmd) "#float32_view " term : command

macro "#float32_view " x:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.html $x))

syntax (name := float32RoundViewCmd) "#float32_round_view " term : command

macro "#float32_round_view " x:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.roundHtml $x))

syntax (name := float32CompareViewCmd) "#float32_compare_view " term ", " term : command

macro "#float32_compare_view " x:term ", " y:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.compareHtml $x $y))

end NN.Widgets
