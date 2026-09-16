/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Model.RealSemantics
public import FloatLib.Floats.Formats.BinaryInterchange.Model.ERealSemantics
public import FloatLib.Floats.Formats.IEEE754.Native
public meta import NN.Widgets.Core.Tensor
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured -- shake: keep
public meta import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime -- shake: keep
public meta import FloatLib.Floats.Formats.IEEE754.Native -- shake: keep
public meta import NN.Widgets.Core.UI -- shake: keep
public meta import ProofWidgets.Component.HtmlDisplay -- shake: keep

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

- `float32Html`: inspect class/fields/bits for one `ExecFloat.Binary 8 23` value.
- `float32RoundHtml`: show `Float64 -> Float32` rounding behavior.
- `float32CompareHtml`: side-by-side bit-level comparison.
- `#float32_view`, `#float32_round_view`, `#float32_compare_view`: command entry points.
-/

public meta section

open scoped ProofWidgets.Jsx

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Widgets

open UI

namespace Float32Internal

/-- Compact rendering of a 32-bit word for the bit-pattern pills. -/
def u32Hex (u : UInt32) : String :=
  "0x" ++ u.toBitVec.toHex

/--
The 64-bit counterpart, used when a widget shows a binary64 input alongside its float32 result.
-/
def u64Hex (u : UInt64) : String :=
  "0x" ++ u.toBitVec.toHex

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
both
light and dark editor themes. -/
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

This is the exact value of a finite float, written the way Flocq and Coq's `Fappli_IEEE` write it,
so
what the widget shows can be compared directly against the proofs. -/
def dyadicString (d : FloatLib.Numerics.Dyadic) : String :=
  let sign := if d.negative then "-" else "+"
  s!"{sign}{d.significand} * 2^{d.exponent}"

end Float32Internal

open Float32Internal

/-- Render an executable float32 (`ExecFloat.Binary 8 23`) as HTML. -/
def float32Html (x : ExecFloat.Binary 8 23) : ProofWidgets.Html :=
  let b := ExecFloat.Binary.toBits32 x
  let s := ExecFloat.Binary.signBit x
  let e := (ExecFloat.Binary.toModel x).expField
  let f := (ExecFloat.Binary.toModel x).fracField
  let cls := classify x
  let f64 := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary32 FloatFormat.binary64 (ExecFloat.Binary.toModel x)
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
      {pill "FloatLib binary32"} {pill s!"class={cls}"} {pill s!"bits={u32Hex b}"} {pill
        s!"asFloat={toString f64}"}
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
    let cls := Float32Internal.classify x
    let v := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
      Model.cast FloatFormat.binary32 FloatFormat.binary64 (ExecFloat.Binary.toModel x)
    let title := s!"class={cls}\nbits={Float32Internal.u32Hex b}\nasFloat={toString v}";
    <span title={title}>{monospace (toString v)}</span>⟩

namespace Float32Internal

/-- Compare two `ExecFloat.Binary 8 23` values at the bit level and render the results as HTML. -/
def float32CompareHtml (x y : ExecFloat.Binary 8 23) : ProofWidgets.Html :=
  let bx := ExecFloat.Binary.toBits32 x
  let byBits := ExecFloat.Binary.toBits32 y
  let diff := bx ^^^ byBits
  let same := decide (bx = byBits);
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "binary32 compare"} {pill s!"sameBits={same}"} {pill s!"xor={u32Hex diff}"}
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

/-- Show how a Lean `Float` (binary64) rounds to an executable float32 (`ExecFloat.Binary 8 23`,
binary32). -/
def float32RoundHtml (x : Float) : ProofWidgets.Html :=
  let b64 : UInt64 := x.toBits
  let x32 : ExecFloat.Binary 8 23 := ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary64 FloatFormat.binary32
      (ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat x))
  let y := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary32 FloatFormat.binary64 (ExecFloat.Binary.toModel x32)
  let err : Float := y - x;
  <span style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <span style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "Float → binary32"}
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
  UI.canonicalCommand <$> `(#html (float32Html $x))

syntax (name := float32RoundViewCmd) "#float32_round_view " term : command

macro "#float32_round_view " x:term : command =>
  UI.canonicalCommand <$> `(#html (Float32Internal.float32RoundHtml $x))

syntax (name := float32CompareViewCmd) "#float32_compare_view " term ", " term : command

macro "#float32_compare_view " x:term ", " y:term : command =>
  UI.canonicalCommand <$> `(#html (Float32Internal.float32CompareHtml $x $y))

end NN.Widgets
