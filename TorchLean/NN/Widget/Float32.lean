/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Classification.Runtime
public meta import TorchLean.NN.Widget.Tensor
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured -- shake: keep
public import TorchLean.NN.Widget.Tensor
import ProofWidgets.Component.HtmlDisplay

/-!
# Float32

HTML views of a configured binary value `x : ExecFloat (Configured.Family format code plan)` with a
`ModelCodec plan (Model format) code` instance, such as any `ExecFloat.Binary` value.

## Main definitions

- `Float32.html`: the view of one value.
- `Float32.compareHtml`: two views side by side with the xor of the interchange words.
- `Float32.roundHtml`: a Lean `Float` rounded to nearest-even in a given format.
- `#float32_view`, `#float32_compare_view`, `#float32_round_view`: these views at binary32.
-/

public meta section

open scoped ProofWidgets.Jsx

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.Widget

open UI

namespace Float32

/-- Render a bit vector in hexadecimal, preserving the width encoded by its type. -/
def hex {width : Nat} (bits : BitVec width) : String :=
  "0x" ++ bits.toHex

/-- The `width` low-order bits of `n` as a binary string, most significant bit first. -/
def bitsPadded (width : Nat) (n : Nat) : String :=
  let rec go (i : Nat) (acc : List Char) : List Char :=
    match i with
    | 0 => acc
    | i + 1 =>
        let j := i
        let c := if Nat.testBit n j then '1' else '0'
        go i (c :: acc)
  String.ofList (go width []).reverse

/-- One labelled, colour-coded run of bits: the sign, exponent and fraction fields each get one.

The colours are translucent `rgba` overlays, so the bits stay readable in light and dark editor
themes. -/
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

/-- The name of `format` in FloatLib's catalog, such as `binary32` or `e4m3fn`, and otherwise
`e<E>m<M>` from its exponent and fraction widths. -/
def formatName (format : FloatFormat) : String :=
  let catalog : List (FloatFormat × String) :=
    [(.binary16, "binary16"), (.bfloat16, "bfloat16"), (.binary32, "binary32"),
      (.binary64, "binary64"), (.binary128, "binary128"), (.binary256, "binary256"),
      (.tf32, "tf32"), (.e5m2, "e5m2"), (.e4m3fn, "e4m3fn"), (.e4m3fnuz, "e4m3fnuz"),
      (.e5m2fnuz, "e5m2fnuz"), (.e2m1, "e2m1"), (.e2m3, "e2m3"), (.e3m2, "e3m2")]
  (catalog.lookup format).getD s!"e{format.expWidth}m{format.fracWidth}"

section

variable {format : FloatFormat}
    {plan : FloatLib.Floats.Formats.BinaryInterchange.Configured.StoragePlan format}
    {code : Type} [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" =>
  ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan)

/-- The `Float` the widgets display for `x`: `x` cast to binary64 by `Model.cast`, which rounds to
nearest-even when binary64 cannot hold `x`. -/
def displayFloat (x : Value) : Float :=
  ExecFloat.Binary.toFloat
    (ExecFloat.Binary.ofModel (Model.cast format .binary64 (ExecFloat.Binary.toModel x)) :
      FloatLib.Floats.Binary64)

/-- The label of the IEEE class `ExecFloat.Binary.classify x`: `sNaN`, `qNaN`, `-Inf`, `+Inf`,
`-0`, `+0`, and `subnormal` or `normal` for either sign. -/
def classify (x : Value) : String :=
  match ExecFloat.Binary.classify x with
  | .signalingNaN => "sNaN"
  | .quietNaN => "qNaN"
  | .negativeInfinity => "-Inf"
  | .positiveInfinity => "+Inf"
  | .negativeZero => "-0"
  | .positiveZero => "+0"
  | .negativeSubnormal | .positiveSubnormal => "subnormal"
  | .negativeNormal | .positiveNormal => "normal"

/-- Render a dyadic rational as `±significand * 2^exponent`, writing the sign even for zero. -/
def dyadicString (d : FloatLib.Numerics.Dyadic) : String :=
  let sign := if d.negative then "-" else "+"
  s!"{sign}{d.significand} * 2^{d.exponent}"

/-- HTML view of `x`: its class, interchange word, sign, exponent and fraction bits, FloatLib's
class flags, the raw fields, and the exact dyadic value when `x` is finite. -/
def html (x : Value) : ProofWidgets.Html :=
  let m := ExecFloat.Binary.toModel x
  let s := ExecFloat.Binary.signBit x
  let e := m.expField
  let f := m.fracField
  let cls := classify x
  let f64 := displayFloat x
  let dyadic? := m.toDyadic?
  let signBits := if s then "1" else "0"
  let expBits := bitsPadded format.expWidth e
  let fracBits := bitsPadded format.fracWidth f;
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
      {pill s!"FloatLib {formatName format}"} {pill s!"class={cls}"} {pill s!"bits={hex m.toBits}"}
      {pill s!"display(binary64)={toString f64}"}
    </div>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {bitPill "sign" signBits "rgba(255, 80, 80, 0.18)"}
      {bitPill "exp" expBits "rgba(80, 160, 255, 0.18)"}
      {bitPill "frac" fracBits "rgba(0, 200, 120, 0.14)"}
    </div>
    <div style={json% {"display": "flex", "gap": "6px", "flex-wrap": "wrap"}}>
      {flagBadge "NaN" (ExecFloat.Binary.isNaN x)} {flagBadge "Inf" (ExecFloat.Binary.isInf x)}
      {flagBadge "Zero" (ExecFloat.Binary.isZero x)} {flagBadge "Finite" (ExecFloat.Binary.isFinite
        x)}
      {flagBadge "qNaN" (ExecFloat.Binary.isQNaN x)} {flagBadge "sNaN"
        (ExecFloat.Binary.isSNaN x)}
    </div>
    <details style={json% {"margin-top": "10px"}} «open»={false}>
      <summary>{.text "Raw fields"}</summary>
      <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "6px",
        "margin-top": "8px"}}>
        <div><b>toBits:</b> {monospace (reprStr m.toNatBits)}</div>
        <div><b>signBit:</b> {monospace (reprStr s)}</div>
        <div><b>expField:</b> {monospace (reprStr e)}</div>
        <div><b>fracField:</b> {monospace (reprStr f)}</div>
        <div><b>quietBit:</b> {monospace (reprStr ((m.toBits &&& FloatFormat.quietBit format) !=
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

/-- Tensor views show a configured binary entry `x` as `displayFloat x`, with its class, its
interchange word and that value in a tooltip. -/
instance : TensorElemView Value :=
  ⟨fun x =>
    let cls := classify x
    let v := displayFloat x
    let bits := hex (ExecFloat.Binary.toModel x).toBits
    let title := s!"class={cls}\nbits={bits}\ndisplay(binary64)={toString v}";
    <span title={title}>{monospace (toString v)}</span>⟩

/-- HTML views of `x` and `y` side by side, with whether their interchange words are equal and the
xor of the two words. -/
def compareHtml (x y : Value) : ProofWidgets.Html :=
  let bx := (ExecFloat.Binary.toModel x).toBits
  let byBits := (ExecFloat.Binary.toModel y).toBits
  let diff := bx ^^^ byBits
  let same := decide (bx = byBits);
  <div style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill s!"{formatName format} compare"} {pill s!"sameBits={same}"} {pill s!"xor={hex diff}"}
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

end

/-- HTML view of the Lean `Float` `x` rounded to nearest-even in `format` by
`ExecFloat.Binary.ofFloat`. The difference shown is `displayFloat` of the result minus `x` in
binary64 arithmetic, which can round. -/
def roundHtml (format : FloatFormat) (x : Float) : ProofWidgets.Html :=
  let b64 : UInt64 := x.toBits
  let rounded : ExecFloat.Binary format.expWidth format.fracWidth format.encoding
      format.exponentBias format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
      format.exponentBias_le_maxFinite := ExecFloat.Binary.ofFloat x
  let y := displayFloat rounded
  let err : Float := y - x;
  <span style={json% {"display": "grid", "grid-template-columns": "1fr", "gap": "10px"}}>
    <span style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill s!"native binary64 → FloatLib {formatName format}"}
      {pill s!"input(native binary64)={toString x}"}
      {pill s!"inputBits={hex b64.toBitVec}"}
      {pill s!"rounded(widened to binary64)={toString y}"}
      {pill s!"difference(binary64)={toString err}"}
    </span>
    {html rounded}
  </span>

end Float32

/-!
## Commands
-/

/-- `#float32_view x` shows the `Float32.html` view of `x` as a binary32 value. -/
macro "#float32_view " x:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.html ($x : FloatLib.Floats.Binary32)))

/-- `#float32_round_view x` shows the `Float32.roundHtml` view of the Lean `Float` `x` rounded to
binary32. -/
macro "#float32_round_view " x:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.roundHtml FloatFormat.binary32 $x))

/-- `#float32_compare_view x, y` shows the `Float32.compareHtml` view of `x` and `y` as binary32
values. -/
macro "#float32_compare_view " x:term ", " y:term : command =>
  UI.canonicalCommand <$> `(#html (Float32.compareHtml ($x : FloatLib.Floats.Binary32)
    ($y : FloatLib.Floats.Binary32)))

end TorchLean.Widget
