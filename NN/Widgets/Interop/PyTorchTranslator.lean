/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# PyTorch Translator Widget

This file implements the editor-side "write PyTorch, see TorchLean" translator workflow.

The goal is deliberately modest and honest:

- accept a Python source file, with a lower-level command for selected `nn.Sequential` /
  `nn.Module` text;
- recognize common layer constructors by name;
- emit a TorchLean skeleton using the public `nn.sequential!` style;
- report the exact boundary between translated layers, layers that need extra shape information,
  and Python code that is outside this lightweight supported-subset assistant.

This is **not** a verified Python parser and it is not a full PyTorch semantic import. For full
graph capture, the right path is still the existing `torch.export` JSON bridge in
`NN.Runtime.PyTorch.Import.TorchExport`. This widget is the fast, friendly "front door" that shows
whether a model is close to the supported TorchLean subset before users commit to the full
capture/import path.

The VS Code extension version can reuse this design:

1. Run a stronger Python-side analyzer (`ast`, `torch.fx`, or `torch.export`).
2. Send the normalized layer/graph report to Lean.
3. Display the same kind of TorchLean skeleton plus trust-boundary diagnostics.

For the in-repo workflow, prefer `#pytorch_translate_file "path/to/model.py"`. That command reads a
real Python source file and renders the report in the Lean infoview. The lower-level
`#pytorch_translate_view someString` command remains useful for tests and for future editor
integrations that already have selected text in memory.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets
namespace PyTorchTranslator

open UI
open Lean Elab Command

/--
A layer shape recognized by the file-based PyTorch supported-subset analyzer.

The constructors intentionally describe *semantic layer families*, not exact Python AST nodes. For
example, `linear 784 128` can come from `nn.Linear(784, 128)` inside `nn.Sequential`, a field
assignment such as `self.fc = nn.Linear(784, 128)`, or a compact documentation snippet. This is what
makes the widget useful for quick editor feedback, while still keeping it honest: anything outside
this small layer vocabulary is represented as `unsupported` and shown to the user as a boundary.
-/
inductive Layer where
  /-- Fully connected layer with numeric `in_features` and `out_features`. -/
  | linear (inDim outDim : Nat)
  /--
  2D convolution with the fields this text-level assistant can reliably infer from common snippets.

  We do not generate executable TorchLean code directly from this constructor because a correct
  `nn.conv` term also needs the input contract: batch size, input height, and input width.
  -/
  | conv2d (inC outC kernel stride padding : Nat)
  /-- 2D max-pooling metadata; executable lowering likewise needs the surrounding image shape. -/
  | maxPool2d (kernel stride : Nat)
  /-- Adaptive average pooling is detected as a named boundary item. -/
  | adaptiveAvgPool2d (out : Nat)
  /-- Flatten layer; translated to `nn.flatten` in vector-style sequential skeletons. -/
  | flatten
  /-- Elementwise ReLU. -/
  | relu
  /-- Elementwise GELU. -/
  | gelu
  /-- Elementwise sigmoid. -/
  | sigmoid
  /-- Elementwise tanh. -/
  | tanh
  /-- Dropout is recognized, but emitted as a boundary comment because mode/seed must be explicit. -/
  | dropout
  /-- A line that looks relevant to PyTorch but is outside the supported translator subset. -/
  | unsupported (raw reason : String)
  deriving Repr, Inhabited

/--
Summary produced by the file-based supported-subset analyzer.

`layers` preserves the order in which relevant lines appear, including unsupported boundary items.
`translated` counts only recognized layer-family rows; `warnings` are global advice such as "this
CNN snippet needs an input image shape"; `unsupported` is a compact list for the red diagnostic
section in the widget.
-/
structure Report where
  /-- Ordered layer/boundary rows extracted from the snippet. -/
  layers : Array Layer := #[]
  /-- Count of rows recognized as part of the supported layer vocabulary. -/
  translated : Nat := 0
  /-- Human-facing warnings about missing shape contracts or mode choices. -/
  warnings : Array String := #[]
  /-- Unsupported PyTorch-looking lines, each paired with the reason it was not translated. -/
  unsupported : Array String := #[]
  deriving Repr, Inhabited

/--
Small substring predicate used by the heuristic parser.

Lean's core string API is enough for this lightweight assistant. A VS Code extension should use
`ast`, `torch.fx`, or `torch.export` on the Python side rather than substring matching.
-/
private def hasSubstr (s needle : String) : Bool :=
  (s.splitOn needle).length > 1

/--
Normalize a source line before matching constructors.

Common PyTorch snippets write layers as assignments:

```python
self.fc1 = nn.Linear(784, 128)
```

The widget only needs the right-hand constructor, so this helper strips a simple assignment prefix
and trims whitespace. It deliberately does not try to understand arbitrary Python expressions.
-/
private def lineClean (s : String) : String :=
  let s := s.trimAscii.toString
  -- Drop the common `self.foo =` or `foo =` prefix so constructor matching is stable.
  match s.splitOn "=" with
  | _lhs :: rhs :: _ => rhs.trimAscii.toString
  | _ => s

/-- Convert an ASCII digit character to its numeric value. Called only after `Char.isDigit`. -/
private def digitVal (c : Char) : Nat :=
  c.toNat - '0'.toNat

/--
Extract decimal natural numbers from a constructor line.

This is enough for layer signatures like `Linear(784, 128)` and `Conv2d(3, 64, 7, stride=2,
padding=3)`. It intentionally ignores floating literals, symbolic dimensions, tuples with different
height/width values, and keyword names; those cases are better handled by the real graph-capture
path. The payoff is that the widget stays total, fast, and easy to inspect.
-/
private def numbersInString (s : String) : Array Nat :=
  let finish (acc : Array Nat) (cur? : Option Nat) : Array Nat :=
    match cur? with
    | some n => acc.push n
    | none => acc
  let (acc, cur?) :=
    s.toList.foldl
      (fun (state : Array Nat × Option Nat) c =>
        let (acc, cur?) := state
        if c.isDigit then
          let d := digitVal c
          let n := match cur? with | some n => n * 10 + d | none => d
          (acc, some n)
        else
          (finish acc cur?, none))
      (#[], none)
  finish acc cur?

private def nth? (xs : Array Nat) (i : Nat) : Option Nat :=
  if h : i < xs.size then some xs[i] else none

/-- Whether a layer row belongs to the recognized vocabulary rather than the unsupported bucket. -/
private def supported (l : Layer) : Bool :=
  match l with
  | .unsupported _ _ => false
  | _ => true

/-- Human-readable layer label used in the report table. -/
private def layerName : Layer → String
  | .linear i o => s!"Linear({i}, {o})"
  | .conv2d i o k s p => s!"Conv2d({i}, {o}, kernel={k}, stride={s}, padding={p})"
  | .maxPool2d k s => s!"MaxPool2d(kernel={k}, stride={s})"
  | .adaptiveAvgPool2d o => s!"AdaptiveAvgPool2d({o})"
  | .flatten => "Flatten"
  | .relu => "ReLU"
  | .gelu => "GELU"
  | .sigmoid => "Sigmoid"
  | .tanh => "Tanh"
  | .dropout => "Dropout"
  | .unsupported raw _ => s!"Unsupported: {raw}"

/--
Render a layer as a direct `nn.sequential!` term when that is safe for the supported subset.

Only vector-shaped elementwise and linear layers are emitted directly. Shape-changing CNN pieces are
not silently guessed, because that would create exactly the kind of misleading "it translated!"
experience TorchLean should avoid.
-/
private def layerTorchLeanTerm? : Layer → Option String
  | .linear i o => some s!"nn.linear {i} {o} (pfx := Spec.Shape.scalar)"
  | .flatten => some "nn.flatten"
  | .relu => some "nn.relu"
  | .gelu => some "nn.gelu"
  | .sigmoid => some "nn.sigmoid"
  | .tanh => some "nn.tanh"
  | _ => none

/--
Render the non-direct pieces as comments in the generated skeleton.

These comments are part of the user-facing translator output. A user should be able to paste the skeleton into a
Lean file and immediately see which information is still missing: image shape, dropout probability
and seed, adaptive-pooling semantics, or an unsupported PyTorch operation.
-/
private def layerBoundaryComment? : Layer → Option String
  | .conv2d i o k s p =>
      some <| s!"-- Conv2d({i}, {o}, kernel_size={k}, stride={s}, padding={p}) detected: " ++
        "add `nn.conv` after choosing `batch`, `inH`, and `inW`."
  | .maxPool2d k s =>
      some s!"-- MaxPool2d(kernel_size={k}, stride={s}) detected: add pooling after choosing channel/spatial dimensions."
  | .adaptiveAvgPool2d o =>
      some s!"-- AdaptiveAvgPool2d({o}) detected: connect this to the specific TorchLean pooling spec you want."
  | .dropout =>
      some "-- Dropout detected: add `nn.dropout p (seed := seed)` after making `p` and mode behavior explicit."
  | .unsupported raw reason =>
      some s!"-- Unsupported PyTorch line: {raw} ({reason})"
  | _ => none

/--
Analyze one source line.

The result has three possible meanings:

- `some layer`: a supported layer or explicit unsupported boundary was found;
- `none`: the line is ordinary Python structure (`class`, `def forward`, `return`, imports, etc.)
  or blank/comment text that should not become a report row;
- `some (.unsupported ...)`: the line looks like a PyTorch operation but is outside the supported
  translator subset.

This distinction keeps the report readable. We do not want to complain about every `class` or
`return`, but we do want to flag `nn.BatchNorm2d` or `torch.reshape` if the user expected a model
translation.
-/
private def analyzeLine (raw : String) : Option Layer :=
  let s := lineClean raw
  if s.isEmpty || s.startsWith "#" then
    none
  else if hasSubstr s "nn.Linear" || hasSubstr s "Linear(" then
    let ns := numbersInString s
    match nth? ns 0, nth? ns 1 with
    | some i, some o => some (.linear i o)
    | _, _ => some (.unsupported s "Linear needs numeric in_features and out_features")
  else if hasSubstr s "nn.Conv2d" || hasSubstr s "Conv2d(" then
    let ns := numbersInString s
    match nth? ns 0, nth? ns 1, nth? ns 2 with
    | some i, some o, some k =>
        let stride := (nth? ns 3).getD 1
        let padding := (nth? ns 4).getD 0
        some (.conv2d i o k stride padding)
    | _, _, _ => some (.unsupported s "Conv2d needs numeric in_channels, out_channels, kernel_size")
  else if hasSubstr s "nn.MaxPool2d" || hasSubstr s "MaxPool2d(" then
    let ns := numbersInString s
    match nth? ns 0 with
    | some k =>
        let stride := (nth? ns 1).getD k
        some (.maxPool2d k stride)
    | none => some (.unsupported s "MaxPool2d needs a numeric kernel_size")
  else if hasSubstr s "nn.AdaptiveAvgPool2d" || hasSubstr s "AdaptiveAvgPool2d(" then
    match nth? (numbersInString s) 0 with
    | some o => some (.adaptiveAvgPool2d o)
    | none => some (.unsupported s "AdaptiveAvgPool2d needs a numeric output size")
  else if hasSubstr s "nn.Flatten" || hasSubstr s "torch.flatten" || hasSubstr s ".flatten(" then
    some .flatten
  else if hasSubstr s "nn.ReLU" || hasSubstr s "F.relu" || hasSubstr s ".relu(" then
    some .relu
  else if hasSubstr s "nn.GELU" || hasSubstr s "F.gelu" then
    some .gelu
  else if hasSubstr s "nn.Sigmoid" || hasSubstr s "torch.sigmoid" then
    some .sigmoid
  else if hasSubstr s "nn.Tanh" || hasSubstr s "torch.tanh" then
    some .tanh
  else if hasSubstr s "nn.Dropout" || hasSubstr s "Dropout(" then
    some .dropout
  else if hasSubstr s "def forward" || hasSubstr s "class " || hasSubstr s "super().__init__" ||
      hasSubstr s "return " || hasSubstr s "import " || hasSubstr s "from " ||
      hasSubstr s "nn.Sequential" || s = ")" || s = "]" || s = "}" then
    none
  else if hasSubstr s "nn." || hasSubstr s "torch." || hasSubstr s "F." then
    some (.unsupported s "not in the supported translator layer subset")
  else
    none

/--
Analyze PyTorch source text using the small supported layer subset.

The analyzer is order-preserving and fail-soft: one unsupported line does not prevent later lines
from being recognized. That matters for editor UX, because users should still get a useful partial
skeleton even when one layer needs manual handling.
-/
def analyze (snippet : String) : Report :=
  let layers := (snippet.splitOn "\n").foldl
    (fun acc line =>
      match analyzeLine line with
      | some l => acc.push l
      | none => acc)
    #[]
  let translated := layers.foldl (fun n l => if supported l then n + 1 else n) 0
  let unsupported := layers.foldl
    (fun acc l =>
      match l with
      | .unsupported raw reason => acc.push s!"{raw}: {reason}"
      | _ => acc)
    #[]
  let warnings : Array String := Id.run do
    let mut warnings : Array String := #[]
    if layers.any (fun l => match l with | .conv2d .. => true | .maxPool2d .. => true | _ => false) then
      warnings := warnings.push
        "CNN layers need an explicit input contract (`batch`, channels, height, width) before the \
        generated TorchLean skeleton can be made executable."
    if layers.any (fun l => match l with | .dropout => true | _ => false) then
      warnings := warnings.push
        "Dropout is mode-dependent; TorchLean asks for an explicit probability/seed and keeps train/eval \
        behavior visible."
    if layers.any (fun l => match l with | .adaptiveAvgPool2d .. => true | _ => false) then
      warnings := warnings.push
        "Adaptive pooling is detected as a shape-changing operation; connect it to the specific \
        TorchLean pooling spec you want before treating the skeleton as executable."
    pure warnings
  { layers, translated, warnings, unsupported }

private def joinLines (xs : List String) : String :=
  String.intercalate "\n" xs

/--
Generate a TorchLean skeleton from the recognized layer sequence.

The emitted code is meant to be a starting point, not a final theorem. It imports the public
TorchLean umbrella, opens the user-facing API namespaces, emits direct sequential terms for the safe
subset, and then appends boundary notes as Lean comments. The next intended step is to add a concrete
shape contract and wrap the model in a `train.Task` / `SeqTask`.
-/
def torchLeanSkeleton (r : Report) (name : String := "translatedModel") : String :=
  let translatedLines :=
    r.layers.toList.filterMap layerTorchLeanTerm?
  let body :=
    match translatedLines with
    | [] => "    -- No directly translatable sequential terms were recognized."
    | first :: rest =>
        joinLines <| ("    " ++ first) :: rest.map (fun line => "  , " ++ line)
  let boundaryComments := r.layers.toList.filterMap layerBoundaryComment?
  let boundaryBlock :=
    if boundaryComments.isEmpty then
      "-- Boundary notes: none for this supported translator subset."
    else
      joinLines ("-- Boundary notes:" :: boundaryComments)
  joinLines [
    "import NN",
    "",
    "open NN.API",
    "open Spec",
    "",
    s!"def {name} :=",
    "  nn.sequential![",
    body,
    "  ]",
    "",
    boundaryBlock,
    "",
    "-- Next steps:",
    "-- 1. Add the concrete input/output shape contract.",
    "-- 2. Choose a loss and wrap this in a `train.Task` / `SeqTask`.",
    "-- 3. If this came from a real PyTorch module, use `torch.export` capture for a checked graph path."
  ]

/-- Render one recognized/unsupported layer row in the HTML report table. -/
private def layerRowHtml (l : Layer) : ProofWidgets.Html :=
  let badge := if supported l then okBadge "recognized" else warnBadge "unsupported"
  let detail :=
    match l with
    | .unsupported _ reason => reason
    | .conv2d .. => "recognized, but executable lowering needs image shape metadata"
    | .maxPool2d .. => "recognized, but executable lowering needs image shape metadata"
    | .adaptiveAvgPool2d .. => "recognized as a boundary item"
    | .dropout => "recognized; probability/seed must be explicit"
    | _ => "direct sequential skeleton"
  ;
  <tr>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {badge}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {monospace (layerName l)}
    </td>
    <td style={json% {"padding": "6px 8px", "border-bottom": "1px solid rgba(127,127,127,0.18)"}}>
      {.text detail}
    </td>
  </tr>

/--
Render a compact warning/error list.

The empty case returns an empty `div` rather than an optional HTML value so the caller can compose
panels in the JSX block without extra branching noise.
-/
private def msgListHtml (title : String) (msgs : Array String) (kind : String) : ProofWidgets.Html :=
  if msgs.isEmpty then
    <div></div>
  else
    let badge := if kind = "warn" then warnBadge title else errBadge title
    let rows := msgs.map (fun msg =>
      <li style={json% {"margin": "4px 0"}}>{.text msg}</li>)
    ;
    <div style={json% {"margin-top": "10px"}}>
      {badge}
      <ul style={json% {"margin-top": "6px"}}>
        {...rows}
      </ul>
    </div>

/--
Render the translator report as an infoview panel.

The panel has four sections:

1. badges that summarize the number of recognized rows;
2. a row-by-row layer table;
3. warnings / unsupported diagnostics;
4. a generated Lean skeleton plus a trust-boundary explanation.

That layout mirrors the intended editor-facing view: useful generated code beside an
equally visible account of what has *not* been checked.
-/
def html (snippet : String) : ProofWidgets.Html :=
  let r := analyze snippet
  let rows := r.layers.map layerRowHtml
  let skeleton := torchLeanSkeleton r
  let allSupported := r.unsupported.isEmpty && !r.layers.isEmpty
  ;
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom": "10px"}}>
      {pill "PyTorch -> TorchLean"}
      {pill s!"layers={r.layers.size}"}
      {pill s!"translated={r.translated}"}
      {if allSupported then okBadge "supported subset" else warnBadge "boundary report"}
    </div>
    <p style={json% {"margin": "0 0 10px 0"}}>
      {.text "Lightweight supported-subset assistant for common PyTorch layer stacks. It generates a TorchLean skeleton and names the parts that still need shape contracts or the full torch.export path."}
    </p>
    <details «open»={true}>
      <summary>{.text "Recognized layers"}</summary>
      <table style={json% {"border-collapse": "collapse", "margin-top": "8px", "width": "100%"}}>
        <thead>
          <tr>
            <th style={json% {"text-align": "left", "padding": "4px 8px"}}>{.text "status"}</th>
            <th style={json% {"text-align": "left", "padding": "4px 8px"}}>{.text "layer"}</th>
            <th style={json% {"text-align": "left", "padding": "4px 8px"}}>{.text "meaning"}</th>
          </tr>
        </thead>
        <tbody>{...rows}</tbody>
      </table>
    </details>
    {msgListHtml "warnings" r.warnings "warn"}
    {msgListHtml "unsupported" r.unsupported "err"}
    <details «open»={true} style={json% {"margin-top": "10px"}}>
      <summary>{.text "Generated TorchLean skeleton"}</summary>
      <pre style={json% {
        "white-space": "pre",
        "overflow-x": "auto",
        "margin-top": "6px",
        "padding": "8px",
        "border-radius": "8px",
        "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
        "background": "var(--vscode-textCodeBlock-background, rgba(127,127,127,0.12))"
      }}>{.text skeleton}</pre>
    </details>
    <details style={json% {"margin-top": "10px"}}>
      <summary>{.text "Trust boundary"}</summary>
      <ul>
        <li>{.text "This widget is a heuristic editor assistant, not a proof about arbitrary Python."}</li>
        <li>{.text "A skeleton becomes executable only after you add the typed input/output shape contract."}</li>
        <li>{.text "For real PyTorch modules, use the existing torch.export JSON bridge to capture and validate the graph."}</li>
      </ul>
    </details>
  </div>

syntax (name := pytorchTranslateViewCmd) "#pytorch_translate_view " term : command

/--
Low-level command frontend for already-selected source text.

This command accepts a Lean `String` term. It is mostly a hook for tests and future editor
integrations that already have selected Python text in memory. For normal in-repo use, prefer
`#pytorch_translate_file`, which reads a real `.py` file.

Usage:

```lean
def snippet : String :=
  "nn.Linear(784, 128)\n" ++
  "nn.ReLU()\n"
#pytorch_translate_view snippet
```

The argument is a Lean term of type `String`, so examples can define reusable snippets rather than
putting large multi-line strings directly in the command.
-/
macro "#pytorch_translate_view " snippet:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (_root_.NN.Widgets.PyTorchTranslator.html $snippet))

private def fileErrorHtml (path msg : String) : ProofWidgets.Html :=
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    {errBadge "file error"} {pill path}
    <p>{.text "The PyTorch translator widget could not read this file."}</p>
    <pre style={json% {
      "white-space": "pre-wrap",
      "padding": "8px",
      "border-radius": "8px",
      "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
      "background": "var(--vscode-textCodeBlock-background, rgba(127,127,127,0.12))"
    }}>{.text msg}</pre>
  </div>

/--
Read a Python source file and render the translator report.

This is the practical in-repo workflow:

```lean
#pytorch_translate_file "NN/Examples/Quickstart/pytorch_translator_mlp.py"
```

The command runs during elaboration, reads the file relative to the current Lake working directory,
and displays the same report as `#pytorch_translate_view`. If the file is missing, the Lean build
does not crash with an opaque IO exception; the widget renders an explicit file-error panel instead.
-/
def htmlFromFile (path : String) : CommandElabM ProofWidgets.Html := do
  try
    let source ← liftIO <| IO.FS.readFile path
    pure (html source)
  catch _ =>
    pure (fileErrorHtml path "IO.FS.readFile failed. Check that the path is relative to the Lake project root and that the file exists.")

syntax (name := pytorchTranslateFileCmd) "#pytorch_translate_file " str : command

/--
Command frontend that reads a `.py` file and renders the translator widget.

This is still not a proof about Python. It is the file-based translator widget over real source
text. For checked model import, use the existing `torch.export` JSON bridge after the report tells
you the model is close to the supported subset.
-/
macro "#pytorch_translate_file " path:str : command =>
  Lean.TSyntax.mkInfoCanonical <$>
    `(#html (_root_.NN.Widgets.PyTorchTranslator.htmlFromFile $path))

end PyTorchTranslator
end NN.Widgets
