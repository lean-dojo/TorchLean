/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Widgets.Core.UI
public import NN.Widgets.Core.UI
import ProofWidgets.Component.HtmlDisplay

/-!
# PyTorch Translator Widget

This file implements the editor-side "write PyTorch, see TorchLean" translator workflow.

The supported scope is deliberately small:

- accept a Python source file, with a lower-level command for selected `nn.Sequential` /
  `nn.Module` text;
- recognize common layer constructors by name;
- emit a TorchLean skeleton using the public `import NN` / `nn.Sequential!` style;
- report the exact boundary between translated layers, layers that need extra shape information,
  and Python code that is outside this supported-subset assistant.

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
assignment such as `self.fc = nn.Linear(784, 128)`, or a compact documentation snippet. The widget
uses this vocabulary to give immediate editor feedback while still marking anything outside the
supported subset as `unsupported`.
-/
inductive Layer where
  /-- Fully connected layer with numeric `in_features` and `out_features`. -/
  | linear (inDim outDim : Nat)
  /-- Convolution metadata with an explicit spatial rank. -/
  | conv (rank inC outC kernel stride padding : Nat)
  /-- Max-pooling metadata with an explicit spatial rank. -/
  | maxPool (rank kernel stride : Nat)
  /-- Adaptive average pooling is detected as a rank-parameterized boundary item. -/
  | adaptiveAvgPool (rank out : Nat)
  /-- Flatten layer; its axis range needs an explicit shape contract. -/
  | flatten
  /-- Elementwise ReLU. -/
  | relu
  /-- Elementwise GELU. -/
  | gelu
  /-- Elementwise sigmoid. -/
  | sigmoid
  /-- Elementwise tanh. -/
  | tanh
  /-- Dropout is recognized, but emitted as a boundary comment because mode/seed must be
  explicit. -/
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

Lean's core string API is enough for this bounded-scope assistant. A VS Code extension should use
`ast`, `torch.fx`, or `torch.export` on the Python side rather than substring matching.
-/
private def hasSubstr (s needle : String) : Bool :=
  (s.splitOn needle).length > 1

/-- Match a complete call name, so `ReLU6` is not mistaken for `ReLU`. -/
private def hasCall (s name : String) : Bool :=
  let head := ((s.splitOn "(").headD s).trimAscii.toString
  head == name || head.endsWith ("." ++ name) || head.endsWith (" " ++ name)

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
  if s.startsWith "#" then s else
  -- Drop the common `self.foo =` or `foo =` prefix so constructor matching is stable.
  match s.splitOn "=" with
  | lhs :: rhs =>
      if hasSubstr lhs "(" then s else (String.intercalate "=" rhs).trimAscii.toString
  | _ => s

/-- Read natural-valued positional or named arguments from one simple constructor call.

Nested tuples, symbolic dimensions, negative or fractional values, duplicate keywords, and unknown
options are rejected rather than guessed. Constructor-name digits are outside the argument list.
-/
private def constructorNatArgs? (s : String) (names : Array String) :
    Option (Array (Option Nat)) := do
  let [_head, body] := s.splitOn "(" | none
  let [args, tail] := body.splitOn ")" | none
  unless tail.trimAscii.toString ∈ ["", ","] do none
  let mut values : Array (Option Nat) := Array.replicate names.size none
  let mut positional : Nat := 0
  let mut sawKeyword := false
  for raw in args.splitOn "," do
    let raw := raw.trimAscii.toString
    if raw.isEmpty then none
    let (index, value) ←
      match raw.splitOn "=" with
      | [value] => do
          if sawKeyword then none
          let index := positional
          positional := positional + 1
          pure (index, value)
      | [key, value] => do
          sawKeyword := true
          let index ← names.findIdx? (· == key.trimAscii.toString)
          pure (index, value)
      | _ => none
    let .none ← values[index]? | none
    let number ← value.trimAscii.toString.toNat?
    values := values.set! index (some number)
  pure values

/-- Whether a layer row belongs to the recognized vocabulary rather than the unsupported bucket. -/
private def supported (l : Layer) : Bool :=
  match l with
  | .unsupported _ _ => false
  | _ => true

/-- Layer label used in the report table. -/
private def layerName : Layer → String
  | .linear i o => s!"Linear({i}, {o})"
  | .conv d i o k s p =>
      s!"Conv(rank={d}, in={i}, out={o}, kernel={k}, stride={s}, padding={p})"
  | .maxPool d k s => s!"MaxPool(rank={d}, kernel={k}, stride={s})"
  | .adaptiveAvgPool d o => s!"AdaptiveAvgPool(rank={d}, output={o})"
  | .flatten => "Flatten"
  | .relu => "ReLU"
  | .gelu => "GELU"
  | .sigmoid => "Sigmoid"
  | .tanh => "Tanh"
  | .dropout => "Dropout"
  | .unsupported raw _ => s!"Unsupported: {raw}"

/--
Render a layer as a direct `nn.Sequential!` term when that is safe for the supported subset.

Only vector-shaped elementwise and linear layers are emitted directly. Shape-changing CNN pieces are
not silently guessed, because that would create exactly the kind of misleading "it translated!"
experience TorchLean should avoid.
-/
private def layerTorchLeanTerm? : Layer → Option String
  | .linear i o => some s!"nn.linear {i} {o}"
  | .relu => some "nn.relu"
  | .sigmoid => some "nn.sigmoid"
  | .tanh => some "nn.tanh"
  | _ => none

/--
Render the non-direct pieces as comments in the generated skeleton.

These comments are part of the user-facing translator output. A user should be able to paste the
skeleton into a Lean file and immediately see which information is still missing: image shape,
dropout probability and seed, adaptive-pooling semantics, or an unsupported PyTorch operation.
-/
private def layerBoundaryComment? : Layer → Option String
  | .flatten =>
      some "-- Flatten detected: choose `nn.flatten` or `nn.flattenAfter` after checking the \
        source axis range and batch shape."
  | .gelu =>
      some "-- GELU detected: `nn.gelu` uses the tanh approximation. Check the source \
        approximation before adding it."
  | .conv d i o k s p =>
      some <| s!"-- Conv(rank={d}, in={i}, out={o}, kernel={k}, stride={s}, padding={p}) " ++
        "detected: add `nn.conv` after choosing the input spatial vector."
  | .maxPool d k s =>
      some <| s!"-- MaxPool(rank={d}, kernel={k}, stride={s}) detected: add `nn.maxPool` " ++
        "after choosing the channel count and input spatial vector."
  | .adaptiveAvgPool d o =>
      some <| s!"-- AdaptiveAvgPool(rank={d}, output={o}) detected: connect it to the " ++
        "rank-polymorphic pooling operation required by the model."
  | .dropout =>
      some "-- Dropout detected: choose `p`, add `nn.dropout p`, and build the model with \
        `nn.build seed` after checking train/eval behavior."
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

Ordinary Python structure is omitted from the report. Unsupported PyTorch operations such as
`nn.BatchNorm2d` or `torch.reshape` are retained because they prevent a complete model translation.
-/
private def analyzeLine (raw : String) : Option Layer :=
  let s := lineClean raw
  if s.isEmpty || s.startsWith "#" then
    none
  else if hasCall s "nn.linear" || hasCall s "Linear" then
    let ns := (constructorNatArgs? s #["in_features", "out_features"]).getD #[]
    match (ns[0]?.getD none), (ns[1]?.getD none) with
    | some i, some o => some (.linear i o)
    | _, _ => some (.unsupported s
        "Linear needs natural in_features/out_features with no extra options")
  else if hasCall s "nn.Conv1d" || hasCall s "Conv1d" ||
      hasCall s "nn.Conv2d" || hasCall s "Conv2d" ||
      hasCall s "nn.Conv3d" || hasCall s "Conv3d" then
    let rank := if hasSubstr s "Conv1d" then 1 else if hasSubstr s "Conv2d" then 2 else 3
    let ns := (constructorNatArgs? s
      #["in_channels", "out_channels", "kernel_size", "stride", "padding"]).getD #[]
    match (ns[0]?.getD none), (ns[1]?.getD none), (ns[2]?.getD none) with
    | some i, some o, some k =>
        let stride := (ns[3]?.getD none).getD 1
        let padding := (ns[4]?.getD none).getD 0
        some (.conv rank i o k stride padding)
    | _, _, _ => some (.unsupported s
        "Conv needs scalar natural dimensions/stride/padding; extra options need manual handling")
  else if hasCall s "nn.MaxPool1d" || hasCall s "MaxPool1d" ||
      hasCall s "nn.MaxPool2d" || hasCall s "MaxPool2d" ||
      hasCall s "nn.MaxPool3d" || hasCall s "MaxPool3d" then
    let rank := if hasSubstr s "MaxPool1d" then 1 else if hasSubstr s "MaxPool2d" then 2 else 3
    let ns := (constructorNatArgs? s #["kernel_size", "stride"]).getD #[]
    match (ns[0]?.getD none) with
    | some k =>
        let stride := (ns[1]?.getD none).getD k
        some (.maxPool rank k stride)
    | none => some (.unsupported s
        "MaxPool needs scalar natural kernel_size/stride and no extra options")
  else if hasCall s "nn.AdaptiveAvgPool1d" || hasCall s "AdaptiveAvgPool1d" ||
      hasCall s "nn.AdaptiveAvgPool2d" || hasCall s "AdaptiveAvgPool2d" ||
      hasCall s "nn.AdaptiveAvgPool3d" || hasCall s "AdaptiveAvgPool3d" then
    let rank := if hasSubstr s "AdaptiveAvgPool1d" then 1
      else if hasSubstr s "AdaptiveAvgPool2d" then 2 else 3
    let ns := (constructorNatArgs? s #["output_size"]).getD #[]
    match (ns[0]?.getD none) with
    | some o => some (.adaptiveAvgPool rank o)
    | none => some (.unsupported s "AdaptiveAvgPool needs a scalar natural output_size")
  else if hasCall s "nn.Flatten" || hasCall s "nn.flatten" ||
      hasCall s "torch.flatten" || hasCall s "flatten" then
    some .flatten
  else if hasCall s "nn.ReLU" || hasCall s "nn.relu" ||
      hasCall s "F.relu" || hasCall s "relu" then
    some .relu
  else if hasCall s "nn.GELU" || hasCall s "nn.gelu" || hasCall s "F.gelu" then
    some .gelu
  else if hasCall s "nn.Sigmoid" || hasCall s "nn.sigmoid" || hasCall s "torch.sigmoid" then
    some .sigmoid
  else if hasCall s "nn.Tanh" || hasCall s "nn.tanh" || hasCall s "torch.tanh" then
    some .tanh
  else if hasCall s "nn.Dropout" || hasCall s "Dropout" then
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
    if layers.any (fun l => match l with | .conv .. => true | .maxPool .. => true | _ => false) then
      warnings := warnings.push
        "Convolutional layers need explicit leading, channel, and spatial shapes before the \
        generated TorchLean skeleton can be made executable."
    if layers.any (fun l => match l with | .dropout => true | _ => false) then
      warnings := warnings.push
        "Dropout is mode-dependent; TorchLean asks for an explicit probability/seed and keeps \
        train/eval behavior visible."
    if layers.any (fun l => match l with | .adaptiveAvgPool .. => true | _ => false) then
      warnings := warnings.push
        "Adaptive pooling is detected as a shape-changing operation; connect it to the specific \
        TorchLean pooling spec you want before treating the skeleton as executable."
    if layers.any (fun l => match l with | .flatten => true | _ => false) then
      warnings := warnings.push
        "Flatten needs an explicit axis range and batch shape; it is left as a boundary note."
    if layers.any (fun l => match l with | .gelu => true | _ => false) then
      warnings := warnings.push
        "GELU needs an explicit approximation choice; TorchLean's `nn.gelu` uses tanh."
    pure warnings
  { layers, translated, warnings, unsupported }

/--
Generate a TorchLean skeleton from the recognized layer sequence.

The emitted code is meant to be a starting point, not a final theorem. It imports the public
TorchLean umbrella, opens the user-facing API namespaces, emits direct sequential terms for the safe
subset, and then appends boundary notes as Lean comments. The next intended step is to add a
concrete shape contract and hand the model to `Trainer.new` with an objective.
-/
def torchLeanSkeleton (r : Report) (name : String := "translatedModel") : String :=
  let translatedLines := r.layers.filterMap layerTorchLeanTerm?
  let body :=
    match translatedLines[0]? with
    | none => "    -- No directly translatable sequential terms were recognized."
    | some first =>
        String.intercalate "\n" <| Array.toList <| #["    " ++ first] ++
          (translatedLines.extract 1 translatedLines.size).map (fun line => "  , " ++ line)
  let boundaryComments := r.layers.filterMap layerBoundaryComment?
  let boundaryBlock :=
    if boundaryComments.isEmpty then
      "-- Boundary notes: none for this supported translator subset."
    else
      String.intercalate "\n" (#["-- Boundary notes:"] ++ boundaryComments).toList
  String.intercalate "\n" <| Array.toList #[
    "import NN",
    "",
    "open TorchLean",
    "",
    s!"def {name} :=",
    "  nn.Sequential![",
    body,
    "  ]",
    "",
    boundaryBlock,
    "",
    "-- Next steps:",
    "-- 1. Add the concrete input/output shape contract.",
    "-- 2. Choose a loss and hand this to `Trainer.new` with that objective.",
    "-- 3. If this came from a real PyTorch module, use `torch.export` capture for a checked \
    graph path."
  ]

/-- Render one recognized/unsupported layer row in the HTML report table. -/
private def layerRowHtml (l : Layer) : ProofWidgets.Html :=
  let badge := if supported l then okBadge "recognized" else warnBadge "unsupported"
  let detail :=
    match l with
    | .unsupported _ reason => reason
    | .conv .. => "recognized; executable lowering needs the input spatial shape"
    | .maxPool .. => "recognized; executable lowering needs the input spatial shape"
    | .adaptiveAvgPool .. => "recognized as a boundary item"
    | .flatten => "recognized; axis range and batch shape must be explicit"
    | .gelu => "recognized; approximation choice must be explicit"
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
private def msgListHtml (title : String) (msgs : Array String) (kind : String) :
    ProofWidgets.Html :=
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

That layout is meant for use inside the editor: useful generated code beside an equally visible
account of what has *not* been checked.
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
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap",
        "margin-bottom": "10px"}}>
      {pill "PyTorch -> TorchLean"}
      {pill s!"layers={r.layers.size}"}
      {pill s!"translated={r.translated}"}
      {if allSupported then okBadge "supported subset" else warnBadge "boundary report"}
    </div>
    <p style={json% {"margin": "0 0 10px 0"}}>
      {.text "Supported-subset assistant for common PyTorch layer stacks. It generates a TorchLean \
        skeleton and names the parts that still need shape contracts or the full torch.export \
        path."}
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
        <li>{.text "This widget is a heuristic editor assistant; checked import uses the explicit \
          artifact bridge."}</li>
        <li>{.text "A skeleton becomes executable only after you add the typed input/output shape \
          contract."}</li>
        <li>{.text "For real PyTorch modules, use the existing torch.export JSON bridge to capture \
          and validate the graph."}</li>
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
  UI.canonicalCommand <$> `(#html (NN.Widgets.PyTorchTranslator.html $snippet))

/-- Panel shown when the snippet file cannot be read at all. -/
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
#pytorch_translate_file "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"
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
    pure (fileErrorHtml path
      "IO.FS.readFile failed. Check that the path is relative to the Lake project root and that \
      the file exists.")

syntax (name := pytorchTranslateFileCmd) "#pytorch_translate_file " str : command

/--
Command frontend that reads a `.py` file and renders the translator widget.

This is the file-based translator widget over real source text. For checked model import, use the
existing `torch.export` JSON bridge after the report tells you the model is close to the supported
subset.
-/
macro "#pytorch_translate_file " path:str : command =>
  UI.canonicalCommand <$>
    `(#html (NN.Widgets.PyTorchTranslator.htmlFromFile $path))

end PyTorchTranslator
end NN.Widgets
