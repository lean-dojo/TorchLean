/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.NN.Widget.UI
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay

/-!
# PyTorch Translator Widget

`#pytorch_translate_file "path/to/model.py"` reads a Python source file and matches common PyTorch
layer constructors line by line. The infoview report lists the layers it recognized, the layers that
need more information before they can be emitted (an input shape, a dropout probability and seed,
the GELU approximation) and the lines outside the supported subset. It also gives a TorchLean
skeleton in the `nn.Sequential!` style of `TorchLean.NN.Sequential.Literal`.
`#pytorch_translate_view s` runs the same analysis on a Lean string `s`.

The matching works on constructor names and substrings, with no Python parser and no model of
PyTorch semantics. Graph capture goes through `torch.export` to a JSON capture document, which
`IR.load` loads. This widget shows whether a model is close to the supported TorchLean
subset before that capture and import.
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget
namespace PyTorchTranslator

open UI
open Lean Elab Command

/--
A layer family the translator recognizes in one source line, with the numeric arguments it read.

The same row can come from several spellings: `linear 784 128` is read from `nn.Linear(784, 128)`
inside an `nn.Sequential`, from a field assignment `self.fc = nn.Linear(784, 128)` or from a bare
snippet line. A PyTorch-looking line outside the vocabulary becomes `unsupported`.
-/
inductive Layer where
  /-- Fully connected layer with numeric `in_features` and `out_features`. -/
  | linear (inDim outDim : Nat)
  /-- `Conv1d`, `Conv2d` or `Conv3d` with its spatial rank, channel counts, kernel size, stride
  (default 1) and padding (default 0). -/
  | conv (rank inC outC kernel stride padding : Nat)
  /-- `MaxPool1d`, `MaxPool2d` or `MaxPool3d` with its spatial rank, kernel size and stride
  (default: the kernel size). -/
  | maxPool (rank kernel stride : Nat)
  /-- `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d` or `AdaptiveAvgPool3d` with its spatial rank and
  output size, left as a comment in the skeleton. -/
  | adaptiveAvgPool (rank out : Nat)
  /-- Flatten, which needs a shape contract for its axis range before it can be emitted. -/
  | flatten
  /-- Elementwise ReLU. -/
  | relu
  /-- Elementwise GELU. -/
  | gelu
  /-- Elementwise sigmoid. -/
  | sigmoid
  /-- Elementwise tanh. -/
  | tanh
  /-- Dropout, left as a comment because its probability, seed and train/eval mode are chosen in
  Lean. -/
  | dropout
  /-- A PyTorch-looking line outside the vocabulary, with the reason it cannot be translated. -/
  | unsupported (raw reason : String)
  deriving Repr, Inhabited

/--
What the translator found in a source text: the rows in source order, the number recognized, the
warnings for the model as a whole and the unsupported lines.
-/
structure Report where
  /-- One row per relevant line, in source order, unsupported lines included. -/
  layers : Array Layer := #[]
  /-- Number of rows that are not `unsupported`. -/
  translated : Nat := 0
  /-- Model-wide warnings about choices the source does not make, such as the input shapes of a
  convolution or the GELU approximation, each given once. -/
  warnings : Array String := #[]
  /-- Unsupported PyTorch-looking lines, each followed by the reason it cannot be translated. -/
  unsupported : Array String := #[]
  deriving Repr, Inhabited

/-- Match a complete call name, so `ReLU6` is not mistaken for `ReLU`. -/
private def hasCall (s name : String) : Bool :=
  let head := ((s.splitOn "(").headD s).trimAscii.toString
  head == name || head.endsWith ("." ++ name) || head.endsWith (" " ++ name)

/--
Trim a source line and strip a simple assignment prefix, so `self.fc1 = nn.Linear(784, 128)`
becomes `nn.Linear(784, 128)`. Comment lines come back trimmed.
-/
private def lineClean (s : String) : String :=
  let s := s.trimAscii.toString
  if s.startsWith "#" then s else
  -- An `=` after a `(` belongs to a keyword argument, so such a line is kept whole.
  match s.splitOn "=" with
  | lhs :: rhs =>
      if lhs.contains "(" then s else (String.intercalate "=" rhs).trimAscii.toString
  | _ => s

/-- The natural-number arguments of a single constructor call, positional or by keyword, in the
order of `names`, with `none` in the slot of an absent argument.

The whole result is `none` when an argument is a nested tuple, a symbol, a negative or fractional
number, a repeated keyword or an unknown option. Digits in the constructor name (`Conv2d`) are not
read as arguments.
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

/-- Whether a row is a recognized layer, that is, anything but `unsupported`. -/
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
The `nn.Sequential!` entry for a linear, ReLU, sigmoid or tanh layer, and `none` for every other
row (those rows become comments in the skeleton).

Entries follow the textual order of the constructors, which need not be the order in which the
model's `forward` applies them.
-/
private def layerTorchLeanTerm? : Layer → Option String
  | .linear i o => some s!"nn.linear {i} {o}"
  | .relu => some "nn.op .relu"
  | .sigmoid => some "nn.op .sigmoid"
  | .tanh => some "nn.op .tanh"
  | _ => none

/--
The skeleton comment for a row with no direct entry: what to add for a flatten, GELU, convolution,
max-pooling, adaptive-pooling or dropout layer, and the reason for an unsupported line.

Pasted into a Lean file, the skeleton then shows what information is missing, such as an input
shape, a dropout probability and seed, or the GELU approximation.
-/
private def layerBoundaryComment? : Layer → Option String
  | .flatten =>
      some "-- Flatten detected: add `nn.op (.reshape h)` after checking the source axis range \
        and batch shape."
  | .gelu =>
      some "-- GELU detected: `nn.op .gelu` uses the tanh approximation. Check the source \
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
Classify one source line.

- `some l` with `l` a recognized layer: the line calls a supported constructor with arguments the
  translator can read.
- `some (.unsupported raw reason)`: the line calls a supported constructor with arguments it cannot
  read, or mentions `nn.`, `torch.` or `F.` outside the vocabulary (`nn.BatchNorm2d`,
  `torch.reshape`). These lines stay in the report because they block a complete translation.
- `none`: a blank or comment line, ordinary Python structure (`class`, `def forward`, `return`,
  imports, `nn.Sequential(`) or any other line, which never becomes a report row.
-/
private def analyzeLine (raw : String) : Option Layer :=
  let s := lineClean raw
  if s.isEmpty || s.startsWith "#" then
    none
  else if hasCall s "nn.Linear" || hasCall s "Linear" then
    let ns := (constructorNatArgs? s #["in_features", "out_features"]).getD #[]
    match (ns[0]?.getD none), (ns[1]?.getD none) with
    | some i, some o => some (.linear i o)
    | _, _ => some (.unsupported s
        "Linear needs natural in_features/out_features with no extra options")
  else if hasCall s "nn.Conv1d" || hasCall s "Conv1d" ||
      hasCall s "nn.Conv2d" || hasCall s "Conv2d" ||
      hasCall s "nn.Conv3d" || hasCall s "Conv3d" then
    let rank := if s.contains "Conv1d" then 1 else if s.contains "Conv2d" then 2 else 3
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
    let rank := if s.contains "MaxPool1d" then 1 else if s.contains "MaxPool2d" then 2 else 3
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
    let rank := if s.contains "AdaptiveAvgPool1d" then 1
      else if s.contains "AdaptiveAvgPool2d" then 2 else 3
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
  else if s.contains "def forward" || s.contains "class " || s.contains "super().__init__" ||
      s.contains "return " || s.contains "import " || s.contains "from " ||
      s.contains "nn.Sequential" || s = ")" || s = "]" || s = "}" then
    none
  else if s.contains "nn." || s.contains "torch." || s.contains "F." then
    some (.unsupported s "not in the supported translator layer subset")
  else
    none

/--
Run `analyzeLine` over every line of `snippet` and collect the rows, the recognized count, the
model-wide warnings and the unsupported lines.

An unsupported line becomes a row and the scan goes on, so one layer that needs manual handling
leaves the rest of the skeleton intact.
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
        "GELU needs an explicit approximation choice; TorchLean's `nn.op .gelu` uses tanh."
    pure warnings
  { layers, translated, warnings, unsupported }

/--
Generate a TorchLean skeleton from the recognized layer sequence.

The skeleton imports `TorchLean.NN.Sequential.Literal` and `TorchLean.NN.Builder`, opens
`TorchLean`, defines `name` as an `nn.Sequential!` of the translated layers and ends with the
boundary notes and the next steps as Lean comments.
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
    "import TorchLean.NN.Sequential.Literal",
    "import TorchLean.NN.Builder",
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

/-- One report table row: a `recognized` or `unsupported` badge, the layer label and a note on how
the skeleton handles it. -/
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
A titled list of messages, badged as warnings when `kind = "warn"` and as errors otherwise. An
empty list renders as an empty `div`, so the caller can place it in the JSX block unconditionally.
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
Analyze `snippet` and render the report as an infoview panel. The panel holds badges with the row
counts, the table of rows, the warnings and unsupported lines, the generated skeleton, and a
collapsed "Trust boundary" section that points to the `torch.export` bridge for checked import.
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
        <li>{.text "This widget is a heuristic editor assistant. Checked import uses the explicit \
          artifact bridge."}</li>
        <li>{.text "A skeleton becomes executable only after you add the typed input/output shape \
          contract."}</li>
        <li>{.text "For real PyTorch modules, use the torch.export JSON bridge to capture and \
          validate the graph."}</li>
      </ul>
    </details>
  </div>

/--
`#pytorch_translate_view s` renders the translator report for Python text given as a Lean term
`s : String`.

```lean
def snippet : String :=
  "nn.Linear(784, 128)\n" ++
  "nn.ReLU()\n"
#pytorch_translate_view snippet
```
-/
macro "#pytorch_translate_view " snippet:term : command =>
  UI.canonicalCommand <$> `(#html (PyTorchTranslator.html $snippet))

/-- Panel shown when the source file cannot be read. -/
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
Read a Python source file during elaboration and render the translator report for its text.

```lean
#pytorch_translate_file "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"
```

A relative path is resolved against the working directory of the Lean process, which is the Lake
project root in a normal build. If the file cannot be read, the panel shows a file error in place
of the report.
-/
def htmlFromFile (path : String) : CommandElabM ProofWidgets.Html := do
  try
    let source ← liftIO <| IO.FS.readFile path
    pure (html source)
  catch _ =>
    pure (fileErrorHtml path
      "IO.FS.readFile failed. Check that the path is relative to the Lake project root and that \
      the file exists.")

/--
`#pytorch_translate_file "model.py"` shows the translator report for a Python source file, read by
`htmlFromFile`.
-/
macro "#pytorch_translate_file " path:str : command =>
  UI.canonicalCommand <$>
    `(#html (PyTorchTranslator.htmlFromFile $path))

end PyTorchTranslator
end TorchLean.Widget
