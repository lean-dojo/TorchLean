/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log
public meta import NN.Widgets.Core.UI
public meta import NN.Widgets.Runtime.Training
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# GPT-2 Training Log Viewer

TorchLean's generic `TrainLog` widget (`#train_log_view` / `#train_log_file_view`) is great for
curves and small scalar summaries, but language-model examples benefit from a "prompt → sample"
panel that keeps the generated text readable.

This module provides GPT-2 specific viewers that:
- extract `prompt=...` and `generated=...` notes (as written by the `torchlean gpt2` example),
- render them as separate monospace blocks, and
- delegate the remaining metrics/notes to the generic training-log renderer.

Usage (after running the executable to write a JSON log):

```lean
#gpt2_train_log_file_view "data/model_zoo/gpt2_trainlog.json"
```
-/

namespace NN.Widgets

public meta section

open scoped ProofWidgets.Jsx

open _root_.Runtime.Training
open UI

namespace Models
namespace Sequence
namespace Gpt2

private def noteValue? (pre : String) (notes : Array String) : Option String :=
  (notes.find? (fun s => s.startsWith pre)).map (fun s => (s.drop pre.length).toString)

private def dropNotePrefix (pre : String) (notes : Array String) : Array String :=
  notes.filter (fun s => !(s.startsWith pre))

private def block (title : String) (body : String) : ProofWidgets.Html :=
  <details style={json% {"margin-top": "10px"}}>
    <summary style={json% {"cursor": "pointer", "user-select": "none"}}>
      {pill title}
    </summary>
    <pre style={json% {
      "white-space": "pre-wrap",
      "word-break": "break-word",
      "margin-top": "8px",
      "padding": "10px",
      "border-radius": "10px",
      "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
      "background": "rgba(120,120,120,0.06)",
      "font-family":
        "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace)",
      "font-size": "12px",
      "line-height": "18px"
    }}>{.text body}</pre>
  </details>

/-- Render a `TrainLog` produced by `torchlean gpt2` with prompt/sample blocks. -/
def gpt2TrainLogHtml (log : TrainLog) : ProofWidgets.Html :=
  let prompt := noteValue? "prompt=" log.notes |>.getD "";
  let generated := noteValue? "generated=" log.notes |>.getD "";
  let notes := log.notes |> dropNotePrefix "prompt=" |> dropNotePrefix "generated=";
  let log' : TrainLog := { log with notes := notes };
  <div style={json% {"padding": "10px"}}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap"}}>
      {pill "gpt2"} {pill log.title}
    </div>
    {if prompt.isEmpty then warnBadge "missing prompt note" else block "prompt" prompt}
    {if generated.isEmpty then warnBadge "missing generated note" else block "generated" generated}
    <details style={json% {"margin-top": "12px"}}>
      <summary style={json% {"cursor": "pointer", "user-select": "none"}}>{pill "metrics"}</summary>
      <div style={json% {"margin-top": "8px"}}>{trainLogHtml log'}</div>
    </details>
  </div>

/-!
## One-shot Prompt Runner

`#gpt2_prompt_view "..."` runs the `torchlean gpt2` executable (CUDA) with a small default
configuration, writes a temporary JSON `TrainLog`, then renders it.

This is meant for quick "does it run?" checks inside the infoview. It is not a replacement for
the CLI interactive loop, and it will still take noticeable time if you increase `--steps`.
-/

private def torchleanBin : System.FilePath :=
  ".lake/build/bin/torchlean"

private def gpt2PromptHtml (prompt : String) (steps : Nat := 5) (generate : Nat := 96) :
    IO ProofWidgets.Html := do
  if !(← torchleanBin.pathExists) then
    pure <|
      <div style={json% {"padding": "10px"}}>
        {warnBadge "gpt2_prompt_view"}
        <div style={json% {"margin-top": "8px"}}>
          {.text "Could not find the TorchLean executable at: "}{monospace torchleanBin.toString}
        </div>
        <div style={json% {"margin-top": "6px", "opacity": "0.9"}}>
          {.text "Build it with "}{monospace "lake build -R -K cuda=true torchlean:exe"}{.text "."}
        </div>
      </div>
  else
    IO.FS.withTempFile (fun h p => do
      -- The temp file is already created; we only need its path. (The handle is closed by
      -- `withTempFile` after this callback returns.)
      h.flush
      let args : Array String :=
        #["gpt2", "--cuda", "--fast-kernels", "--tiny-shakespeare",
          "--steps", toString steps,
          "--windows", "64",
          "--lr", "0.0005",
          "--prompt", prompt,
          "--generate", toString generate,
          "--temperature", "0.9",
          "--top-k", "12",
          "--sample-seed", "7",
          "--ascii-only", "true",
          "--log", p.toString]
      let procOut ← IO.Process.output { cmd := torchleanBin.toString, args := args }
      if procOut.exitCode != 0 then
        return (
          <div style={json% {"padding": "10px"}}>
            {errBadge "gpt2_prompt_view failed"}
            <div style={json% {"margin-top": "8px"}}>{monospace procOut.stderr}</div>
          </div>)
      else
        let log ← TrainLog.readJson p
        let html := gpt2TrainLogHtml log
        return (
          <div>
            <details style={json% {"margin-bottom": "10px"}}>
              <summary style={json% {"cursor": "pointer", "user-select": "none"}}>{pill "run output"}</summary>
              <pre style={json% {
                "white-space": "pre-wrap",
                "word-break": "break-word",
                "margin-top": "8px",
                "padding": "10px",
                "border-radius": "10px",
                "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
                "background": "rgba(120,120,120,0.06)",
              "font-family":
                "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace)",
              "font-size": "12px",
              "line-height": "18px"
            }}>{.text (procOut.stdout.trimAsciiEnd.toString)}</pre>
            </details>
            {html}
          </div>)
    )

/-!
## Commands
-/

syntax (name := gpt2TrainLogViewCmd) "#gpt2_train_log_view " term : command
syntax (name := gpt2TrainLogFileViewCmd) "#gpt2_train_log_file_view " term : command
syntax (name := gpt2PromptViewCmd) "#gpt2_prompt_view " term : command
syntax (name := gpt2ShortPromptViewCmd) "#gpt2 " term : command

macro "#gpt2_train_log_view " log:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (gpt2TrainLogHtml $log))

macro "#gpt2_train_log_file_view " path:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (do
    let p : System.FilePath := $path
    try
      let log ← TrainLog.readJson p
      pure (gpt2TrainLogHtml log)
    catch e =>
      pure <|
        <div style={json% {"padding": "10px"}}>
          {warnBadge "gpt2_train_log_file_view"}
          <div style={json% {"margin-top": "8px"}}>
            {.text "Could not read a TrainLog JSON file at: "}
            {monospace p.toString}
          </div>
          <div style={json% {"margin-top": "6px", "opacity": "0.9"}}>
            {.text "Tip: create this file by running "}
            {monospace "lake exe torchlean gpt2 --cuda --log <path> ..."}
            {.text "."}
          </div>
          <div style={json% {"margin-top": "6px"}}>{monospace (toString e)}</div>
        </div>))

macro "#gpt2_prompt_view " prompt:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (gpt2PromptHtml $prompt))

macro "#gpt2 " prompt:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (gpt2PromptHtml $prompt))

end Gpt2
end Sequence
end Models

end
