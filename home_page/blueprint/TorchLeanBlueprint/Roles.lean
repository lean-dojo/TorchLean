import VersoManual

/-!
Roles for linking to files in the TorchLean GitHub repository from Verso prose.

Verso link targets and role arguments must fit on one source line, and many repository paths
are too long for a full `https://github.com/...` URL to fit within the 100-column limit that
the repository lint enforces. These roles build the URL from the repository-relative path, so
only the path has to appear in the document source.
-/

open Lean
open Verso Doc Elab ArgParse

namespace TorchLeanBlueprint.Roles

/-- Base URL of the TorchLean repository on GitHub. -/
def repoUrl : String := "https://github.com/lean-dojo/TorchLean/"

/-- Links `text` to a file in the TorchLean repository at `blob/main/PATH`. -/
def repoFileLink (path : String) (inlines : TSyntaxArray `inline) :
    DocElabM (Array (TSyntax `term)) := do
  let url := repoUrl ++ "blob/main/" ++ path
  let content ← inlines.mapM elabInline
  pure #[← ``(Verso.Doc.Inline.link #[$content,*] $(quote url))]

/-- Links `text` to a directory in the TorchLean repository at `tree/main/PATH`. -/
def repoDirLink (path : String) (inlines : TSyntaxArray `inline) :
    DocElabM (Array (TSyntax `term)) := do
  let url := repoUrl ++ "tree/main/" ++ path
  let content ← inlines.mapM elabInline
  pure #[← ``(Verso.Doc.Inline.link #[$content,*] $(quote url))]

end TorchLeanBlueprint.Roles

open TorchLeanBlueprint.Roles

/-- `{src "NN/API.lean"}[text]` links `text` to the file `NN/API.lean` on GitHub. -/
@[role_expander src]
def src : RoleExpander
  | args, inlines => do
    let path ← ArgParse.run (.positional `path .string) args
    repoFileLink path inlines

/-- `{srcDir "NN/Proofs"}[text]` links `text` to the directory `NN/Proofs` on GitHub. -/
@[role_expander srcDir]
def srcDir : RoleExpander
  | args, inlines => do
    let path ← ArgParse.run (.positional `path .string) args
    repoDirLink path inlines


open Verso.Genre Manual

/- A shell command or a recorded terminal result, kept distinct from checked Lean examples. -/
block_extension terminalBlock (output : Bool) where
  data := toJson output
  traverse _ _ _ := pure none
  toHtml :=
    open Verso.Output.Html in
    some <| fun _ blockHtml _ data content => do
      let output := (fromJson? data : Except String Bool).toOption.getD false
      pure {{
        <div class="tl-terminal" data-output={{if output then "true" else "false"}}>
          <div class="tl-terminal-heading">
            {{if output then "Recorded output" else "Terminal"}}
          </div>
          {{← content.mapM blockHtml}}
        </div>
      }}
  toTeX := some <| fun _ blockTeX _ _ content => .seq <$> content.mapM blockTeX

/-- Display shell input with `terminal`, or a recorded result with `terminal +output`.

These blocks display text; they do not execute commands during the guide build.
-/
@[code_block_expander terminal]
def terminal : CodeBlockExpander
  | args, str => do
    let output ← ArgParse.run (.flag `output false) args
    let block ←
      ``(Verso.Doc.Block.other (terminalBlock $(quote output))
        #[Verso.Doc.Block.code $(quote str.getString)])
    pure #[block]
