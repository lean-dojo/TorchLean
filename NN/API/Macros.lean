/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean

/-!
# Small Convenience Macros

This file contains only **general-purpose** syntactic sugar:
- `seq! a, b, c` for composing TorchLean `Seq` models without chaining `>>>` manually.
- `tlist! x, y, ...` for building `TorchLean.TList` values without `.cons ... .nil` boilerplate.

Both macros expand to fully-qualified names under `NN.API.TorchLean.*`, so in practice you will
usually import `NN.API.Public` (or at least `NN.API.Runtime`) alongside this module.

We avoid layer-specific "proof-eliding" macros here; prefer the named-field APIs in `NN.API.Public`
for clarity and stable documentation.
-/

@[expose] public section


namespace NN
namespace API

/-- Compose `Seq` models without chaining `>>>` manually. -/
syntax (name := seqLit) "seq!" term,+ : term

/-!
## Sequential Literals

TorchLean sequential models are *shape-indexed* (`Seq σ τ`), so we cannot use a plain `List` of
layers like PyTorch does (a `List` would require every element to have the same type).

Instead we provide macros that expand to ordinary `Seq` composition while still letting users
write “list-shaped” model definitions.
-/

-- NOTE: This must *not* reserve the keyword `nn.Sequential`, because that breaks parsing of
-- expressions like `nn.Sequential σ τ` where `nn.Sequential` is used as a constant/type name.
--
-- So we provide only the `...!` forms (with `!`): `nn.Sequential![...]` and `nn.sequential![...]`.
syntax (name := nnSequentialBangLit) "nn.Sequential!" "[" term,+ "]" : term

/-!
For naming-convention friendliness, we also provide the lowercase alias `nn.sequential![...]`.
It expands to the same seeded-builder composition as `nn.Sequential![...]`.
-/
syntax (name := nnSequentialBangLitLower) "nn.sequential!" "[" term,+ "]" : term

private meta def mkGlobalIdent (val : Lean.Name) : Lean.Ident :=
  -- Use a *macro-scope-free* identifier, so expansions refer to the actual constant name
  -- (e.g. `...compAny`) instead of a macro-scoped one (`...compAny✝`).
  ⟨Lean.Syntax.ident Lean.SourceInfo.none (toString val).toRawSubstring val []⟩

macro_rules (kind := nnSequentialBangLitLower)
  | `(nn.sequential![$ts:term,*]) =>
      `(nn.Sequential![$ts,*])

macro_rules (kind := nnSequentialBangLit)
  | `(nn.Sequential![$a:term]) =>
      let f := mkGlobalIdent `NN.API.TorchLean.NN.AsSeqK.asSeq
      `(do
        let a ← ($a)
        pure ($f a))
  | `(nn.Sequential![$a:term, $b:term]) =>
      let f := mkGlobalIdent `NN.API.TorchLean.NN.compAny
      `(do
        let a ← ($a)
        let b ← ($b)
        pure ($f a b))
  | `(nn.Sequential![$a:term, $b:term, $rest:term,*]) =>
      let f := mkGlobalIdent `NN.API.TorchLean.NN.compAny
      `(do
        let a ← ($a)
        let bc ← (nn.Sequential![$b, $rest,*])
        pure ($f a bc))

macro_rules (kind := seqLit)
  | `(seq! $a:term) => `($a)
  | `(seq! $a:term, $b:term) =>
      let f := mkGlobalIdent `NN.API.TorchLean.NN.compAny
      `($f $a $b)
  | `(seq! $a:term, $b:term, $rest:term,*) =>
      let f := mkGlobalIdent `NN.API.TorchLean.NN.compAny
      `($f $a (seq! $b, $rest,*))

/--
Build a `TorchLean.TList` from comma-separated tensors (avoids `.cons ... .nil` boilerplate).

This stays compact: it is just a syntactic convenience for examples and small programs.
-/
syntax (name := tlistLit) "tlist!" term,+ : term


macro_rules (kind := tlistLit)
  | `(tlist! $x:term) =>
      let f := mkGlobalIdent `NN.API.TorchLean.tlist1
      `($f $x)
  | `(tlist! $x:term, $y:term) =>
      let f := mkGlobalIdent `NN.API.TorchLean.tlist2
      `($f $x $y)
  | `(tlist! $x:term, $y:term, $rest:term,*) => `(.cons $x (tlist! $y, $rest,*))

end API
end NN
