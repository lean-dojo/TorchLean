/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init

/-!
# Differential checks on an explicit corpus

`compareOn` runs two implementations on the same inputs. A mismatch reports the zero-based case
index, input, and both results. Callers choose the comparison: exact equality, floating-point bit
identity, or a domain-specific tolerance. In particular, no tolerance or NaN policy is implicit.

`NN.Testing.Command` provides the editor command for pure functions. Keep the reference independent
of the implementation being tested. These checks execute code; they do not produce proofs.
-/

public section

namespace NN.Testing

/-- Compare effectful implementations on a nonempty, caller-supplied corpus.
The reference runs first; each side must start from equivalent state if it is stateful.
Exceptions and empty corpora fail the check rather than count as agreement. -/
def compareOn {α β γ : Type} [Repr α] [Repr β] [Repr γ] (cases : Array α)
    (candidate : α → IO β) (reference : α → IO γ) (agrees : β → γ → Bool) : IO Unit := do
  if cases.isEmpty then
    throw <| IO.userError "comparison needs at least one case"
  for h : index in [:cases.size] do
    let input := cases[index]
    let expected ← try reference input catch error =>
      throw <| IO.userError s!"reference failed at case {index}, input {repr input}: {error}"
    let actual ← try candidate input catch error =>
      throw <| IO.userError s!"candidate failed at case {index}, input {repr input}: {error}"
    unless agrees actual expected do
      throw <| IO.userError s!"mismatch at case {index}, input {repr input}\n\
        candidate: {repr actual}\nreference: {repr expected}"

end NN.Testing
