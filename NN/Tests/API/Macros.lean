/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.API.Seeded

/-!
# Sequential literal diagnostics

These compile-time tests preserve shape inference, the original composition tree, and monadic
execution order. Expected failures check the adjacent entry numbers and both incompatible shapes.
-/

namespace NN.Tests.API.Macros

open TorchLean

example (first : nn.Sequential [2] [3]) (second : nn.Sequential [3] [4])
    (third : nn.Sequential [4] [1]) :
    nn.compose![first, second, third] = nn.compose first (nn.compose second third) := rfl

example (layer : nn.Layer [2] [3]) : nn.compose![layer] = layer := rfl

example (first : nn.Layer [2] [3]) (second : nn.Sequential [3] [1]) :
    nn.Sequential [2] [1] :=
  nn.compose![first, second]

-- Expected types still resolve shape-polymorphic entries, including singleton literals.
example : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.relu, nn.linear 3 1]

example : nn.Builder (nn.Sequential [7, 3] [7, 3]) := nn.Sequential![nn.relu]

example (n : Nat) : nn.Builder (nn.Sequential [n, 2] [n, 1]) :=
  nn.Sequential![
    nn.linear 2 3 (batchShape := [n]),
    nn.relu,
    nn.linear 3 1 (batchShape := [n])]

example (first : nn.Sequential [2] [3]) (second : nn.Sequential [3] [4])
    (third : nn.Sequential [4] [1]) : nn.Sequential [2] [1] :=
  nn.compose![first, nn.compose![second, third]]

-- Definitional shape equality remains sufficient, without a syntactic comparison of dimensions.
example (n : Nat) (first : nn.Sequential [2] [n + 0])
    (second : nn.Sequential [n] [1]) : nn.Sequential [2] [1] :=
  nn.compose![first, second]

private structure Wrapped (input output : Shape) where
  model : nn.Sequential input output

private instance : nn.AsSequential Wrapped where
  asSequential := Wrapped.model

example (first : Wrapped [2] [3]) (second : nn.Sequential [3] [1]) :
    nn.Sequential [2] [1] :=
  nn.compose![first, second]

-- This equality holds for every monad, including those whose bind is not associative.
example {m : Type 1 → Type 1} [Monad m]
    (first : m (nn.Sequential [2] [3])) (second : m (nn.Sequential [3] [4]))
    (third : m (nn.Sequential [4] [1])) :
    (nn.Sequential![first, second, third] : m (nn.Sequential [2] [1])) =
      (do
        let a ← first
        let bc ← (do
          let b ← second
          let c ← third
          pure (nn.compose b c))
        pure (nn.compose a bc)) := rfl

private def record {input output : Shape} (index : Nat)
    (model : nn.Sequential input output) :
    StateM (ULift.{1} (List Nat)) (nn.Sequential input output) := do
  modify fun state => ⟨state.down ++ [index]⟩
  pure model

example (first : nn.Sequential [2] [3]) (second : nn.Sequential [3] [4])
    (third : nn.Sequential [4] [5]) (fourth : nn.Sequential [5] [1]) :
    ((nn.Sequential![record 1 first, record 2 second, record 3 third, record 4 fourth] :
      StateM (ULift.{1} (List Nat)) (nn.Sequential [2] [1])).run ⟨[]⟩).2.down =
      [1, 2, 3, 4] := rfl

/--
@ +3:21...27
error: nn.compose!: layer 2 expects input shape [5], but layer 1 outputs [3].
Change layer 2's input shape or insert a layer that converts [3] to [5].
-/
#guard_msgs (positions := true) in
example (first : nn.Sequential [2] [3]) (second : nn.Sequential [5] [1]) :
    nn.Sequential [2] [1] :=
  nn.compose![first, second]

/--
error: nn.compose!: layer 3 expects input shape [5], but layer 2 outputs [4].
Change layer 3's input shape or insert a layer that converts [4] to [5].
-/
#guard_msgs in
example (first : nn.Sequential [2] [3]) (second : nn.Sequential [3] [4])
    (third : nn.Sequential [5] [1]) : nn.Sequential [2] [1] :=
  nn.compose![first, second, third]

/--
@ +2:32...45
error: nn.Sequential!: layer 2 expects input shape [5], but layer 1 outputs [3].
Change layer 2's input shape or insert a layer that converts [3] to [5].
-/
#guard_msgs (positions := true) in
example : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.linear 5 1]

/--
error: nn.Sequential!: layer 3 expects input shape [5], but layer 2 outputs [4].
Change layer 3's input shape or insert a layer that converts [4] to [5].
-/
#guard_msgs in
example : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 3, nn.linear 3 4, nn.linear 5 1]

/--
error: nn.compose!: layer 2 expects input shape [3], but layer 1 outputs [2, 3].
Change layer 2's input shape or insert a layer that converts [2, 3] to [3].
-/
#guard_msgs in
example (first : nn.Sequential [2] [2, 3]) (second : nn.Sequential [3] [1]) :
    nn.Sequential [2] [1] :=
  nn.compose![first, second]

/--
error: nn.compose!: layer 2 expects input shape [5], but layer 1 outputs [3].
Change layer 2's input shape or insert a layer that converts [3] to [5].
-/
#guard_msgs in
example (first : Wrapped [2] [3]) (second : nn.Sequential [5] [1]) :
    nn.Sequential [2] [1] :=
  nn.compose![first, second]

/-- error: Unknown identifier `missingLayer` -/
#guard_msgs in
example (first : nn.Sequential [2] [3]) : nn.Sequential [2] [1] :=
  nn.compose![first, missingLayer]

end NN.Tests.API.Macros
