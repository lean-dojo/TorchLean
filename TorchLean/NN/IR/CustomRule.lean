/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table
public import TorchLean.NN.Kernel.Tensor

/-!
# The rule of a custom node

A custom node (`Op.custom name inputs output`) runs a checked scalar program. The program is data
of one scalar type, so the graph stores it for the node and the operator value carries only the
descriptor (name, input shapes, output shape). A node's identity is that descriptor together with
the program in the graph's payload, and two custom nodes with equal operator values can compute
different values when their payloads hold different programs.

`Op.rule?` gives `none` for a custom node. `Op.customRule` builds the rule from the program: the
forward map evaluates the program once per output entry, `validate` reports the program's
evaluation error, and the pullback is the caller's `backward`, an executable policy that need not
be the program's derivative. `Op.customCheck` is the checked entry, which evaluates the program
once and returns the value with the pullback, and it equals the rule's `check`
(`Op.customCheck_eq`).
-/

@[expose] public section

namespace TorchLean

namespace Values

variable {α : Type} [Storage α]

/-- The values of the value ports of `shapes`, as a tensor pack. -/
def toPack : (shapes : List Shape) → Values α (shapes.map Port.value) → TensorPack α shapes
  | [], _ => .nil
  | _ :: shapes, (x, xs) => .cons x (toPack shapes xs)

/-- The values of the value ports of `shapes`, from a tensor pack. -/
def ofPack : (shapes : List Shape) → TensorPack α shapes → Values α (shapes.map Port.value)
  | [], .nil => ()
  | _ :: shapes, .cons x xs => (x, ofPack shapes xs)

/-- Reading a tensor pack's values back gives the pack. -/
@[simp] theorem toPack_ofPack : (shapes : List Shape) → (pack : TensorPack α shapes) →
    toPack shapes (ofPack shapes pack) = pack
  | [], .nil => rfl
  | _ :: shapes, .cons _ xs => congrArg _ (toPack_ofPack shapes xs)

end Values

variable {α : Type} [Storage α] [Context α]

/-- The rule of the custom node `name` with the checked program `program` and the pullback
`backward`. The forward map is the program's result where it evaluates and zeros where `validate`
fails. -/
def Op.customRule (name : String) (program : Kernel.Program α) (inputs : List Shape)
    (output : Shape)
    (backward : Values α (inputs.map Port.value) → Tangents α [Port.value output] →
      Tangents α (inputs.map Port.value)) :
    Op.Rule α (inputs.map Port.value) [Port.value output] where
  forward xs :=
    match program.eval (Values.toPack inputs xs) output with
    | .ok y => (y, ())
    | .error _ => (Tensor.full output 0, ())
  backward := backward
  validate xs :=
    match program.eval (Values.toPack inputs xs) output with
    | .ok _ => .ok ()
    | .error error => .error s!"custom {name}: {repr error}"

/-- The checked entry of a custom node: one evaluation of the program, giving the value with the
pullback of the active inputs, or the program's error. -/
def Op.customCheck (name : String) (program : Kernel.Program α) (inputs : List Shape)
    (output : Shape)
    (backward : Values α (inputs.map Port.value) → Tangents α [Port.value output] →
      Tangents α (inputs.map Port.value))
    (xs : Values α (inputs.map Port.value)) (flags : Flags (inputs.map Port.value)) :
    Except String (Op.Prepared α (inputs.map Port.value) [Port.value output]) :=
  match program.eval (Values.toPack inputs xs) output with
  | .ok y => .ok ⟨(y, ()), fun g => Tangents.select _ flags (backward xs g)⟩
  | .error error => .error s!"custom {name}: {repr error}"

/-- The checked entry of a custom node is its rule's `check`. -/
theorem Op.customCheck_eq (name : String) (program : Kernel.Program α) (inputs : List Shape)
    (output : Shape) backward (xs : Values α (inputs.map Port.value))
    (flags : Flags (inputs.map Port.value)) :
    Op.customCheck name program inputs output backward xs flags =
      (Op.customRule name program inputs output backward).check xs flags := by
  simp only [Op.customCheck, Op.Rule.check, Op.customRule]
  split <;> simp_all

/-- Where the program evaluates to `y`, the custom rule's forward value is `y`. -/
theorem Op.customRule_forward (name : String) (program : Kernel.Program α)
    (inputs : List Shape) (output : Shape) backward (xs : Values α (inputs.map Port.value))
    (y : Tensor α output)
    (h : program.eval (Values.toPack inputs xs) output =
      .ok y) :
    (Op.customRule name program inputs output backward).forward xs = (y, ()) := by
  simp only [Op.customRule, h]
  rfl

/-- The custom rule's check fails exactly where the program does not evaluate. -/
theorem Op.customRule_validate (name : String) (program : Kernel.Program α)
    (inputs : List Shape) (output : Shape) backward (xs : Values α (inputs.map Port.value)) :
    ((Op.customRule name program inputs output backward).validate xs).isOk =
      (program.eval (Values.toPack inputs xs) output).isOk := by
  simp only [Op.customRule]
  split <;> simp_all [Except.isOk, Except.toBool]

end TorchLean
