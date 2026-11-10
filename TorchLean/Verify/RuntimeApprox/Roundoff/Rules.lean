/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Shape-dependent reads for `roundoff`

`roundoff_read` registers the coordinate read of a producer that depends on the structure of its
shape, or a bound on its entries, used where that shape is not literal. The attribute is separate
from the tensor adapter, so rule modules can register reads and bounds without importing the
analysis. The attribute for all other rules, `roundoff_rule`, is FloatLib's
(`FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension`).
-/

public meta section

namespace TorchLean.Roundoff

/-- The `roundoff_read` label: coordinate reads of producers whose definition depends on the
structure of their shape, and bounds on their entries that are also `roundoff_rule`s. The analysis
applies them only where the shape is not built from `.scalar` and `.dim`, such as a shape
variable. Elsewhere the producer unfolds as usual and its read or bound is not used. -/
public initialize reads : Lean.LabelExtension ←
  Lean.registerLabelAttr `roundoff_read
    "register a coordinate read of a shape-recursive producer, or a bound on its entries"

end TorchLean.Roundoff
