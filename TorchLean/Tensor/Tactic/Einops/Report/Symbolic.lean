/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops.Report.Decode
public import TorchLean.Tensor.Tactic.Einops.Report.Decode

/-!
# Symbolic reports

Reports for checked plans whose dimensions are symbolic in the local context. Shapes, axes and
lengths stay symbolic in the report, and no concrete value is invented for them.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/--
Assemble the report of a checked plan with symbolic local dimensions from the sections the caller
supplies. The correctness section names the theorem and states that the native program equals
`denotation` for every valid checked plan.
-/
def symbolic (operation : String)
    (typeEntries : List (String × String))
    (obligations logicalStages workEstimate : List String)
    (nativeExecution correctnessTheorem denotation : String) : String :=
  String.intercalate "\n" <|
    [s!"{operation} (symbolic checked plan)"] ++
      typeCheckSection typeEntries ++
      ["  Concrete shapes, axes, and lengths remain symbolic in the local \
          checked certificate."] ++
      formatSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      formatSection "Generated execution strategy" [nativeExecution] ++
      formatSection "Shape-derived work estimate" workEstimate ++
      formatSection "Correctness"
        [s!"Theorem: {correctnessTheorem}.",
         s!"Proven result: the native program equals {denotation} for every \
            valid checked plan."] ++
      performanceFooter

end TorchLean.Einops.Report
