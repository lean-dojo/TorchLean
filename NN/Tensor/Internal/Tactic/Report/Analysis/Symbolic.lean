/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Common

/-!
# Symbolic reports

This module describes checked plans whose dimensions remain symbolic in the
local context without inventing concrete values.
-/

public meta section

namespace TorchLean.Tensor.Internal.Report.Impl

open Lean Elab Tactic Meta

/--
Describe a symbolic checked plan without pretending that its local dimensions
are concrete.
-/
def symbolicReport (operation : String)
    (typeEntries : List (String × String))
    (obligations logicalStages workEstimate : List String)
    (nativeExecution correctnessTheorem denotation : String) : String :=
  String.intercalate "\n" <|
    [s!"{operation} (symbolic checked plan)"] ++
      typeCheckSection typeEntries ++
      ["  Concrete shapes, axes, and lengths remain symbolic in the local \
          checked certificate."] ++
      reportSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      reportSection "Generated execution strategy" [nativeExecution] ++
      reportSection "Shape-derived work estimate" workEstimate ++
      reportSection "Correctness"
        [s!"Theorem: {correctnessTheorem}.",
         s!"Proven result: the native program equals {denotation} for every \
            valid checked plan."] ++
      performanceFooter

end TorchLean.Tensor.Internal.Report.Impl
