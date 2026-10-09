/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.API.Runtime
public import NN.Spec.Core.Shape
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public meta import Lean.Meta.Closure
public meta import Lean.Meta.Eqns
public meta import NN.Kernel.Frontend

/-!
# Ordinary calculations on the existing autograd tape

`Function.run` uses this internal frontend when recording an ordinary scalar or tensor function.
Supported arithmetic, reductions and matrix products become existing `Ops` calls. Shared local
bindings stay shared. Eager execution records those calls on its tape; no second differentiation
engine or gradient representation is introduced.

The frontend checks equality of the translated operation recipe in Lean's kernel.
No ring simplification, reassociation, precision conversion or speculative derivative is used.
This equality is not a proof of a foreign backend's implementation. Analytic correctness still
uses the existing local derivative laws and graph-composition theorems.

The call site's input supplies the element type and shape. Captured scalars and tensors are
constants, not learnable parameters. Quotients and direct negation use the existing operation
interface. Scalar conditionals inspect the current value and record only the selected branch;
on GPU, this decision synchronizes to Lean. Fixed-parameter recurrences record their steps on
the same tape. Neither feature supplies a derivative at a discontinuous branch boundary.
-/

public section

namespace TorchLean.autograd

/-- A forward value and its recorded input pullback. Backward consumes the recording;
`close` releases it without differentiation. The value remains usable after either operation. -/
structure Result (α : Type) [Storage α] (input output : Shape) where
  /-- Materialized forward value, retaining the requested element type. -/
  value : Tensor α output
  /-- Effective device, including a precision-preserving CPU fallback. -/
  device : NN.Backend.Device
  /-- Consume the existing tape with an explicit output cotangent. -/
  pullback : Tensor α output → IO (Tensor α input)
  /-- Release an unused recording. Calling this more than once is harmless. -/
  close : IO Unit

/-- Differentiate with the supplied cotangent. The default differentiates the sum of output entries.
The recording is released even when backward fails; a consumed recording cannot be reused. -/
def Result.backward {α : Type} [Storage α] [One α] {input output : Shape}
    (result : Result α input output) (seed : Tensor α output := Tensor.full output 1) :
    IO (Tensor α input) :=
  result.pullback seed

instance {α : Type} [Storage α] {input output : Shape} [Repr (Tensor α output)] :
    Repr (Result α input output) where
  reprPrec result prec := reprPrec result.value prec

namespace Internal

/-- Record each step on the existing tape, retaining the order of a scalar or tensor recurrence. -/
def iterate {m : Type → Type} [Monad m] {β : Type} (step : Nat → β → m β) :
    Nat → m β → m β
  | 0, initial => initial
  | n + 1, initial => do
    let value ← iterate step n initial
    step n value

/-- Constructor equations identify a recurrence without imposing algebraic laws or reassociation. -/
theorem recurrence_eq {α : Type} (f : Nat → α) (initial : α) (step : Nat → α → α)
    (hzero : f 0 = initial) (hsucc : ∀ n, f (n + 1) = step n (f n)) (n : Nat) :
    Nat.rec initial step n = f n := by
  induction n with
  | zero => exact hzero.symm
  | succ n ih => simpa only [Nat.rec_add_one, ih] using (hsucc n).symm

/-- An element context and a program using only the existing operation interface. -/
structure Recording (α : Type) [Storage α] (input output : Shape) where
  context : Context α
  program : ∀ {m : Type → Type}, [Monad m] →
    [@Runtime.Autograd.Torch.Ops m α _ context] →
    @Runtime.ValueRef m α _ context _ _ input →
    m (@Runtime.ValueRef m α _ context _ _ output)

/-- Lift scalar control flow through the existing balanced tensor traversal. No unselected branch
is evaluated. GPU arithmetic and backward stay on the GPU; scalar decisions synchronize to Lean. -/
def Recording.elementwise {α : Type} [Storage α] {shape : Shape}
    (recording : Recording α [] []) : Recording α shape shape :=
  letI : Context α := recording.context
  { context := recording.context
    program := fun {_m} _ _ x => do
      let flat ← Runtime.reshape (s₂ := [shape.size]) x (by simp [Shape.size])
      let output ← Runtime.Autograd.Torch.mapBatch
        (σ := []) (τ := []) recording.program flat
      Runtime.reshape (s₂ := shape) output (by simp [Shape.size]) }

/-- Execute through the existing eager session, owned until backward or explicit close. -/
def Recording.run {α : Type} [Storage α]
    [transfer : Runtime.Autograd.Torch.TensorTransfer α]
    {input output : Shape} (recording : Recording α input output)
    (value : Tensor α input) (device : NN.Backend.Device) (grad : Bool := true) :
    IO (Result α input output) :=
  letI : Context α := recording.context
  letI : Runtime.Autograd.Torch.TensorTransfer α :=
    if Runtime.Autograd.Torch.TensorTransfer.supportsEncodedGpu (α := α) then
      { transfer with dtype? := some .encoded }
    else transfer
  do
    let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) { device := device }
    try
      let argument ← session.input value (requiresGrad := grad)
      let result ← recording.program (m := Runtime.Autograd.Torch.Internal.EagerM α)
        argument session
      let value ← session.getValue result
      let openTape ← IO.mkRef true
      let close : IO Unit := do
        if ← openTape.swap false then session.resetTape
      let pullback (seed : Tensor α output) : IO (Tensor α input) := do
        unless ← openTape.swap false do
          throw <| IO.userError "autograd: this recording has already been consumed or closed"
        try
          let gradients ← session.backwardDenseAll result seed
          Runtime.Autograd.Torch.Internal.EagerSession.grad gradients argument
        finally
          session.resetTape
      return ⟨value, session.options.device, pullback, close⟩
    catch error =>
      session.resetTape
      throw error

end Internal

end TorchLean.autograd

public meta section

namespace TorchLean.autograd.Frontend

open Lean Meta Elab Term

private abbrev RecordM := StateRefT (Array Lean.Expr) TermElabM

-- Arithmetic may expose the representation behind the public tensor alias. Both carry the
-- exact shape; never infer rank from a flattened buffer's length.
private def isTensor (type : Lean.Expr) : Bool :=
  type.isAppOf ``TorchLean.Tensor || type.isAppOf ``TorchLean.Tensor.Internal.Rep

/-- Expand to recorded primitives and retain their scalar operation recipe for certification. -/
private def record (scalar m inputShape : Lean.Expr) (tensorMode : Bool)
    (scope : List (Lean.Expr × Lean.Expr)) : Nat → Lean.Expr →
    RecordM (Lean.Expr × Lean.Expr)
  | 0, _ => throwError "recording: calculation exceeds the traversal limit"
  | fuel + 1, source => do
    let source := source.consumeMData.headBeta
    let sourceType ← inferType source
    let shape ← if !tensorMode then pure inputShape else do
      if isTensor sourceType then
        unless ← isDefEq sourceType.getAppArgs[0]! scalar do
          throwError "recording: element-type conversion needs a recorded derivative"
        pure sourceType.getAppArgs[1]!
      else
        unless ← isDefEq sourceType scalar do
          throwError "recording: expected a scalar or tensor calculation"
        pure (mkConst ``Spec.Shape.scalar)
    let inputType ← mkAppM ``TorchLean.Runtime.ValueRef #[m, scalar, shape]
    let resultType := mkApp m inputType
    let pureRef (value : Lean.Expr) : TermElabM Lean.Expr := do
      elabTermEnsuringType (← `(pure $(← exprToSyntax value))) (some resultType)
    if let some (_, ref) := scope.find? (·.1 == source) then
      return (← pureRef ref, source)
    let dependsOnInput := source.hasAnyFVar fun id =>
      scope.any fun entry => entry.1.isFVar && entry.1.fvarId! == id
    if !dependsOnInput && !source.hasLooseBVars && !source.hasMVar then
      let value ← if isTensor sourceType then exprToSyntax source else
        `(TorchLean.Tensor.full $(← exprToSyntax shape) $(← exprToSyntax source))
      let operation ← elabTermEnsuringType (← `(TorchLean.Runtime.const $value)) (some resultType)
      return (operation, source)
    let bind (value : Lean.Expr) (body : Lean.Expr → RecordM Lean.Expr) :
        RecordM Lean.Expr := do
      let refType := (← inferType value).getAppArgs.back!
      withLocalDeclD `value refType fun ref => do
        let body ← body ref
        synthesizeSyntheticMVarsNoPostponing
        let continuation ← mkLambdaFVars #[ref] (← instantiateMVars body)
        mkAppM ``Bind.bind #[value, continuation]
    if let .letE name type value body _ := source then
      let (value, recipe) ← record scalar m inputShape tensorMode scope fuel value
      let refType := (← inferType value).getAppArgs.back!
      return ← withLocalDeclD name type fun entry => do
        withLocalDeclD `value refType fun ref => do
          let (body, recordedRecipe) ←
            record scalar m inputShape tensorMode ((entry, ref) :: scope) fuel
              (body.instantiate1 entry)
          synthesizeSyntheticMVarsNoPostponing
          let continuation ← mkLambdaFVars #[ref] (← instantiateMVars body)
          let computation ← mkAppM ``Bind.bind #[value, continuation]
          return (computation, (← instantiateMVars recordedRecipe).replaceFVar entry recipe)
    let args := source.getAppArgs
    -- A locally named function is still an ordinary Lean function. Expand its binding,
    -- leaving the function body's shared scalar/tensor lets for the recorder below.
    if source.getAppFn.isFVar then
      let decl ← source.getAppFn.fvarId!.getDecl
      if let some value := decl.value? then
        return ← record scalar m inputShape tensorMode scope fuel (mkAppN value args).headBeta
    let operations := [( ``HAdd.hAdd, ``TorchLean.Runtime.add),
      (``HSub.hSub, ``TorchLean.Runtime.sub), (``HMul.hMul, ``TorchLean.Runtime.mul),
      (``HDiv.hDiv, ``TorchLean.Runtime.div)]
    if let some (_, operation) := operations.find? (some ·.1 == source.getAppFn.constName?) then
      let (left, leftRecipe) ←
        record scalar m inputShape tensorMode scope fuel args[args.size - 2]!
      let (right, rightRecipe) ← record scalar m inputShape tensorMode scope fuel args.back!
      let computation ← bind left fun left => bind right fun right => do
        let call := Syntax.mkCApp operation #[← exprToSyntax left, ← exprToSyntax right]
        elabTermEnsuringType call (some resultType)
      let left ← exprToSyntax leftRecipe
      let right ← exprToSyntax rightRecipe
      let scalarSyntax ← exprToSyntax sourceType
      let recipeSyntax ← if operation == ``TorchLean.Runtime.add then
          `(($left + $right : $scalarSyntax))
        else if operation == ``TorchLean.Runtime.sub then
          `(($left - $right : $scalarSyntax))
        else if operation == ``TorchLean.Runtime.mul then `(($left * $right : $scalarSyntax))
        else `(($left / $right : $scalarSyntax))
      let recipe ← elabTermEnsuringType recipeSyntax (some sourceType)
      return (computation, recipe)
    if source.isAppOf ``Neg.neg then
      let (value, recipe) ← record scalar m inputShape tensorMode scope fuel args.back!
      let computation ← bind value fun ref => do
        elabTermEnsuringType
          (Syntax.mkCApp ``TorchLean.Runtime.neg #[← exprToSyntax ref]) (some resultType)
      return (computation, ← mkAppM ``Neg.neg #[recipe])
    if tensorMode then
      let name := source.getAppFn.constName?.getD .anonymous
      let unary := [( ``TorchLean.Tensor.sum, ``TorchLean.Runtime.sum),
        (``TorchLean.Tensor.sumSpec, ``TorchLean.Runtime.sum),
        (``TorchLean.Tensor.relu, ``TorchLean.Runtime.relu),
        (``TorchLean.Tensor.sigmoid, ``TorchLean.Runtime.sigmoid),
        (``TorchLean.Tensor.tanh, ``TorchLean.Runtime.tanh)]
      if let some (_, operation) := unary.find? (·.1 == name) then
        let (value, recipe) ← record scalar m inputShape tensorMode scope fuel args.back!
        let computation ← bind value fun ref => do
          elabTermEnsuringType (Syntax.mkCApp operation #[← exprToSyntax ref]) (some resultType)
        let recipe ← mkAppM name #[recipe]
        return (computation, recipe)
      if name == ``TorchLean.Tensor.scalar then
        let (computation, recipe) ← record scalar m inputShape tensorMode scope fuel args.back!
        return (computation, ← mkAppM ``TorchLean.Tensor.scalar #[recipe])
      if name == ``TorchLean.Tensor.square || name == ``TorchLean.Tensor.squareSpec then
        let (value, recipe) ← record scalar m inputShape tensorMode scope fuel args.back!
        let computation ← bind value fun ref => do
          elabTermEnsuringType
            (← `(TorchLean.Runtime.mul $(← exprToSyntax ref) $(← exprToSyntax ref)))
            (some resultType)
        return (computation, ← mkAppM name #[recipe])
      if name == ``TorchLean.Tensor.matmul then
        let (left, leftRecipe) ←
          record scalar m inputShape tensorMode scope fuel args[args.size - 2]!
        let (right, rightRecipe) ← record scalar m inputShape tensorMode scope fuel args.back!
        let computation ← bind left fun left => bind right fun right => do
          elabTermEnsuringType
            (← `(TorchLean.Runtime.matmul (batchA := []) (batchB := []) (batch := [])
              (mDim := $(← exprToSyntax args[args.size - 5]!))
              (nDim := $(← exprToSyntax args[args.size - 4]!))
              (pDim := $(← exprToSyntax args[args.size - 3]!))
              $(← exprToSyntax left) $(← exprToSyntax right)))
            (some resultType)
        return (computation, ← mkAppM name #[leftRecipe, rightRecipe])
    if [``ite, ``dite].contains
        (source.getAppFn.constName?.getD .anonymous) then
      if source.isAppOf ``ite then
        let condition := args[1]!
        let dependent := condition.hasAnyFVar (fun id =>
          scope.any fun entry => entry.1.isFVar && entry.1.fvarId! == id)
        if dependent && (tensorMode || !(← isDefEq inputShape (mkConst ``Spec.Shape.scalar))) then
          throwError "recording: input-dependent branches need scalar traversal"
        let (yes, yesRecipe) ← record scalar m inputShape tensorMode scope fuel
          args[args.size - 2]!
        let (no, noRecipe) ← record scalar m inputShape tensorMode scope fuel args.back!
        let branch (condition decision : Lean.Expr) : RecordM Lean.Expr :=
          pure (mkAppN source.getAppFn #[resultType, condition, decision, yes, no])
        let rec observe (remaining : List (Lean.Expr × Lean.Expr))
            (condition decision : Lean.Expr) : RecordM Lean.Expr := do
          match remaining with
          | [] => branch condition decision
          | (entry, ref) :: remaining =>
            if entry.isFVar && condition.hasAnyFVar (· == entry.fvarId!) then
              let observed ← mkAppM ``Runtime.Autograd.Torch.Ops.observe #[ref]
              bind observed fun value => observe remaining (condition.replaceFVar entry value)
                (decision.replaceFVar entry value)
            else observe remaining condition decision
        let computation ← if dependent then observe scope condition args[2]! else
          branch condition args[2]!
        let recipe := mkAppN source.getAppFn
          #[sourceType, condition, args[2]!, yesRecipe, noRecipe]
        return (computation, recipe)
      throwError "recording: input-dependent branches need \
        faithful recorded primitives"
    if let some name := source.getAppFn.constName? then
      if ← isRecursiveDefinition name then
        unless !args.isEmpty &&
            (← isDefEq (← inferType args.back!) (mkConst ``Nat)) do
          throwError "recording: a recurrence needs a final natural-number count"
        let count := args.back!
        unless !(count.hasAnyFVar fun id =>
            scope.any fun entry => entry.1.isFVar && entry.1.fvarId! == id) do
          throwError "recording: the recurrence count must not depend on differentiable inputs"
        let some equations ← getEqnsFor? name |
          throwError "recording: no constructor equations for the recurrence"
        let apply (n : Lean.Expr) := mkAppN source.getAppFn (args.set! (args.size - 1) n)
        let (zero, zeroEquation) ← NN.Kernel.Frontend.equationBody (apply (mkNatLit 0)) equations
        let (initial, initialRecipe) ← record scalar m inputShape tensorMode scope fuel zero
        return ← withLocalDeclD `n (mkConst ``Nat) fun n => do
          let previous := apply n
          let (successor, successorEquation) ←
            NN.Kernel.Frontend.equationBody (apply (← mkAppM ``Nat.succ #[n])) equations
          let previous ← instantiateMVars previous
          withLocalDeclD `acc sourceType fun acc => do
            let body := successor.replace fun term =>
              if term == previous then some acc else none
            if (body.find? (·.isConstOf name)).isSome then
              throwError "recording: recursive calls must use the predecessor and fixed parameters"
            withLocalDeclD `value inputType fun ref => do
              let (body, bodyRecipe) ← record scalar m inputShape tensorMode
                ((acc, ref) :: scope) fuel body
              let step ← mkLambdaFVars #[n, ref] (← instantiateMVars body)
              let stepRecipe ← mkLambdaFVars #[n, acc] (← instantiateMVars bodyRecipe)
              let reference ← mkLambdaFVars #[n] previous
              let initialSyntax ← exprToSyntax initialRecipe
              let stepSyntax ← exprToSyntax stepRecipe
              let referenceSyntax ← exprToSyntax reference
              let countSyntax ← exprToSyntax count
              let zeroEquation := mkIdent zeroEquation
              let successorEquation := mkIdent successorEquation
              -- A generic shape may select homogeneous tensor addition, while a concrete shape
              -- selects its promotion instance. Prove their pointwise agreement, not new laws.
              let bridge ← elabTerm (← `(
                TorchLean.autograd.Internal.recurrence_eq $referenceSyntax $initialSyntax
                  $stepSyntax (by simp only [$zeroEquation:term])
                  (by
                    intro n
                    simp only [$successorEquation:term] <;> first
                    | rfl
                    | apply TorchLean.Tensor.Internal.Rep.ext
                      intro i
                      simp only [TorchLean.Tensor.Internal.Rep.hAdd_apply,
                        TorchLean.Tensor.Internal.Rep.add_apply,
                        TorchLean.Tensor.Internal.Rep.hSub_apply,
                        TorchLean.Tensor.Internal.Rep.sub_apply,
                        TorchLean.Tensor.Internal.Rep.mul_apply,
                        TorchLean.Tensor.Internal.Rep.div_apply,
                        TorchLean.Tensor.Internal.Rep.neg_apply]) $countSyntax)) none
              synthesizeSyntheticMVarsNoPostponing
              let bridge ← instantiateMVars bridge
              if bridge.hasSorry then throwError "recording: recurrence correspondence is unproved"
              let bridgeType ← inferType bridge
              let entries := scope.reverse.toArray.map (·.1) |>.filter fun entry =>
                entry.isFVar && bridgeType.hasAnyFVar (· == entry.fvarId!)
              let bridge ← mkLambdaFVars entries bridge
              modify (·.push bridge)
              let computation ← mkAppM ``TorchLean.autograd.Internal.iterate #[step, count, initial]
              let motive ← mkLambdaFVars #[n] sourceType
              let recipe ← mkAppOptM ``Nat.rec
                #[some motive, some initialRecipe, some stepRecipe, some count]
              return (computation, recipe)
    if let some unfolded ← unfoldDefinition? source then
      unless unfolded == source do
        return ← record scalar m inputShape tensorMode scope fuel unfolded
    throwError "recording: unsupported recorded calculation {source}; \
      use an existing Runtime operation"

end TorchLean.autograd.Frontend

namespace TorchLean.autograd.Internal

/-- Internal call-site translation; ordinary function definitions need no recording annotation. -/
syntax (name := programStx) "record_function% " term:max : term

open Lean Meta Elab Term

@[term_elab programStx]
def elaborateProgram : TermElab := fun stx expectedType? => withRef stx do
  let `(record_function% $source) := stx | throwUnsupportedSyntax
  let some expectedType := expectedType? | throwError "recording: missing program type"
  forallTelescopeReducing expectedType fun locals resultType => do
    let input := locals.back!
    let inputType ← inferType input
    unless inputType.isAppOf ``TorchLean.Runtime.ValueRef do
      throwError "recording: expected a reference input"
    let arguments := inputType.getAppArgs
    let scalar := arguments[1]!
    -- Expand a locally bound function before comparing its recurrence recipe. Otherwise
    -- simplification sees the local alias instead of the constructor equations we certified.
    let function ← whnf (← elabTerm source none)
    let .forallE _ domain _ _ ← whnf (← inferType function) |
      throwError "recording: expected an ordinary function"
    let tensorMode := Frontend.isTensor domain
    if tensorMode then
      let tensor ← mkConstWithFreshMVarLevels ``TorchLean.Tensor
      let tensorType := mkAppN tensor #[scalar, arguments.back!, arguments[2]!]
      unless ← isDefEq domain tensorType do
        throwError "recording: the tensor function accepts {domain}, but the input is {tensorType}"
    else
      unless ← isDefEq domain scalar do
        throwError "recording: the scalar function must preserve its element type"
    synthesizeSyntheticMVarsNoPostponing
    let function ← instantiateMVars function
    let domain ← instantiateMVars domain
    let (computation, proof) ← withLocalDeclD `entry domain fun entry => do
      let ((computation, recipe), bridges) ← (Frontend.record scalar arguments[0]!
        arguments.back! tensorMode [(entry, input)] 4096 ((mkApp function entry).headBeta)).run #[]
      synthesizeSyntheticMVarsNoPostponing
      let computation ← instantiateMVars computation
      let recipe ← instantiateMVars recipe
      let equality ← mkEq recipe (mkApp function entry)
      let proof ← if ← isDefEq recipe (mkApp function entry) then
          mkExpectedTypeHint (← mkEqRefl recipe) equality
        else
          let goal ← mkFreshExprSyntheticOpaqueMVar equality
          let bridges ← bridges.mapM fun bridge => do
            let term ← exprToSyntax bridge
            `(Parser.Tactic.simpLemma| $term:term)
          let remaining ← Lean.Elab.Tactic.run goal.mvarId! do
            Lean.Elab.Tactic.evalTactic (← `(tactic| simp only [$[$bridges],*]))
          unless remaining.isEmpty do
            throwError "recording: translated operations do not match the original function:\n\
              {equality}"
          instantiateMVars goal
      return (computation, ← mkLambdaFVars #[entry] proof)
    discard <| mkAuxTheorem (← inferType proof) proof (zetaDelta := true)
      (kind? := (← getMainModule) ++ `_recordedArithmetic)
    unless ← isDefEq (← inferType computation) resultType do
      throwError "recording: output shape does not match the function"
    mkLambdaFVars locals computation

end TorchLean.autograd.Internal
