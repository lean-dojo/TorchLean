/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Session.Selection
public import TorchLean.Tensor.Shape
public import TorchLean.NN.ScalarTrainer.EagerOps
public import TorchLean.NN.Session.Optimizer
public meta import Lean.Meta.Closure
public meta import Lean.Meta.Eqns
public meta import TorchLean.CUDA.Kernel.Elab
public import TorchLean.CUDA.Kernel.Elab
public import TorchLean.Tensor.Promotion

/-!
# Recording ordinary calculations on the autograd tape

`Function.run` uses this internal frontend when recording an ordinary scalar or tensor function.
It maps `+`, `-`, `*`, `/` and negation to `Op.add`, `Op.sub`, `Op.mul`, `Op.div` and `Op.neg`,
and on tensors `Tensor.relu`, `Tensor.sigmoid`, `Tensor.tanh`, `Tensor.sum` and `Tensor.matmul`
to their operators and `Tensor.square` to `Op.mul` of the operand with itself. Other definitions
are unfolded and recorded through their bodies, and a term that does not unfold is rejected.
Shared local bindings stay shared. Eager execution records the calls on its tape, so the usual
differentiation engine and gradient representation handle them.

Every recording of a function `f` adds a kernel-checked theorem about `f` itself: the recorded
program, run by the pure interpreter `TorchLean.instOpsId` in the recording's element context, maps
every input `t` to `Tensor.map f t` for a scalar function, to `f t` for a tensor function and to
`Tensor.scalar (f t)` for a function from a tensor to a scalar. The interpreter applies each
operator's rule, and no other `Ops Id` instance in scope takes its place. The recorder proves the
theorem node by node. At every recorded operation it proves that the operator's forward, in the
recording's element context and at arbitrary operands, equals the source application with the
source's own instances: definitionally within a heartbeat bound, through FloatLib's
`Capability.run_eq_spec` when two capabilities of a configured format have one specification, or by
an operator lemma such as `pureApply_matmul`. An operation whose source instance computes something
other than the operator in that context, for example a local `Mul` instance that subtracts, is an
error. No ring simplification, reassociation, precision conversion or speculative derivative is
used.

The theorem covers the pure interpreter only. When a scalar function does not record at its input's
shape, `Function.run` records it at shape `[]` and lifts that recording to every entry with
`Recording.elementwise`, and no theorem covers the lifted program. That the eager tape computes the
same values, and that its backward is the derivative, rest on the local derivative laws and the
graph-composition theorems and are not part of this check. A foreign backend's implementation lies
outside it. The element context is the recording's own, which `Function.run` finds by instance
search, local instances included. The theorem holds in that context, and the gradient is the
backward of the operators' rules in it. The derivative laws are proved over `ℝ`, so in a context
whose multiplication is not the scalar's the gradient need not be the derivative of the source.

The call site's input supplies the element type and shape. Captured scalars and tensors are
constants and receive no gradient. Scalar conditionals inspect the current value and record only
the selected branch. On GPU, this decision synchronizes to Lean. Fixed-parameter recurrences
record their steps on the same tape. Neither feature supplies a derivative at a discontinuous
branch boundary.
-/

public section

namespace TorchLean.autograd

/--
A forward value and its recorded input pullback. `Result.backward` consumes the recording, and
`close` releases it without differentiation. The value remains usable after either operation.
-/
structure Result (α : Type) [Storage α] (input output : Shape) where
  /-- Materialized forward value, retaining the requested element type. -/
  value : Tensor α output
  /-- Effective device, including a precision-preserving CPU fallback. -/
  device : Backend.Device
  /-- Consume the recording's tape with an explicit output cotangent. -/
  pullback : Tensor α output → IO (Tensor α input)
  /-- Release an unused recording. Calling this more than once is harmless. -/
  close : IO Unit

/--
Differentiate with the cotangent `seed`, which defaults to all ones, so that the default
differentiates the sum of the output entries. The recording is released even when backward fails,
and a consumed recording cannot be reused. PyTorch analogue:
`torch.autograd.grad(output, input, grad_outputs=seed)`.
-/
def Result.backward {α : Type} [Storage α] [One α] {input output : Shape}
    (result : Result α input output) (seed : Tensor α output := Tensor.full output 1) :
    IO (Tensor α input) :=
  result.pullback seed

instance {α : Type} [Storage α] {input output : Shape} [Repr (Tensor α output)] :
    Repr (Result α input output) where
  reprPrec result prec := reprPrec result.value prec

/-- Record each step on the tape, retaining the order of a scalar or tensor recurrence. -/
def iterate {m : Type → Type} [Monad m] {β : Type} (step : Nat → β → m β) :
    Nat → m β → m β
  | 0, initial => initial
  | n + 1, initial => do
    let value ← iterate step n initial
    step n value

/-- Induction over `iterate` in the pure monad `Id`: a relation that holds at step `0` for
`initial` and that every `step` preserves holds after any number of steps. -/
theorem iterate_induction {β : Type} (R : Nat → β → Prop) (step : Nat → β → Id β)
    (initial : Id β) (zero : R 0 initial)
    (succ : ∀ n value, R n value → R (n + 1) (step n value)) :
    ∀ n, R n (iterate step n initial) := by
  intro n
  induction n with
  | zero => exact zero
  | succ n ih => exact succ n _ ih

/-- A function `f` with `f 0 = initial` and `f (n + 1) = step n (f n)` for every `n` equals the
recursion `Nat.rec initial step` at every `n`. The equations impose no algebraic law or
reassociation. -/
theorem recurrence_eq {α : Type} (f : Nat → α) (initial : α) (step : Nat → α → α)
    (hzero : f 0 = initial) (hsucc : ∀ n, f (n + 1) = step n (f n)) (n : Nat) :
    Nat.rec initial step n = f n :=
  iterate_induction (fun n (_ : α) => Nat.rec initial step n = f n) step initial hzero.symm
    (fun n _ ih => by simpa only [Nat.rec_add_one, ih] using (hsucc n).symm) n

/-- The pure interpreter's product of two matrices, without batch axes and for any capability
record and broadcast proofs, is `Tensor.matmul` with the element context's arithmetic. -/
theorem pureApply_matmul {α : Type} [Storage α] [Context α] {m n p : Nat}
    (hl : false = true → m = 1) (hr : false = true → p = 1)
    (broadcastA broadcastB : Shape.CanBroadcastTo .scalar .scalar) (caps : Op.Capabilities α)
    (A : Tensor α [m, n]) (B : Tensor α [n, p]) :
    Ops.pureApply (Op.matmul (batchA := .scalar) (batchB := .scalar) (batch := .scalar) false
      false hl hr broadcastA broadcastB) caps (A, B, ()) = Tensor.matmul A B := by
  have h : Tensor.batchMatmul broadcastA broadcastB A B = Tensor.matmul A B := by
    simp only [Tensor.batchMatmul, Tensor.broadcast_self, Tensor.matmulCommonBatch_scalar]
  exact h

/-- An element context and a program written against the `Ops` interface. -/
structure Recording (α : Type) [Storage α] (input output : Shape) where
  /-- The element context the program runs in. -/
  context : Context α
  /-- The program from an input reference to an output reference, for every monad with an
  `Ops` instance over `context`. -/
  program : ∀ {m : Type → Type}, [Monad m] →
    [@Ops m α _ context] →
    @Ops.Ref m α _ context _ input →
    m (@Ops.Ref m α _ context _ output)

/--
Lift scalar control flow to every entry of a tensor through `Ops.mapBatch`, a
balanced traversal of the flattened entries. No unselected branch is evaluated. GPU arithmetic and
backward stay on the GPU, and scalar decisions synchronize to Lean.
-/
def Recording.elementwise {α : Type} [Storage α] {shape : Shape}
    (recording : Recording α [] []) : Recording α shape shape :=
  letI : Context α := recording.context
  { context := recording.context
    program := fun {_m} _ _ x => do
      let flat ← Ops.apply
          (.reshape (t := [shape.size]) (by simp)) (x, ())
      let output ← Ops.mapBatch
        (σ := []) (τ := []) recording.program flat
      Ops.apply (.reshape (t := shape) (by simp)) (output, ()) }

/-- Execute in a fresh eager session, owned until backward or explicit close. -/
def Recording.run {α : Type} [Storage α]
    [transfer : TensorTransfer α]
    {input output : Shape} (recording : Recording α input output)
    (value : Tensor α input) (device : Backend.Device) (grad : Bool := true) :
    IO (Result α input output) :=
  letI : Context α := recording.context
  letI : TensorTransfer α :=
    if TensorTransfer.supportsEncodedGPU (α := α) then
      { transfer with dtype? := some .encoded }
    else transfer
  do
    let session ← Session.create (α := α) { device := device }
    try
      let argument ← session.input value (requiresGrad := grad)
      let result ← recording.program (m := SessionM α)
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
          Session.grad gradients argument
        finally
          session.resetTape
      return ⟨value, session.options.device, pullback, close⟩
    catch error =>
      session.resetTape
      throw error

end TorchLean.autograd

public meta section

namespace TorchLean.autograd

open Lean Meta Elab Term

-- Arithmetic may expose the representation behind the public tensor alias. Both carry the full
-- shape in their type, so the rank never has to be inferred from a flattened buffer's length.
private def isTensor (type : Lean.Expr) : Bool :=
  type.isAppOf ``TorchLean.Tensor || type.isAppOf ``Tensor.Rep

/-- Fail unless `source` applies `name` to all the arguments of its type. -/
private def expectArity (name : Name) (source : Lean.Expr) : MetaM Unit := do
  let arity := (← getConstInfo name).type.getForallArity
  unless source.getAppNumArgs == arity do
    throwError "recording: `{.ofConstName name}` is applied to {source.getAppNumArgs} \
      arguments, expected {arity}"

/-- The error for an operation that is not what its operator computes in the recording. -/
private def reject {β : Type} (source : Lean.Expr) (operator : Name) (scalar : Lean.Expr) :
    MetaM β :=
  throwError "recording: the operation{indentExpr source}\ndoes not compute what \
    `{.ofConstName operator}` computes on `{scalar}` in the recording's context. An operation \
    is recorded only when its instance agrees with that context's."

/-- Run `k` with fresh local variables of the given types. -/
private def withFresh {β : Type} (types : Array Lean.Expr) (k : Array Lean.Expr → TermElabM β) :
    TermElabM β :=
  go types.toList #[]
where
  go : List Lean.Expr → Array Lean.Expr → TermElabM β
    | [], xs => k xs
    | type :: rest, xs =>
      withLocalDeclD (.mkSimple s!"w{xs.size}") type fun x => go rest (xs.push x)

/-- The recording's interface, the pure interpreter's instances, and the coordinate at which
scalar nodes are related to their values. -/
private structure Env where
  scalar : Lean.Expr
  storage : Lean.Expr
  context : Lean.Expr
  monad : Lean.Expr
  monadInst : Lean.Expr
  ops : Lean.Expr
  idM : Lean.Expr
  idMonad : Lean.Expr
  idOps : Lean.Expr
  inputShape : Lean.Expr
  tensorMode : Bool
  coord : Lean.Expr
  deriving Inhabited

/-- A source variable in scope: its reference, its value in `Id` and the proof relating them. -/
private structure ScopeEntry where
  source : Lean.Expr
  ref : Lean.Expr
  value : Lean.Expr
  proof : Lean.Expr
  shape : Lean.Expr
  tensorKind : Bool
  deriving Inhabited

/-- A recorded node: the computation, its value in `Id`, and a proof of `rel value source`. -/
private structure Recorded where
  comp : Lean.Expr
  value : Lean.Expr
  proof : Lean.Expr
  shape : Lean.Expr
  tensorKind : Bool
  deriving Inhabited

private def tensorTy (env : Env) (shape : Lean.Expr) : Lean.Expr :=
  mkAppN (mkConst ``TorchLean.Tensor [Level.zero]) #[env.scalar, shape, env.storage]

private def getAt (env : Env) (shape v c : Lean.Expr) : Lean.Expr :=
  mkAppN (mkConst ``Tensor.Rep.get [Level.zero]) #[env.scalar, env.storage, shape, v, c]

private def refType (env : Env) (shape : Lean.Expr) : Lean.Expr :=
  mkAppN (mkConst ``Ops.Ref) #[env.monad, env.scalar, env.storage, env.context, env.ops, shape]

/-- The relation between a node's value and its source: `v = s` for tensor nodes, and for scalar
nodes `v` at the recording's coordinate is `s`. -/
private def rel (env : Env) (shape : Lean.Expr) (tensorKind : Bool) (v s : Lean.Expr) :
    Lean.Expr :=
  if tensorKind then mkApp3 (mkConst ``Eq [Level.one]) (tensorTy env shape) v s
  else mkApp3 (mkConst ``Eq [Level.one]) env.scalar (getAt env shape v env.coord) s

/-- The computation `e` in the pure monad `Id` and its `Ops` instance. -/
private def idSubst (env : Env) (e : Lean.Expr) : Lean.Expr :=
  e.replaceFVars #[env.monad, env.monadInst, env.ops] #[env.idM, env.idMonad, env.idOps]

/-- Run `k` with a budget of `budget` thousand heartbeats, counted from the call. The result is
`none` on any failure, and then the state is restored. -/
private def bounded {β : Type} (budget : Nat) (k : MetaM β) : MetaM (Option β) := do
  let saved ← saveState
  tryCatchRuntimeEx
    (withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := budget * 1000 })
      (withCurrHeartbeats (some <$> k)))
    (fun _ => do saved.restore; return none)

/-- Bounded definitional equality at `mode`, treating the caller's metavariables as constants. -/
private def boundedDefEq (mode : TransparencyMode) (budget : Nat) (a b : Lean.Expr) :
    MetaM Bool :=
  return (← bounded budget (withNewMCtxDepth (withTransparency mode (isDefEq a b)))) == some true

/-- `rfl` at the type `type`. -/
private def reflHint (a type : Lean.Expr) : MetaM Lean.Expr := do
  mkExpectedTypeHint (← mkEqRefl a) type

private def getOf (ty inst shape x c : Lean.Expr) : MetaM Lean.Expr := do
  return mkAppN (mkConst ``Tensor.Rep.get [← getDecLevel ty]) #[ty, inst, shape, x, c]

private def pwStop (e : Lean.Expr) : Bool :=
  e.isFVar || e.isAppOfArity ``Tensor.zipWith 10 || e.isAppOfArity ``Tensor.map 7 ||
    e.isAppOfArity ``Tensor.full 4 || e.isAppOfArity ``Tensor.Pointwise.apply 11 ||
    e.isAppOfArity ``Tensor.Rep.ofFn 4

/-- The entry of the tensor `e` at `c` as a scalar expression, with a proof that
`get e c` is that expression. -/
private partial def pw (ty inst shape e c : Lean.Expr) : MetaM (Lean.Expr × Lean.Expr) := do
  let r ← whnfHeadPred e (fun x => return !pwStop x)
  let a := r.getAppArgs
  let lhs ← getOf ty inst shape e c
  let finish (lem step rhs : Lean.Expr) : MetaM (Lean.Expr × Lean.Expr) := do
    return (rhs, ← mkExpectedTypeHint (← mkEqTrans lem step) (← mkEq lhs rhs))
  if r.isAppOfArity ``Tensor.zipWith 10 then
    let lem := mkAppN (mkConst ``Tensor.zipWith_apply r.getAppFn.constLevels!) (a.push c)
    let (ex, px) ← pw a[0]! a[1]! a[6]! a[8]! c
    let (ey, py) ← pw a[2]! a[3]! a[6]! a[9]! c
    finish lem (← mkCongr (← mkCongrArg a[7]! px) py) (mkApp2 a[7]! ex ey).headBeta
  else if r.isAppOfArity ``Tensor.map 7 then
    let lem := mkAppN (mkConst ``Tensor.map_apply r.getAppFn.constLevels!) (a.push c)
    let (ex, px) ← pw a[0]! a[1]! a[4]! a[6]! c
    finish lem (← mkCongrArg a[5]! px) (mkApp a[5]! ex).headBeta
  else if r.isAppOfArity ``Tensor.full 4 then
    let lem := mkAppN (mkConst ``Tensor.full_apply r.getAppFn.constLevels!) (a.push c)
    return (a[3]!, ← mkExpectedTypeHint lem (← mkEq lhs a[3]!))
  else if r.isAppOfArity ``Tensor.Pointwise.apply 11 then
    let lem := mkAppN (mkConst ``Tensor.Pointwise.apply_at) (a.push c)
    let f := mkAppN (mkConst ``Tensor.Pointwise.scalar) (a.extract 0 8)
    let (ex, px) ← pw a[1]! a[4]! a[8]! a[9]! c
    let (ey, py) ← pw a[2]! a[5]! a[8]! a[10]! c
    finish lem (← mkCongr (← mkCongrArg f px) py) (mkApp2 f ex ey)
  else if r.isAppOfArity ``Tensor.Rep.ofFn 4 then
    let lem := mkAppN (mkConst ``Tensor.Rep.get_ofFn r.getAppFn.constLevels!) (a.push c)
    let rhs := (mkApp a[3]! c).headBeta
    return (rhs, ← mkExpectedTypeHint lem (← mkEq lhs rhs))
  else
    let rhs ← getOf ty inst shape r c
    return (rhs, ← reflHint lhs (← mkEq lhs rhs))

mutual
/-- A proof of `lhs = rhs`: definitional at instance transparency, through two FloatLib
capabilities with the same specification, or definitional within a heartbeat bound. -/
private partial def agree (lhs rhs : Lean.Expr) : MetaM (Option Lean.Expr) := do
  let goal ← mkEq lhs rhs
  if lhs == rhs then return some (← reflHint lhs goal)
  if ← boundedDefEq .instances 2000 lhs rhs then return some (← reflHint lhs goal)
  let l? ← bounded 2000 (whnfUntil lhs ``FloatLib.Floats.ExecFloat.Capability.run)
  let r? ← bounded 2000 (whnfUntil rhs ``FloatLib.Floats.ExecFloat.Capability.run)
  if let (some (some l), some (some r)) := (l?, r?) then
    if let some p ← agreeRun l r then
      return some (← mkExpectedTypeHint p goal)
  if ← boundedDefEq .default 20000 lhs rhs then return some (← reflHint lhs goal)
  return none

/-- A proof that two applications of `Capability.run` agree: same format, encoding and operation,
specifications that agree definitionally, and arguments that `agree`. -/
private partial def agreeRun (l r : Lean.Expr) : MetaM (Option Lean.Expr) := do
  let la := l.getAppArgs
  let ra := r.getAppArgs
  unless la.size == ra.size && la.size ≥ 5 do return none
  unless ← boundedDefEq .default 2000 la[0]! ra[0]! do return none
  unless ← boundedDefEq .instances 2000 la[1]! ra[1]! do return none
  unless ← boundedDefEq .default 2000 la[3]! ra[3]! do return none
  let lv := l.getAppFn.constLevels!
  let spec (x : Array Lean.Expr) :=
    mkAppN (mkConst ``FloatLib.Floats.ExecFloat.Capability.spec lv) (x.extract 0 5)
  unless ← boundedDefEq .default 20000 (spec la) (spec ra) do return none
  let runSpec (x : Array Lean.Expr) :=
    mkAppN (mkConst ``FloatLib.Floats.ExecFloat.Capability.run_eq_spec lv) (x.extract 0 5)
  let h ← reflHint (spec la) (← mkEq (spec la) (spec ra))
  let mut proof ← mkEqTrans (runSpec la) (← mkEqTrans h (← mkEqSymm (runSpec ra)))
  for i in [5:la.size] do
    let some p ← agree la[i]! ra[i]! | return none
    proof ← mkCongr proof p
  return some proof
end

/-- `f a₁ … aₙ = f b₁ … bₙ` from proofs of `aᵢ = bᵢ`. -/
private def congrN (f : Lean.Expr) (proofs : Array Lean.Expr) : MetaM Lean.Expr := do
  let mut h ← mkCongrArg f proofs[0]!
  for p in proofs.extract 1 proofs.size do
    h ← mkCongr h p
  return h

mutual
/-- A proof that two instances of one class agree: definitional at instance transparency, or field
by field (function fields by extensionality, values through `agree`). -/
private partial def instEq (a b : Lean.Expr) : MetaM (Option Lean.Expr) := do
  let goal ← mkEq a b
  if a == b then return some (← reflHint a goal)
  if ← boundedDefEq .instances 2000 a b then return some (← reflHint a goal)
  let type ← whnfR (← inferType a)
  let .const cls us := type.getAppFn | return none
  let env ← getEnv
  let some info := getStructureInfo? env cls | return none
  let params := type.getAppArgs
  let mut proofs := #[]
  for field in info.fieldNames do
    let some proj := getProjFnForField? env cls field | return none
    let some p ← fieldEq (mkApp (mkAppN (mkConst proj us) params) a)
      (mkApp (mkAppN (mkConst proj us) params) b) | return none
    proofs := proofs.push p
  if proofs.isEmpty then return none
  let ctor := mkAppN (mkConst (getStructureCtor env cls).name us) params
  let some p ← (try some <$> congrN ctor proofs catch _ => pure none) | return none
  return some (← mkExpectedTypeHint p goal)

/-- A proof that two fields of instances agree. -/
private partial def fieldEq (l r : Lean.Expr) : MetaM (Option Lean.Expr) := do
  let type ← inferType l
  if ← isProp type then return some (← reflHint l (← mkEq l r))
  if (← isClass? type).isSome then return ← instEq l r
  forallTelescopeReducing type fun xs _ => do
    let some p ← agree (mkAppN l xs).headBeta (mkAppN r xs).headBeta | return none
    let mut p := p
    for x in xs.reverse do
      p ← mkFunExt (← mkLambdaFVars #[x] p)
    return some p
end

/-- The recording context's instance of the class `type`, when `Context` extends it. -/
private def contextInstance (env : Env) (type : Lean.Expr) : MetaM (Option Lean.Expr) := do
  let .const cls _ := type.getAppFn | return none
  let some path := getPathToBaseStructure? (← getEnv) cls ``TorchLean.Context | return none
  let mut e := env.context
  for proj in path do
    let arity := (← getConstInfo proj).type.getForallArity
    e ← mkAppOptM proj ((Array.replicate (arity - 1) none).push (some e))
  return some e

/-- The application `source` with each instance argument of a class that `Context` extends
replaced by the recording context's, and a proof that the result equals `source`. -/
private def contextForm (env : Env) (source : Lean.Expr) :
    MetaM (Option (Lean.Expr × Lean.Expr)) := do
  let fn := source.getAppFn
  let args := source.getAppArgs
  let info ← getFunInfoNArgs fn args.size
  let mut positions := #[]
  let mut replaced := #[]
  let mut proofs := #[]
  for i in [0:args.size] do
    let some param := info.paramInfo[i]? | continue
    unless param.binderInfo.isInstImplicit && !param.hasFwdDeps do continue
    let some c ← contextInstance env (← inferType args[i]!) | continue
    if c == args[i]! then continue
    let some p ← instEq c args[i]! | return none
    positions := positions.push i
    replaced := replaced.push c
    proofs := proofs.push p
  if positions.isEmpty then return some (source, ← reflHint source (← mkEq source source))
  let types ← positions.mapM fun i => inferType args[i]!
  let lam ← withLocalDeclsDND (types.map (`z, ·)) fun zs => do
    let mut a := args
    for (i, z) in positions.zip zs do
      a := a.set! i z
    mkLambdaFVars zs (mkAppN fn a)
  let form := lam.beta replaced
  return some (form, ← mkExpectedTypeHint (← congrN lam proofs) (← mkEq form source))

/-- Operators whose forward in `Id` is related to the source operation by a lemma. -/
private def forwardLemmas : List (Name × Name) :=
  [(``TorchLean.Op.matmul, ``TorchLean.autograd.pureApply_matmul)]

/-- `forward = rhs` by the forward lemma of `operator`, if it has one that applies. -/
private def forwardStep (operator : Name) (forward : Lean.Expr) :
    MetaM (Option (Lean.Expr × Lean.Expr)) := do
  let some (_, name) := forwardLemmas.find? (·.1 == operator) | return none
  let lem ← mkConstWithFreshMVarLevels name
  let (ms, _, type) ← forallMetaTelescopeReducing (← inferType lem)
  let some (_, lhs, rhs) := type.eq? | return none
  let unify (mode : TransparencyMode) (budget : Nat) : MetaM Bool := do
    return (← bounded budget (withTransparency mode (isDefEq lhs forward))) == some true
  unless (← unify .instances 2000) || (← unify .default 20000) do return none
  let rhs ← instantiateMVars rhs
  let proof ← instantiateMVars (mkAppN lem ms)
  if rhs.hasExprMVar || proof.hasExprMVar then return none
  return some (rhs, ← mkExpectedTypeHint proof (← mkEq forward rhs))

/-- A proof of `rel (top at fresh operands) (source head at fresh operands)`, or the error of
`reject`. Nodes marked `exact` are compared as a whole after the context's instances replace the
source's. The others are also compared entry by entry (`pw`, `agree`). -/
private def proveFact (env : Env) (exact : Bool) (shape topW : Lean.Expr) (tensorKind : Bool)
    (srcW source : Lean.Expr) (operator : Name) : MetaM Lean.Expr := do
  let lhs := if tensorKind then topW else getAt env shape topW env.coord
  let goal := rel env shape tensorKind topW srcW
  if ← boundedDefEq .instances 2000 lhs srcW then return ← reflHint lhs goal
  let fallback : MetaM Lean.Expr := do
    if ← boundedDefEq .default 20000 lhs srcW then reflHint lhs goal
    else reject source operator env.scalar
  if exact then
    let some (form, hForm) ← contextForm env srcW | reject source operator env.scalar
    let (rhs, hForward) ← match ← forwardStep operator lhs with
      | some r => pure r
      | none => pure (lhs, ← reflHint lhs (← mkEq lhs lhs))
    unless (← boundedDefEq .instances 2000 rhs form) ||
        (← boundedDefEq .default 20000 rhs form) do
      reject source operator env.scalar
    let middle ← reflHint rhs (← mkEq rhs form)
    return ← mkExpectedTypeHint (← mkEqTrans hForward (← mkEqTrans middle hForm)) goal
  if !tensorKind then
    let some (e, p) ← bounded 20000 (pw env.scalar env.storage shape topW env.coord) |
      return ← fallback
    let some q ← agree e srcW | reject source operator env.scalar
    return ← mkExpectedTypeHint (← mkEqTrans p q) goal
  withLocalDeclD `i (← mkAppM ``Shape.Coord #[shape]) fun i => do
    let some (e₁, p₁) ← bounded 20000 (pw env.scalar env.storage shape topW i) | fallback
    let some (e₂, p₂) ← bounded 20000 (pw env.scalar env.storage shape srcW i) | fallback
    let some q ← agree e₁ e₂ | reject source operator env.scalar
    let pi ← mkExpectedTypeHint (← mkEqTrans p₁ (← mkEqTrans q (← mkEqSymm p₂)))
      (← mkEq (getAt env shape topW i) (getAt env shape srcW i))
    mkExpectedTypeHint (mkAppN (mkConst ``Tensor.Rep.ext [Level.zero])
      #[env.scalar, env.storage, shape, topW, srcW, ← mkLambdaFVars #[i] pi]) goal

/-- The `Ops.apply` call built by `build` from the argument tuple of `refs`. -/
private def call (resultType : Lean.Expr) (refs : Array Lean.Expr)
    (build : Syntax.Term → TermElabM Syntax.Term) : TermElabM Lean.Expr := do
  let mut tuple ← `(())
  for ref in refs.reverse do
    tuple ← `(($(← exprToSyntax ref), $tuple))
  let value ← elabTermEnsuringType (← build tuple) (some resultType)
  synthesizeSyntheticMVarsNoPostponing
  let value ← instantiateMVars value
  if value.hasExprMVar then throwError "recording: unresolved operator call {value}"
  let some apply := value.find? (·.isAppOfArity ``Ops.apply 11) |
    throwError "recording: no operator call in {value}"
  let applyArgs := apply.getAppArgs
  if (applyArgs.extract 5 10).any (·.hasAnyFVar fun id => refs.any (·.fvarId! == id)) then
    throwError "recording: the operator {applyArgs[8]!} reads its operands"
  return value

/-- Bind the computations of `children` in order and continue with their references. -/
private partial def bindAll (env : Env) (children : List Recorded) (refs : Array Lean.Expr)
    (k : Array Lean.Expr → TermElabM Lean.Expr) : TermElabM Lean.Expr := do
  match children with
  | [] => k refs
  | c :: rest =>
    withLocalDeclD `value (refType env c.shape) fun ref => do
      let body ← bindAll env rest (refs.push ref) k
      mkAppM ``Bind.bind #[c.comp, ← mkLambdaFVars #[ref] body]

/-- Observe the scalar values of the scope entries a branch condition reads, newest last. -/
private partial def observeAll (env : Env) (branch : Lean.Expr → Lean.Expr → Lean.Expr) :
    List ScopeEntry → Lean.Expr → Lean.Expr → TermElabM Lean.Expr
  | [], c, dec => return branch c dec
  | e :: rest, c, dec => do
    let observed := mkAppN (mkConst ``Ops.observe)
      #[env.monad, env.scalar, env.storage, env.context, env.ops, e.ref]
    withLocalDeclD `o env.scalar fun o => do
      let body ← observeAll env branch rest (c.replaceFVar e.source o)
        (dec.replaceFVar e.source o)
      mkAppM ``Bind.bind #[observed, ← mkLambdaFVars #[o] body]

/-- The constructor equation of a recurrence at `source`: its body and a proof that `source`
equals it. -/
private def equation (source : Lean.Expr) (equations : Array Name) :
    TermElabM (Lean.Expr × Lean.Expr) := do
  try
    let (body, _, proof) ← Kernel.Elab.equationBody source equations
    return (body, proof)
  catch _ => throwError "recording: no constructor equation for{indentExpr source}"

/-- Record `source` as computations on the `Ops` interface. The result also holds the value of
the computation in `Id` and a proof that it is `source`, relative to the scope's values. -/
private partial def record (env : Env) (scope : List ScopeEntry) (fuel : Nat)
    (source : Lean.Expr) : TermElabM Recorded := do
  if fuel == 0 then throwError "recording: calculation exceeds the traversal limit"
  let fuel := fuel - 1
  let source := source.consumeMData.headBeta
  let sourceType ← instantiateMVars (← inferType source)
  let typeEq (a b : Lean.Expr) : MetaM Bool := boundedDefEq .default 20000 a b
  let (shape, tensorKind) ← if !env.tensorMode then do
      unless ← typeEq sourceType env.scalar do
        throwError "recording: `{source}` has type `{sourceType}`, but a scalar function \
          records calculations in `{env.scalar}`"
      pure (env.inputShape, false)
    else if isTensor sourceType then do
      unless ← typeEq sourceType.getAppArgs[0]! env.scalar do
        throwError "recording: element-type conversion needs a recorded derivative"
      let shape := sourceType.getAppArgs[1]!
      let expected := tensorTy env shape
      unless ← typeEq sourceType expected do
        throwError "recording: `{source}` has type `{sourceType}`, but the recording holds \
          `{expected}`"
      pure (shape, true)
    else do
      unless ← typeEq sourceType env.scalar do
        throwError "recording: expected a scalar or tensor calculation"
      pure (mkConst ``Shape.scalar, false)
  let resultType := mkApp env.monad (refType env shape)
  let goalOf (v : Lean.Expr) := rel env shape tensorKind v source
  let inScope (e : Lean.Expr) : Bool :=
    e.hasAnyFVar fun id => scope.any (·.source.fvarId! == id)
  if let some entry := scope.find? (·.source == source) then
    let comp ← mkAppOptM ``Pure.pure #[some env.monad, none, none, some entry.ref]
    return { comp, value := entry.value, proof := entry.proof, shape, tensorKind }
  if !inScope source && !source.hasLooseBVars && !source.hasMVar then
    let value := if tensorKind then source else
      mkAppN (mkConst ``Tensor.full [Level.zero]) #[env.scalar, env.storage, shape, source]
    let comp := mkAppN (mkConst ``Ops.const)
      #[env.monad, env.scalar, env.storage, env.context, env.ops, shape, value]
    let proof ← if tensorKind then reflHint source (goalOf value) else
      mkExpectedTypeHint (mkAppN (mkConst ``Tensor.full_apply [Level.zero])
        #[env.scalar, env.storage, shape, source, env.coord]) (goalOf value)
    return { comp, value, proof, shape, tensorKind }
  if let .letE name type val body _ := source then
    let e ← record env scope fuel val
    return ← withLocalDeclD name type fun y => do
      withLocalDeclD `value (refType env e.shape) fun ref => do
      withLocalDeclD `v (tensorTy env e.shape) fun v => do
      withLocalDeclD `h (rel env e.shape e.tensorKind v y) fun h => do
        let entry : ScopeEntry :=
          { source := y, ref, value := v, proof := h, shape := e.shape,
            tensorKind := e.tensorKind }
        let b ← record env (entry :: scope) fuel (body.instantiate1 y)
        if b.value.containsFVar y.fvarId! || b.value.containsFVar h.fvarId! ||
            b.value.containsFVar ref.fvarId! || b.proof.containsFVar ref.fvarId! ||
            b.comp.containsFVar v.fvarId! || b.comp.containsFVar y.fvarId! ||
            b.comp.containsFVar h.fvarId! then
          throwError "recording: internal error, the recorded let body mixes levels"
        let comp ← mkAppM ``Bind.bind #[e.comp, ← mkLambdaFVars #[ref] b.comp]
        let value := Lean.mkLet `v (tensorTy env e.shape) e.value (b.value.abstract #[v])
        let proof ← mkExpectedTypeHint
          (mkAppN (← mkLambdaFVars #[v, y, h] b.proof) #[e.value, val, e.proof]) (goalOf value)
        return { comp, value, proof, shape, tensorKind }
  let args := source.getAppArgs
  let fn := source.getAppFn
  -- A locally named function is an ordinary Lean function: record its body.
  if fn.isFVar then
    if let some value := (← fn.fvarId!.getDecl).value? then
      let r ← record env scope fuel (mkAppN value args).headBeta
      return { r with proof := ← mkExpectedTypeHint r.proof (goalOf r.value) }
  let head := fn.constName?.getD .anonymous
  -- An operator node: record the last `k` arguments, call the operator on their references in
  -- the order `pattern`, and prove that its forward is the source head at arbitrary operands.
  let opNode (k : Nat) (pattern : Array Nat) (exact : Bool) (operator : Name)
      (build : Syntax.Term → TermElabM Syntax.Term) : TermElabM Recorded := do
    expectArity head source
    let n := args.size
    let srcF := mkAppN fn (args.extract 0 (n - k))
    let children ← (args.extract (n - k) n).toList.mapM (record env scope fuel)
    let stored ← IO.mkRef (none : Option Lean.Expr)
    let comp ← bindAll env children #[] fun refs => do
      let value ← call resultType (pattern.map (refs[·]!)) build
      stored.set (some (← mkLambdaFVars refs value))
      return value
    let some top := (← stored.get).map (idSubst env) |
      throwError "recording: internal error, no operator call"
    let kids := children.toArray
    let value := top.beta (kids.map (·.value))
    let fact ← withFresh (kids.map (tensorTy env ·.shape)) fun ws => do
      let images := (kids.zip ws).map fun (c, w) =>
        if c.tensorKind then w else getAt env c.shape w env.coord
      let p ← proveFact env exact shape (top.beta ws) tensorKind (mkAppN srcF images) source
        operator
      mkLambdaFVars ws p
    let proof ← mkExpectedTypeHint
      (← mkEqTrans (mkAppN fact (kids.map (·.value))) (← congrN srcF (kids.map (·.proof))))
      (goalOf value)
    return { comp, value, proof, shape, tensorKind }
  let arith := [(``HAdd.hAdd, ``TorchLean.Op.add), (``HSub.hSub, ``TorchLean.Op.sub),
    (``HMul.hMul, ``TorchLean.Op.mul), (``HDiv.hDiv, ``TorchLean.Op.div)]
  if let some (_, op) := arith.find? (·.1 == head) then
    expectArity head source
    unless (← typeEq args[0]! sourceType) && (← typeEq args[1]! sourceType) do
      throwError "recording: `{source}` combines `{args[0]!}` and `{args[1]!}`; only \
        operands of its result type `{sourceType}` are recorded"
    return ← opNode 2 #[0, 1] false op fun t => `(Ops.apply $(mkCIdent op) $t)
  if head == ``Neg.neg then
    return ← opNode 1 #[0] false ``TorchLean.Op.neg fun t => `(Ops.apply TorchLean.Op.neg $t)
  if env.tensorMode then
    let unary := [(``TorchLean.Tensor.relu, ``TorchLean.Op.relu),
      (``TorchLean.Tensor.sigmoid, ``TorchLean.Op.sigmoid),
      (``TorchLean.Tensor.tanh, ``TorchLean.Op.tanh)]
    if let some (_, op) := unary.find? (·.1 == head) then
      return ← opNode 1 #[0] false op fun t => `(Ops.apply $(mkCIdent op) $t)
    if head == ``TorchLean.Tensor.sum then
      return ← opNode 1 #[0] true ``TorchLean.Op.reduce fun t =>
        `(Ops.Ref.cast (Shape.reduce_finRange _) <$>
          Ops.apply
            (TorchLean.Op.reduce .sum (List.finRange _) false
              (fun h => by rcases h with h | h <;> cases h))
            $t)
    if head == ``TorchLean.Tensor.scalar then
      expectArity head source
      let c ← record env scope fuel args[2]!
      let item ← mkAppOptM ``Tensor.item #[some env.scalar, some env.storage, some c.value]
      let p ← mkExpectedTypeHint c.proof (← mkEq item args[2]!)
      let scalarItem := mkAppN (mkConst ``Tensor.scalar_item) #[env.scalar, env.storage, c.value]
      let proof ← mkExpectedTypeHint
        (← mkEqTrans (← mkEqSymm scalarItem)
          (← mkCongrArg (mkAppN (mkConst ``Tensor.scalar) #[env.scalar, env.storage]) p))
        (goalOf c.value)
      return { c with proof, shape, tensorKind }
    if head == ``TorchLean.Tensor.square then
      return ← opNode 1 #[0, 0] false ``TorchLean.Op.mul fun t =>
        `(Ops.apply TorchLean.Op.mul $t)
    if head == ``TorchLean.Tensor.matmul then
      expectArity head source
      return ← opNode 2 #[0, 1] true ``TorchLean.Op.matmul fun t => do
        `(Ops.apply
          (TorchLean.Op.matmul (batchA := []) (batchB := []) (batch := [])
            (m := $(← exprToSyntax args[5]!))
            (n := $(← exprToSyntax args[6]!))
            (p := $(← exprToSyntax args[7]!)) false false (fun h => nomatch h)
            (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
          $t)
  if head == ``ite then
    expectArity head source
    let c := args[1]!
    let dec := args[2]!
    let dependent := inScope c || inScope dec
    if dependent && (env.tensorMode ||
        !(← boundedDefEq .default 2000 env.inputShape (mkConst ``Shape.scalar))) then
      throwError "recording: input-dependent branches need scalar traversal"
    let yes ← record env scope fuel args[3]!
    let no ← record env scope fuel args[4]!
    let T := tensorTy env shape
    let iteOf (ty c dec : Lean.Expr) := mkApp3 (mkConst ``ite [Level.one]) ty c dec
    let g ← withLocalDeclD `v T fun v => mkLambdaFVars #[v] (getAt env shape v env.coord)
    if !dependent then
      let comp := mkApp2 (iteOf resultType c dec) yes.comp no.comp
      let value := mkApp2 (iteOf T c dec) yes.value no.value
      let branches ← mkCongr
        (← mkCongrArg (iteOf (if tensorKind then T else env.scalar) c dec) yes.proof) no.proof
      let proof ← if tensorKind then pure branches else
        mkEqTrans (mkAppN (mkConst ``apply_ite [Level.one, Level.one])
          #[T, env.scalar, g, c, dec, yes.value, no.value]) branches
      let proof ← mkExpectedTypeHint proof (goalOf value)
      return { comp, value, proof, shape, tensorKind }
    -- The program observes the scalar values the condition reads, and the proof replaces
    -- each observed value by its source variable, newest first.
    let mentioned := (scope.reverse.filter fun e =>
      c.containsFVar e.source.fvarId! || dec.containsFVar e.source.fvarId!).toArray
    let comp ← observeAll env (fun c dec => mkApp2 (iteOf resultType c dec) yes.comp no.comp)
      mentioned.toList c dec
    let sources := mentioned.map (·.source)
    let items ← mentioned.mapM fun e =>
      mkAppOptM ``Tensor.item #[some env.scalar, some env.storage, some e.value]
    let c'' := c.replaceFVars sources items
    let dec'' := dec.replaceFVars sources items
    let value := mkApp2 (iteOf T c'' dec'') yes.value no.value
    let h₁ := mkAppN (mkConst ``apply_ite [Level.one, Level.one])
      #[T, env.scalar, g, c'', dec'', yes.value, no.value]
    let h₂ ← mkCongr (← mkCongrArg (iteOf env.scalar c'' dec'') yes.proof) no.proof
    let final := mkApp2 (iteOf env.scalar c'' dec'') args[3]! args[4]!
    let mut h₃ ← reflHint final (← mkEq final final)
    for i' in [0:mentioned.size] do
      let i := mentioned.size - 1 - i'
      let e := mentioned[i]!
      let motive ← withLocalDeclD `z (← inferType e.source) fun z => do
        let from_ := (sources.extract 0 i).push e.source
        let to_ := (items.extract 0 i).push z
        mkLambdaFVars #[z] (mkApp2 (iteOf env.scalar (c.replaceFVars from_ to_)
          (dec.replaceFVars from_ to_)) args[3]! args[4]!)
      let p ← mkExpectedTypeHint e.proof (← mkEq items[i]! e.source)
      h₃ ← mkEqTrans h₃ (← mkCongrArg motive p)
    let proof ← mkEqTrans h₁ (← mkEqTrans h₂ h₃)
    let proof ← mkExpectedTypeHint proof (goalOf value)
    return { comp, value, proof, shape, tensorKind }
  if head == ``dite then
    throwError "recording: input-dependent branches need faithful recorded primitives"
  if let some name := fn.constName? then
    if ← isRecursiveDefinition name then
      unless !args.isEmpty && (← typeEq (← inferType args.back!) (mkConst ``Nat)) do
        throwError "recording: a recurrence needs a final natural-number count"
      let count := args.back!
      if inScope count then
        throwError "recording: the recurrence count must not depend on differentiable inputs"
      let some equations ← getEqnsFor? name |
        throwError "recording: no constructor equations for the recurrence"
      let apply (k : Lean.Expr) := mkAppN fn (args.set! (args.size - 1) k)
      let (zeroBody, eq₀) ← equation (apply (mkNatLit 0)) equations
      let init ← record env scope fuel zeroBody
      let T := tensorTy env shape
      return ← withLocalDeclD `n (mkConst ``Nat) fun n => do
        let previous := apply n
        let next := apply (mkNatAdd n (mkNatLit 1))
        let (succBody, eqS) ← equation next equations
        withLocalDeclD `acc sourceType fun acc => do
          let body := succBody.replace fun e => if e == previous then some acc else none
          if (body.find? (·.isConstOf name)).isSome then
            throwError "recording: recursive calls must use the predecessor and fixed \
              parameters"
          withLocalDeclD `value (refType env shape) fun ref => do
          withLocalDeclD `w T fun w => do
          withLocalDeclD `hw (rel env shape tensorKind w acc) fun hw => do
            let entry : ScopeEntry :=
              { source := acc, ref, value := w, proof := hw, shape, tensorKind }
            let b ← record env (entry :: scope) fuel body
            if b.value.containsFVar ref.fvarId! || b.value.containsFVar acc.fvarId! ||
                b.value.containsFVar hw.fvarId! || b.proof.containsFVar ref.fvarId! ||
                b.comp.containsFVar w.fvarId! || b.comp.containsFVar hw.fvarId! ||
                b.comp.containsFVar acc.fvarId! then
              throwError "recording: internal error, the recorded recurrence body mixes levels"
            let comp ← mkAppM ``TorchLean.autograd.iterate
              #[← mkLambdaFVars #[n, ref] b.comp, count, init.comp]
            let stepV ← mkLambdaFVars #[n, w] b.value
            let value ← mkAppOptM ``TorchLean.autograd.iterate
              #[some env.idM, some env.idMonad, some T, some stepV, some count, some init.value]
            let R ← withLocalDeclD `k (mkConst ``Nat) fun k => withLocalDeclD `x T fun x =>
              mkLambdaFVars #[k, x] (rel env shape tensorKind x (apply k))
            let zero ← mkExpectedTypeHint (← mkEqTrans init.proof (← mkEqSymm eq₀))
              (rel env shape tensorKind init.value (apply (mkNatLit 0)))
            let succ ← withLocalDeclD `hw' (rel env shape tensorKind w previous) fun hw' => do
              let inner ← mkExpectedTypeHint
                (mkAppN (← mkLambdaFVars #[acc, hw] b.proof) #[previous, hw'])
                (rel env shape tensorKind b.value succBody)
              let step ← mkExpectedTypeHint (← mkEqTrans inner (← mkEqSymm eqS))
                (rel env shape tensorKind (mkApp2 stepV n w) next)
              mkLambdaFVars #[n, w, hw'] step
            let proof := mkAppN (mkConst ``TorchLean.autograd.iterate_induction)
              #[T, R, stepV, init.value, zero, succ, count]
            let proof ← mkExpectedTypeHint proof (goalOf value)
            return { comp, value, proof, shape, tensorKind }
  if let some unfolded ← unfoldDefinition? source then
    unless unfolded == source do
      let r ← record env scope fuel unfolded
      return { r with proof := ← mkExpectedTypeHint r.proof (goalOf r.value) }
  throwError "recording: unsupported recorded calculation {source}; \
    use an existing Runtime operation"

/-- Add the generated equality as an auxiliary theorem, with the kernel check synchronous and
bounded. -/
private def addChecked (statement proof : Lean.Expr) (kind : Name) : MetaM Lean.Expr :=
  tryCatchRuntimeEx
    (withOptions (fun o => Elab.async.set (maxHeartbeats.set o 50000) false)
      (mkAuxTheorem statement proof (zetaDelta := true) (kind? := kind)))
    (fun ex => throwError "recording: the kernel rejected the generated equality: \
      {ex.toMessageData}")

end TorchLean.autograd

namespace TorchLean.autograd

/--
Internal call-site translation for recorded functions. Ordinary function definitions need no
recording annotation.
-/
syntax (name := programStx) "record_function% " term:max : term

open Lean Meta Elab Term

/-- Elaborates `record_function% f` as an `Ops` computation of the expected program type, and adds
a kernel-checked auxiliary theorem (kind `_recordedArithmetic`) that the computation, run by the
pure interpreter `TorchLean.instOpsId`, is `f`: `∀ t, program t = Tensor.map f t` for a scalar
function `f`, `program t = f t` for a tensor function and `program t = Tensor.scalar (f t)` for a
function from a tensor to a scalar. An operation whose source instance computes something other
than its recorded operator is an error. -/
@[term_elab programStx]
def elaborateProgram : TermElab := fun stx expectedType? => withRef stx do
  let `(record_function% $source) := stx | throwUnsupportedSyntax
  let some expectedType := expectedType? | throwError "recording: missing program type"
  forallTelescopeReducing expectedType fun locals resultType => do
    let some input := locals.back? | throwError "recording: expected a reference input"
    let inputType ← inferType input
    unless inputType.isAppOf ``Ops.Ref do
      throwError "recording: expected a reference input"
    let monadType :=
      mkApp (Lean.mkConst ``Monad [Level.zero, Level.zero]) inputType.getAppArgs[0]!
    let some monadInst := (← getLCtx).findDecl? fun decl =>
        if decl.type == monadType then some decl.toExpr else none |
      throwError "recording: expected a program over a monad with an `Ops` instance"
    let arguments := inputType.getAppArgs
    let scalar := arguments[1]!
    let original ← elabTerm source none
    -- Expand a locally bound function, so that its body is recorded.
    let function ← whnf original
    let .forallE binder domain _ _ ← whnf (← inferType function) |
      throwError "recording: expected an ordinary function"
    let tensorMode := isTensor domain
    if tensorMode then
      let tensor ← mkConstWithFreshMVarLevels ``TorchLean.Tensor
      let tensorType := mkAppN tensor #[scalar, arguments.back!, arguments[2]!]
      unless ← isDefEq domain tensorType do
        throwError "recording: the tensor function accepts {domain}, but the input is \
          {tensorType}"
    else
      unless ← isDefEq domain scalar do
        throwError "recording: the scalar function must preserve its element type"
    synthesizeSyntheticMVarsNoPostponing
    let original ← instantiateMVars original
    let function ← instantiateMVars function
    let domain ← instantiateMVars domain
    let arguments ← arguments.mapM instantiateMVars
    for (part, label) in [(arguments[1]!, "element type"), (arguments[2]!, "storage"),
        (arguments[3]!, "element context"), (arguments.back!, "input shape")] do
      if part.hasExprMVar then
        throwError "recording: the {label} is not known when the program is recorded"
    if function.hasExprMVar then
      throwError "recording: the function has unknown parts: {function}"
    -- The theorem is about the pure interpreter itself, never an `Ops Id` instance in scope.
    let idM := Lean.mkConst ``Id [Level.zero]
    let idMonad := Lean.mkConst ``Id.instMonad [Level.zero]
    let idOps := mkAppN (Lean.mkConst ``TorchLean.instOpsId)
      #[arguments[1]!, arguments[2]!, arguments[3]!]
    let inputShape := arguments.back!
    let tensorT := mkAppN (mkConst ``TorchLean.Tensor [Level.zero])
      #[arguments[1]!, inputShape, arguments[2]!]
    let binder := if binder.isAnonymous || binder.hasMacroScopes then `x else binder
    withLocalDeclD `t tensorT fun t => do
    withLocalDeclD `j (← mkAppM ``Shape.Coord #[inputShape]) fun j => do
      let env : Env :=
        { scalar := arguments[1]!, storage := arguments[2]!, context := arguments[3]!
          monad := arguments[0]!, monadInst, ops := arguments[4]!, idM, idMonad
          idOps, inputShape, tensorMode
          coord := if tensorMode then mkConst ``PUnit.unit [Level.one] else j }
      withLocalDeclD binder domain fun entry => do
      withLocalDeclD `hin (rel env inputShape tensorMode t entry) fun hin => do
        let top : ScopeEntry :=
          { source := entry, ref := input, value := t, proof := hin, shape := inputShape,
            tensorKind := tensorMode }
        let root ← record env [top] 4096 (mkApp function entry).headBeta
        let comp ← instantiateMVars root.comp
        if [entry, hin, j].any (root.value.containsFVar ·.fvarId!) then
          throwError "recording: internal error, the recorded value depends on the source input"
        if [t, j, entry, hin].any (comp.containsFVar ·.fvarId!) then
          throwError "recording: internal error, the recorded program depends on proof locals"
        unless ← isDefEq (← inferType comp) resultType do
          throwError "recording: output shape does not match the function"
        let program ← mkLambdaFVars locals comp
        let run := (idSubst env (← mkLambdaFVars #[input] comp)).beta #[t]
        let atInput (x : Lean.Expr) : MetaM Lean.Expr := do
          let hx ← reflHint x (← mkEq x x)
          return mkAppN (← mkLambdaFVars #[entry, hin] root.proof) #[x, hx]
        let (target, proof) ← if !tensorMode then do
            let target := mkAppN (mkConst ``Tensor.map [Level.zero, Level.zero])
              #[env.scalar, env.storage, env.scalar, env.storage, inputShape, original, t]
            let mapApply := mkAppN (mkConst ``Tensor.map_apply [Level.zero, Level.zero])
              #[env.scalar, env.storage, env.scalar, env.storage, inputShape, original, t, j]
            let pj ← mkExpectedTypeHint
              (← mkEqTrans (← atInput (getAt env inputShape t j)) (← mkEqSymm mapApply))
              (← mkEq (getAt env inputShape root.value j) (getAt env inputShape target j))
            pure (target, mkAppN (mkConst ``Tensor.Rep.ext [Level.zero])
              #[env.scalar, env.storage, inputShape, root.value, target,
                ← mkLambdaFVars #[j] pj])
          else if root.tensorKind then do
            pure ((mkApp original t).headBeta, ← atInput t)
          else do
            let value := (mkApp original t).headBeta
            let target := mkAppN (mkConst ``Tensor.scalar) #[env.scalar, env.storage, value]
            let item ← mkAppOptM ``Tensor.item
              #[some env.scalar, some env.storage, some root.value]
            let h ← mkEqTrans (← mkExpectedTypeHint (← atInput t) (← mkEq item value))
              (← mkEqSymm
                (mkAppN (mkConst ``Tensor.item_scalar) #[env.scalar, env.storage, value]))
            pure (target, mkAppN (mkConst ``Tensor.ext_scalar)
              #[env.scalar, env.storage, root.value, target, h])
        let equation := mkApp3 (mkConst ``Eq [Level.one]) (tensorTy env root.shape) run target
        let statement ← mkForallFVars #[t] equation
        let proof ← mkLambdaFVars #[t] (← mkExpectedTypeHint proof equation)
        discard <| addChecked statement proof ((← getMainModule) ++ `_recordedArithmetic)
        return program

end TorchLean.autograd
