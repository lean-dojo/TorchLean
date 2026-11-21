/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Eval
public import TorchLean.Tensor.Op.Table.Conv
public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.Tensor.Op.Table.Reduce

/-!
# Graphs in the left fold

`Op.leftFold` sets the order of each reduction of the operator table (`reduce`, `mean`,
`contract`, `matmul`, `scaledMatmul`, `pool`, `cumsum`, `cumprod`, `cummax`, `cummin`) to
`Tensor.ReduceOrder.leftFold` and keeps every other operator, and `IR.Graph.leftFold` does so at
every node. In the left fold each arm of `Op.rule?` is the arm before orders
(`Op.rule?_reduce_leftFold` and the others, by `rfl`), so a graph whose orders are all the left fold
evaluates as it did, bit for bit, at every scalar.

Over `ℝ` the order does not matter. `Op.check_leftFold`: an operator that passes `Op.check` passes
it in the left fold. `Op.rule?_leftFold_real`: an operator whose orders cover their counts, as
`Op.check` requires, has the rule of the left fold at `ℝ` (the `_real` theorems of
`TorchLean.Tensor.Reduce.Order` and `Attention.scaledMatmulIn_real`). `Op.invoke_leftFold_real`,
`IR.Node.eval_leftFold_real`, `IR.runNodes_leftFold_real`, `IR.Graph.run_leftFold_real` and
`IR.Graph.eval_leftFold_real`: every successful invocation, node, run and evaluation at `ℝ` is one
of the left-fold operator or graph, with the same outputs.
-/

@[expose] public section

namespace TorchLean

open Tensor (ReduceOrder)

/-- The operator with the order of its reduction, if it has one, set to the left fold. -/
def Op.leftFold : {ins outs : List Port} → Op ins outs → Op ins outs
  | _, _, .reduce op dims keepdim h _ => .reduce op dims keepdim h
  | _, _, .mean dims keepdim _ => .mean dims keepdim
  | _, _, .contract dims batchDims h _ => .contract dims batchDims h
  | _, _, .matmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB _ =>
      .matmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB
  | _, _, .scaledMatmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB _ =>
      .scaledMatmul (m := m) (n := n) (p := p) lv rv hl hr broadcastA broadcastB
  | _, _, .pool op dims kernel stride padLow padHigh dilation ceilMode _ =>
      .pool op dims kernel stride padLow padHigh dilation ceilMode
  | _, _, .cumsum dim _ => .cumsum dim
  | _, _, .cumprod dim _ => .cumprod dim
  | _, _, .cummax dim _ => .cummax dim
  | _, _, .cummin dim _ => .cummin dim
  | _, _, op => op

private theorem ok_of_bind_ok {x y : Except String Unit} (h : (x >>= fun _ => y) = .ok ()) :
    x = .ok () ∧ y = .ok () := by
  cases x with
  | error e => cases h
  | ok u => exact ⟨rfl, h⟩

private theorem bind_ok_of {x : Except String Unit} (h : x = .ok ()) :
    (x >>= fun _ => (.ok () : Except String Unit)) = .ok () := by
  rw [h]
  rfl

private theorem ok_of_ite_ok {p : Prop} [Decidable p] {y : Except String Unit} {e : String}
    (h : (if p then y else ((throw e : Except String Unit) >>= fun _ => y)) = .ok ()) :
    p ∧ y = .ok () := by
  by_cases hp : p
  · rw [ite_eq_left hp] at h
    exact ⟨hp, h⟩
  · rw [ite_eq_right hp] at h
    cases h

/-- An operator that passes `Op.check` passes it in the left fold. -/
theorem Op.check_leftFold {ins outs : List Port} {op : Op ins outs} (hc : op.check = .ok ()) :
    op.leftFold.check = .ok () := by
  cases op with
  | reduce | mean =>
      simp only [Op.check] at hc
      simp only [Op.leftFold, Op.check, ite_eq_left (ok_of_ite_ok hc).1]
      exact (Op.checkOrder_eq_ok _ _ _).2 (ReduceOrder.covers_leftFold _)
  | pool => exact bind_ok_of (ok_of_bind_ok hc).1
  | contract | matmul | scaledMatmul | cumsum | cumprod | cummax | cummin => rfl
  | _ => exact hc

/-- Over `ℝ`, an operator that passes `Op.check`, so that its order covers its count, has the
rule of the left fold. -/
theorem Op.rule?_leftFold_real {ins outs : List Port} {op : Op ins outs}
    (hc : op.check = .ok ()) : op.rule? (α := ℝ) = op.leftFold.rule? := by
  cases op with
  | reduce rop dims keepdim h order =>
      simp only [Op.check] at hc
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 (ok_of_ite_ok hc).2
      show _ = (Op.reduce rop dims keepdim h).rule?
      rw [Op.rule?_reduce _ _ _ _ order, Op.rule?_reduce_leftFold]
      cases rop <;> refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.reduceIn_sum_real (hcov := hcov),
          Tensor.reduceIn_prod_real (hcov := hcov), Tensor.reduceIn_max_real (hcov := hcov),
          Tensor.reduceIn_min_real (hcov := hcov)] <;> rfl
  | mean dims keepdim order =>
      simp only [Op.check] at hc
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 (ok_of_ite_ok hc).2
      show _ = (Op.mean dims keepdim).rule?
      rw [Op.rule?_mean _ _ order, Op.rule?_mean_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.meanIn_real (hcov := hcov)]
      rfl
  | contract dims batchDims h order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.contract dims batchDims h).rule?
      rw [Op.rule?_contract _ _ _ order, Op.rule?_contract_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.contractIn_real (hcov := hcov)]
      rfl
  | matmul lv rv hl hr broadcastA broadcastB order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.matmul lv rv hl hr broadcastA broadcastB).rule?
      rw [Op.rule?_matmul _ _ _ _ _ _ order, Op.rule?_matmul_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.batchMatmulIn_eq_batchMatmul (hcov := hcov)]
      rfl
  | scaledMatmul lv rv hl hr broadcastA broadcastB order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.scaledMatmul lv rv hl hr broadcastA broadcastB).rule?
      rw [Op.rule?_scaledMatmul _ _ _ _ _ _ order, Op.rule?_scaledMatmul_leftFold]
      refine congrArg some (Op.Rule.ext_forward (funext fun xs => ?_) rfl)
      rw [Attention.scaledMatmulIn_real (hcov := hcov)]
  | pool pop dims kernel stride padLow padHigh dilation ceilMode order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 (ok_of_bind_ok hc).2
      show _ = (Op.pool pop dims kernel stride padLow padHigh dilation ceilMode).rule?
      rw [Op.rule?_pool _ _ _ _ _ _ _ _ order, Op.rule?_pool_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.poolIn_real (hcov := hcov)]
      rfl
  | cumsum dim order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.cumsum dim).rule?
      rw [Op.rule?_cumsum _ order, Op.rule?_cumsum_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.scanIn_add_real (hcov := hcov)]
      rfl
  | cumprod dim order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.cumprod dim).rule?
      rw [Op.rule?_cumprod _ order, Op.rule?_cumprod_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ ?_) <;>
        simp only [Tensor.scanIn_mul_real (hcov := hcov)]
      rfl
  | cummax dim order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.cummax dim).rule?
      rw [Op.rule?_cummax _ order, Op.rule?_cummax_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ rfl)
      simp only [Tensor.cummaxIn_real (hcov := hcov)]
      rfl
  | cummin dim order =>
      have hcov := (Op.checkOrder_eq_ok _ _ _).1 hc
      show _ = (Op.cummin dim).rule?
      rw [Op.rule?_cummin _ order, Op.rule?_cummin_leftFold]
      refine congrArg some (Op.Rule.ext_forward ?_ rfl)
      simp only [Tensor.cumminIn_real (hcov := hcov)]
      rfl
  | _ => rfl

/-- Over `ℝ`, a successful invocation is one of the operator in the left fold, with the same
outputs. -/
theorem Op.invoke_leftFold_real {ins outs : List Port} {op : Op ins outs}
    {inputs : Except String (List (Op.Value ℝ))} {outputs : List (Op.Value ℝ)}
    (h : op.invoke inputs = .ok outputs) : op.leftFold.invoke inputs = .ok outputs := by
  obtain ⟨values, xs, rule, hc, rfl, hx, hs, hv, rfl⟩ := Op.invoke_eq_ok h
  rw [Op.rule?_leftFold_real hc] at hs
  simp only [Op.invoke, Op.check_leftFold hc, hx, hs, hv, Op.Rule.check, bind, Except.bind, pure,
    Except.pure, rule.prepare_value]

namespace IR

/-- The node with its operator in the left fold. -/
def Node.leftFold : Node → Node
  | .op ⟨ins, outs, f⟩ refs => .op ⟨ins, outs, f.leftFold⟩ refs
  | n => n

/-- The graph with every operator in the left fold. -/
def Graph.leftFold (g : Graph) : Graph := { g with nodes := g.nodes.map Node.leftFold }

/-- Over `ℝ`, a node that evaluates evaluates to the same outputs in the left fold. -/
theorem Node.eval_leftFold_real {payload : Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {id : Nat} {n : Node} {vs : List (Op.Value ℝ)}
    (h : n.eval payload inputs earlier id = .ok vs) :
    n.leftFold.eval payload inputs earlier id = .ok vs := by
  cases n with
  | input | const => exact h
  | op o refs =>
      obtain ⟨ins, outs, f⟩ := o
      simp only [Node.eval, Node.leftFold] at h ⊢
      cases f with
      | custom => exact h
      | _ => exact Op.invoke_leftFold_real h

/-- Over `ℝ`, nodes that run run to the same outputs in the left fold. -/
theorem runNodes_leftFold_real {payload : Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {nodes : List Node}
    {values : Array (List (Op.Value ℝ))}
    (h : runNodes payload inputs earlier nodes = .ok values) :
    runNodes payload inputs earlier (nodes.map Node.leftFold) = .ok values := by
  induction nodes generalizing earlier with
  | nil => exact h
  | cons n rest ih =>
      rw [runNodes_cons] at h
      obtain ⟨x, hx, hrest⟩ := bind_eq_ok.1 h
      rw [List.map_cons, runNodes_cons, Node.eval_leftFold_real hx]
      exact ih hrest

/-- Over `ℝ`, a graph that runs runs to the same outputs in the left fold. -/
theorem Graph.run_leftFold_real {g : Graph} {payload : Payload ℝ} {inputs : List (Op.Value ℝ)}
    {values : Array (List (Op.Value ℝ))} (h : g.run payload inputs = .ok values) :
    g.leftFold.run payload inputs = .ok values := by
  simpa only [Graph.run, Graph.leftFold, Array.toList_map] using runNodes_leftFold_real h

/-- Over `ℝ`, a graph that evaluates evaluates to the same outputs in the left fold. -/
theorem Graph.eval_leftFold_real {g : Graph} {payload : Payload ℝ} {inputs : List (Op.Value ℝ)}
    {outputs : List (Op.Value ℝ)} (h : g.eval payload inputs = .ok outputs) :
    g.leftFold.eval payload inputs = .ok outputs := by
  obtain ⟨values, hv, ho⟩ := bind_eq_ok.1 h
  exact bind_eq_ok.2 ⟨values, Graph.run_leftFold_real hv, ho⟩

end IR

end TorchLean
