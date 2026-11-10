/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Launch
public import TorchLean.CUDA.Kernel.Program

/-!
# Unary functions in kernels as checked table searches

The expression language of kernels compares and loads scalars, but it has no operation that reads
the word of a scalar and no transcendental function. A `Table` lets a kernel compute a unary
function `f` of a scalar type of at most `Table.maxWidth` bits from comparisons and loads alone.
Its keys are scalars of the type, which the search reads as if they increased, and its values are
the results at the keys. `Table.apply` reads both from two inputs of the kernel and finds its
argument among the keys by a binary search unrolled into one load and one comparison for each
of the `depth` halvings of its `2 ^ depth` keys. Two kinds of input skip the search. One is an
input that is not equal to itself, a NaN. The other is an input equal to the scalar `zero`, so
that encodings that the comparisons cannot tell apart, like the two signed zeros of IEEE 754, get
their own result. For each kind the table gives the input itself, one scalar, an operation of
the input with one scalar on either side, or the negation of the input. A NaN input has no other
way to reach the result, as every comparison with it is false. The functions of FloatLib return a
NaN input made quiet with its sign and payload, which is `x + 0` in the arithmetic of FloatLib.

Nothing in a `Table` says that its keys are sorted or that its search finds its input, and we prove
neither. `Table.make` evaluates `f` once at every word of the type, builds a candidate table, runs
it at every word and returns it only when it gives `f` at every one, bit for bit
(`Table.eval_eq_of_check`). At the NaNs the candidate tries the input, the input plus `zero`, the
result at the first NaN, the negation of the input, `zero` minus the input and that first result
divided by the input, in this order, and keeps the first that is `f` at all of them. It does the
same at the inputs equal to `zero`, where the last form gives the reciprocal its two infinities.
So `make` returns `none` for a function whose results there take none of these forms, such as the
absolute value, which clears the sign of a NaN, and for an arithmetic whose comparisons the search
cannot follow. A `Table.Of A f` is a table with the proof that it gives `f` at every input.
`Table.Of.eval_apply` shows that such a table, read from two inputs, computes `f` of its argument
inside any expression, failures of the argument included.

For an IEEE format, a table of FloatLib's correctly rounded `exp`, `expMinus1`, `log` or `logPlus1`
makes a kernel compute that function rounded once to nearest, as `Table.Of.toReal_exp` and the
theorems after it state. The tests find these tables at every word of binary16, bfloat16, a 12-bit
format and the 8-bit formats e4m3 and e5m2. They find them also in e4m3fn and e4m3fnuz, which have
no infinities and whose encodings the theorems of rounding do not cover. Other unary functions,
such as TorchLean's correctly rounded `tan`, the reciprocal and negation, give tables the same way
when `make` returns one. Wider formats have no tables, and `make` returns `none` for them.

A kernel reads a table as two of its inputs, so one launch can apply several tables and mix them
with arithmetic. `Table.Of.runDevice` runs the kernel of one table over an array on the device.
The theorems are about the evaluation of kernels in Lean, and the device run is trusted: the CUDA
source that `runBytes` renders, which for the configured formats no theorem relates to
`Expr.eval`, the bytes of the inputs and the results, NVRTC, the launch and its record of failed
reads, and the agreement of the device arithmetic of `Scalar.precision` with `Scalar.arithmetic`,
which the class `Scalar` does not state. `Table.Of.runDevice` refuses words wider than their
storage before the launch and result words wider than the format after it. The tests run the
tables on the device at every word of their formats and compare the results with the functions
bit for bit.
-/

@[expose] public section

namespace TorchLean.Kernel

variable {α : Type} {Γ : List Ty}

/-- Reader `R` holds the array `a` as input `op`: every entry of `a` reads back as it is. -/
def Reader.Holds (R : Reader α) (op : Nat) (a : Array α) : Prop :=
  ∀ (i : UInt64) (h : i.toNat < a.size), R op i = .ok a[i.toNat]

namespace Table

/-- What a table gives at an input that its search skips: the input itself, one scalar, an
operation of the input with one scalar on either side, or the negation of the input. -/
inductive Special (α : Type) where
  /-- The input itself. -/
  | input
  /-- The scalar `y`. -/
  | value (y : α)
  /-- The operation `op` of the input and the scalar `y`. -/
  | binary (op : ScalarOp) (y : α)
  /-- The operation `op` of the scalar `y` and the input. -/
  | flip (op : ScalarOp) (y : α)
  /-- The negation of the input. -/
  | neg
  deriving Inhabited

/-- The scalar that `s` gives at the input `x` in the arithmetic `A`. -/
def Special.result (A : Arithmetic α) : Special α → α → α
  | .input, x => x
  | .value y, _ => y
  | .binary op y, x => A.binary op x y
  | .flip op y, x => A.binary op y x
  | .neg, x => A.neg x

/-- The expression of `s` at the input that is the newest variable. -/
def Special.expr : Special α → Expr α (.scalar :: Γ) .scalar
  | .input => .var .zero
  | .value y => .scalar y
  | .binary op y => .binary op (.var .zero) (.scalar y)
  | .flip op y => .binary op (.scalar y) (.var .zero)
  | .neg => .neg (.var .zero)

/-- The expression of `s` gives the scalar of `s` at the newest variable. -/
@[simp] theorem Special.eval_expr (s : Special α) (A : Arithmetic α) (R : Reader α)
    (env : Env α Γ) (x : α) :
    (s.expr (Γ := Γ)).eval A R (env.push x) = .ok (s.result A x) := by
  cases s <;> rfl

end Table

/-- A candidate table of a unary function: the number of halvings of its search, its keys and its
values, a scalar `zero`, and what the table gives at the inputs equal to `zero` and at the inputs
not equal to themselves. At every other input the search `Table.searchFrom` picks a value. Nothing
here says that the table gives a function, and a `Table.Of A f` carries the proof that it gives
`f`. -/
structure Table (α : Type) where
  /-- The number of halvings of the search. -/
  depth : Nat
  /-- The keys, which the search reads as if they increased. -/
  keys : Array α
  /-- The values: the search gives value `i` at the inputs it sends to position `i`. -/
  values : Array α
  /-- The scalar whose equal inputs skip the search. -/
  zero : α
  /-- What the table gives at the inputs equal to `zero`. -/
  atZero : Table.Special α
  /-- What the table gives at the inputs that are not equal to themselves. -/
  atNaN : Table.Special α

namespace Table

/-- The step `2 ^ j` as an index. -/
def step (j : Nat) : UInt64 :=
  (2 ^ j : Nat).toUInt64

/-- The search for the input `x` over the readers `keys` and `values`, from the position `lo` with
`j` halvings left. A halving reads the key at `lo + 2 ^ (j - 1) - 1`, the last of the next
`2 ^ (j - 1)` keys, and moves `lo` past those keys when that key is below `x`. With no halvings
left the search reads the value at `lo`. -/
def searchFrom (A : Arithmetic α) (keys values : UInt64 → Except Error α) (x : α) :
    Nat → UInt64 → Except Error α
  | 0, lo => values lo
  | j + 1, lo => do
      let key ← keys (lo + (step j - 1))
      searchFrom A keys values x j (bif A.compare .lt key x then lo + step j else lo)

/-- The search as an expression, with the position as the newest variable and the input as the
one below it, reading the keys from input `keys` and the values from input `values`. -/
def searchExpr (keys values : Nat) : Nat → Expr α (.index :: .scalar :: Γ) .scalar
  | 0 => .load values (.var .zero)
  | j + 1 =>
      .letIn
        (.cond
          (.compare .lt (.load keys (.indexBinary .add (.var .zero) (.index (step j - 1))))
            (.var (.succ .zero)))
          (.indexBinary .add (.var .zero) (.index (step j)))
          (.var .zero))
        ((searchExpr keys values j).rename (Renaming.lift fun v => .succ v))

/-- The table at the input that is the newest variable, reading the keys from input `keys` and the
values from input `values`. -/
def body (t : Table α) (keys values : Nat) : Expr α (.scalar :: Γ) .scalar :=
  .cond (.compare .eq (.var .zero) (.var .zero))
    (.cond (.compare .eq (.var .zero) (.scalar t.zero)) t.atZero.expr
      (.letIn (.index 0) (searchExpr keys values t.depth)))
    t.atNaN.expr

/-- The table applied to the scalar expression `x`, reading the keys from input `keys` and the
values from input `values`. -/
def apply (t : Table α) (keys values : Nat) (x : Expr α Γ .scalar) : Expr α Γ .scalar :=
  .letIn x (t.body keys values)

/-- The kernel that applies the table to entry `i` of input `0` at the output index `i`, reading
the keys from input `1` and the values from input `2`. -/
def kernel (t : Table α) : Expr α [.index] .scalar :=
  t.apply 1 2 (.load 0 (.var .zero))

/-- What the table gives at `x` over the readers `keys` and `values`. -/
def evalWith (t : Table α) (A : Arithmetic α) (keys values : UInt64 → Except Error α) (x : α) :
    Except Error α :=
  bif A.compare .eq x x then
    bif A.compare .eq x t.zero then .ok (t.atZero.result A x)
    else searchFrom A keys values x t.depth 0
  else .ok (t.atNaN.result A x)

/-- The reader of the array `a` as input `op`: entry `i`, or a failed read at or past the end. -/
def reader (a : Array α) (op : Nat) (i : UInt64) : Except Error α :=
  if h : i.toNat < a.size then .ok a[i.toNat] else .error (.bounds op i a.size)

/-- What the table gives at `x`, reading its keys as input `0` and its values as input `1`. -/
def eval (t : Table α) (A : Arithmetic α) (x : α) : Except Error α :=
  t.evalWith A (reader t.keys 0) (reader t.values 1) x

/-- The search expression with `j` halvings left evaluates to `searchFrom` over the readers of its
inputs. -/
theorem eval_searchExpr (A : Arithmetic α) (R : Reader α) (keys values : Nat) (env : Env α Γ)
    (x : α) : ∀ (j : Nat) (lo : UInt64),
      (searchExpr (Γ := Γ) keys values j).eval A R
          ((env.push (t := .scalar) x).push (t := .index) lo) =
        searchFrom A (R keys) (R values) x j lo
  | 0, _ => rfl
  | j + 1, lo => by
      simp only [searchExpr, searchFrom, Expr.eval, Expr.eval_rename, Env.get_push_zero,
        Env.get_push_succ, IndexOp.eval, Env.pull_push, Env.pull_succ_push, Bind.bind,
        Except.bind, Pure.pure, Except.pure]
      cases R keys (lo + (step j - 1)) with
      | error e => rfl
      | ok key =>
          dsimp only
          cases A.compare .lt key x with
          | false => exact eval_searchExpr A R keys values env x j lo
          | true => exact eval_searchExpr A R keys values env x j (lo + step j)

/-- The body of a table evaluates to `evalWith` over the readers of its inputs. -/
theorem eval_body (t : Table α) (A : Arithmetic α) (R : Reader α) (keys values : Nat)
    (env : Env α Γ) (x : α) :
    (t.body keys values).eval A R (env.push x) = t.evalWith A (R keys) (R values) x := by
  simp only [body, evalWith, Expr.eval, Env.get_push_zero, Special.eval_expr, Bind.bind,
    Except.bind, Pure.pure, Except.pure]
  cases A.compare .eq x x
  · rfl
  · cases A.compare .eq x t.zero
    · exact eval_searchExpr A R keys values env x t.depth 0
    · rfl

/-- The table applied to an expression evaluates the expression and then gives `evalWith` over
the readers of its inputs at the result, so a failure of the expression is the failure of the
whole. -/
theorem eval_apply (t : Table α) (A : Arithmetic α) (R : Reader α) (keys values : Nat)
    (x : Expr α Γ .scalar) (env : Env α Γ) :
    (t.apply keys values x).eval A R env =
      (x.eval A R env).bind (t.evalWith A (R keys) (R values)) := by
  simp only [apply, Expr.eval, Bind.bind]
  cases x.eval A R env with
  | error e => rfl
  | ok v => exact eval_body t A R keys values env v

/-- A search that succeeds over the readers `keys` and `values` succeeds with the same result over
readers that agree with them wherever they succeed. -/
theorem searchFrom_of_ok (A : Arithmetic α) {keys values keys' values' : UInt64 → Except Error α}
    (hk : ∀ i y, keys i = .ok y → keys' i = .ok y)
    (hv : ∀ i y, values i = .ok y → values' i = .ok y) (x : α) :
    ∀ {j : Nat} {lo : UInt64} {y : α}, searchFrom A keys values x j lo = .ok y →
      searchFrom A keys' values' x j lo = .ok y
  | 0, lo, y, h => hv lo y h
  | j + 1, lo, y, h => by
      simp only [searchFrom, Bind.bind] at h ⊢
      cases hkey : keys (lo + (step j - 1)) with
      | error e =>
          rw [hkey] at h
          cases h
      | ok key =>
          rw [hkey] at h
          rw [hk _ _ hkey]
          exact searchFrom_of_ok A hk hv x h

end Table

/-- A reader that holds `a` as input `op` reads what the reader of `a` reads wherever that read
succeeds. -/
theorem Reader.Holds.eq_ok_of_reader {R : Reader α} {op : Nat} {a : Array α} (h : R.Holds op a)
    (op' : Nat) (i : UInt64) (y : α) (hy : Table.reader a op' i = .ok y) : R op i = .ok y := by
  unfold Table.reader at hy
  split at hy
  · cases hy
    exact h i ‹_›
  · cases hy

namespace Table

/-- A table of the function `f` in the arithmetic `A`: a table with the proof that it gives `f` at
every input. -/
structure Of (A : Arithmetic α) (f : α → α) where
  /-- The table. -/
  table : Table α
  /-- At every input the table gives `f`. -/
  eval_eq : ∀ x, table.eval A x = .ok (f x)

/-- Over a reader that holds the keys of a table of `f` as input `keys` and its values as input
`values`, the table gives `f` at every input. -/
theorem Of.evalWith_eq {A : Arithmetic α} {f : α → α} (T : Of A f) {R : Reader α}
    {keys values : Nat} (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (x : α) : T.table.evalWith A (R keys) (R values) x = .ok (f x) := by
  have h := T.eval_eq x
  unfold Table.eval Table.evalWith at h
  unfold Table.evalWith
  cases hn : A.compare .eq x x <;> cases hz : A.compare .eq x T.table.zero <;>
    simp only [hn, hz, Bool.cond_true, Bool.cond_false] at h ⊢
  all_goals first
    | exact h
    | exact searchFrom_of_ok A (hk.eq_ok_of_reader 0) (hv.eq_ok_of_reader 1) x h

/-- Inside any expression, a table of `f` read from the inputs `keys` and `values` gives `f` of its
argument, and a failure of the argument is the failure of the whole. -/
theorem Of.eval_apply {A : Arithmetic α} {f : α → α} (T : Of A f) {R : Reader α}
    {keys values : Nat} (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (x : Expr α Γ .scalar) (env : Env α Γ) :
    (T.table.apply keys values x).eval A R env = (x.eval A R env).map f := by
  rw [Table.eval_apply]
  cases x.eval A R env with
  | error e => rfl
  | ok v => exact T.evalWith_eq hk hv v

/-- The kernel of a table of `f` gives `f` of entry `i` of input `0` at the output index `i`, when
inputs `1` and `2` hold the keys and the values. -/
theorem Of.evaluate_kernel {A : Arithmetic α} {f : α → α} (T : Of A f) {R : Reader α}
    (hk : R.Holds 1 T.table.keys) (hv : R.Holds 2 T.table.values) (i : UInt64) :
    evaluate A T.table.kernel R i = (R 0 i).map f :=
  T.eval_apply hk hv _ _

/-- The widest words that `make` tabulates. The tables of a type of `w` bits have up to `2 ^ w`
entries. -/
def maxWidth : Nat := 20

section Make

variable [Numeric.Word α]

/-- The words below `2 ^ width` in the order of the values of IEEE encodings of `width` bits:
the words with the top bit set from the largest down, then the others from `0` up. -/
def signMagnitude : Nat → Array Nat
  | 0 => #[0]
  | w + 1 => (Array.range (2 ^ w)).reverse.map (2 ^ w + ·) ++ Array.range (2 ^ w)

/-- The entries `es` sorted by the `<` of `A` on their first components: `es` itself when they
increase, which takes one comparison per entry, and otherwise merge sorted. -/
def sortKeys (A : Arithmetic α) (es : Array (α × α)) : Array (α × α) :=
  if (es.zip (es.extract 1 es.size)).all fun (a, b) => A.compare .lt a.1 b.1 then es
  else (es.toList.mergeSort fun a b => A.compare .le a.1 b.1).toArray

/-- The table that `make` checks, from the results `results` of a function at the words `0` to
`results.size - 1`. The keys are the words equal to themselves and not to word `0`, sorted with
the `<` of `A` by `sortKeys` from the order of `signMagnitude`, and the values are the results at
them, both padded to a power of two with their last entry. Word `0` is the scalar `zero`. At the
inputs equal to it and at the inputs not equal to themselves, the table gives the first of these
forms that is every result there: the input, the input plus `zero`, the first result there, the
negation of the input, `zero` minus the input, and the first result there divided by the input.
When none is, it gives the first result there, and `check` rejects the table. -/
def build (A : Arithmetic α) (results : Array α) : Table α :=
  let zero : α := Numeric.Word.ofNatBits 0
  let entries : Array (α × α) := results.mapIdx fun i y => (Numeric.Word.ofNatBits i, y)
  let ordinary := fun (e : α × α) => A.compare .eq e.1 e.1
  let byValue := (signMagnitude (Numeric.Word.width (α := α))).filterMap fun i => entries[i]?
  let sorted := sortKeys A (byValue.filter fun e => ordinary e && !A.compare .eq e.1 zero)
  let depth := if sorted.size ≤ 1 then 0 else (sorted.size - 1).log2 + 1
  let last := sorted.back?.getD (zero, zero)
  let pad := 2 ^ depth - sorted.size
  let fits (es : Array (α × α)) (s : Special α) : Bool :=
    es.all fun e => Numeric.Word.toBits e.2 == Numeric.Word.toBits (s.result A e.1)
  let special (es : Array (α × α)) : Special α :=
    let y := (es[0]?.map (·.2)).getD zero
    (List.find? (fits es) [.input, .binary .add zero, .value y, .neg, .flip .sub zero,
      .flip .div y]).getD (.value y)
  { depth
    keys := sorted.map (·.1) ++ Array.replicate pad last.1
    values := sorted.map (·.2) ++ Array.replicate pad last.2
    zero
    atZero := special (entries.filter fun e => ordinary e && A.compare .eq e.1 zero)
    atNaN := special (entries.filter fun e => !ordinary e) }

/-- Whether the table `t` gives `results[i]` at word `i` for every `i` below `results.size`,
comparing scalars by their words. -/
def check (t : Table α) (A : Arithmetic α) (results : Array α) : Bool :=
  (List.range results.size).all fun i =>
    match t.eval A (Numeric.Word.ofNatBits i), results[i]? with
    | .ok y, some z => Numeric.Word.toBits y == Numeric.Word.toBits z
    | _, _ => false

/-- The results of `f` at all the words of `α`, in the order of the words. -/
def results (f : α → α) : Array α :=
  Array.ofFn fun i : Fin (2 ^ Numeric.Word.width (α := α)) => f (Numeric.Word.ofNatBits i.val)

/-- The value of the word of `x` read as a natural number is `x`. -/
theorem ofNatBits_toNat_toBits (x : α) :
    Numeric.Word.ofNatBits (Numeric.Word.toBits x).toNat = x := by
  rw [Numeric.Word.ofNatBits_eq, BitVec.ofNat_toNat, BitVec.setWidth_eq,
    Numeric.Word.ofBits_toBits]

/-- A table that `check` accepts against the results of `f` at all the words gives `f` at every
input. -/
theorem eval_eq_of_check (t : Table α) (A : Arithmetic α) (f : α → α)
    (h : t.check A (results f) = true) (x : α) : t.eval A x = .ok (f x) := by
  have hlt := (Numeric.Word.toBits x).isLt
  unfold check at h
  rw [List.all_eq_true] at h
  have hi := h (Numeric.Word.toBits x).toNat (List.mem_range.2 (by simpa [results] using hlt))
  rw [ofNatBits_toNat_toBits, results, Array.getElem?_ofFn, dite_eq_left hlt] at hi
  simp only [ofNatBits_toNat_toBits] at hi
  cases he : t.eval A x with
  | error e =>
      rw [he] at hi
      cases hi
  | ok y =>
      rw [he] at hi
      have hb : Numeric.Word.toBits y = Numeric.Word.toBits (f x) := by simpa using hi
      rw [← Numeric.Word.ofBits_toBits y, hb, Numeric.Word.ofBits_toBits]

/-- The table of `f` in the arithmetic `A`, when `α` has at most `maxWidth` bits. `f` is evaluated
once at every word, and the table of `build` is returned when `check` finds that it gives `f` at
every word. -/
def make (A : Arithmetic α) (f : α → α) : Option (Of A f) :=
  if Numeric.Word.width (α := α) ≤ maxWidth then
    let rs := results f
    let t := build A rs
    if h : t.check A rs = true then some ⟨t, eval_eq_of_check t A f h⟩ else none
  else none

end Make

section CorrectlyRounded

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]
  {A : Arithmetic (ExecFloat (Configured.Family format code plan))}
  {R : Reader (ExecFloat (Configured.Family format code plan))} {keys values : Nat}
  {x : Expr (ExecFloat (Configured.Family format code plan)) Γ .scalar}
  {env : Env (ExecFloat (Configured.Family format code plan)) Γ}
  {v : ExecFloat (Configured.Family format code plan)}

/-- Inside any expression of an IEEE format, a table of FloatLib's correctly rounded exponential
gives the exponential of a finite argument rounded once to nearest, when the exact exponential is
below the overflow threshold of the format. -/
theorem Of.toReal_exp (T : Of A (ExecFloat.Binary.CorrectlyRounded.exp · {}))
    (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (hx : x.eval A R env = .ok v) (hfinite : ExecFloat.Binary.isFinite v = true)
    (hrange : Real.exp (ExecFloat.Binary.toReal v) <
      Model.Transcendentals.Certified.overflowThreshold format)
    (hformat : format.isIEEE = true) :
    ∃ y, (T.table.apply keys values x).eval A R env = .ok y ∧
      ExecFloat.Binary.toReal y = Model.roundAt format (Real.exp (ExecFloat.Binary.toReal v)) :=
  ⟨_, by rw [T.eval_apply hk hv, hx]; rfl,
    ExecFloat.Binary.CorrectlyRounded.toReal_exp_eq_roundAt hfinite hrange hformat⟩

/-- Inside any expression of an IEEE format, a table of FloatLib's correctly rounded `expMinus1`
gives `exp v - 1` of a finite argument `v` rounded once to nearest, when the exact result is below
the overflow threshold of the format. -/
theorem Of.toReal_expMinus1 (T : Of A (ExecFloat.Binary.CorrectlyRounded.expMinus1 · {}))
    (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (hx : x.eval A R env = .ok v) (hfinite : ExecFloat.Binary.isFinite v = true)
    (hrange : Real.exp (ExecFloat.Binary.toReal v) - 1 <
      Model.Transcendentals.Certified.overflowThreshold format)
    (hformat : format.isIEEE = true) :
    ∃ y, (T.table.apply keys values x).eval A R env = .ok y ∧
      ExecFloat.Binary.toReal y =
        Model.roundAt format (Real.exp (ExecFloat.Binary.toReal v) - 1) :=
  ⟨_, by rw [T.eval_apply hk hv, hx]; rfl,
    ExecFloat.Binary.CorrectlyRounded.toReal_expMinus1_eq_roundAt hfinite hrange hformat⟩

/-- Inside any expression of an IEEE format, a table of FloatLib's correctly rounded logarithm
gives the logarithm of a finite positive argument rounded once to nearest, when the exact
logarithm is below the overflow threshold of the format in absolute value. -/
theorem Of.toReal_log (T : Of A (ExecFloat.Binary.CorrectlyRounded.log · {}))
    (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (hx : x.eval A R env = .ok v) (hfinite : ExecFloat.Binary.isFinite v = true)
    (hpositive : 0 < ExecFloat.Binary.toReal v)
    (hrange : |Real.log (ExecFloat.Binary.toReal v)| <
      Model.Transcendentals.Certified.overflowThreshold format)
    (hformat : format.isIEEE = true) :
    ∃ y, (T.table.apply keys values x).eval A R env = .ok y ∧
      ExecFloat.Binary.toReal y = Model.roundAt format (Real.log (ExecFloat.Binary.toReal v)) :=
  ⟨_, by rw [T.eval_apply hk hv, hx]; rfl,
    ExecFloat.Binary.CorrectlyRounded.toReal_log_eq_roundAt hfinite hpositive hrange hformat⟩

/-- Inside any expression of an IEEE format, a table of FloatLib's correctly rounded `logPlus1`
gives `log (1 + v)` of a finite argument `v` above `-1` rounded once to nearest, when the exact
result is below the overflow threshold of the format in absolute value. -/
theorem Of.toReal_logPlus1 (T : Of A (ExecFloat.Binary.CorrectlyRounded.logPlus1 · {}))
    (hk : R.Holds keys T.table.keys) (hv : R.Holds values T.table.values)
    (hx : x.eval A R env = .ok v) (hfinite : ExecFloat.Binary.isFinite v = true)
    (hdomain : -1 < ExecFloat.Binary.toReal v)
    (hrange : |Real.log (1 + ExecFloat.Binary.toReal v)| <
      Model.Transcendentals.Certified.overflowThreshold format)
    (hformat : format.isIEEE = true) :
    ∃ y, (T.table.apply keys values x).eval A R env = .ok y ∧
      ExecFloat.Binary.toReal y =
        Model.roundAt format (Real.log (1 + ExecFloat.Binary.toReal v)) :=
  ⟨_, by rw [T.eval_apply hk hv, hx]; rfl,
    ExecFloat.Binary.CorrectlyRounded.toReal_logPlus1_eq_roundAt hfinite hdomain hrange hformat⟩

end CorrectlyRounded

/-- `f` at every entry of `xs`, computed on the device by the kernel of the table over `xs`, with
the keys and the values of the table as inputs `1` and `2`. It fails when the device has no
arithmetic for the precision of `α`, when the words of `α` are wider than their storage, when
`runBytes` fails, which it does without LibTorch also for an empty array, and when a result word
is not below `2 ^ Numeric.Word.width α`. No Lean theorem covers the device run, as the module doc
says. -/
@[no_expose] def Of.runDevice [Scalar α] {f : α → α} (T : Of (Scalar.arithmetic (α := α)) f)
    (xs : Array α) : IO (Array α) := do
  let precision := Scalar.precision α
  unless precision.supportsGPU do
    throw <| IO.userError "table: the device has no arithmetic for the scalar format"
  let width := precision.bytes
  unless Numeric.Word.width α ≤ 8 * width do
    throw <| IO.userError "table: the scalar words are wider than their storage"
  let words (a : Array α) : ByteArray :=
    a.foldl (fun b x => Bytes.pushWordLE b width (Numeric.Word.toBits x).toNat) .empty
  let out ← runBytes inferInstance T.table.kernel
    #[words xs, words T.table.keys, words T.table.values] xs.size.toUInt64
  unless out.size == xs.size * width do
    throw <| IO.userError s!"table: expected {xs.size * width} bytes, received {out.size}"
  let bound := 2 ^ Numeric.Word.width α
  let results := Array.ofFn fun i : Fin xs.size => Bytes.wordFromLE out (i.val * width) width
  unless results.all (· < bound) do
    throw <| IO.userError "table: the device returned a word wider than the format"
  return results.map Numeric.Word.ofNatBits

end Table

end TorchLean.Kernel
