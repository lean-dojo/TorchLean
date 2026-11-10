/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Kolmogorov-Arnold Networks

KAN layers replace each scalar edge by a small trainable one-dimensional function. TorchLean keeps
that structure visible: an edge family first expands every scalar input into basis features, and the
KAN layer learns one coefficient per `(output, input, basis)` edge. `kan` builds the network from a
`KAN.Config`.

The built-in family `nn.PiecewiseLinear.edgeFamily` uses triangular piecewise-linear hats. Any
other `KAN.EdgeFamily` consists of a basis size and a TorchLean model from `[inputWidth]` to
`[inputWidth * basisSize]`.

Our source for the layer is Liu et al., Section 2.2. Eq. (2.5) sums one learned function per edge,
and eq. (2.12) writes each function as `Σ c_i B_i(x)` with learned `c_i`, which here are the
weights of the linear map (plus one bias per output). Our edges are simpler than the paper's in
three ways. The hats `ReLU(1 - |s x - k|)` are degree-1 B-splines on the integer knots, where the
paper usually takes cubic ones (`k = 3`). Eqs. (2.10) and (2.11) add `w_b · silu(x)` to every
edge, and we leave that term out. And the grid is fixed, while the paper updates it from the
activations. `KAN.stack` also puts a `tanh` between layers, which the paper's network doesn't
have. With the default `gridSize := 8` and scale `1`, an input of `2.5` lights hats 2 and 3 at
height `0.5` each.

References:

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025, arXiv:2404.19756, Section 2.2,
  eqs. (2.5) and (2.10) to (2.12).
- C. de Boor, "A Practical Guide to Splines", Applied Mathematical Sciences, Springer, 1978,
  doi:10.1007/978-1-4612-6333-3.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean


open TorchLean.Tensor

namespace nn

/--
Backend-compatible KAN edge family.

An edge family turns each scalar input coordinate into `basisSize` features, and a KAN layer then
applies a learned linear map to all expanded features. The basis is a TorchLean model fragment, so
the resulting KAN runs on the eager, typed graph, CPU and CUDA training paths that the underlying
operations support.

For the usual KAN interpretation, the basis must expand each input coordinate independently. The
shape signature alone does not enforce this, and a custom basis model can mix coordinates.
-/
structure KAN.EdgeFamily where
  /-- Short label shown in model summaries and training metadata. -/
  name : String
  /-- Number of basis features produced per scalar input coordinate. Must be positive. -/
  basisSize : Nat
  /-- Basis model for an unbatched feature vector. -/
  basis : (inputWidth : Nat) →
    nn.Sequential [inputWidth] [inputWidth * basisSize]

namespace PiecewiseLinear

/--
Configuration for triangular piecewise-linear KAN edge bases.

The basis functions are hats centered at the integer knots
$0,\ldots,\mathrm{gridSize}-1$. The input is multiplied by `inputScale` before the hats are
evaluated. For normalized data in $[0,1]$, setting
$\mathrm{inputScale}=\mathrm{gridSize}-1$ spreads the grid across the full interval. The built-in
family uses a nonnegative integer scale so the same definition works for every TorchLean scalar
backend through `Context`'s natural-number conversion.
-/
structure Config where
  /--
  Number of knots, hence the number of basis functions per scalar coordinate. Must be positive.
  -/
  gridSize : Nat
  /--
  Nonnegative integer scale applied before basis evaluation, such as $\mathrm{gridSize}-1$ for
  normalized $[0,1]$ inputs.
  -/
  inputScale : Nat := 1
deriving Repr

/--
Expand a tensor of shape `[inputWidth]` to all triangular basis features.

The output is flattened row-major from a `(gridSize × inputWidth)` table:
$[\operatorname{basis}_0(x_0),\ldots,\operatorname{basis}_0(x_n),
\operatorname{basis}_1(x_0),\ldots]$.

Each basis value is $\operatorname{ReLU}(1-|\mathrm{inputScale}\,x_i-k|)$, computed with ordinary
TorchLean ops.
-/
def layer (config : Config) (inputWidth : Nat) :
    nn.Sequential [inputWidth] [inputWidth * config.gridSize] :=
  nn.Sequential.fromLayer
    { kind := s!"KAN.PiecewiseLinear(grid={config.gridSize},scale={config.inputScale})"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      validateConfig := do
        if inputWidth = 0 then
          throw "KAN.PiecewiseLinear: input width must be positive"
        if config.gridSize = 0 then
          throw "KAN.PiecewiseLinear: grid size must be positive"
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            ((do
              let zeros : Tensor α [config.gridSize, inputWidth] :=
                Tensor.full [config.gridSize, inputWidth] (0 : α)
              let scaledInput ← Ops.apply (m := m) (α := α) .scale
                  (x, .data
                    (Ops.dataConst (m := m) (α := α)
                      (Tensor.scalar ((config.inputScale : Nat) : α))), ())
              let initialBasis ← Ops.const (m := m) (α := α) zeros
              let basisValues ← (List.finRange config.gridSize).foldlM
                (init := initialBasis) (fun acc k => do
                let centerValues : Tensor α [inputWidth] :=
                  Tensor.full [inputWidth] ((k.val : Nat) : α)
                let unitValues : Tensor α [inputWidth] :=
                  Tensor.full [inputWidth] (1 : α)
                let center ← Ops.const (m := m) (α := α) centerValues
                let ones ← Ops.const (m := m) (α := α) unitValues
                let shifted ← Ops.apply (m := m) (α := α) .sub
                    (scaledInput, center, ())
                let distance ← Ops.apply (m := m) (α := α) .abs (shifted, ())
                let height ← Ops.apply (m := m) (α := α) .sub
                    (ones, distance, ())
                let basis ← Ops.apply (m := m) (α := α) .relu (height, ())
                let basisRow ← Ops.apply (m := m) (α := α)
                    (.reshape (s := [inputWidth]) (t := [1, inputWidth])
                      (by simp)) (basis, ())
                let index : Tensor (Fin config.gridSize) [1] :=
                  TorchLean.Tensor.ofFn (fun _ : Fin 1 => k)
                -- Add the row into `acc` at `index` along axis 0: a scatter sum over every axis.
                Ops.apply (m := m) (α := α)
                  (.scatter (s := [config.gridSize, inputWidth]) (t := [1, inputWidth]) .sum
                    (List.finRange (Shape.rank [config.gridSize, inputWidth]))
                    (List.nodup_finRange _))
                  (acc, Ops.dataConst (m := m) (α := α)
                      (TorchLean.Tensor.map (TorchLean.Op.indexOfAxisTuples
                          (Shape.extent [config.gridSize, inputWidth]) _)
                        (TorchLean.Tensor.selectSources (s := [config.gridSize, inputWidth])
                          ⟨0, Shape.AxisInBounds.proof⟩ index)),
                    Ops.Ref.cast (m := m) (α := α)
                      (by rw [Shape.eraseAxes_finRange, List.append_nil]) basisRow, ()))
              Ops.apply (m := m) (α := α)
                  (.reshape (s := [config.gridSize, inputWidth])
                    (t := [inputWidth * config.gridSize]) (by
                  simp [Nat.mul_comm])) (basisValues, ())
            ) : m (Ops.Ref (m := m) (α := α)
              [inputWidth * config.gridSize]))
    }

/-- Turn piecewise-linear triangular bases into a general KAN edge family. -/
def edgeFamily (config : Config) : KAN.EdgeFamily :=
  { name := s!"KAN.PiecewiseLinear(grid={config.gridSize},scale={config.inputScale})"
    basisSize := config.gridSize
    basis := layer config }

end PiecewiseLinear

/-- Architecture of a Kolmogorov-Arnold network over feature vectors. -/
structure KAN.Config where
  /-- Number of scalar input coordinates. Must be positive. -/
  inputWidth : Nat
  /-- Hidden KAN widths. Each must be positive and creates one KAN layer followed by `tanh`. -/
  hiddenWidths : List Nat := []
  /-- Number of output coordinates/classes. Must be positive. -/
  outputWidth : Nat
  /-- Edge basis family. The default is a compact triangular piecewise-linear basis. -/
  edge : EdgeFamily := PiecewiseLinear.edgeFamily { gridSize := 8 }

namespace KAN.Config

/-- Validate every architecture width and the selected edge family before construction. -/
def validate (config : Config) : Except String Unit := do
  if config.inputWidth = 0 then
    throw "KAN: input width must be positive"
  if ∃ width ∈ config.hiddenWidths, width = 0 then
    throw "KAN: hidden widths must be positive"
  if config.outputWidth = 0 then
    throw "KAN: output width must be positive"
  if config.edge.basisSize = 0 then
    throw "KAN: edge basis size must be positive"

/-- Typed input boundary with arbitrary batch axes. -/
abbrev inputShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.inputWidth]

/-- Typed output boundary with the same batch axes as the input. -/
abbrev outputShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.outputWidth]

end KAN.Config

/--
One KAN layer over a feature vector.

The layer first applies the selected edge basis to every input coordinate, then learns coefficients
with an ordinary linear map from the expanded features to `outputWidth`.
-/
def KAN.layer (inputWidth outputWidth : Nat) (edge : EdgeFamily) :
  nn.Builder (nn.Sequential [inputWidth] [outputWidth]) :=
  if inputWidth = 0 then
    pure <| nn.Sequential.invalid [inputWidth] [outputWidth]
      "KAN" "KAN: input width must be positive"
  else if outputWidth = 0 then
    pure <| nn.Sequential.invalid [inputWidth] [outputWidth]
      "KAN" "KAN: output width must be positive"
  else if edge.basisSize = 0 then
    pure <| nn.Sequential.invalid [inputWidth] [outputWidth]
      "KAN" "KAN: edge basis size must be positive"
  else
    nn.Sequential![
      pure (edge.basis inputWidth),
      nn.linear (inputWidth * edge.basisSize) outputWidth
    ]

namespace KAN

/-- Recursive KAN stack over one feature vector. Hidden layers use `tanh`. -/
def stack (edge : EdgeFamily) :
    (inputWidth : Nat) → (hiddenWidths : List Nat) → (outputWidth : Nat) →
      nn.Builder (nn.Sequential [inputWidth] [outputWidth])
  | inputWidth, .nil, outputWidth => layer inputWidth outputWidth edge
  | inputWidth, .cons hiddenWidth remainingWidths, outputWidth =>
      nn.Sequential![
        layer inputWidth hiddenWidth edge,
        nn.op .tanh,
        stack edge hiddenWidth remainingWidths outputWidth
      ]

end KAN

/--
Build a KAN over any `batchShape`.

The model fixes no task, so regression and classification share this constructor and differ only
in the `Trainer.create` objective, such as `objective := .mse`, `.oneHotCrossEntropy axis` or
`.custom ...`.
-/
def kan (config : KAN.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "KAN" message
  | .ok () => do
      let sample ←
        KAN.stack config.edge config.inputWidth config.hiddenWidths
          config.outputWidth
      pure (by
        simpa [KAN.Config.inputShape, KAN.Config.outputShape] using
          nn.mapLeading batchShape sample)

end nn

end TorchLean
