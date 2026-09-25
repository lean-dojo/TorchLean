/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Kolmogorov-Arnold Networks

KAN layers replace each scalar edge by a small trainable one-dimensional function. TorchLean keeps
that structure visible: an edge family first expands every scalar input into basis features, and the
KAN layer learns one coefficient per `(output, input, basis)` edge.

The built-in family uses triangular piecewise-linear hats. Another family consists of a basis
size and a TorchLean model from `[inputWidth]` to `[inputWidth * basisSize]`.

References:

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756.
- C. de Boor, "A Practical Guide to Splines", Springer, 1978/2001.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models
namespace KAN

/--
Backend-compatible KAN edge family.

An edge family turns each scalar input coordinate into `basisSize` features. A KAN layer then
applies a learned linear map to all expanded features. The basis is a TorchLean model fragment, not
an arbitrary Lean callback, so the resulting KAN can run in eager, typed graph, CPU, and CUDA
training paths supported by the underlying operations.
-/
structure EdgeFamily where
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
  Nonnegative integer scale applied before basis evaluation; use $\mathrm{gridSize}-1$ for
  normalized $[0,1]$ inputs.
  -/
  inputScale : Nat := 1
deriving Repr

/--
Expand a tensor of shape `[inputWidth]` to all triangular basis features.

The output is flattened row-major from a `(gridSize × inputWidth)` table:
$[\operatorname{basis}_0(x_0),\ldots,\operatorname{basis}_0(x_n),
\operatorname{basis}_1(x_0),\ldots]$.

Each basis value is
$\operatorname{ReLU}(1-|\mathrm{inputScale}\,x_i-k|)$, expressed directly in the ordinary
TorchLean op language rather than through an opaque spline evaluator.
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
              let xBasis ← Runtime.Autograd.Torch.scale (m := m) (α := α) x
                ((config.inputScale : Nat) : α)
              let out0 ← Runtime.Autograd.Torch.const (m := m) (α := α) zeros
              let out ← (List.finRange config.gridSize).foldlM (init := out0) (fun acc k => do
                let centerT : Tensor α [inputWidth] :=
                  Tensor.full [inputWidth] ((k.val : Nat) : α)
                let oneT : Tensor α [inputWidth] :=
                  Tensor.full [inputWidth] (1 : α)
                let c ← Runtime.Autograd.Torch.const (m := m) (α := α) centerT
                let ones ← Runtime.Autograd.Torch.const (m := m) (α := α) oneT
                let shifted ← Runtime.Autograd.Torch.sub (m := m) (α := α) xBasis c
                let dist ← Runtime.Autograd.Torch.abs (m := m) (α := α) shifted
                let raw ← Runtime.Autograd.Torch.sub (m := m) (α := α) ones dist
                let basis ← Runtime.Autograd.Torch.relu (m := m) (α := α) raw
                let basisRow ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := [inputWidth]) (s₂ := [1, inputWidth]) basis
                  (by simp [Spec.Shape.size])
                let index : Tensor (Fin config.gridSize) [1] :=
                  TorchLean.Tensor.ofFn (fun _ : Fin 1 => k)
                Runtime.Autograd.Torch.scatterAdd (m := m) (α := α)
                  (s := [config.gridSize, inputWidth]) 0 1 acc basisRow
                  (Runtime.Autograd.Torch.dataConst (m := m) (α := α) index))
              let flat ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [config.gridSize, inputWidth])
                (s₂ := [config.gridSize * inputWidth])
                out (by
                  simp [Spec.Shape.size, Nat.mul_comm])
              Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [config.gridSize * inputWidth])
                (s₂ := [inputWidth * config.gridSize])
                flat (by
                  simp [Spec.Shape.size, Nat.mul_comm])
            ) : m (Runtime.Autograd.Model.RefTy (m := m) (α := α)
              [inputWidth * config.gridSize]))
    }

/-- Turn piecewise-linear triangular bases into a general KAN edge family. -/
def edgeFamily (config : Config) : EdgeFamily :=
  { name := s!"KAN.PiecewiseLinear(grid={config.gridSize},scale={config.inputScale})"
    basisSize := config.gridSize
    basis := layer config }

end PiecewiseLinear

/-- Architecture of a Kolmogorov-Arnold network over feature vectors. -/
structure Config where
  /-- Number of scalar input coordinates. Must be positive. -/
  inputWidth : Nat
  /-- Hidden KAN widths. Each must be positive and creates one KAN layer followed by `tanh`. -/
  hiddenWidths : List Nat := []
  /-- Number of output coordinates/classes. Must be positive. -/
  outputWidth : Nat
  /-- Edge basis family. The default is a compact triangular piecewise-linear basis. -/
  edge : EdgeFamily := PiecewiseLinear.edgeFamily { gridSize := 8 }

namespace Config

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
  batchShape.appendDim config.inputWidth

/-- Typed output boundary with the same batch axes as the input. -/
abbrev outputShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.outputWidth

end Config

/--
One KAN layer over a feature vector.

The layer first applies the selected edge basis to every input coordinate, then learns coefficients
with an ordinary linear map from the expanded features to `outputWidth`.
-/
def layer (inputWidth outputWidth : Nat) (edge : EdgeFamily) :
  nn.Builder (nn.Sequential [inputWidth] [outputWidth]) :=
  if inputWidth = 0 then
    pure <| nn.Internal.invalidConfiguration [inputWidth] [outputWidth]
      "KAN" "KAN: input width must be positive"
  else if outputWidth = 0 then
    pure <| nn.Internal.invalidConfiguration [inputWidth] [outputWidth]
      "KAN" "KAN: output width must be positive"
  else if edge.basisSize = 0 then
    pure <| nn.Internal.invalidConfiguration [inputWidth] [outputWidth]
      "KAN" "KAN: edge basis size must be positive"
  else
    nn.Sequential![
      pure (edge.basis inputWidth),
      nn.linear (inputWidth * edge.basisSize) outputWidth
    ]

namespace Internal

/-- Recursive KAN stack over one feature vector. Hidden layers use `tanh`. -/
def stack (edge : EdgeFamily) :
    (inputWidth : Nat) → (hiddenWidths : List Nat) → (outputWidth : Nat) →
      nn.Builder (nn.Sequential [inputWidth] [outputWidth])
  | inputWidth, .nil, outputWidth => layer inputWidth outputWidth edge
  | inputWidth, .cons hiddenWidth remainingWidths, outputWidth =>
      nn.Sequential![
        layer inputWidth hiddenWidth edge,
        nn.tanh,
        stack edge hiddenWidth remainingWidths outputWidth
      ]

end Internal

end KAN

/--
Build a KAN over any `batchShape`.

Task semantics are deliberately not baked into the model name: use `Trainer.new` with
`objective := .mse`, `.oneHotCrossEntropy axis`, or `.custom ...` with the same KAN
constructor.
-/
def kan (config : KAN.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "KAN" message
  | .ok () => do
      let sample ←
        KAN.Internal.stack config.edge config.inputWidth config.hiddenWidths
          config.outputWidth
      pure (by
        simpa [KAN.Config.inputShape, KAN.Config.outputShape, Shape.appendDim_eq_concat] using
          nn.mapLeading batchShape sample)

end models
end nn

end TorchLean
