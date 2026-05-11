import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Approximation Theory" =>
%%%
tag := "approximation-theory"
%%%

The word "approximation" appears in several places in TorchLean, and the meanings are different.
Runtime approximation asks how far executable arithmetic may differ from ideal semantics. Verifier
approximation asks for sound enclosures of activations or margins. Approximation theory asks a more
classical mathematical question: what functions can networks represent, and with what error?

The quantifiers are different in each setting:

- *CROWN style verifier approximation*: one network, one input region, and sound bounds; the
  artifact is a certificate over activations or margins.
- *Runtime approximation*: one executable path compared with one specification path; the artifact
  is a tolerance proof for Float32 or `IEEE32Exec`.
- *Universal approximation*: for a target function and error, there exists a network; the artifact
  is an existence theorem over a function class.

Keeping the quantifiers visible is the whole point. A theorem saying that ReLU networks are dense in
a function space is not a deployment certificate for one trained model, and a CROWN certificate for
one model is not a universal approximation theorem.

The three forms are:

$$`\text{universal approximation:}\quad
\forall f,\forall\varepsilon>0,\exists N_\theta,\;
\sup_x |N_\theta(x)-f(x)|<\varepsilon`

$$`\text{runtime approximation:}\quad
\forall x,\;
\|N_{\mathrm{run}}(x)-N_{\mathrm{spec}}(x)\|\le \varepsilon`

$$`\text{verifier approximation:}\quad
x\in B\Longrightarrow N(x)\in \mathcal A(B).`

# Universal Approximation

The basic one dimensional theorem has the familiar form:

$$`\forall f:[a,b]\to\mathbb{R},\quad
\forall \varepsilon>0,\quad
\exists h_{\mathrm{ReLU}},\quad
\forall x\in[a,b],\quad
\left|h_{\mathrm{ReLU}}(x)-f(x)\right|<\varepsilon`

The [universal approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximation.lean)
contains the real valued statement `relu_universal_approximation_Icc`. The rate file
[universal approximation with rates](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationRate.lean)
adds `reluApproximationWidth` and proves rate style theorems such as
`relu_universal_approximation_Icc_rate`: the width is chosen as an explicit function of the
Lipschitz constant, interval size, and error target.

The multidimensional file
[universal approximation in higher dimension](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationND.lean)
uses coordinate subalgebras and Stone-Weierstrass style reasoning. The shape of the theorem changes
from an interval on the line to compact domains and tensor valued coordinates, but the reader
should keep the same mental picture: networks are used as a dense class of functions under stated
topological hypotheses.

The float32 and `IEEE32Exec` variants express the same conceptual theorem after the approximating
network is tied to a concrete scalar model. The clean real theorem and the executable bridge are
related, but they are not the same statement.

The float versions are intentionally explicit about their extra obligations:

$$`\text{real approximation error}
+ \text{dyadic parameter quantization error}
+ \text{rounded execution error}
\le \text{requested tolerance}`

For example, the [IEEE32Exec approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32Exec.lean)
contains theorems such as `reluApproximationIccIEEE32Exec_threeTerm`,
`hinge_fun_dyadic_quantization_error_le_Icc`, and
`reluApproximationIccIEEE32Exec_dyadicHalfUlp`. These statements make the finite precision bridge
visible instead of claiming that a theorem over `ℝ` automatically transfers to binary32 code.

# Float Interval Approximation

Float interval approximation asks for a sound set interpretation of finite values. The interval
semantics starts with an interpretation map:

$$`\gamma_I : \operatorname{Interval}\to\operatorname{Set}(\operatorname{IEEE32Exec}),
\qquad
\gamma : \operatorname{Box}(d)\to
\operatorname{Set}\!\left(\operatorname{Fin}(d)\to\operatorname{IEEE32Exec}\right)`

An abstract operation is sound when every concrete result produced by inputs in the concrete sets
is still inside the abstract output. In the
[float interval semantics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean),
the central theorems are `add_sound`, `mul_sound`, `relu_sound`, `aff_sound`, and `eval_sound`.
The theorem shape is:

$$`x\in\gamma(B)\quad\Longrightarrow\quad
\operatorname{eval}(net,x)\in\gamma_I(\operatorname{evalSharp}(net,B))`

That is the same soundness pattern as verification certificates, but here the carrier is finite
float semantics and interval images. The
[exact image theorem API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/FloatInterval/ExactImageTheorem.lean)
then states stronger exact image style conditions, including `ExactIntervalImage` and
`roundedTargetExactIntervalImage_of_correctRounding`. The
[constant target example](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/FloatInterval/ConstantTarget.lean)
is the small case readers can use to see the definitions without the full network machinery.

This is close in spirit to verification, but it is still a different layer. A verifier certificate
usually checks a specific network on a specific input region. The approximation theory layer is
about representability and semantic approximation statements.

# Three Meanings We Keep Separate

TorchLean uses "approximation" in at least three different ways:

- *Approximation theory*: networks as function approximators.
- *Runtime approximation*: executable arithmetic versus ideal semantics.
- *Verifier approximation*: sound overapproximations of reachable activations or output margins.

Keeping those meanings separate prevents ambiguity: an approximation theorem, a Float32 tolerance,
and a CROWN bound are different claims with different proof obligations.
