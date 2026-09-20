import VersoManual
import VersoBlueprint

/-!
Bibliography entries for the TorchLean guide.

Every work that the guide leans on for a technical claim is registered here once, with an
`@[bib "label"]` attribute, and cited from prose with `{Informal.citet label}[]` or
`{Informal.citep label}[]`. Keeping the metadata in one file means a paper's authors and year are
written once, the rendered bibliography lists each work a single time, and the bibliography page
can show which sections cite it.

Verso's `Citable` type covers papers: conference proceedings, journal articles, arXiv preprints,
and theses. Books and living web resources (standards pages, project repositories, framework
documentation) have no faithful `Citable` shape, so those stay as ordinary links in the prose that
discusses them rather than being forced into a paper-shaped record here.
-/

open Verso.Genre.Manual.Bibliography

namespace TorchLeanBlueprint.Bib

/-! ## TorchLean and the proof assistant it is written in -/

/-- The TorchLean paper; the same entry appears in `README.md` as BibTeX. -/
@[bib "torchlean2026"]
def torchlean2026 : Citable := .arXiv
  { title := inlines!"TorchLean: Formalizing Neural Networks in Lean"
    authors := #[inlines!"Robert Joseph George", inlines!"Jennifer Cruden",
      inlines!"Will Adkisson", inlines!"Xiangru Zhong", inlines!"Huan Zhang",
      inlines!"Anima Anandkumar"]
    year := 2026
    id := "2602.22631" }

/-- The language paper for Lean 4, which is both the proof assistant and the host language. -/
@[bib "lean4"]
def lean4 : Citable := .inProceedings
  { title := inlines!"The Lean 4 Theorem Prover and Programming Language"
    authors := #[inlines!"Leonardo de Moura", inlines!"Sebastian Ullrich"]
    year := 2021
    booktitle := inlines!"Automated Deduction (CADE 28)"
    url := some "https://lean-lang.org/papers/lean4.pdf" }

/-- Mathlib, which supplies the real analysis, measure theory, and algebra the proofs build on. -/
@[bib "mathlib2020"]
def mathlib2020 : Citable := .inProceedings
  { title := inlines!"The Lean Mathematical Library"
    authors := #[inlines!"The mathlib Community"]
    year := 2020
    booktitle := inlines!"Certified Programs and Proofs (CPP)"
    url := some "https://doi.org/10.1145/3372885.3373824" }

/--
Lean's reference-counting story. We cite it whenever the guide claims that an immutable interface
does not force a copy: uniquely owned values are updated in place by the compiled code.
-/
@[bib "immutablebeans2019"]
def immutablebeans2019 : Citable := .inProceedings
  { title := inlines!"Counting Immutable Beans: Reference Counting Optimized for Purely " ++
      inlines!"Functional Programming"
    authors := #[inlines!"Sebastian Ullrich", inlines!"Leonardo de Moura"]
    year := 2019
    booktitle := inlines!"Implementation and Application of Functional Languages (IFL)"
    url := some "https://arxiv.org/abs/1908.05647" }

/-! ## Deep learning frameworks and automatic differentiation -/

/-- PyTorch, the framework whose surface syntax and idioms TorchLean deliberately echoes. -/
@[bib "pytorch2019"]
def pytorch2019 : Citable := .arXiv
  { title := inlines!"PyTorch: An Imperative Style, High-Performance Deep Learning Library"
    authors := #[inlines!"Adam Paszke", inlines!"Sam Gross", inlines!"Francisco Massa",
      inlines!"Adam Lerer", inlines!"James Bradbury", inlines!"Gregory Chanan",
      inlines!"Trevor Killeen", inlines!"Zeming Lin", inlines!"Natalia Gimelshein",
      inlines!"Luca Antiga", inlines!"Alban Desmaison", inlines!"Andreas Köpf",
      inlines!"Edward Yang", inlines!"Zachary DeVito", inlines!"Martin Raison",
      inlines!"Alykhan Tejani", inlines!"Sasank Chilamkurthy", inlines!"Benoit Steiner",
      inlines!"Lu Fang", inlines!"Junjie Bai", inlines!"Soumith Chintala"]
    year := 2019
    id := "1912.01703" }

/--
PyTorch's own op-tagged graph capture. We cite it in the IR chapter because `torch.fx` is the
analogue of `NN.IR.Graph`. FX represents function, method, and module calls; targets may be Python
callables or names, depending on the node kind. Shape metadata is separate from the node's type.
-/
@[bib "fx2022"]
def fx2022 : Citable := .inProceedings
  { title := inlines!"torch.fx: Practical Program Capture and Transformation for Deep Learning " ++
      inlines!"in Python"
    authors := #[inlines!"James K. Reed", inlines!"Zachary DeVito", inlines!"Horace He",
      inlines!"Ansley Ussery", inlines!"Jason Ansel"]
    year := 2022
    booktitle := inlines!"Machine Learning and Systems (MLSys)"
    url := some "https://arxiv.org/abs/2112.08429" }

/--
The compiler infrastructure that made "operations are data, with declared types and a verifier" the
default way to build an ML IR. TorchLean's node array is far smaller, but the design decisions it
copies, a closed operation vocabulary and shape metadata that a checker validates, come from this
line of work.
-/
@[bib "mlir2021"]
def mlir2021 : Citable := .inProceedings
  { title := inlines!"MLIR: Scaling Compiler Infrastructure for the End of Moore's Law"
    authors := #[inlines!"Chris Lattner", inlines!"Mehdi Amini", inlines!"Uday Bondhugula",
      inlines!"Albert Cohen", inlines!"Andy Davis", inlines!"Jacques Pienaar",
      inlines!"River Riddle", inlines!"Tatiana Shpeisman", inlines!"Nicolas Vasilache",
      inlines!"Oleksandr Zinenko"]
    year := 2021
    booktitle := inlines!"Code Generation and Optimization (CGO)"
    url := some "https://doi.org/10.1109/CGO51591.2021.9370308" }

/-- The survey that fixes the vocabulary of forward mode, reverse mode, and the tape. -/
@[bib "baydin2018"]
def baydin2018 : Citable := .article
  { title := inlines!"Automatic Differentiation in Machine Learning: a Survey"
    authors := #[inlines!"Atılım Güneş Baydin", inlines!"Barak A. Pearlmutter",
      inlines!"Alexey Andreyevich Radul", inlines!"Jeffrey Mark Siskind"]
    journal := inlines!"Journal of Machine Learning Research"
    year := 2018
    month := none
    volume := inlines!"18"
    number := inlines!"153"
    pages := some (1, 43)
    url := some "https://jmlr.org/papers/v18/17-468.html" }

/--
Reverse mode as a program transformation on a functional language rather than as a mutable tape.
TorchLean's eager engine is closer to this reading than to a framework tape: `Tape` is a pure
grow-only array, and `backward` is a function of it, so replaying a reverse pass cannot observe a
different forward value than the one that was recorded.
-/
@[bib "pearlmutter2008"]
def pearlmutter2008 : Citable := .article
  { title := inlines!"Reverse-Mode AD in a Functional Framework: Lambda the Ultimate Backpropagator"
    authors := #[inlines!"Barak A. Pearlmutter", inlines!"Jeffrey Mark Siskind"]
    journal := inlines!"ACM Transactions on Programming Languages and Systems"
    year := 2008
    month := none
    volume := inlines!"30"
    number := inlines!"2"
    pages := none
    url := some "https://doi.org/10.1145/1330017.1330018" }

/--
What checkpointing actually is when it is implemented: a schedule that trades recomputation for
storage during the reverse sweep. We cite it where `nn.functional.checkpoint` is discussed, because
that hook currently has the mathematical meaning and none of the schedule.
-/
@[bib "griewank2000"]
def griewank2000 : Citable := .article
  { title := inlines!"Algorithm 799: Revolve. An Implementation of Checkpointing for " ++
      inlines!"the Reverse or Adjoint Mode of Computational Differentiation"
    authors := #[inlines!"Andreas Griewank", inlines!"Andrea Walther"]
    journal := inlines!"ACM Transactions on Mathematical Software"
    year := 2000
    month := none
    volume := inlines!"26"
    number := inlines!"1"
    pages := some (19, 45)
    url := some "https://doi.org/10.1145/347837.347846" }

/--
The paper that made the ReLU-at-zero question respectable. It builds a calculus of "conservative
fields" in which the value automatic differentiation returns at a kink is a legitimate derivative
object, and shows that gradient methods converge with it. We cite it wherever this manual says
that a nonsmooth rule needs a stated convention rather than an apology.
-/
@[bib "bolte2020"]
def bolte2020 : Citable := .inProceedings
  { title := inlines!"A Mathematical Model for Automatic Differentiation in Machine Learning"
    authors := #[inlines!"Jérôme Bolte", inlines!"Edouard Pauwels"]
    year := 2020
    booktitle := inlines!"Neural Information Processing Systems (NeurIPS)"
    url := some "https://arxiv.org/abs/2006.02080" }

/--
The tracing-and-transforms design that `torch.func` and TorchLean's `autograd` transforms both
follow: differentiation is a function on functions, not a method on a mutable tensor. Worth reading
next to our `autograd.Function` type, which is the same idea expressed as scalar polymorphism.
-/
@[bib "jax2018"]
def jax2018 : Citable := .inProceedings
  { title := inlines!"Compiling Machine Learning Programs via High-Level Tracing"
    authors := #[inlines!"Roy Frostig", inlines!"Matthew James Johnson",
      inlines!"Chris Leary"]
    year := 2018
    booktitle := inlines!"Systems for Machine Learning (SysML)"
    url := some "https://mlsys.org/Conferences/doc/2018/146.pdf" }

/-- FlashAttention, the reference point for a fused attention kernel with an exact specification. -/
@[bib "flashattention2022"]
def flashattention2022 : Citable := .arXiv
  { title := inlines!"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    authors := #[inlines!"Tri Dao", inlines!"Daniel Y. Fu", inlines!"Stefano Ermon",
      inlines!"Atri Rudra", inlines!"Christopher Ré"]
    year := 2022
    id := "2205.14135" }

/-! ## Floating-point arithmetic and its formalizations -/

/-- Flocq, whose separation of formats from rounding operators shaped TorchLean's float layer. -/
@[bib "flocq2011"]
def flocq2011 : Citable := .inProceedings
  { title := inlines!"Flocq: A Unified Library for Proving Floating-Point Algorithms in Coq"
    authors := #[inlines!"Sylvie Boldo", inlines!"Guillaume Melquiond"]
    year := 2011
    booktitle := inlines!"20th IEEE Symposium on Computer Arithmetic (ARITH)"
    url := some "https://doi.org/10.1109/ARITH.2011.40" }

/-- Goldberg's survey, still the shortest correct answer to "why is float arithmetic like this". -/
@[bib "goldberg1991"]
def goldberg1991 : Citable := .article
  { title := inlines!"What Every Computer Scientist Should Know About Floating-Point Arithmetic"
    authors := #[inlines!"David Goldberg"]
    journal := inlines!"ACM Computing Surveys"
    year := 1991
    month := none
    volume := inlines!"23"
    number := inlines!"1"
    pages := some (5, 48)
    url := some "https://doi.org/10.1145/103162.103163" }

/--
The CompCert float work, which is the closest existing answer to "how do I connect a proof about
rounded arithmetic to the code a compiler actually emitted".
-/
@[bib "boldo2015"]
def boldo2015 : Citable := .article
  { title := inlines!"Verified Compilation of Floating-Point Computations"
    authors := #[inlines!"Sylvie Boldo", inlines!"Jacques-Henri Jourdan",
      inlines!"Xavier Leroy", inlines!"Guillaume Melquiond"]
    journal := inlines!"Journal of Automated Reasoning"
    year := 2015
    month := none
    volume := inlines!"54"
    number := inlines!"2"
    pages := some (135, 163)
    url := some "https://doi.org/10.1007/s10817-014-9317-x" }

/-! ## Robustness verification and bound propagation -/

/-- The paper that made adversarial examples a standard reason to want verified bounds. -/
@[bib "szegedy2014"]
def szegedy2014 : Citable := .arXiv
  { title := inlines!"Intriguing Properties of Neural Networks"
    authors := #[inlines!"Christian Szegedy", inlines!"Wojciech Zaremba",
      inlines!"Ilya Sutskever", inlines!"Joan Bruna", inlines!"Dumitru Erhan",
      inlines!"Ian Goodfellow", inlines!"Rob Fergus"]
    year := 2014
    id := "1312.6199" }

/-- The convex outer adversarial polytope, the dual view of a linear relaxation. -/
@[bib "wongkolter2018"]
def wongkolter2018 : Citable := .arXiv
  { title :=
      inlines!"Provable Defenses against Adversarial Examples via the \
        Convex Outer Adversarial Polytope"
    authors := #[inlines!"Eric Wong", inlines!"J. Zico Kolter"]
    year := 2018
    id := "1711.00851" }

/-- Interval bound propagation, the cheapest sound bound TorchLean implements. -/
@[bib "gowal2018"]
def gowal2018 : Citable := .arXiv
  { title :=
      inlines!"On the Effectiveness of Interval Bound Propagation for Training \
        Verifiably Robust Models"
    authors := #[inlines!"Sven Gowal", inlines!"Krishnamurthy Dvijotham",
      inlines!"Robert Stanforth", inlines!"Rudy Bunel", inlines!"Chongli Qin",
      inlines!"Jonathan Uesato", inlines!"Relja Arandjelovic",
      inlines!"Timothy Mann", inlines!"Pushmeet Kohli"]
    year := 2018
    id := "1810.12715" }

/-- CROWN, the linear relaxation TorchLean's affine bound propagation follows. -/
@[bib "crown2018"]
def crown2018 : Citable := .arXiv
  { title :=
      inlines!"Efficient Neural Network Robustness Certification with General Activation Functions"
    authors := #[inlines!"Huan Zhang", inlines!"Tsui-Wei Weng", inlines!"Pin-Yu Chen",
      inlines!"Cho-Jui Hsieh", inlines!"Luca Daniel"]
    year := 2018
    id := "1811.00866" }

/-- auto_LiRPA, which generalizes bound propagation from layer stacks to general graphs. -/
@[bib "autolirpa2020"]
def autolirpa2020 : Citable := .arXiv
  { title := inlines!"Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"
    authors := #[inlines!"Kaidi Xu", inlines!"Zhouxing Shi", inlines!"Huan Zhang",
      inlines!"Yihan Wang", inlines!"Kai-Wei Chang", inlines!"Minlie Huang",
      inlines!"Bhavya Kailkhura", inlines!"Xue Lin", inlines!"Cho-Jui Hsieh"]
    year := 2020
    id := "2002.12920" }

/-- Beta-CROWN, the branch-and-bound method behind the alpha-beta-CROWN toolchain. -/
@[bib "betacrown2021"]
def betacrown2021 : Citable := .arXiv
  { title :=
      inlines!"Beta-CROWN: Efficient Bound Propagation with Per-neuron Split \
        Constraints for Complete and Incomplete Neural Network Robustness \
        Verification"
    authors := #[inlines!"Shiqi Wang", inlines!"Huan Zhang", inlines!"Kaidi Xu",
      inlines!"Xue Lin", inlines!"Suman Jana", inlines!"Cho-Jui Hsieh",
      inlines!"J. Zico Kolter"]
    year := 2021
    id := "2103.06624" }

/--
Branch and bound as the search strategy whose leaves an exported certificate records. Cited where
the leaf-artifact format is described, since the format only makes sense against this search.
-/
@[bib "bunel2020"]
def bunel2020 : Citable := .arXiv
  { title := inlines!"Branch and Bound for Piecewise Linear Neural Network Verification"
    authors := #[inlines!"Rudy Bunel", inlines!"Jingyue Lu", inlines!"Ilker Turkaslan",
      inlines!"Philip H. S. Torr", inlines!"Pushmeet Kohli",
      inlines!"M. Pawan Kumar"]
    year := 2019
    id := "1909.06588" }

/--
The paper that made "which arithmetic is the theorem about" more than a stylistic question: verified
networks attacked through the floating-point error their verifiers ignored.
-/
@[bib "jiarinard2020"]
def jiarinard2020 : Citable := .arXiv
  { title := inlines!"Exploiting Verified Neural Networks via Floating Point Numerical Error"
    authors := #[inlines!"Kai Jia", inlines!"Martin Rinard"]
    year := 2020
    id := "2003.03021" }

/-- Neural Lyapunov control, the learn-a-certificate loop the two-stage runners follow. -/
@[bib "neurallyapunov2019"]
def neurallyapunov2019 : Citable := .inProceedings
  { title := inlines!"Neural Lyapunov Control"
    authors := #[inlines!"Ya-Chien Chang", inlines!"Nima Roohi", inlines!"Sicun Gao"]
    year := 2019
    booktitle := inlines!"Neural Information Processing Systems (NeurIPS)"
    url := some "https://arxiv.org/abs/2005.00611" }

/-! ## Proof-carrying artifacts and testing of tensor frameworks -/

/-- Proof-carrying code, the earliest statement of the idea a certificate checker implements. -/
@[bib "necula1997"]
def necula1997 : Citable := .inProceedings
  { title := inlines!"Proof-Carrying Code"
    authors := #[inlines!"George C. Necula"]
    year := 1997
    booktitle := inlines!"Principles of Programming Languages (POPL)"
    url := some "https://doi.org/10.1145/263699.263712" }

/-- TensorFuzz, coverage-guided fuzzing for neural networks. -/
@[bib "tensorfuzz2019"]
def tensorfuzz2019 : Citable := .inProceedings
  { title := inlines!"TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing"
    authors := #[inlines!"Augustus Odena", inlines!"Catherine Olsson",
      inlines!"David G. Andersen", inlines!"Ian Goodfellow"]
    year := 2019
    booktitle := inlines!"International Conference on Machine Learning (ICML)"
    url := some "https://proceedings.mlr.press/v97/odena19a.html" }

/-- NNSmith, which finds deep-learning compiler bugs by generating well-typed graphs. -/
@[bib "nnsmith2023"]
def nnsmith2023 : Citable := .arXiv
  { title :=
      inlines!"NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers"
    authors := #[inlines!"Jiawei Liu", inlines!"Jinkun Lin", inlines!"Fabian Ruffy",
      inlines!"Cheng Tan", inlines!"Jinyang Li", inlines!"Aurojit Panda",
      inlines!"Lingming Zhang"]
    year := 2023
    id := "2207.13066" }

/-! ## Language semantics and generated proof certificates -/

/-- Matching logic, the logic underlying the K framework's language definitions. -/
@[bib "matchinglogic2017"]
def matchinglogic2017 : Citable := .article
  { title := inlines!"Matching Logic"
    authors := #[inlines!"Grigore Roșu"]
    journal := inlines!"Logical Methods in Computer Science"
    year := 2017
    month := none
    volume := inlines!"13"
    number := inlines!"4"
    pages := none
    url := some "https://doi.org/10.23638/LMCS-13(4:28)2017" }

/-- The K framework, one executable semantics reused as interpreter and as verifier. -/
@[bib "kframework2010"]
def kframework2010 : Citable := .article
  { title := inlines!"An Overview of the K Semantic Framework"
    authors := #[inlines!"Grigore Roșu", inlines!"Traian Florin Șerbănuță"]
    journal := inlines!"Journal of Logic and Algebraic Programming"
    year := 2010
    month := none
    volume := inlines!"79"
    number := inlines!"6"
    pages := some (397, 434)
    url := some "https://doi.org/10.1016/j.jlap.2010.03.012" }

/-- Proof generation for a semantics-based language framework, checkable outside the tool. -/
@[bib "kproofgen2021"]
def kproofgen2021 : Citable := .inProceedings
  { title :=
      inlines!"Towards a Trustworthy Semantics-Based Language Framework via Proof Generation"
    authors := #[inlines!"Xiaohong Chen", inlines!"Zhengyao Lin",
      inlines!"Minh-Thai Trinh", inlines!"Grigore Roșu"]
    year := 2021
    booktitle := inlines!"Computer Aided Verification (CAV)"
    url := some "https://doi.org/10.1007/978-3-030-81688-9_38" }

/-! ## Typed syntax with several interpretations -/

/--
The tagless-final style: one object language, several interpreters, with the object language's
types carried by the host language. GraphSpec's primitive record is this idea with tensor shapes
as the type index, which is why one architecture term can be read as a pure function and as an
executable program.
-/
@[bib "kiselyov2012"]
def kiselyov2012 : Citable := .inProceedings
  { title := inlines!"Typed Tagless Final Interpreters"
    authors := #[inlines!"Oleg Kiselyov"]
    year := 2012
    booktitle := inlines!"Generic and Indexed Programming (Spring School)"
    url := some "https://okmij.org/ftp/tagless-final/" }

/--
Intrinsically typed term representations with de Bruijn variables. GraphSpec's DAG variables carry
their tensor shape in the type for exactly the reason this paper gives: the well-typedness
invariant lives in the syntax, so evaluation and renaming need no dependent casts through a
separate typing judgement.
-/
@[bib "benton2012"]
def benton2012 : Citable := .article
  { title := inlines!"Strongly Typed Term Representations in Coq"
    authors := #[inlines!"Nick Benton", inlines!"Chung-Kil Hur",
      inlines!"Andrew J. Kennedy", inlines!"Conor McBride"]
    journal := inlines!"Journal of Automated Reasoning"
    year := 2012
    month := none
    volume := inlines!"49"
    number := inlines!"2"
    pages := some (141, 159)
    url := none }

/-! ## Optimization and learning theory -/

/--
Stochastic approximation, the origin of the update rule every optimizer in this guide specializes.
We cite it in the training chapter because the one-sample step is not an approximation of a
full-batch step invented for speed; it is the older idea.
-/
@[bib "robbins1951"]
def robbins1951 : Citable := .article
  { title := inlines!"A Stochastic Approximation Method"
    authors := #[inlines!"Herbert Robbins", inlines!"Sutton Monro"]
    journal := inlines!"The Annals of Mathematical Statistics"
    year := 1951
    month := none
    volume := inlines!"22"
    number := inlines!"3"
    pages := some (400, 407)
    url := some "https://doi.org/10.1214/aoms/1177729586" }

/--
Adam. The training chapter states the moment recurrences and the bias correction exactly as this
paper defines them, because TorchLean's checkpoint story only makes sense once you can see that the
step counter is part of the update.
-/
@[bib "adam2015"]
def adam2015 : Citable := .inProceedings
  { title := inlines!"Adam: A Method for Stochastic Optimization"
    authors := #[inlines!"Diederik P. Kingma", inlines!"Jimmy Ba"]
    year := 2015
    booktitle := inlines!"International Conference on Learning Representations (ICLR)"
    url := some "https://arxiv.org/abs/1412.6980" }

/-- Cosine annealing, the shape of the decay half of TorchLean's `warmupCosine` schedule. -/
@[bib "sgdr2017"]
def sgdr2017 : Citable := .arXiv
  { title := inlines!"SGDR: Stochastic Gradient Descent with Warm Restarts"
    authors := #[inlines!"Ilya Loshchilov", inlines!"Frank Hutter"]
    year := 2017
    id := "1608.03983" }

/--
Learning-rate warm-up. The ramp in `warmupCosine` is not a TorchLean invention; it is the recipe
this paper made standard for large-batch training.
-/
@[bib "goyal2017"]
def goyal2017 : Citable := .arXiv
  { title := inlines!"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    authors := #[inlines!"Priya Goyal", inlines!"Piotr Dollár", inlines!"Ross Girshick",
      inlines!"Pieter Noordhuis", inlines!"Lukasz Wesolowski", inlines!"Aapo Kyrola",
      inlines!"Andrew Tulloch", inlines!"Yangqing Jia", inlines!"Kaiming He"]
    year := 2017
    id := "1706.02677" }

/--
Xavier (Glorot) uniform initialization, the default TorchLean gives an affine layer. The bound
$\sqrt{6/(n_{in}+n_{out})}$ that the training chapter checks against a printed weight matrix is
this paper's.
-/
@[bib "glorot2010"]
def glorot2010 : Citable := .inProceedings
  { title := inlines!"Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    authors := #[inlines!"Xavier Glorot", inlines!"Yoshua Bengio"]
    year := 2010
    booktitle := inlines!"Artificial Intelligence and Statistics (AISTATS)"
    url := some "https://proceedings.mlr.press/v9/glorot10a.html" }

/--
Kaiming initialization, which is what PyTorch's `Linear` uses by default. We cite it beside
`glorot2010` so the training chapter can explain why the same architecture and seed give two
different initial losses in the two frameworks.
-/
@[bib "he2015"]
def he2015 : Citable := .inProceedings
  { title := inlines!"Delving Deep into Rectifiers: Surpassing Human-Level Performance on " ++
      inlines!"ImageNet Classification"
    authors := #[inlines!"Kaiming He", inlines!"Xiangyu Zhang", inlines!"Shaoqing Ren",
      inlines!"Jian Sun"]
    year := 2015
    booktitle := inlines!"International Conference on Computer Vision (ICCV)"
    url := some "https://doi.org/10.1109/ICCV.2015.123" }

/-- Decoupled weight decay, the difference between Adam and AdamW that the optimizer proofs see. -/
@[bib "adamw2019"]
def adamw2019 : Citable := .arXiv
  { title := inlines!"Decoupled Weight Decay Regularization"
    authors := #[inlines!"Ilya Loshchilov", inlines!"Frank Hutter"]
    year := 2019
    id := "1711.05101" }

/-- The definition of differential privacy that the mechanism statements instantiate. -/
@[bib "dwork2006"]
def dwork2006 : Citable := .inProceedings
  { title := inlines!"Calibrating Noise to Sensitivity in Private Data Analysis"
    authors := #[inlines!"Cynthia Dwork", inlines!"Frank McSherry", inlines!"Kobbi Nissim",
      inlines!"Adam Smith"]
    year := 2006
    booktitle := inlines!"Theory of Cryptography (TCC)"
    url := some "https://doi.org/10.1007/11681878_14" }

/-- Uniform stability, the generalization route the stability predicates follow. -/
@[bib "bousquet2002"]
def bousquet2002 : Citable := .article
  { title := inlines!"Stability and Generalization"
    authors := #[inlines!"Olivier Bousquet", inlines!"André Elisseeff"]
    journal := inlines!"Journal of Machine Learning Research"
    year := 2002
    month := none
    volume := inlines!"2"
    number := inlines!""
    pages := some (499, 526)
    url := some "https://jmlr.org/papers/v2/bousquet02a.html" }

/-- Momentum, the second state variable that turns a gradient step into a two-line recurrence. -/
@[bib "polyak1964"]
def polyak1964 : Citable := .article
  { title := inlines!"Some Methods of Speeding Up the Convergence of Iteration Methods"
    authors := #[inlines!"Boris T. Polyak"]
    journal := inlines!"USSR Computational Mathematics and Mathematical Physics"
    year := 1964
    month := none
    volume := inlines!"4"
    number := inlines!"5"
    pages := some (1, 17)
    url := some "https://doi.org/10.1016/0041-5553(64)90137-5" }

/--
Dropout, our standard example of a layer that denotes two different functions. The inverted scaling
by `1 / (1 - p)` that TorchLean and PyTorch both use comes from this paper's section 10.
-/
@[bib "dropout2014"]
def dropout2014 : Citable := .article
  { title := inlines!"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    authors := #[inlines!"Nitish Srivastava", inlines!"Geoffrey Hinton",
      inlines!"Alex Krizhevsky", inlines!"Ilya Sutskever", inlines!"Ruslan Salakhutdinov"]
    journal := inlines!"Journal of Machine Learning Research"
    year := 2014
    month := none
    volume := inlines!"15"
    number := inlines!"56"
    pages := some (1929, 1958)
    url := some "https://jmlr.org/papers/v15/srivastava14a.html" }

/--
Batch normalization, the other layer whose behavior depends on the mode, and the reason a layer
needs somewhere to put running statistics.
-/
@[bib "batchnorm2015"]
def batchnorm2015 : Citable := .arXiv
  { title := inlines!"Batch Normalization: Accelerating Deep Network Training by Reducing " ++
      inlines!"Internal Covariate Shift"
    authors := #[inlines!"Sergey Ioffe", inlines!"Christian Szegedy"]
    year := 2015
    id := "1502.03167" }

/-! ## Universal approximation -/

/--
Cybenko's density theorem for continuous functions on a finite-dimensional unit cube and continuous
sigmoidal activations. TorchLean's hinge construction instead supplies a width bound for univariate
Lipschitz functions using ReLU. Their different hypotheses matter when comparing these results.
-/
@[bib "cybenko1989"]
def cybenko1989 : Citable := .article
  { title := inlines!"Approximation by Superpositions of a Sigmoidal Function"
    authors := #[inlines!"George Cybenko"]
    journal := inlines!"Mathematics of Control, Signals and Systems"
    year := 1989
    month := none
    volume := inlines!"2"
    number := inlines!"4"
    pages := some (303, 314)
    url := some "https://doi.org/10.1007/BF02551274" }

/--
Hornik's multilayer version, which removes the sigmoid-specific hypothesis. Cited alongside
Cybenko because the two together are what "universal approximation" normally means in the
literature, and neither of them is the finite-precision statement TorchLean needs.
-/
@[bib "hornik1991"]
def hornik1991 : Citable := .article
  { title := inlines!"Approximation Capabilities of Multilayer Feedforward Networks"
    authors := #[inlines!"Kurt Hornik"]
    journal := inlines!"Neural Networks"
    year := 1991
    month := none
    volume := inlines!"4"
    number := inlines!"2"
    pages := some (251, 257)
    url := some "https://doi.org/10.1016/0893-6080(91)90009-T" }

/-! ## Architectures used by the worked examples -/

/-- Residual learning, the skip connection the ResNet example builds. -/
@[bib "resnet2016"]
def resnet2016 : Citable := .arXiv
  { title := inlines!"Deep Residual Learning for Image Recognition"
    authors := #[inlines!"Kaiming He", inlines!"Xiangyu Zhang", inlines!"Shaoqing Ren",
      inlines!"Jian Sun"]
    year := 2016
    id := "1512.03385" }

/-- The transformer, whose attention block the causal-mask proofs are about. -/
@[bib "transformer2017"]
def transformer2017 : Citable := .arXiv
  { title := inlines!"Attention Is All You Need"
    authors := #[inlines!"Ashish Vaswani", inlines!"Noam Shazeer", inlines!"Niki Parmar",
      inlines!"Jakob Uszkoreit", inlines!"Llion Jones", inlines!"Aidan N. Gomez",
      inlines!"Łukasz Kaiser", inlines!"Illia Polosukhin"]
    year := 2017
    id := "1706.03762" }

/--
Byte-pair encoding, the merge rule that the GPT-2 vocabulary and merge files spell out. We cite it
in the data chapter because a tokenizer is a semantic dependency of a text dataset, not a parsing
detail.
-/
@[bib "sennrich2016"]
def sennrich2016 : Citable := .inProceedings
  { title := inlines!"Neural Machine Translation of Rare Words with Subword Units"
    authors := #[inlines!"Rico Sennrich", inlines!"Barry Haddow", inlines!"Alexandra Birch"]
    year := 2016
    booktitle := inlines!"Association for Computational Linguistics (ACL)"
    url := some "https://aclanthology.org/P16-1162/" }

/-- The vision transformer, the patch-token layout in the image examples. -/
@[bib "vit2021"]
def vit2021 : Citable := .arXiv
  { title :=
      inlines!"An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    authors := #[inlines!"Alexey Dosovitskiy", inlines!"Lucas Beyer",
      inlines!"Alexander Kolesnikov", inlines!"Dirk Weissenborn", inlines!"Xiaohua Zhai",
      inlines!"Thomas Unterthiner", inlines!"Mostafa Dehghani", inlines!"Matthias Minderer",
      inlines!"Georg Heigold", inlines!"Sylvain Gelly", inlines!"Jakob Uszkoreit",
      inlines!"Neil Houlsby"]
    year := 2021
    id := "2010.11929" }

/-- LoRA, the source of the low-rank adapter contract in the model chapter. -/
@[bib "lora2022"]
def lora2022 : Citable := .arXiv
  { title := inlines!"LoRA: Low-Rank Adaptation of Large Language Models"
    authors := #[inlines!"Edward J. Hu", inlines!"Yelong Shen", inlines!"Phillip Wallis",
      inlines!"Zeyuan Allen-Zhu", inlines!"Yuanzhi Li", inlines!"Shean Wang",
      inlines!"Lu Wang", inlines!"Weizhu Chen"]
    year := 2022
    id := "2106.09685" }

/--
Hopfield's original paper. Cited for the energy argument the `Spec.Hopfield` development formalizes;
the update rule there is asynchronous and the weights are symmetric with zero diagonal, which is
exactly the setting our `SymmetricW` and `DiagonalZero` hypotheses name.
-/
@[bib "hopfield1982"]
def hopfield1982 : Citable := .article
  { title := inlines!"Neural Networks and Physical Systems with Emergent Collective " ++
      inlines!"Computational Abilities"
    authors := #[inlines!"John J. Hopfield"]
    journal := inlines!"Proceedings of the National Academy of Sciences"
    year := 1982
    month := none
    volume := inlines!"79"
    number := inlines!"8"
    pages := some (2554, 2558)
    url := some "https://doi.org/10.1073/pnas.79.8.2554" }

/--
The modern continuous-state Hopfield layer. Cited where the guide says our discrete energy theorem
does not transfer: that layer stores exponentially many patterns and updates all coordinates at
once, so it needs a different energy and a different convergence argument.
-/
@[bib "modernhopfield2021"]
def modernhopfield2021 : Citable := .inProceedings
  { title := inlines!"Hopfield Networks is All You Need"
    authors := #[inlines!"Hubert Ramsauer", inlines!"Bernhard Schäfl",
      inlines!"Johannes Lehner", inlines!"Philipp Seidl", inlines!"Michael Widrich",
      inlines!"Thomas Adler", inlines!"Lukas Gruber", inlines!"Markus Holzleitner",
      inlines!"Milena Pavlović", inlines!"Geir Kjetil Sandve",
      inlines!"Victor Greiff", inlines!"David Kreil", inlines!"Michael Kopp",
      inlines!"Günter Klambauer", inlines!"Johannes Brandstetter",
      inlines!"Sepp Hochreiter"]
    year := 2021
    booktitle := inlines!"International Conference on Learning Representations (ICLR)"
    url := some "https://arxiv.org/abs/2008.02217" }

/-- Structured state spaces (S4), the recurrence `DiagonalS4Spec` is the diagonal case of. -/
@[bib "s4_2022"]
def s4_2022 : Citable := .inProceedings
  { title := inlines!"Efficiently Modeling Long Sequences with Structured State Spaces"
    authors := #[inlines!"Albert Gu", inlines!"Karan Goel", inlines!"Christopher Ré"]
    year := 2022
    booktitle := inlines!"International Conference on Learning Representations (ICLR)"
    url := some "https://arxiv.org/abs/2111.00396" }

/-- Mamba, the selective state-space model behind the causality proof. -/
@[bib "mamba2024"]
def mamba2024 : Citable := .arXiv
  { title := inlines!"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    authors := #[inlines!"Albert Gu", inlines!"Tri Dao"]
    year := 2024
    id := "2312.00752" }

/-- The Fourier neural operator, the scientific-ML example with a spectral layer. -/
@[bib "fno2021"]
def fno2021 : Citable := .arXiv
  { title :=
      inlines!"Fourier Neural Operator for Parametric Partial Differential Equations"
    authors := #[inlines!"Zongyi Li", inlines!"Nikola Kovachki",
      inlines!"Kamyar Azizzadenesheli", inlines!"Burigede Liu",
      inlines!"Kaushik Bhattacharya", inlines!"Andrew Stuart", inlines!"Anima Anandkumar"]
    year := 2021
    id := "2010.08895" }

/-- Physics-informed neural networks, the source of the residual objective. -/
@[bib "pinn2019"]
def pinn2019 : Citable := .article
  { title :=
      inlines!"Physics-informed neural networks: A deep learning framework for \
        solving forward and inverse problems involving nonlinear partial \
        differential equations"
    authors := #[inlines!"Maziar Raissi", inlines!"Paris Perdikaris",
      inlines!"George Em Karniadakis"]
    journal := inlines!"Journal of Computational Physics"
    year := 2019
    month := none
    volume := inlines!"378"
    number := inlines!""
    pages := some (686, 707)
    url := some "https://doi.org/10.1016/j.jcp.2018.10.045" }

/-! ## Generative models -/

/-- Denoising diffusion probabilistic models, the forward process the proofs formalize. -/
@[bib "ddpm2020"]
def ddpm2020 : Citable := .arXiv
  { title := inlines!"Denoising Diffusion Probabilistic Models"
    authors := #[inlines!"Jonathan Ho", inlines!"Ajay Jain", inlines!"Pieter Abbeel"]
    year := 2020
    id := "2006.11239" }

/-- Denoising diffusion implicit models, the deterministic sampler variant. -/
@[bib "ddim2021"]
def ddim2021 : Citable := .arXiv
  { title := inlines!"Denoising Diffusion Implicit Models"
    authors := #[inlines!"Jiaming Song", inlines!"Chenlin Meng", inlines!"Stefano Ermon"]
    year := 2021
    id := "2010.02502" }

/-- The variational autoencoder and its reparameterized objective. -/
@[bib "vae2014"]
def vae2014 : Citable := .arXiv
  { title := inlines!"Auto-Encoding Variational Bayes"
    authors := #[inlines!"Diederik P. Kingma", inlines!"Max Welling"]
    year := 2014
    id := "1312.6114" }

/-- Vector-quantized representation learning, the discrete-codebook example. -/
@[bib "vqvae2017"]
def vqvae2017 : Citable := .arXiv
  { title := inlines!"Neural Discrete Representation Learning"
    authors := #[inlines!"Aäron van den Oord", inlines!"Oriol Vinyals",
      inlines!"Koray Kavukcuoglu"]
    year := 2017
    id := "1711.00937" }

/-- Least-squares GAN, the adversarial objective used in the generative chapter. -/
@[bib "lsgan2017"]
def lsgan2017 : Citable := .arXiv
  { title := inlines!"Least Squares Generative Adversarial Networks"
    authors := #[inlines!"Xudong Mao", inlines!"Qing Li", inlines!"Haoran Xie",
      inlines!"Raymond Y. K. Lau", inlines!"Zhen Wang", inlines!"Stephen Paul Smolley"]
    year := 2017
    id := "1611.04076" }

/-- Masked autoencoders, one of the self-supervised objectives with a shape statement. -/
@[bib "mae2022"]
def mae2022 : Citable := .arXiv
  { title := inlines!"Masked Autoencoders Are Scalable Vision Learners"
    authors := #[inlines!"Kaiming He", inlines!"Xinlei Chen", inlines!"Saining Xie",
      inlines!"Yanghao Li", inlines!"Piotr Dollár", inlines!"Ross Girshick"]
    year := 2022
    id := "2111.06377" }

/-- I-JEPA, the joint-embedding predictive objective the latent-target contract is modelled on. -/
@[bib "ijepa2023"]
def ijepa2023 : Citable := .arXiv
  { title :=
      inlines!"Self-Supervised Learning from Images with a Joint-Embedding \
        Predictive Architecture"
    authors := #[inlines!"Mahmoud Assran", inlines!"Quentin Duval", inlines!"Ishan Misra",
      inlines!"Piotr Bojanowski", inlines!"Pascal Vincent", inlines!"Michael Rabbat",
      inlines!"Yann LeCun", inlines!"Nicolas Ballas"]
    year := 2023
    id := "2301.08243" }

/-- VICReg, whose variance and covariance terms the self-supervised proofs bound. -/
@[bib "vicreg2022"]
def vicreg2022 : Citable := .arXiv
  { title :=
      inlines!"VICReg: Variance-Invariance-Covariance Regularization for \
        Self-Supervised Learning"
    authors := #[inlines!"Adrien Bardes", inlines!"Jean Ponce", inlines!"Yann LeCun"]
    year := 2022
    id := "2105.04906" }

/-- Barlow Twins, the redundancy-reduction objective compared against VICReg. -/
@[bib "barlowtwins2021"]
def barlowtwins2021 : Citable := .arXiv
  { title := inlines!"Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
    authors := #[inlines!"Jure Zbontar", inlines!"Li Jing", inlines!"Ishan Misra",
      inlines!"Yann LeCun", inlines!"Stéphane Deny"]
    year := 2021
    id := "2103.03230" }

/-! ## Reinforcement learning -/

/-- Deep Q-networks, the value-based baseline in the reinforcement-learning chapter. -/
@[bib "dqn2015"]
def dqn2015 : Citable := .article
  { title := inlines!"Human-level control through deep reinforcement learning"
    authors := #[inlines!"Volodymyr Mnih", inlines!"Koray Kavukcuoglu",
      inlines!"David Silver", inlines!"Andrei A. Rusu", inlines!"Joel Veness",
      inlines!"Marc G. Bellemare", inlines!"Alex Graves", inlines!"Martin Riedmiller",
      inlines!"Andreas K. Fidjeland", inlines!"Georg Ostrovski", inlines!"Stig Petersen",
      inlines!"Charles Beattie", inlines!"Amir Sadik", inlines!"Ioannis Antonoglou",
      inlines!"Helen King", inlines!"Dharshan Kumaran", inlines!"Daan Wierstra",
      inlines!"Shane Legg", inlines!"Demis Hassabis"]
    journal := inlines!"Nature"
    year := 2015
    month := none
    volume := inlines!"518"
    number := inlines!"7540"
    pages := some (529, 533)
    url := some "https://doi.org/10.1038/nature14236" }

/-- Generalized advantage estimation, the advantage recursion the rollout code implements. -/
@[bib "gae2015"]
def gae2015 : Citable := .arXiv
  { title :=
      inlines!"High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    authors := #[inlines!"John Schulman", inlines!"Philipp Moritz", inlines!"Sergey Levine",
      inlines!"Michael I. Jordan", inlines!"Pieter Abbeel"]
    year := 2015
    id := "1506.02438" }

/-- Proximal policy optimization, the clipped surrogate objective. -/
@[bib "ppo2017"]
def ppo2017 : Citable := .arXiv
  { title := inlines!"Proximal Policy Optimization Algorithms"
    authors := #[inlines!"John Schulman", inlines!"Filip Wolski",
      inlines!"Prafulla Dhariwal", inlines!"Alec Radford", inlines!"Oleg Klimov"]
    year := 2017
    id := "1707.06347" }

end TorchLeanBlueprint.Bib
