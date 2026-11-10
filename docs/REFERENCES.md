# References

We wrote TorchLean's specifications by hand. Where one follows a paper, a book, a standard or a
documentation page, its docstring names the source down to the equation, section or clause.
This file lists those sources, each with the files whose specifications follow it.

## Standards

- IEEE Std 754-2019, "IEEE Standard for Floating-Point Arithmetic",
  doi:10.1109/IEEESTD.2019.8766229. The specs cite clauses 3.4, 4.3.1, 5.12 and 9.6 and name the
  operations they follow.
  Specs: `TorchLean/NN/RL/Numerics/Intervals.lean`, `TorchLean/NN/RL/Numerics/Returns.lean`,
  `TorchLean/NN/RL/Numerics/Types.lean`, `TorchLean/NN/RL/PolicyGradient.lean`,
  `TorchLean/NN/RL/ValueLearning.lean`, `TorchLean/Tensor/ConfiguredBinary.lean`,
  `TorchLean/Tensor/Context.lean`, `TorchLean/Tensor/Elementwise.lean`,
  `TorchLean/Tensor/FromFloat.lean`, `TorchLean/Tensor/Layers/Activation.lean`,
  `TorchLean/Tensor/Layers/Loss.lean`, `TorchLean/Tensor/Layers/Normalization.lean`,
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`, `TorchLean/Tensor/Layers/Softmax.lean`,
  `TorchLean/Tensor/Numeric/Quotient.lean`, `TorchLean/Tensor/Reduce.lean`,
  `TorchLean/Tensor/Scatter.lean`, `TorchLean/Tensor/Window.lean`,
  `TorchLean/Verify/CROWN/BoundOps.lean`, `TorchLean/Verify/CROWN/BoundOps/Lawful.lean`,
  `TorchLean/Verify/CROWN/FlatBox/Transfer.lean`, `TorchLean/Verify/Certificate/CROWNNode.lean`,
  `TorchLean/Verify/Certificate/IBPNode.lean`, `TorchLean/Verify/DecimalRounding.lean`,
  `TorchLean/Verify/Floats/FP32.lean`, `TorchLean/Verify/Floats/IEEE754.lean`,
  `TorchLean/Verify/Floats/IEEE754/Expressions.lean` and `TorchLean/Verify/LibTorch/Contract.lean`.
- WG14 N1256, ISO/IEC 9899:TC3 Committee Draft (2007), the C99 text: clause 7.12 and Annexes F and
  G.
  Specs: `TorchLean/Tensor/Complex.lean`, `TorchLean/Tensor/ConfiguredBinary.lean`,
  `TorchLean/Tensor/ConfiguredBinary/Elementary.lean`,
  `TorchLean/Tensor/ConfiguredBinary/InverseTrigonometric.lean`, `TorchLean/Tensor/Context.lean`,
  `TorchLean/Tensor/Context/Real.lean`, `TorchLean/Tensor/FromFloat.lean`,
  `TorchLean/Tensor/NF.lean`, `TorchLean/Tensor/Numeric/InverseTrig.lean` and
  `TorchLean/Tensor/Numeric/InverseTrig/Real.lean`.
- WG14 N3096, the C23 working draft (2023): 7.12.11.1, 7.24.1.5 and F.3.
  Specs: `TorchLean/Tensor/Context.lean`, `TorchLean/Tensor/Numeric/Quotient.lean` and
  `TorchLean/Verify/DecimalRounding.lean`.

## Papers

- Ahrens, Demmel and Nguyen (2020), "Algorithms for Efficient Reproducible Floating Point
  Summation", ACM TOMS 46(3), doi:10.1145/3389360.
  Specs: `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- Alshiekh, Bloem, Ehlers, Könighofer, Niekum and Topcu (2018), "Safe Reinforcement Learning via
  Shielding", AAAI 2018, 32(1), arXiv:1708.08611.
  Specs: `TorchLean/NN/RL/Environment/Safety.lean`.
- Edward Anderson (2017), "Algorithm 978: Safe Scaling in the Level 1 BLAS", ACM Transactions on
  Mathematical Software 44(1), 1-28, doi:10.1145/3061665.
  Specs: `TorchLean/Tensor/LinearAlgebra/Algorithms.lean`.
- Martin Arjovsky, Soumith Chintala and Léon Bottou (2017), "Wasserstein Generative Adversarial
  Networks", ICML 2017, PMLR 70, pp. 214-223, arXiv:1701.07875.
  Specs: `TorchLean/NN/Models/GAN/Loss.lean`.
- Auer, Cesa-Bianchi and Fischer (2002), "Finite-time Analysis of the Multiarmed Bandit Problem",
  Machine Learning 47(2-3):235-256, doi:10.1023/A:1013689704352.
  Specs: `TorchLean/NN/RL/Bandits.lean`.
- Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey E. Hinton (2016), "Layer Normalization",
  arXiv:1607.06450.
  Specs: `TorchLean/NN/Models/Transformer.lean`, `TorchLean/Tensor/Layers/Normalization.lean` and
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`.
- Barron, Gnanapragasam and Tschannen (2026), "A Fix for Softmax and Cross-Entropy",
  arXiv:2610.31642.
  Specs: `TorchLean/Tensor/Layers/Activation.lean`, `TorchLean/Tensor/Layers/Loss.lean` and
  `TorchLean/Tensor/Layers/Loss/Axis.lean`.
- Baum and Petrie (1966), "Statistical Inference for Probabilistic Functions of Finite State Markov
  Chains", Annals of Mathematical Statistics 37(6), 1554-1563, doi:10.1214/aoms/1177699147.
  Specs: `TorchLean/NN/Models/HMM.lean`.
- Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul and Jeffrey Mark Siskind
  (2018), "Automatic Differentiation in Machine Learning: a Survey", JMLR 18(153):1-43,
  arXiv:1502.05767.
  Specs: `TorchLean/NN/AdjointOp.lean` and `TorchLean/NN/VJPOp.lean`.
- Bengio, Ducharme, Vincent and Jauvin (2003), "A Neural Probabilistic Language Model", JMLR
  3:1137-1155.
  Specs: `TorchLean/Tensor/Layers/Embedding.lean`.
- Bezanson, Edelman, Karpinski and Shah (2014), "Julia: A Fresh Approach to Numerical Computing",
  arXiv:1411.1607.
  Specs: `TorchLean/Tensor/Context.lean`.
- Blanchard, Higham and Higham (2019), "Accurate Computation of the Log-Sum-Exp and Softmax
  Functions", arXiv:1909.03469.
  Specs: `TorchLean/Tensor/Layers/Activation.lean` and `TorchLean/Tensor/Layers/Softmax.lean`.
- Boldo and Melquiond (2011), "Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq", ARITH 2011, pp. 243-252, doi:10.1109/ARITH.2011.40.
  Specs: `TorchLean/Tensor/NF.lean`, `TorchLean/Verify/Floats/FP32.lean`,
  `TorchLean/Verify/Floats/IEEE754/Expressions.lean` and
  `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- G. E. P. Box and Mervin E. Muller (1958), "A Note on the Generation of Random Normal Deviates",
  The Annals of Mathematical Statistics 29(2), 610-611, doi:10.1214/aoms/1177706645.
  Specs: `TorchLean/Tensor/Random.lean`.
- Breiman, Friedman, Olshen and Stone (1984), "Classification and Regression Trees", Wadsworth,
  Belmont, CA.
  Specs: `TorchLean/NN/Models/GradientBoostedTrees.lean`.
- Breiman (2001), "Random Forests", Machine Learning 45(1), 5-32, doi:10.1023/A:1010933404324.
  Specs: `TorchLean/NN/Models/RandomForest.lean`.
- Brockman et al. (2016), "OpenAI Gym", arXiv:1606.01540.
  Specs: `TorchLean/NN/RL/Boundary.lean`, `TorchLean/NN/RL/Gymnasium/Client.lean` and
  `TorchLean/NN/RL/Gymnasium/Session.lean`.
- Chen and Guestrin (2016), "XGBoost: A Scalable Tree Boosting System", Proc. 22nd ACM SIGKDD (KDD
  2016), 785-794, doi:10.1145/2939672.2939785, arXiv:1603.02754.
  Specs: `TorchLean/NN/Models/GradientBoostedTrees.lean`.
- Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
  Schwenk and Yoshua Bengio (2014), "Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation", EMNLP 2014, pages 1724-1734, doi:10.3115/v1/D14-1179.
  Specs: `TorchLean/NN/Models/Recurrent.lean`, `TorchLean/NN/Models/Seq2Seq.lean` and
  `TorchLean/Tensor/Layers/GRU.lean`.
- Christodoulou (2019), "Soft Actor-Critic for Discrete Action Settings", arXiv:1910.07207.
  Specs: `TorchLean/NN/RL/PolicyGradient.lean`.
- Chung, Gulcehre, Cho and Bengio (2014), "Empirical Evaluation of Gated Recurrent Neural Networks
  on Sequence Modeling", NIPS 2014 Deep Learning and Representation Learning Workshop,
  arXiv:1412.3555.
  Specs: `TorchLean/Tensor/Layers/GRU.lean`.
- Matteo Cipollina, Michail Karatarakis and Freek Wiedijk (2025), "Formalized Hopfield Networks and
  Boltzmann Machines", LPAR-26, EPiC Series in Computing, arXiv:2512.07766.
  Specs: `TorchLean/NN/Dynamics.lean` and `TorchLean/NN/Models/Hopfield.lean`.
- Clevert, Unterthiner and Hochreiter (2016), "Fast and Accurate Deep Network Learning by
  Exponential Linear Units (ELUs)", ICLR 2016, arXiv:1511.07289.
  Specs: `TorchLean/Tensor/Layers/Activation.lean`.
- Cortes and Vapnik (1995), "Support-Vector Networks", Machine Learning 20(3), 273-297,
  doi:10.1007/BF00994018.
  Specs: `TorchLean/NN/Models/SVM.lean`.
- Cover and Hart (1967), "Nearest Neighbor Pattern Classification", IEEE Transactions on Information
  Theory 13(1), 21-27, doi:10.1109/TIT.1967.1053964.
  Specs: `TorchLean/NN/Models/KNN.lean`.
- Dao, Fu, Ermon, Rudra and Ré (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness", arXiv:2205.14135.
  Specs: `TorchLean/Tensor/Layers/Attention.lean`.
- Dao (2023), "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
  arXiv:2307.08691.
  Specs: `TorchLean/Tensor/Layers/Attention.lean` and `TorchLean/Tensor/Layers/Softmax.lean`.
- Tri Dao and Albert Gu (2024), "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality" (Mamba-2), ICML 2024, arXiv:2405.21060.
  Specs: `TorchLean/NN/Models/Mamba/Block.lean` and `TorchLean/Tensor/Layers/StateSpace.lean`.
- Demmel and Nguyen (2013), "Fast Reproducible Floating-Point Summation", ARITH 2013,
  doi:10.1109/ARITH.2013.9.
  Specs: `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- Dempster, Laird and Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm",
  Journal of the Royal Statistical Society, Series B 39(1), doi:10.1111/j.2517-6161.1977.tb01600.x.
  Specs: `TorchLean/NN/Models/GMM.lean`.
- Alexey Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale", ICLR 2021, arXiv:2010.11929.
  Specs: `TorchLean/NN/Models/ViT.lean`.
- Duchi, Hazan and Singer (2011), "Adaptive Subgradient Methods for Online Learning and Stochastic
  Optimization", JMLR 12(61):2121-2159, https://jmlr.org/papers/v12/duchi11a.html.
  Specs: `TorchLean/NN/Optimizer/Update.lean`.
- Richard Durstenfeld (1964), "Algorithm 235: Random permutation", Communications of the ACM 7(7),
  420, doi:10.1145/364520.364540.
  Specs: `TorchLean/Tensor/Random/Permutation.lean`.
- Elfwing, Uchibe and Doya (2017), "Sigmoid-Weighted Linear Units for Neural Network Function
  Approximation in Reinforcement Learning", arXiv:1702.03118.
  Specs: `TorchLean/Tensor/Layers/Activation.lean`.
- Elliott (2018), "The Simple Essence of Automatic Differentiation", arXiv:1804.00746.
  Specs: `TorchLean/Tensor/Context.lean`.
- Jeffrey L. Elman (1990), "Finding Structure in Time", Cognitive Science 14(2):179-211,
  doi:10.1207/s15516709cog1402_1.
  Specs: `TorchLean/NN/Models/Recurrent/Elman.lean` and `TorchLean/Tensor/Layers/RNN.lean`.
- Agner Fog (2018), "NaN payload propagation - unresolved issues", background document of the IEEE
  754-2019 revision,
  https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/nan-propagation.pdf.
  Specs: `TorchLean/Tensor/Numeric/Widening.lean`.
- Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine", Annals of
  Statistics 29(5), 1189-1232, doi:10.1214/aos/1013203451.
  Specs: `TorchLean/NN/Models/GradientBoostedTrees.lean`.
- Fujimoto, van Hoof and Meger (2018), "Addressing Function Approximation Error in Actor-Critic
  Methods", ICML 2018, arXiv:1802.09477.
  Specs: `TorchLean/NN/RL/ValueLearning.lean`.
- Gers, Schmidhuber and Cummins (2000), "Learning to Forget: Continual Prediction with LSTM", Neural
  Computation 12(10):2451-2471, doi:10.1162/089976600300015015.
  Specs: `TorchLean/Tensor/Layers/LSTM.lean`.
- Goldberg (1991), "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM
  Computing Surveys 23(1), 5-48, doi:10.1145/103162.103163.
  Specs: `TorchLean/NN/RL/Numerics/Types.lean`, `TorchLean/Verify/Floats/IEEE754/Expressions.lean`
  and `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- Ian J. Goodfellow et al. (2014), "Generative Adversarial Nets", NIPS 2014, arXiv:1406.2661.
  Specs: `TorchLean/NN/Models/GAN.lean` and `TorchLean/NN/Models/GAN/Loss.lean`.
- Gowal, Dvijotham, Stanforth, Bunel, Qin, Uesato, Arandjelović, Mann and Kohli (2018), "On the
  Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models",
  arXiv:1810.12715.
  Specs: `TorchLean/Verify/CROWN/Activation.lean`, `TorchLean/Verify/CROWN/MLP.lean`,
  `TorchLean/Verify/CROWN/Op/Cert.lean`, `TorchLean/Verify/CROWN/Op/Exec.lean` and
  `TorchLean/Verify/Certificate/IBP.lean`.
- Albert Gu, Karan Goel and Christopher Ré (2022), "Efficiently Modeling Long Sequences with
  Structured State Spaces", ICLR 2022, arXiv:2111.00396.
  Specs: `TorchLean/NN/Models/S4.lean`, `TorchLean/NN/StateSpace.lean` and
  `TorchLean/Tensor/Layers/StateSpace.lean`.
- Albert Gu and Tri Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  COLM 2024, arXiv:2312.00752.
  Specs: `TorchLean/NN/Models/Mamba.lean`, `TorchLean/NN/Models/Mamba/Block.lean`,
  `TorchLean/NN/StateSpace.lean` and `TorchLean/Tensor/Layers/StateSpace.lean`.
- Haarnoja, Zhou, Abbeel and Levine (2018), "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor", ICML 2018, arXiv:1801.01290.
  Specs: `TorchLean/NN/RL/ValueLearning.lean`.
- Haarnoja, Zhou, Hartikainen et al. (2018), "Soft Actor-Critic Algorithms and Applications",
  arXiv:1812.05905.
  Specs: `TorchLean/NN/RL/PolicyGradient.lean` and `TorchLean/NN/RL/ValueLearning.lean`.
- van Hasselt (2010), "Double Q-learning", NIPS 23, pp. 2613-2621.
  Specs: `TorchLean/NN/RL/Tabular.lean`.
- van Hasselt, Guez and Silver (2016), "Deep Reinforcement Learning with Double Q-learning", AAAI
  2016, arXiv:1509.06461.
  Specs: `TorchLean/NN/RL/DQN.lean` and `TorchLean/NN/RL/ValueLearning.lean`.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun (2016), "Deep Residual Learning for Image
  Recognition", CVPR 2016, pp. 770-778, doi:10.1109/CVPR.2016.90, arXiv:1512.03385.
  Specs: `TorchLean/NN/Models/ResNet.lean` and `TorchLean/NN/Models/Transformer.lean`.
- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár and Ross Girshick (2022), "Masked
  Autoencoders Are Scalable Vision Learners", CVPR 2022, pp. 15979-15988,
  doi:10.1109/CVPR52688.2022.01553, arXiv:2111.06377.
  Specs: `TorchLean/NN/Models/MaskedPatchReconstructor.lean`.
- Dan Hendrycks and Kevin Gimpel (2016), "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415.
  Specs: `TorchLean/NN/Models/Transformer/FeedForward.lean`,
  `TorchLean/Tensor/Layers/Activation.lean`, `TorchLean/Tensor/Op/Pointwise.lean`,
  `TorchLean/Tensor/Op/Pointwise/Real.lean` and `TorchLean/Tensor/Op/Table.lean`.
- Higham (1993), "The Accuracy of Floating Point Summation", SIAM J. Sci. Comput. 14(4), 783-799,
  doi:10.1137/0914050.
  Specs: `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- Hinton, with Srivastava and Swersky, "Neural Networks for Machine Learning", lecture 6e slides,
  https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
  Specs: `TorchLean/NN/Optimizer/Config.lean` and `TorchLean/NN/Optimizer/Update.lean`.
- Jonathan Ho, Ajay Jain and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239.
  Specs: `TorchLean/NN/Models/Diffusion.lean`, `TorchLean/NN/Models/Diffusion/ForwardProcess.lean`,
  `TorchLean/NN/Models/Diffusion/ForwardProcess/Gaussian.lean`,
  `TorchLean/NN/Models/Diffusion/ImageDDIM/Training.lean`,
  `TorchLean/NN/Models/Diffusion/Loss.lean`, `TorchLean/NN/Models/Diffusion/NoisePredictor.lean`,
  `TorchLean/NN/Models/Diffusion/ReverseDDPM.lean`, `TorchLean/NN/Models/Diffusion/Samplers.lean`
  and `TorchLean/NN/Models/Diffusion/Schedule.lean`.
- Hochreiter and Schmidhuber (1997), "Long Short-Term Memory", Neural Computation 9(8):1735-1780,
  doi:10.1162/neco.1997.9.8.1735.
  Specs: `TorchLean/Tensor/Layers/LSTM.lean`.
- Hoerl and Kennard (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
  Technometrics 12(1), 55-67, doi:10.1080/00401706.1970.10488634.
  Specs: `TorchLean/NN/Models/LinearRegression.lean`.
- Hopfield (1982), "Neural networks and physical systems with emergent collective computational
  abilities", Proceedings of the National Academy of Sciences 79(8), 2554-2558.
  Specs: `TorchLean/NN/Models/Hopfield.lean` and `TorchLean/NN/Models/Hopfield/Energy.lean`.
- Hopfield (1984), "Neurons with graded response have collective computational properties like those
  of two-state neurons", Proceedings of the National Academy of Sciences 81(10), 3088-3092.
  Specs: `TorchLean/NN/Models/Hopfield.lean`.
- Hotelling (1933), "Analysis of a complex of statistical variables into principal components",
  Journal of Educational Psychology 24(6), 417-441, doi:10.1037/h0071325.
  Specs: `TorchLean/NN/Models/PCA.lean`.
- Huber (1964), "Robust Estimation of a Location Parameter", Ann. Math. Statist. 35(1):73-101,
  doi:10.1214/aoms/1177703732.
  Specs: `TorchLean/Tensor/Layers/Loss.lean`.
- Ioffe and Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift", arXiv:1502.03167.
  Specs: `TorchLean/Tensor/Layers/Normalization.lean` and
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`.
- Jiang et al. (2024), "Mixtral of Experts", arXiv:2401.04088.
  Specs: `TorchLean/Tensor/Layers/MoE.lean`.
- Jordan, Jin, Boza, You, Cesista, Newhouse and Bernstein (2024), "Muon: An optimizer for hidden
  layers in neural networks", blog post, https://kellerjordan.github.io/posts/muon/.
  Specs: `TorchLean/NN/Optimizer/Muon.lean`, `TorchLean/NN/Optimizer/Muon/NewtonSchulz.lean` and
  `TorchLean/NN/Optimizer/Update.lean`.
- W. Kahan (1987), "Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign
  Bit", in A. Iserles and M. J. D. Powell (eds.), The State of the Art in Numerical Analysis, Inst.
  Math. Appl. Conf. Ser., New Ser. 9, Clarendon Press, Oxford, 165-211.
  Specs: `TorchLean/Tensor/Complex.lean`.
- Ke, Meng, Finley, Wang, Chen, Ma, Ye and Liu (2017), "LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree", Advances in Neural Information Processing Systems 30.
  Specs: `TorchLean/NN/Models/GradientBoostedTrees.lean`.
- Diederik P. Kingma and Max Welling (2014), "Auto-Encoding Variational Bayes", ICLR 2014,
  arXiv:1312.6114.
  Specs: `TorchLean/NN/Latent.lean`, `TorchLean/NN/Models/VAE.lean` and
  `TorchLean/NN/Models/VAE/Elbo.lean`.
- Kingma and Ba (2015), "Adam: A Method for Stochastic Optimization", ICLR 2015, arXiv:1412.6980.
  Specs: `TorchLean/NN/Optimizer/Config.lean` and `TorchLean/NN/Optimizer/Update.lean`.
- Kipf and Welling (2017), "Semi-Supervised Classification with Graph Convolutional Networks", ICLR
  2017, arXiv:1609.02907.
  Specs: `TorchLean/NN/Models/GCN.lean` and `TorchLean/Tensor/Layers/GCN.lean`.
- Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew
  Stuart and Anima Anandkumar (2021), "Fourier Neural Operator for Parametric Partial Differential
  Equations", ICLR 2021, arXiv:2010.08895.
  Specs: `TorchLean/NN/Models/FNO.lean` and `TorchLean/NN/Models/FNO/Block.lean`.
- Lillicrap et al. (2015), "Continuous Control with Deep Reinforcement Learning", ICLR 2016,
  arXiv:1509.02971.
  Specs: `TorchLean/NN/RL/DQN.lean` and `TorchLean/NN/RL/ValueLearning.lean`.
- Lin (1992), "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and
  Teaching", Machine Learning 8(3-4):293-321, doi:10.1007/BF00992699.
  Specs: `TorchLean/NN/RL/Replay.lean`.
- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025, arXiv:2404.19756.
  Specs: `TorchLean/NN/Models/KAN.lean`.
- Loshchilov and Hutter (2019), "Decoupled Weight Decay Regularization", ICLR 2019,
  arXiv:1711.05101.
  Specs: `TorchLean/NN/Optimizer/Config.lean`, `TorchLean/NN/Optimizer/GD/Convergence.lean` and
  `TorchLean/NN/Optimizer/Update.lean`.
- Xudong Mao et al. (2017), "Least Squares Generative Adversarial Networks", ICCV 2017,
  arXiv:1611.04076.
  Specs: `TorchLean/NN/Models/GAN.lean` and `TorchLean/NN/Models/GAN/Loss.lean`.
- McCormick (1976), "Computability of global solutions to factorable nonconvex programs: Part I.
  Convex underestimating problems", Mathematical Programming 10(1), 147-175, doi:10.1007/BF01580665.
  Specs: `TorchLean/Verify/CROWN/Op/ExecCROWN.lean`.
- Mikolov, Chen, Corrado and Dean (2013), "Efficient Estimation of Word Representations in Vector
  Space", arXiv:1301.3781.
  Specs: `TorchLean/Tensor/Layers/Embedding.lean`.
- Mnih et al. (2015), "Human-level control through deep reinforcement learning", Nature
  518(7540):529-533, doi:10.1038/nature14236.
  Specs: `TorchLean/NN/RL/DQN.lean`, `TorchLean/NN/RL/DQN/Autograd.lean`,
  `TorchLean/NN/RL/DQN/SoftUpdate.lean`, `TorchLean/NN/RL/Replay.lean` and
  `TorchLean/NN/RL/ValueLearning.lean`.
- Mnih et al. (2016), "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016,
  arXiv:1602.01783.
  Specs: `TorchLean/NN/RL/PolicyGradient.lean`.
- Nair and Hinton (2010), "Rectified Linear Units Improve Restricted Boltzmann Machines", Proc. 27th
  ICML.
  Specs: `TorchLean/Tensor/Layers/Activation.lean`.
- Aaron van den Oord, Oriol Vinyals and Koray Kavukcuoglu (2017), "Neural Discrete Representation
  Learning", NIPS 2017, arXiv:1711.00937.
  Specs: `TorchLean/NN/Latent.lean`, `TorchLean/NN/Models/VQVAE.lean`,
  `TorchLean/NN/Models/VQVAE/Autograd.lean` and `TorchLean/NN/Models/VQVAE/Loss.lean`.
- Pearson (1901), "On Lines and Planes of Closest Fit to Systems of Points in Space", The London,
  Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2(11), 559-572,
  doi:10.1080/14786440109462720.
  Specs: `TorchLean/NN/Models/PCA.lean`.
- Polyak and Juditsky (1992), "Acceleration of Stochastic Approximation by Averaging", SIAM J.
  Control Optim. 30(4):838-855, doi:10.1137/0330046.
  Specs: `TorchLean/NN/RL/DQN.lean` and `TorchLean/NN/RL/DQN/SoftUpdate.lean`.
- Ofir Press and Lior Wolf (2017), "Using the Output Embedding to Improve Language Models", EACL
  2017, arXiv:1608.05859.
  Specs: `TorchLean/NN/Models/CausalTransformer/LanguageModel.lean`.
- Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech
  Recognition", Proceedings of the IEEE 77(2), 257-286, doi:10.1109/5.18626.
  Specs: `TorchLean/NN/Models/HMM.lean`.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever (2019),
  "Language Models are Unsupervised Multitask Learners", OpenAI report.
  Specs: `TorchLean/NN/Models/CausalTransformer.lean`.
- Ramachandran, Zoph and Le (2017), "Searching for Activation Functions", arXiv:1710.05941.
  Specs: `TorchLean/Tensor/Layers/Activation.lean`.
- Ramsauer et al. (2020), "Hopfield Networks is All You Need", arXiv:2008.02217.
  Specs: `TorchLean/NN/Models/Hopfield.lean`.
- Danilo Jimenez Rezende, Shakir Mohamed and Daan Wierstra (2014), "Stochastic Backpropagation and
  Approximate Inference in Deep Generative Models", ICML 2014, arXiv:1401.4082.
  Specs: `TorchLean/NN/Latent.lean`, `TorchLean/NN/Models/VAE.lean` and
  `TorchLean/NN/Models/VAE/Elbo.lean`.
- O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015, LNCS, pp. 234-241, doi:10.1007/978-3-319-24574-4_28, arXiv:1505.04597.
  Specs: `TorchLean/NN/Models/UNet.lean`.
- Rummery and Niranjan (1994), "On-line Q-learning using connectionist systems", Cambridge
  University Engineering Department, CUED/F-INFENG/TR 166.
  Specs: `TorchLean/NN/RL/Tabular.lean`.
- Salimans and Kingma (2016), "Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks", arXiv:1602.07868.
  Specs: `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`.
- Schaul, Quan, Antonoglou and Silver (2015), "Prioritized Experience Replay", ICLR 2016,
  arXiv:1511.05952.
  Specs: `TorchLean/NN/RL/Replay.lean`.
- Schulman et al. (2015), "High-Dimensional Continuous Control Using Generalized Advantage
  Estimation", ICLR 2016, arXiv:1506.02438.
  Specs: `TorchLean/NN/RL/Numerics/Advantage.lean`, `TorchLean/NN/RL/Numerics/Intervals.lean`,
  `TorchLean/NN/RL/PPO/Collect.lean`, `TorchLean/NN/RL/PPO/Rollout.lean`,
  `TorchLean/NN/RL/PolicyGradient.lean` and `TorchLean/NN/RL/PolicyGradient/Autograd.lean`.
- Schulman et al. (2015), "Trust Region Policy Optimization", ICML 2015, arXiv:1502.05477.
  Specs: `TorchLean/NN/RL/PolicyGradient.lean`.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov (2017), "Proximal
  Policy Optimization Algorithms", arXiv:1707.06347.
  Specs: `TorchLean/NN/Models/PPO.lean`, `TorchLean/NN/RL/Boundary.lean`,
  `TorchLean/NN/RL/Numerics/Intervals.lean`, `TorchLean/NN/RL/Numerics/PPO.lean`,
  `TorchLean/NN/RL/PPO/ActorCritic.lean`, `TorchLean/NN/RL/PPO/Collect.lean`,
  `TorchLean/NN/RL/PPO/Rollout.lean`, `TorchLean/NN/RL/PPO/Training.lean`,
  `TorchLean/NN/RL/PolicyGradient.lean` and `TorchLean/NN/RL/PolicyGradient/Autograd.lean`.
- Shah, Bikshandi, Zhang, Thakkar, Ramani and Dao (2024), "FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-precision", arXiv:2407.08608.
  Specs: `TorchLean/Tensor/Layers/Attention.lean`.
- Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton and Dean (2017), "Outrageously Large Neural
  Networks: The Sparsely-Gated Mixture-of-Experts Layer", arXiv:1701.06538.
  Specs: `TorchLean/Tensor/Layers/MoE.lean`.
- Shi, Zhang, Chang, Huang and Hsieh (2020), "Robustness Verification for Transformers", ICLR 2020,
  arXiv:2002.06622.
  Specs: `TorchLean/Verify/CROWN/Op/BackwardProduct.lean` and
  `TorchLean/Verify/CROWN/Op/ExecCROWN.lean`.
- Singh, Gehr, Püschel and Vechev (2019), "An Abstract Domain for Certifying Neural Networks", Proc.
  ACM Program. Lang. 3 (POPL), pp. 1-30, doi:10.1145/3290354.
  Specs: `TorchLean/Verify/CROWN/Activation.lean`.
- Jiaming Song, Chenlin Meng and Stefano Ermon (2021), "Denoising Diffusion Implicit Models", ICLR
  2021, arXiv:2010.02502.
  Specs: `TorchLean/NN/Models/Diffusion/ImageDDIM.lean`.
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon and Ben Poole
  (2021), "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021,
  arXiv:2011.13456.
  Specs: `TorchLean/NN/Models/Diffusion.lean`, `TorchLean/NN/Models/Diffusion/PFODE.lean` and
  `TorchLean/NN/Models/Diffusion/Samplers.lean`.
- Spitters and van der Weegen (2011), "Type classes for mathematics in type theory", Mathematical
  Structures in Computer Science 21(4), 795-825, doi:10.1017/S0960129511000119.
  Specs: `TorchLean/Tensor/Context.lean`.
- Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014), "Dropout: A Simple Way to
  Prevent Neural Networks from Overfitting", JMLR 15(56):1929-1958.
  Specs: `TorchLean/Tensor/Layers/Dropout.lean`.
- Guy L. Steele Jr., Doug Lea and Christine H. Flood (2014), "Fast splittable pseudorandom number
  generators", Proceedings of the 2014 ACM International Conference on Object Oriented Programming
  Systems Languages & Applications (OOPSLA), 453-472, doi:10.1145/2660193.2660195.
  Specs: `TorchLean/Tensor/Random.lean`.
- Su, Lu, Pan, Murtadha, Wen and Liu (2021), "RoFormer: Enhanced Transformer with Rotary Position
  Embedding", arXiv:2104.09864.
  Specs: `TorchLean/Tensor/Layers/PositionalEncoding.lean`.
- Sutskever, Martens, Dahl and Hinton (2013), "On the importance of initialization and momentum in
  deep learning", ICML 2013, PMLR 28(3):1139-1147.
  Specs: `TorchLean/NN/Optimizer/Update.lean`.
- Ilya Sutskever, Oriol Vinyals and Quoc V. Le (2014), "Sequence to Sequence Learning with Neural
  Networks", NIPS 2014, arXiv:1409.3215.
  Specs: `TorchLean/NN/Models/Recurrent/Stack.lean` and `TorchLean/NN/Models/Seq2Seq.lean`.
- Sutton (1988), "Learning to Predict by the Methods of Temporal Differences", Machine Learning
  3(1):9-44, doi:10.1023/A:1022633531479.
  Specs: `TorchLean/NN/RL/Tabular.lean`.
- Sutton (1996), "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse
  Coding", NIPS 8, pp. 1038-1044, MIT Press.
  Specs: `TorchLean/NN/RL/Tabular.lean`.
- Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso", Journal of the Royal
  Statistical Society, Series B 58(1), 267-288, doi:10.1111/j.2517-6161.1996.tb02080.x.
  Specs: `TorchLean/NN/Models/LinearRegression.lean`.
- Towers, Kwiatkowski, Terry et al. (2024), "Gymnasium: A Standard Interface for Reinforcement
  Learning Environments", arXiv:2407.17032.
  Specs: `TorchLean/NN/RL/Boundary.lean`.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz
  Kaiser and Illia Polosukhin (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762.
  Specs: `TorchLean/NN/Models/Transformer.lean`,
  `TorchLean/NN/Models/Transformer/DecoderBlock.lean`,
  `TorchLean/NN/Models/Transformer/EncoderBlock.lean`,
  `TorchLean/NN/Models/Transformer/FeedForward.lean`,
  `TorchLean/NN/Models/Transformer/PostNorm.lean`,
  `TorchLean/NN/Models/Transformer/ResidualAttention.lean`, `TorchLean/Tensor/Layers/Attention.lean`
  and `TorchLean/Tensor/Layers/PositionalEncoding.lean`.
- Wang, Zhang, Xu, Lin, Jana, Hsieh and Kolter (2021), "Beta-CROWN: Efficient Bound Propagation with
  Per-neuron Split Constraints for Complete and Incomplete Neural Network Robustness Verification",
  NeurIPS 2021, arXiv:2103.06624.
  Specs: `TorchLean/Verify/Certificate/AbCROWNLeaf.lean`.
- Watkins and Dayan (1992), "Q-learning", Machine Learning 8(3-4):279-292, doi:10.1007/BF00992698.
  Specs: `TorchLean/NN/RL/Tabular.lean`.
- Williams (1992), "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning", Machine Learning 8(3-4):229-256, doi:10.1023/A:1022672621406.
  Specs: `TorchLean/NN/RL/Bandits.lean`, `TorchLean/NN/RL/PolicyGradient.lean` and
  `TorchLean/NN/RL/PolicyGradient/Autograd.lean`.
- Wu and He (2018), "Group Normalization", arXiv:1803.08494.
  Specs: `TorchLean/Tensor/Layers/Normalization.lean`,
  `TorchLean/Tensor/Layers/Normalization/GroupNorm.lean` and
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`.
- Xu, Shi, Zhang, Wang, Chang, Huang, Kailkhura, Lin and Hsieh (2020), "Automatic Perturbation
  Analysis for Scalable Certified Robustness and Beyond", NeurIPS 2020, arXiv:2002.12920.
  Specs: `TorchLean/Verify/CROWN/FlatBox/Affine.lean`,
  `TorchLean/Verify/CROWN/FlatBox/Backward.lean`, `TorchLean/Verify/CROWN/Graph.lean`,
  `TorchLean/Verify/CROWN/MLP.lean`, `TorchLean/Verify/CROWN/Op/Backward.lean`,
  `TorchLean/Verify/CROWN/Op/ExecCROWN.lean` and `TorchLean/Verify/Certificate/CROWNQuery.lean`.
- Xu, Zhang, Wang, Wang, Jana, Lin and Hsieh (2021), "Fast and Complete: Enabling Complete Neural
  Network Verification with Rapid and Massively Parallel Incomplete Verifiers", ICLR 2021,
  arXiv:2011.13824.
  Specs: `TorchLean/Verify/CROWN/FlatBox/Affine.lean`, `TorchLean/Verify/CROWN/Op/Alpha.lean`,
  `TorchLean/Verify/CROWN/Op/ExecCROWN.lean`, `TorchLean/Verify/Certificate/CROWNNode.lean` and
  `TorchLean/Verify/Certificate/CROWNQuery.lean`.
- Zeiler (2012), "ADADELTA: An Adaptive Learning Rate Method", arXiv:1212.5701.
  Specs: `TorchLean/NN/Optimizer/Update.lean`.
- Zhang, Weng, Chen, Hsieh and Daniel (2018), "Efficient Neural Network Robustness Certification
  with General Activation Functions", NeurIPS 2018, arXiv:1811.00866.
  Specs: `TorchLean/Verify/CROWN/Activation.lean`, `TorchLean/Verify/CROWN/FlatBox/Affine.lean`,
  `TorchLean/Verify/CROWN/Graph.lean`, `TorchLean/Verify/CROWN/MLP.lean` and
  `TorchLean/Verify/CROWN/Op/ExecCROWN.lean`.
- Zhang and Sennrich (2019), "Root Mean Square Layer Normalization", NeurIPS 2019, arXiv:1910.07467.
  Specs: `TorchLean/Tensor/Layers/Normalization.lean` and
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`.
- Zhao, Zhang, Chen, Wang, Anandkumar and Tian (2024), "GaLore: Memory-Efficient LLM Training by
  Gradient Low-Rank Projection", ICML 2024, arXiv:2403.03507.
  Specs: `TorchLean/NN/Optimizer/Update.lean`.
- Zou and Hastie (2005), "Regularization and Variable Selection via the Elastic Net", Journal of the
  Royal Statistical Society, Series B 67(2), 301-320, doi:10.1111/j.1467-9868.2005.00503.x.
  Specs: `TorchLean/NN/Models/LinearRegression.lean`.

## Books

- Bellman (1957), Dynamic Programming, Princeton University Press.
  Specs: `TorchLean/NN/RL/FiniteMDP.lean` and `TorchLean/NN/RL/StochasticMDP.lean`.
- Bertsekas (2012), Dynamic Programming and Optimal Control, Vol. II, 4th ed., Athena Scientific.
  Specs: `TorchLean/NN/RL/FiniteMDP/Bellman.lean`, `TorchLean/NN/RL/MDP.lean`,
  `TorchLean/NN/RL/MDP/Bellman.lean` and `TorchLean/NN/RL/StochasticMDP/Bellman.lean`.
- Bishop (2006), Pattern Recognition and Machine Learning, Springer, doi:10.1007/978-0-387-45528-0.
  Specs: `TorchLean/NN/Models/GMM.lean`, `TorchLean/NN/Models/LogisticRegression.lean` and
  `TorchLean/NN/Models/PCA.lean`.
- C. de Boor, A Practical Guide to Splines, Applied Mathematical Sciences, Springer, 1978,
  doi:10.1007/978-1-4612-6333-3.
  Specs: `TorchLean/NN/Models/KAN.lean`.
- Boyd and Vandenberghe (2004), Convex Optimization, Cambridge University Press.
  Specs: `TorchLean/NN/Optimizer/GD/StrongConvexity.lean`.
- NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov/.
  Specs: `TorchLean/Tensor/Erf.lean`.
- Gene H. Golub and Charles F. Van Loan (2013), Matrix Computations, 4th ed., Johns Hopkins
  University Press, ISBN 978-1-4214-0794-4.
  Specs: `TorchLean/Tensor/Factorization.lean`.
- Andreas Griewank and Andrea Walther (2008), Evaluating Derivatives: Principles and Techniques of
  Algorithmic Differentiation, 2nd ed., SIAM, doi:10.1137/1.9780898717761.
  Specs: `TorchLean/NN/AdjointOp.lean`.
- E. Hairer, S. P. Nørsett and G. Wanner (1993), Solving Ordinary Differential Equations I, Springer
  Series in Computational Mathematics, doi:10.1007/978-3-540-78862-1.
  Specs: `TorchLean/NN/Models/Diffusion/Samplers.lean`.
- Higham (2002), Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM,
  doi:10.1137/1.9780898718027.
  Specs: `TorchLean/Verify/Floats/IEEE754/Expressions.lean` and
  `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- Meyer (1997), Object-Oriented Software Construction, 2nd ed., Prentice Hall.
  Specs: `TorchLean/NN/RL/Boundary/Soundness.lean`.
- Moore, Kearfott and Cloud (2009), Introduction to Interval Analysis, SIAM,
  doi:10.1137/1.9780898717716.
  Specs: `TorchLean/Tensor/Numeric/Enclosure.lean` and `TorchLean/Verify/Numeric/Interval.lean`.
- Puterman (1994), Markov Decision Processes: Discrete Stochastic Dynamic Programming, Wiley,
  doi:10.1002/9780470316887.
  Specs: `TorchLean/NN/RL/FiniteMDP.lean`, `TorchLean/NN/RL/FiniteMDP/Bellman.lean`,
  `TorchLean/NN/RL/MDP.lean`, `TorchLean/NN/RL/MDP/Bellman.lean`,
  `TorchLean/NN/RL/PPO/ActorCritic.lean`, `TorchLean/NN/RL/StochasticMDP.lean` and
  `TorchLean/NN/RL/StochasticMDP/Bellman.lean`.
- Sutton and Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., MIT Press,
  http://incompleteideas.net/book/the-book-2nd.html.
  Specs: `TorchLean/NN/RL/Bandits.lean`, `TorchLean/NN/RL/Boundary/Soundness.lean`,
  `TorchLean/NN/RL/Environment.lean`, `TorchLean/NN/RL/Eval.lean`, `TorchLean/NN/RL/FiniteMDP.lean`,
  `TorchLean/NN/RL/FiniteMDP/Bellman.lean`, `TorchLean/NN/RL/MDP.lean`,
  `TorchLean/NN/RL/MDP/Bellman.lean`, `TorchLean/NN/RL/Numerics/Advantage.lean`,
  `TorchLean/NN/RL/Numerics/Intervals.lean`, `TorchLean/NN/RL/Numerics/Returns.lean`,
  `TorchLean/NN/RL/PPO/ActorCritic.lean`, `TorchLean/NN/RL/PolicyGradient.lean`,
  `TorchLean/NN/RL/PolicyGradient/Autograd.lean`, `TorchLean/NN/RL/StochasticMDP.lean`,
  `TorchLean/NN/RL/StochasticMDP/Bellman.lean` and `TorchLean/NN/RL/Tabular.lean`.
- Vapnik (1995), The Nature of Statistical Learning Theory, Springer, doi:10.1007/978-1-4757-2440-0.
  Specs: `TorchLean/NN/Models/SVM.lean`.

## Software and documentation

- alpha-beta-CROWN verifier, https://github.com/Verified-Intelligence/alpha-beta-CROWN.
  Specs: `TorchLean/Verify/Certificate/AbCROWNLeaf.lean`.
- CPython 3.12 source, Objects/floatobject.c.
  Specs: `TorchLean/Tensor/Op/Pointwise.lean`.
- ExBLAS, reproducible BLAS kernels, https://github.com/riakymch/exblas.
  Specs: `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- glibc source, sysdeps/ieee754/flt-32/s_nextafterf.c.
  Specs: `TorchLean/Tensor/ConfiguredBinary.lean`.
- Gymnasium documentation (Farama Foundation), https://gymnasium.farama.org/, and the Gymnasium
  v1.0.0 source.
  Specs: `TorchLean/NN/RL/Boundary.lean`, `TorchLean/NN/RL/Boundary/Json.lean`,
  `TorchLean/NN/RL/Boundary/Soundness.lean`, `TorchLean/NN/RL/Environment.lean`,
  `TorchLean/NN/RL/Environment/Safety.lean`, `TorchLean/NN/RL/Eval.lean`,
  `TorchLean/NN/RL/FiniteMDP.lean`, `TorchLean/NN/RL/Gymnasium/Client.lean`,
  `TorchLean/NN/RL/Gymnasium/Session.lean`, `TorchLean/NN/RL/Numerics/Types.lean`,
  `TorchLean/NN/RL/PPO/ActorCritic.lean`, `TorchLean/NN/RL/PPO/Collect.lean` and
  `TorchLean/NN/RL/Session.lean`.
- Hugging Face Transformers source, src/transformers/models/mixtral/modeling_mixtral.py (main,
  unpinned).
  Specs: `TorchLean/Tensor/Layers/MoE.lean`.
- JAX documentation, the jax.lax module.
  Specs: `TorchLean/Tensor/Elementwise.lean`.
- Lean 4.35.0 source: src/Init/Data/Float/Float.lean, src/Init/Data/Float/Float32.lean,
  src/include/lean/lean.h and src/runtime/object.cpp.
  Specs: `TorchLean/Tensor/FromFloat.lean`, `TorchLean/Tensor/Numeric/InverseTrig.lean`,
  `TorchLean/Tensor/Numeric/Quotient.lean` and `TorchLean/Tensor/Numeric/Widening.lean`.
- Mathlib at commit 56a005855ea17e0b0958f73ec801ae140b0a1679.
  Specs: `TorchLean/NN/RL/MDP.lean`, `TorchLean/NN/RL/MDP/Bellman.lean` and
  `TorchLean/Tensor/Numeric/InverseTrig/Real.lean`.
- openai/gpt-2 source, https://github.com/openai/gpt-2, src/model.py.
  Specs: `TorchLean/NN/Models/CausalTransformer.lean` and
  `TorchLean/NN/Models/CausalTransformer/LanguageModel.lean`.
- PyTorch 2.8 documentation, https://docs.pytorch.org/docs/2.8/, and the PyTorch v2.8.0 source.
  Specs: `TorchLean/NN/AdjointOp.lean`, `TorchLean/NN/Models/Autoencoder.lean`,
  `TorchLean/NN/Models/CausalTransformer/LanguageModel.lean`,
  `TorchLean/NN/Models/Diffusion/Loss.lean`, `TorchLean/NN/Models/LinearRegression.lean`,
  `TorchLean/NN/Models/LogisticRegression.lean`, `TorchLean/NN/Models/MLP.lean`,
  `TorchLean/NN/Models/Recurrent.lean`, `TorchLean/NN/Models/Recurrent/Elman.lean`,
  `TorchLean/NN/Models/Recurrent/Stack.lean`, `TorchLean/NN/Models/SVM.lean`,
  `TorchLean/NN/Models/Seq2Seq.lean`, `TorchLean/NN/Models/Transformer.lean`,
  `TorchLean/NN/Models/Transformer/FeedForward.lean`,
  `TorchLean/NN/Models/Transformer/PostNorm.lean`,
  `TorchLean/NN/Models/Transformer/ResidualAttention.lean`, `TorchLean/NN/Models/ViT.lean`,
  `TorchLean/NN/Optimizer/Config.lean`, `TorchLean/NN/Optimizer/Constructor.lean`,
  `TorchLean/NN/Optimizer/Update.lean`, `TorchLean/NN/RL/DQN/Autograd.lean`,
  `TorchLean/NN/RL/Eval.lean`, `TorchLean/NN/RL/PolicyGradient/Autograd.lean`,
  `TorchLean/NN/RL/ValueLearning.lean`, `TorchLean/NN/VJPOp.lean`, `TorchLean/Tensor/ArgMax.lean`,
  `TorchLean/Tensor/ConfiguredBinary.lean`, `TorchLean/Tensor/Context.lean`,
  `TorchLean/Tensor/Contract.lean`, `TorchLean/Tensor/Cumulative.lean`,
  `TorchLean/Tensor/Elementwise.lean`, `TorchLean/Tensor/Factorization.lean`,
  `TorchLean/Tensor/Layers/Activation.lean`, `TorchLean/Tensor/Layers/Attention.lean`,
  `TorchLean/Tensor/Layers/Conv.lean`, `TorchLean/Tensor/Layers/Conv/Window.lean`,
  `TorchLean/Tensor/Layers/Dropout.lean`, `TorchLean/Tensor/Layers/Embedding.lean`,
  `TorchLean/Tensor/Layers/GRU.lean`, `TorchLean/Tensor/Layers/LSTM.lean`,
  `TorchLean/Tensor/Layers/Linear.lean`, `TorchLean/Tensor/Layers/Loss.lean`,
  `TorchLean/Tensor/Layers/Loss/Axis.lean`, `TorchLean/Tensor/Layers/MoE.lean`,
  `TorchLean/Tensor/Layers/Normalization.lean`,
  `TorchLean/Tensor/Layers/Normalization/GroupNorm.lean`,
  `TorchLean/Tensor/Layers/Normalization/LayerNorm.lean`, `TorchLean/Tensor/Layers/Pooling.lean`,
  `TorchLean/Tensor/Layers/RNN.lean`, `TorchLean/Tensor/Layers/RealFFT.lean`,
  `TorchLean/Tensor/Layers/Softmax.lean`, `TorchLean/Tensor/LinearAlgebra/Algorithms.lean`,
  `TorchLean/Tensor/Op/Pointwise.lean`, `TorchLean/Tensor/Op/Rule.lean`,
  `TorchLean/Tensor/Op/Table.lean`, `TorchLean/Tensor/Op/Table/Elementwise.lean`,
  `TorchLean/Tensor/Reduce.lean`,
  `TorchLean/Tensor/Reduce/Backward.lean`, `TorchLean/Tensor/Scatter.lean`,
  `TorchLean/Tensor/Shape.lean`, `TorchLean/Tensor/Sort.lean`, `TorchLean/Tensor/Statistics.lean`,
  `TorchLean/Tensor/Window.lean`, `TorchLean/Tensor/Word.lean`,
  `TorchLean/Verify/CROWN/FlatBox/Conv.lean`, `TorchLean/Verify/CROWN/Op.lean`,
  `TorchLean/Verify/LibTorch/Contract.lean` and `TorchLean/Verify/Numeric/ApproxEq.lean`.
- PyTorch Geometric documentation, https://pytorch-geometric.readthedocs.io/en/latest/ (unpinned).
  Specs: `TorchLean/NN/Models/GCN.lean` and `TorchLean/Tensor/Layers/GCN.lean`.
- ReproBLAS, binned reproducible BLAS, https://bebop.cs.berkeley.edu/reproblas/.
  Specs: `TorchLean/Verify/Floats/IEEE754/Reductions.lean`.
- scikit-learn 1.9.1, its user guide, API reference and source.
  Specs: `TorchLean/NN/Models/GMM.lean`, `TorchLean/NN/Models/GradientBoostedTrees.lean`,
  `TorchLean/NN/Models/KNN.lean`, `TorchLean/NN/Models/LinearRegression.lean`,
  `TorchLean/NN/Models/NaiveBayes.lean`, `TorchLean/NN/Models/PCA.lean`,
  `TorchLean/NN/Models/RandomForest.lean` and `TorchLean/NN/Models/SVM.lean`.
- Sebastiano Vigna (2015), splitmix.c, https://prng.di.unimi.it/splitmix.c.
  Specs: `TorchLean/Tensor/Random.lean`.
- StableHLO, the CHLO dialect (chlo.next_after).
  Specs: `TorchLean/Tensor/ConfiguredBinary.lean`.
- state-spaces/mamba source, https://github.com/state-spaces/mamba,
  mamba_ssm/modules/mamba_simple.py.
  Specs: `TorchLean/NN/Models/Mamba.lean` and `TorchLean/NN/Models/Mamba/Block.lean`.
- TorchRL documentation, https://docs.pytorch.org/rl/.
  Specs: `TorchLean/NN/RL/Environment.lean`, `TorchLean/NN/RL/FiniteMDP.lean`,
  `TorchLean/NN/RL/MDP.lean` and `TorchLean/NN/RL/StochasticMDP.lean`.
- XLA operation semantics, ConvGeneralDilated.
  Specs: `TorchLean/Tensor/Window.lean`.
