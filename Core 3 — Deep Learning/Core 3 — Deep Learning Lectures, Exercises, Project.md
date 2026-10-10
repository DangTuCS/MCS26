# Core 3 — Deep Learning: Lectures, Exercises, Project

Oct 7, 2026 · @Dang Tu

## How to use this pack

This pack covers the 10 handbook topics of Core 3 as one lecture each, with exercises and a real-world project. Core 3 is the gate for NLP, Computer Vision, HPC and RL, so keep pace with it in weeks 1–6.

**Handbook grading:** assignments 20% (four, implementing learning algorithms from first principles and training networks), group project 30%, closed-book exam 50%.

**Exercise levels:** (A) by hand, exam style; (B) code in NumPy first, then PyTorch; (C) stretch.

| Book / resource | Use it for |
| --- | --- |
| Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2023) | Main text |
| Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free at d2l.ai) | Code for every lecture |
| Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free PDF) | Lectures 1–4 |
| Goodfellow, Bengio & Courville, *Deep Learning* (free at deeplearningbook.org) | Lectures 5–9 |
| Murphy, *Probabilistic Machine Learning: An Introduction* (free PDF) | Lectures 1, 4, 10 |
| Karpathy, "Neural Networks: Zero to Hero" (YouTube) | Lectures 5, 8 |

**Setup:** Python, `numpy`, `scikit-learn`, `torch`, `torchvision`, `matplotlib`. A free GPU on Google Colab or Kaggle is enough for every exercise; the faculty cluster is for the project.

If you have never trained a neural network, do d2l.ai ch. 2–3 before week 1 (the handbook's "bridging offer" is still TBD).

## Lecture 1 — Statistical foundations of learning

**Goal:** state precisely what "learning" means: a hypothesis class, a loss, an optimiser, and the gap between training and test error. Reading: Bishop & Bishop ch. 1, 4.1; ESL ch. 2; Murphy ch. 5.

### 1.1 Settings

| Setting | Data | Goal | Example |
| --- | --- | --- | --- |
| Supervised | (x, y) pairs | Predict y from x | Spam filter, price prediction |
| Unsupervised | x only | Find structure | Customer segments |
| Self-supervised | x only, labels made from x | Learn representations | Predict masked words or image patches |
| Reinforcement | States, actions, rewards | Maximise return | Game playing (Elective 1) |

### 1.2 Risk and empirical risk

Data come from an unknown distribution P(x, y). The true risk of a predictor f with loss ℓ is R(f) = E\[ℓ(f(x), y)\]. We cannot compute it, so we minimise the empirical risk on n samples:

```latex
\hat R(f) = \frac{1}{n} \sum_{i=1}^{n} \ell\big(f(x_i), y_i\big), \qquad \hat f = \arg\min_{f \in \mathcal{F}} \hat R(f)
```

This is empirical risk minimisation (ERM). Everything in this module is a choice of hypothesis class F, loss ℓ and optimisation procedure.

### 1.3 Loss functions

| Loss | Formula | Task | Probabilistic view (Core 1 L8) |
| --- | --- | --- | --- |
| Squared | (y − ŷ)² | Regression | Gaussian noise MLE |
| Absolute | \|y − ŷ\| | Robust regression | Laplace noise; predicts the median |
| Cross-entropy | −Σ yₖ log pₖ | Classification | Categorical MLE |
| Hinge | max(0, 1 − y·f(x)) | Classification (SVM) | Margin, no probabilities |

The 0–1 loss is what we care about for classification, but it is not differentiable, so we minimise a convex surrogate.

### 1.4 Bias–variance decomposition

For squared loss, with y = f(x) + noise of variance σ², and ̂f trained on a random dataset D:

```latex
\mathbb{E}_{D}\big[(y - \hat f(x))^2\big] = \underbrace{\big(f(x) - \mathbb{E}_D[\hat f(x)]\big)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}_D[\hat f(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}
```

Simple models have high bias; flexible models have high variance. Deep networks complicate this picture: very large models trained long enough can show "double descent", where test error falls again after the interpolation point.

### 1.5 Sources of error

Total error = approximation error (F does not contain the best predictor) + estimation error (finite data) + optimisation error (we don't find the minimiser). Diagnosing which one dominates guides what to do next: a bigger model, more data, or better training.

### Exercises

1. (A) Show that the predictor minimising expected squared loss is E\[y | x\] and the one minimising absolute loss is the conditional median.
2. (A) Derive the bias–variance decomposition above.
3. (A) For a 3-class problem with true label 2 and predicted probabilities (0.2, 0.7, 0.1), compute cross-entropy and the gradient with respect to the logits.
4. (A) Classify each as supervised, unsupervised, self-supervised or RL: (a) Grab ETA prediction, (b) grouping Shopee buyers, (c) BERT pre-training, (d) traffic light control.
5. (B) Simulate y = sin(2πx) + noise with n = 20. Fit polynomials of degree 1–15 over 200 random datasets; plot bias², variance and test error against degree.
6. (B) Repeat with n = 200. Explain how the best degree changes.
7. (C) Reproduce double descent with random-features regression (Belkin et al., 2019): plot test error against number of features around the interpolation threshold.

## Lecture 2 — Generalisation, regularisation and model selection

**Goal:** control overfitting, choose hyperparameters honestly, and know what learning theory guarantees. Reading: ESL ch. 7; Bishop & Bishop ch. 9; Shalev-Shwartz & Ben-David, *Understanding Machine Learning* ch. 2–6 (free PDF).

### 2.1 Capacity

Capacity is how many different functions a model class can fit. Too little underfits; too much memorises noise. Measures: number of parameters (crude), VC dimension (for classifiers), Rademacher complexity (data-dependent). A linear classifier in d dimensions has VC dimension d + 1.

### 2.2 A generalisation bound

For a finite class of H hypotheses and 0–1 loss, with probability at least 1 − δ every h satisfies (Hoeffding + union bound, Core 1 L9):

```latex
R(h) \le \hat R(h) + \sqrt{\frac{\ln |\mathcal{H}| + \ln(1/\delta)}{2n}}
```

For infinite classes, ln|H| is replaced by a VC or Rademacher term. The message: the gap shrinks like 1/√n and grows with capacity. Classic bounds are vacuous for modern deep nets, which generalise well despite having more parameters than examples; implicit regularisation by SGD and architecture is an active research area.

### 2.3 Regularisation

| Method | Penalty | Effect |
| --- | --- | --- |
| Ridge (L2, weight decay) | λ‖w‖² | Shrinks weights smoothly; Gaussian prior |
| Lasso (L1) | λ‖w‖₁ | Sets some weights to exactly zero; feature selection |
| Elastic net | L1 + L2 | Sparse, stable with correlated features |
| Early stopping | Stop when validation loss rises | Acts like L2 for gradient descent |
| Data augmentation, dropout | See Lecture 9 | Deep-learning specific |

### 2.4 Validation protocols

- **Train / validation / test split:** tune on validation; touch the test set once, at the end.
- **k-fold cross-validation** (k = 5 or 10): average validation error over k splits; better for small data.
- **Stratified** folds keep class proportions; **grouped** folds keep all data from one patient, user or farm in the same fold; **time-series** splits always train on the past and test on the future.
- **Nested CV** when you need an unbiased estimate of the whole tuning procedure.

The most common error in student projects is leakage: information from the test set reaching training through preprocessing (scaling on the full data), duplicates, or random splits of time-ordered data.

### 2.5 Model selection criteria

AIC = 2k − 2 log L and BIC = k log n − 2 log L trade fit against parameter count for likelihood models. For ML practice, cross-validated error with a confidence interval is the default.

### Exercises

1. (A) What is the VC dimension of intervals on the real line? Of axis-aligned rectangles in 2D? Prove both.
2. (A) Using the finite-class bound, how many examples guarantee a gap below 0.05 with 95% confidence for |H| = 10⁶?
3. (A) Show that the lasso solution for one feature with orthonormal design is soft-thresholding.
4. (A) A student standardises the whole dataset, then splits into train/test. Explain the leak and its likely effect.
5. (B) On the scikit-learn diabetes data, trace ridge and lasso coefficient paths against λ. Choose λ with 10-fold CV.
6. (B) Compare a random split and a time-based split for predicting electricity demand (e.g. a public hourly load dataset). Explain the difference in reported error.
7. (C) Implement nested CV for an SVM's C and γ and compare its estimate with plain CV on a small dataset.

## Lecture 3 — Classical methods as baselines

**Goal:** master the methods every deep model must beat, especially on tabular data. Reading: ESL ch. 3, 4, 9.2, 10, 12, 15; Bishop & Bishop ch. 4–5.

### 3.1 Linear and logistic regression

Linear regression: ŷ = wᵀx, closed form w = (XᵀX + λI)⁻¹Xᵀy (Core 1 L1, L8). Logistic regression models P(y = 1 | x) = σ(wᵀx) with σ(z) = 1/(1 + e⁻ᶻ); the loss is convex (Core 1 L2 ex. 7) and solved by Newton/IRLS or L-BFGS. Softmax regression generalises it to K classes. Coefficients are interpretable as log-odds changes, which matters in credit scoring and medicine.

### 3.2 Kernel methods

Map x to features φ(x) and work only with inner products k(x, x') = φ(x)ᵀφ(x'). The representer theorem says the regularised solution has the form f(x) = Σαᵢk(xᵢ, x).

| Kernel | Formula | Feature space |
| --- | --- | --- |
| Linear | xᵀx' | Original |
| Polynomial | (xᵀx' + c)^d | All monomials up to degree d |
| RBF (Gaussian) | exp(−‖x − x'‖²/(2ℓ²)) | Infinite-dimensional |

Kernel methods cost O(n²) memory and O(n³) time, so they suit up to about 10⁴–10⁵ examples. Random Fourier features approximate RBF kernels at scale.

### 3.3 Support vector machines

Find the hyperplane with the largest margin. Soft-margin primal:

```latex
\min_{w, b} \; \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max\big(0, 1 - y_i (w^\top x_i + b)\big)
```

The dual (Core 1 L4) involves only kernels; only support vectors (αᵢ > 0) define the boundary. Tune C and the kernel width γ jointly on a log grid.

### 3.4 Decision trees and ensembles

A tree splits on the feature and threshold that most reduce impurity (Gini or entropy for classification, variance for regression). Single trees overfit and are unstable.

- **Random forest:** average many deep trees on bootstrap samples with random feature subsets; reduces variance; few hyperparameters.
- **Gradient boosting** (XGBoost, LightGBM, CatBoost): add shallow trees one at a time, each fitting the negative gradient of the loss; reduces bias. Usually the strongest model on tabular data and a must-have baseline.

### 3.5 k-nearest neighbours and naive Bayes

k-NN predicts from the k closest training points; no training, slow prediction, poor in high dimensions. Naive Bayes assumes features are conditionally independent given the class; fast and surprisingly strong for text.

### Exercises

1. (A) Derive the gradient and Hessian of logistic loss and one Newton (IRLS) step.
2. (A) Show that the polynomial kernel (xᵀx' + 1)² in 2D equals an inner product of explicit 6-dimensional features.
3. (A) For a small 2D dataset, identify the support vectors and the margin of the hard-margin SVM by drawing.
4. (A) Compute the Gini gain of splitting 10 examples (6 positive) into groups (4+, 1−) and (2+, 3−).
5. (B) Implement logistic regression with gradient descent in NumPy and match scikit-learn's coefficients.
6. (B) On a real tabular dataset (e.g. UCI Adult income or a Kaggle credit-default set), compare logistic regression, RBF-SVM, random forest and LightGBM with 5-fold CV. Report accuracy, ROC-AUC and training time.
7. (C) Implement gradient boosting with regression stumps for squared loss from scratch and compare with scikit-learn.

## Lecture 4 — Unsupervised and representation learning

**Goal:** find structure without labels, and learn representations that make downstream tasks easy. Reading: ESL ch. 14; Bishop & Bishop ch. 15–16; Murphy ch. 20–21.

### 4.1 Clustering

**k-means** alternates: assign each point to its nearest centroid, then move each centroid to the mean of its points. It decreases the within-cluster sum of squares every step and converges to a local optimum. Use k-means++ initialisation and several restarts. Assumes round clusters of similar size.

**Gaussian mixture models** fit K Gaussians with the EM algorithm: the E-step computes soft responsibilities, the M-step re-estimates means, covariances and weights. GMMs give probabilities and elliptical clusters; k-means is the limit of EM with equal spherical covariances.

| Method | Shape | Needs K? | Handles noise? |
| --- | --- | --- | --- |
| k-means | Spherical | Yes | No |
| GMM (EM) | Elliptical, soft | Yes (pick by BIC) | Partly |
| Hierarchical (Ward) | Dendrogram | No (cut later) | No |
| DBSCAN / HDBSCAN | Arbitrary, density-based | No | Yes, labels outliers |
| Spectral | Graph-connected (Core 1 L10) | Yes | Partly |

Evaluate with silhouette score when there are no labels, and with adjusted Rand index when there are.

### 4.2 Principal component analysis

PCA finds orthogonal directions of maximum variance: the top eigenvectors of the covariance matrix, or equivalently the top right singular vectors of the centred data (Core 1 L1). Projecting onto k components minimises reconstruction error among all rank-k linear maps.

```latex
\text{explained variance of component } j = \frac{\sigma_j^2}{\sum_i \sigma_i^2}
```

Always standardise features on different scales first. t-SNE and UMAP are non-linear methods for 2D visualisation only: distances between clusters in their plots are not meaningful.

### 4.3 Autoencoders

An encoder compresses x to a code z, a decoder reconstructs x̂; train on ‖x − x̂‖². A linear autoencoder with squared loss recovers the PCA subspace. Non-linear autoencoders learn curved manifolds; denoising autoencoders reconstruct clean inputs from corrupted ones. High reconstruction error flags anomalies (fraud, machine faults). Variational autoencoders come in Spec 2.

### 4.4 Self-supervised learning

Create a supervised task from unlabelled data, then reuse the learned encoder:

- **Pretext tasks:** predict rotation, solve jigsaw puzzles, colourise.
- **Masked modelling:** hide part of the input and predict it (BERT for text, MAE for images).
- **Contrastive learning:** pull two augmentations of the same image together and push others apart (SimCLR, InfoNCE loss).

Evaluate representations with a linear probe: freeze the encoder and train only a linear classifier on few labels.

### Exercises

1. (A) Prove each k-means step does not increase the objective, so k-means converges.
2. (A) Derive the EM updates for a 1D mixture of two Gaussians.
3. (A) Show that the first principal component maximises wᵀΣw subject to ‖w‖ = 1 (use a Lagrange multiplier).
4. (A) Why does t-SNE not preserve global distances? Give one misleading reading of a t-SNE plot.
5. (B) Implement k-means and EM for GMMs in NumPy; compare on a synthetic dataset with elongated clusters.
6. (B) Run PCA on MNIST: plot explained variance, reconstruct digits with 10, 50, 150 components, and visualise with PCA, t-SNE and UMAP.
7. (B) Train a small autoencoder on normal transactions of the Kaggle credit-card fraud dataset and use reconstruction error to detect fraud; report precision–recall.
8. (C) Implement a minimal SimCLR on CIFAR-10 and evaluate a linear probe with 1% of labels.

## Lecture 5 — Feed-forward neural networks and backpropagation

**Goal:** build a multilayer perceptron from scratch, understand why it is expressive, and train it with backpropagation. Reading: Bishop & Bishop ch. 6–8; d2l.ai ch. 5; Goodfellow ch. 6.

### 5.1 The multilayer perceptron

Each layer applies an affine map and a non-linearity:

```latex
h^{(0)} = x, \qquad h^{(\ell)} = \phi\big(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\big), \qquad \hat y = \text{softmax}\big(W^{(L)} h^{(L-1)} + b^{(L)}\big)
```

Without φ the whole network collapses to one linear map. Parameters per layer: n\_in × n\_out + n\_out.

### 5.2 Activation functions

| Activation | Formula | Notes |
| --- | --- | --- |
| Sigmoid | 1/(1 + e⁻ˣ) | Saturates; gradient ≤ 0.25; use only at binary outputs |
| tanh | (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ) | Zero-centred but saturates |
| ReLU | max(0, x) | Default; cheap; "dead" units possible |
| Leaky ReLU | max(0.01x, x) | Avoids dead units |
| GELU / SiLU | x·Φ(x), x·σ(x) | Smooth; used in transformers |

### 5.3 Universal approximation

A one-hidden-layer network with enough units can approximate any continuous function on a compact set to any accuracy (Cybenko 1989, Hornik 1991). It says nothing about how many units are needed or whether training finds them. Depth helps: some functions need exponentially many units in a shallow network but only polynomially many in a deep one. Intuition with ReLU: each unit adds a "kink", so the network is piecewise linear, and depth multiplies the number of pieces.

### 5.4 Backpropagation

Backprop is reverse-mode automatic differentiation (Core 1 L3) on the network's computational graph. For layer ℓ with pre-activation a = Wh + b and upstream gradient δ = ∂L/∂a:

- ∂L/∂W = δ hᵀ, ∂L/∂b = δ
- gradient to the previous layer: ∂L/∂h = Wᵀδ, then multiply element-wise by φ'(a\_prev)

For softmax with cross-entropy the output gradient is simply ŷ − y. With mini-batches, use matrices: rows are examples, and the bias gradient sums over the batch.

### 5.5 Implementation habits

- Check gradients with finite differences on a tiny network (Core 1 L2 ex. 5).
- Overfit a single batch first: if loss won't go near zero, there is a bug.
- Use log-sum-exp for numerically stable softmax/cross-entropy.
- In PyTorch: `model.train()` / `model.eval()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

### Exercises

1. (A) Count the parameters of an MLP 784 → 256 → 128 → 10.
2. (A) Show that a network with only linear activations of any depth is equivalent to a single linear layer.
3. (A) Construct by hand a 2-layer ReLU network that computes XOR.
4. (A) Do one forward and backward pass by hand for a 2–2–1 network with sigmoid units on one example; give all weight gradients.
5. (A) Derive ∂L/∂z = softmax(z) − y for softmax + cross-entropy.
6. (B) Implement an MLP with ReLU, softmax and mini-batch SGD in NumPy only. Reach > 97% on MNIST. Verify gradients numerically.
7. (B) Re-implement in PyTorch and compare speed on CPU vs GPU.
8. (C) Visualise the piecewise-linear regions of a small ReLU network on 2D inputs as depth increases with width fixed.

## Lecture 6 — Optimising deep networks

**Goal:** make deep networks train reliably: optimiser, learning rate, initialisation and normalisation. Reading: Bishop & Bishop ch. 7; d2l.ai ch. 12; Goodfellow ch. 8. Core 1 Lecture 5 covers the optimiser theory.

### 6.1 Optimisers in practice

| Optimiser | Typical settings | When |
| --- | --- | --- |
| SGD + momentum 0.9 | lr 0.1 (batch 256), weight decay 5e−4 | CNNs on images; often best final accuracy |
| Adam | lr 1e−3, betas (0.9, 0.999) | Quick prototyping, small models |
| AdamW | lr 1e−4 to 3e−4, weight decay 0.01–0.1 | Transformers, fine-tuning |

The learning rate matters most. Find it with an LR range test: increase lr exponentially over one epoch and pick a value about 10× below where loss starts to explode.

### 6.2 Learning-rate schedules

- **Warm-up:** ramp lr up linearly over the first \~1–5% of steps; prevents early instability with Adam and large batches.
- **Step decay:** ×0.1 at fixed epochs.
- **Cosine decay:** lrₜ = lr\_min + ½(lr\_max − lr\_min)(1 + cos(πt/T)).
- **One-cycle:** up then down; fast convergence.

Larger batches allow a proportionally larger learning rate up to a point (linear scaling rule).

### 6.3 Initialisation

Initialise so that activations and gradients keep roughly the same variance through layers.

```latex
\text{Xavier/Glorot (tanh): } \mathrm{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad \text{He/Kaiming (ReLU): } \mathrm{Var}(W) = \frac{2}{n_{\text{in}}}
```

All-zero initialisation fails because every unit in a layer stays identical (symmetry is never broken).

### 6.4 Vanishing and exploding gradients

The backward pass multiplies many Jacobians; if their norms are mostly < 1, gradients vanish, if > 1 they explode. Remedies: ReLU-family activations, careful initialisation, normalisation layers, residual connections (Lecture 7), and gradient clipping (rescale when ‖g‖ exceeds a threshold, standard for RNNs and transformers).

### 6.5 Normalisation layers

| Layer | Normalises over | Used in |
| --- | --- | --- |
| Batch norm | The batch, per channel | CNNs; behaves differently in train and eval mode |
| Layer norm | The features of one example | Transformers, RNNs |
| Group norm | Groups of channels per example | Detection/segmentation with small batches |
| RMSNorm | Root mean square of features (no centring) | Modern LLMs |

Each has learnable scale γ and shift β. They smooth the loss landscape and allow larger learning rates.

### 6.6 Mixed precision

Train in float16/bfloat16 with a float32 master copy of weights: about 2× faster and half the memory on modern GPUs. In PyTorch use `torch.autocast` and, for float16, a gradient scaler.

### Exercises

1. (A) Show that with He initialisation the variance of ReLU pre-activations is preserved from layer to layer.
2. (A) A 50-layer network has Jacobians with spectral norm about 0.9. Estimate the gradient shrinkage at the first layer.
3. (A) Write the batch-norm forward pass and explain why eval mode uses running statistics.
4. (B) Train a 10-layer MLP on CIFAR-10 with zero, too-small, Xavier and He initialisation; plot per-layer activation and gradient histograms.
5. (B) Run an LR range test for a ResNet-18 on CIFAR-10, then compare constant, step, cosine and one-cycle schedules for 30 epochs.
6. (B) Compare SGD+momentum, Adam and AdamW on the same task; report final test accuracy over 3 seeds.
7. (C) Measure speed and memory with and without mixed precision on a GPU.

## Lecture 7 — Convolutional networks

**Goal:** understand why convolutions suit images and signals, compute shapes and costs, and know the main architecture families. Reading: Bishop & Bishop ch. 10; d2l.ai ch. 7–8; Stanford CS231n notes.

### 7.1 Why convolution

An MLP on a 224×224×3 image needs \~150k weights per hidden unit and ignores spatial structure. A convolution layer uses small kernels shared across positions:

- **Local connectivity:** each output sees a small patch.
- **Weight sharing:** the same detector everywhere, so far fewer parameters.
- **Translation equivariance:** shifting the input shifts the feature map.

### 7.2 Shapes and costs

For input size W, kernel K, padding P, stride S:

```latex
W_{\text{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1, \qquad \text{params} = K \cdot K \cdot C_{\text{in}} \cdot C_{\text{out}} + C_{\text{out}}
```

Multiply-adds = W\_out · H\_out · K² · C\_in · C\_out. A 3×3 conv with P = 1, S = 1 keeps the size. The receptive field grows with depth: stacking two 3×3 convs sees 5×5, three see 7×7, with fewer parameters than one large kernel.

### 7.3 Pooling and downsampling

Max pooling (2×2, stride 2) halves resolution and adds a little translation invariance. Strided convolutions do the same with learned weights. Global average pooling at the end replaces large fully connected layers.

### 7.4 Residual connections

A residual block computes y = x + F(x). Gradients flow through the identity path unchanged, so networks with 50–150+ layers train well (He et al., 2016). Bottleneck blocks use 1×1 convs to reduce and restore channels around a 3×3 conv.

### 7.5 Architecture families

| Family | Year | Key idea |
| --- | --- | --- |
| LeNet | 1998 | Conv + pool for digits |
| AlexNet | 2012 | ReLU, dropout, GPUs; won ImageNet |
| VGG | 2014 | Stacks of 3×3 convs |
| ResNet | 2015 | Residual connections |
| MobileNet / EfficientNet | 2017–2019 | Depthwise separable convs; compound scaling; phones |
| ConvNeXt | 2022 | Modernised ResNet competing with vision transformers |

Depthwise separable convolution splits a K×K×C\_in×C\_out conv into a per-channel K×K conv and a 1×1 conv, cutting cost by about K² times. That is how models run on phones, as in this module's project.

### 7.6 1D convolutions

The same ideas apply to audio, ECG and sensor time series with 1D kernels; dilated convolutions (WaveNet) grow the receptive field exponentially.

### Exercises

1. (A) An input 32×32×3 passes through conv(5×5, 16 filters, P = 0, S = 1), maxpool 2×2, conv(3×3, 32, P = 1, S = 1). Give every output shape and parameter count.
2. (A) Compute the receptive field after three 3×3 convs with stride 1, then with stride 2 in the second.
3. (A) Compare multiply-adds of a standard 3×3 conv and a depthwise separable one for 56×56×128 → 128.
4. (A) Explain why the gradient of y = x + F(x) cannot vanish through the identity path.
5. (B) Implement 2D convolution with loops, then with im2col + matrix multiply in NumPy; check against `torch.nn.functional.conv2d` and time both.
6. (B) Train a small CNN and a ResNet-18 from scratch on CIFAR-10; report accuracy, parameters and training time.
7. (B) Visualise first-layer filters and Grad-CAM heatmaps for a pretrained ResNet on your own photos.
8. (C) Train a 1D CNN on the MIT-BIH ECG arrhythmia dataset.

## Lecture 8 — Sequence models, attention and the transformer encoder

**Goal:** model sequences with recurrence, see why it struggles, and build the attention-based transformer encoder. Reading: Bishop & Bishop ch. 12; d2l.ai ch. 9–11; Vaswani et al., "Attention Is All You Need" (2017); Jay Alammar, "The Illustrated Transformer". NLP and Spec 2 build directly on this lecture.

### 8.1 Recurrent networks

An RNN keeps a hidden state: hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b). Training unrolls it through time (backpropagation through time). Gradients pass through T copies of Wₕ, so they vanish or explode for long sequences.

**LSTM** adds a cell state cₜ updated additively and controlled by forget, input and output gates; **GRU** is a lighter variant with two gates. They hold information over hundreds of steps but still process tokens one at a time, which limits parallelism on GPUs.

### 8.2 Attention

Each position builds a weighted average of all positions, with weights from query–key similarity:

```latex
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \qquad Q = XW_Q,\; K = XW_K,\; V = XW_V
```

Dividing by √dₖ keeps dot products from growing with dimension and saturating softmax. Any two positions interact in one step, and all positions compute in parallel. Cost is O(L²d) time and O(L²) memory in sequence length L.

### 8.3 Multi-head attention

Run h attention heads with separate projections of size d/h, concatenate, and project: MultiHead(X) = Concat(head₁, …, headₕ)W\_O. Different heads learn different relations (syntax, coreference, nearby tokens).

### 8.4 The transformer encoder block

1. Add positional information to token embeddings (sinusoidal or learned; rotary in modern models).
2. x ← x + MultiHeadAttention(LayerNorm(x))
3. x ← x + FFN(LayerNorm(x)), with FFN(x) = W₂ GELU(W₁x), hidden size about 4d.
4. Stack N blocks. For classification, read out a \[CLS\] token or mean-pool.

This is the pre-norm variant used in practice. Without positional encodings attention is permutation-invariant: it treats the input as a set.

### 8.5 Comparison

| Model | Long-range dependencies | Parallel training | Cost in L |
| --- | --- | --- | --- |
| RNN | Poor | No | O(L) |
| LSTM/GRU | Moderate | No | O(L) |
| 1D CNN | Grows with depth/dilation | Yes | O(L) |
| Transformer | Direct | Yes | O(L²) |

The transformer encoder also works on image patches (Vision Transformer), audio and tabular data.

### Exercises

1. (A) Show why gradients in a vanilla RNN scale like products of Wₕᵀdiag(tanh') and when they vanish.
2. (A) Compute scaled dot-product attention by hand for 3 tokens with given 2D Q, K, V.
3. (A) Count parameters of one encoder block with d = 512, h = 8, FFN size 2048.
4. (A) Show that self-attention without positional encoding is permutation-equivariant.
5. (B) Train a character-level LSTM on a Vietnamese text corpus (e.g. news articles) and sample from it.
6. (B) Implement multi-head self-attention and a transformer encoder block from scratch in PyTorch; check output shapes and compare with `nn.TransformerEncoderLayer`.
7. (B) Train your encoder for sentiment classification on IMDb or a Vietnamese review dataset (e.g. UIT-VSFC) and compare with an LSTM.
8. (C) Visualise attention maps of your trained model on a few sentences.

## Lecture 9 — Regularisation and training practice: dropout, augmentation, transfer learning

**Goal:** get good results with limited data and compute, the normal situation in real projects. Reading: Bishop & Bishop ch. 9; d2l.ai ch. 14.1–14.2; Karpathy, "A Recipe for Training Neural Networks" (blog, 2019).

### 9.1 Dropout

During training, zero each unit with probability p and scale the rest by 1/(1 − p) (inverted dropout); at test time use all units. It prevents units from co-adapting and approximates averaging an ensemble of sub-networks. Typical p: 0.1 in transformers, 0.5 in large fully connected heads; rarely used inside modern CNNs, which rely on batch norm and augmentation.

### 9.2 Weight decay and early stopping

Weight decay adds λ‖w‖² (with AdamW, applied directly to weights). Early stopping keeps the checkpoint with the best validation metric and stops after a patience of N epochs without improvement.

### 9.3 Data augmentation

Augmentation encodes invariances you know are true for the task.

| Data | Augmentations |
| --- | --- |
| Images | Random crop and resize, horizontal flip, colour jitter, rotation, RandAugment, Mixup, CutMix |
| Text | Back-translation, synonym replacement, random masking |
| Audio | Time shift, speed change, noise, SpecAugment |
| Time series | Jitter, scaling, window slicing |

Only use label-preserving transforms: a vertical flip is fine for leaf photos but wrong for digits (6 vs 9). Never augment the validation or test set (except test-time augmentation, applied deliberately).

### 9.4 Transfer learning and fine-tuning

Train a model on a large dataset (ImageNet, web text), then adapt it:

1. **Feature extraction:** freeze the backbone, train only a new head. Best with very little data (tens to hundreds of images per class).
2. **Fine-tuning:** unfreeze some or all layers with a small learning rate (about 10× lower than the head). Best with more data.
3. **Discriminative learning rates:** lower lr for early layers (generic edges and textures), higher for later layers (task-specific).
4. **Gradual unfreezing:** train the head, then unfreeze top blocks, then the rest.

Transfer works best when the source and target domains are related. Preprocess exactly as the pretrained model expects (input size and normalisation statistics). Libraries: `torchvision.models`, `timm`, Hugging Face `transformers`.

### 9.5 The training recipe

1. Look at the data: label errors, class balance, duplicates.
2. Set up evaluation and a dumb baseline first.
3. Overfit one batch, then a small subset.
4. Add regularisation until validation error stops improving.
5. Tune learning rate first, then the rest; change one thing at a time; log every run (Weights & Biases, TensorBoard, MLflow).
6. Fix random seeds and report means over several seeds.

### Exercises

1. (A) Show that inverted dropout keeps the expected activation unchanged between training and testing.
2. (A) For each, say whether the augmentation preserves the label: rotating X-rays 90°, flipping street signs horizontally, colour jitter on ripe vs unripe fruit.
3. (A) You have 300 labelled images over 5 classes. Choose between feature extraction and full fine-tuning and justify.
4. (B) On a 10%-subset of CIFAR-10, compare a CNN with no regularisation, + augmentation, + dropout, + weight decay, + all. Plot the learning curves.
5. (B) Fine-tune a pretrained ResNet-50 or EfficientNet with `timm` on Oxford Flowers-102. Compare feature extraction, full fine-tuning and gradual unfreezing.
6. (B) Measure how accuracy changes with 5, 20, 50 and 100 images per class with and without pretraining.
7. (C) Implement Mixup and CutMix and measure their effect on accuracy and calibration (Lecture 10).

## Lecture 10 — Evaluation, calibration and diagnosis

**Goal:** measure models honestly, compare them with statistics, and diagnose why they fail. Reading: Murphy ch. 5.1, 14.2; Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017); Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science" (2023).

### 10.1 Classification metrics

| Metric | Formula / meaning | Use when |
| --- | --- | --- |
| Accuracy | Correct / total | Balanced classes, equal error costs |
| Precision | TP / (TP + FP) | False alarms are costly |
| Recall | TP / (TP + FN) | Misses are costly (disease, fraud) |
| F1 | Harmonic mean of P and R | Imbalanced, single number |
| ROC-AUC | Ranking quality over all thresholds | Balanced-ish data |
| PR-AUC | Area under precision–recall | Rare positives |
| Macro vs micro averages | Per-class mean vs pooled | Multi-class with imbalance: report macro |

Always show the confusion matrix. Choose the decision threshold from costs, not the default 0.5.

### 10.2 Regression and ranking metrics

Regression: MAE (robust, in the target's units), RMSE (penalises large errors), MAPE (relative; breaks near zero), R². Ranking and retrieval: Precision@k, Recall@k, MRR, NDCG@k.

### 10.3 Calibration

A model is calibrated if, among predictions with confidence 0.8, about 80% are correct. Modern deep networks are often overconfident.

```latex
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|
```

Fix with temperature scaling: divide logits by a single T fitted on the validation set by minimising NLL. It changes confidence but not accuracy. Proper scoring rules (log loss, Brier score) reward both accuracy and calibration. Calibration matters whenever a probability drives a decision: medical triage, credit, deciding when to ask a human.

### 10.4 Statistically sound comparison

- Report mean ± standard deviation over at least 3–5 seeds.
- Give bootstrap confidence intervals on the test metric (Core 1 L8).
- Compare two classifiers on the same test set with McNemar's test; across CV folds use a corrected paired t-test.
- Compare against strong baselines (Lecture 3), not only weak ones.

### 10.5 Diagnosis

| Symptom | Likely cause | Action |
| --- | --- | --- |
| High train and validation error | Underfitting / optimisation | Bigger model, higher lr, train longer |
| Low train, high validation error | Overfitting | More data, augmentation, regularisation |
| Validation good, deployment bad | Distribution shift | Collect target-domain data, domain adaptation |
| Test score suspiciously high | Data leakage, duplicates | Check splits, deduplicate, group splits |
| Loss becomes NaN | Exploding gradients, bad lr | Lower lr, clip gradients, check inputs |
| Good accuracy, useless predictions | Class imbalance | Class weights, resampling, PR-AUC, threshold tuning |

Error analysis: inspect the 50 most confident mistakes and group them by cause. It usually reveals label noise or a missing data slice faster than any metric.

### Exercises

1. (A) A fraud model on data with 0.5% fraud has 99.5% accuracy. Explain why that can be useless and which metrics to report.
2. (A) From a confusion matrix (TP = 40, FP = 10, FN = 20, TN = 930), compute precision, recall, F1 and accuracy.
3. (A) Compute ECE with 3 bins for a given set of 12 predictions.
4. (A) Show that temperature scaling does not change the predicted class.
5. (B) Train a ResNet on CIFAR-10, plot a reliability diagram, apply temperature scaling, and report ECE before and after.
6. (B) Compare two models on the same test set with McNemar's test and bootstrap CIs on the accuracy difference.
7. (B) Simulate distribution shift by testing a CIFAR-10 model on CIFAR-10-C corruptions; plot accuracy by corruption severity.
8. (C) Find leakage in a provided notebook (or one from Kaggle) and quantify how much the reported score drops after fixing it.

## Project — Rice leaf disease detector that runs on a farmer's phone

**What you build:** a farmer photographs a rice leaf, and the app names the disease (e.g. leaf blast, brown spot, bacterial leaf blight, tungro, or healthy) with a calibrated confidence and a "not sure, ask an extension officer" option. Rice is Vietnam's main crop and these diseases cause large losses in the Mekong Delta. The project covers every Core 3 lecture: baselines (L3), CNNs (L7), transfer learning and augmentation (L9), optimisation (L6), self-supervised pre-training (L4) and honest evaluation with calibration (L10).

**Data:** start with public rice-leaf disease image sets on Kaggle and Mendeley Data, often taken in labs on plain backgrounds. Then collect at least 200 field photos yourself or with a partner (a farm cooperative, or an agriculture faculty such as Can Tho University) to test real-world performance. Record which farm and date each photo comes from.

### Stages

1. **Data audit (week 1):** check class balance, duplicates and near-duplicates (perceptual hashing), and label errors. Split by source/farm (grouped split, L2), never randomly across near-identical photos.
2. **Baselines (week 1–2):** colour-histogram + texture features with logistic regression and LightGBM (L3). Then a small CNN trained from scratch (L7).
3. **Transfer learning (weeks 2–3):** fine-tune MobileNetV3 and EfficientNet-B0 from `timm` with augmentation suited to field photos (lighting, blur, rotation, background clutter). Compare feature extraction vs fine-tuning (L9) and tune the learning-rate schedule (L6).
4. **Robustness (week 4):** evaluate on your field photos (distribution shift, L10). Use Grad-CAM to check the model looks at lesions, not backgrounds. Try self-supervised pre-training on unlabelled field photos (L4) if the shift is large.
5. **Calibration and abstention (week 4):** temperature scaling, a reliability diagram, and a confidence threshold below which the app says "not sure". Report the accuracy–coverage trade-off.
6. **Deployment (weeks 5–6):** export to ONNX or TFLite with int8 quantisation, measure size and latency on an Android phone, and build a simple app (Flutter, or a Gradio web demo) with a Vietnamese interface and care advice for each disease.

### Evaluation

| Metric | Why |
| --- | --- |
| Macro-F1 and per-class recall on the lab test set | Fair across rare diseases |
| Macro-F1 on field photos | What farmers will actually get |
| ECE before/after temperature scaling | Trustworthy confidence |
| Accuracy at 80% / 90% coverage | Value of abstaining |
| Model size (MB), latency on phone (ms) | Runs offline in the field |

Report all metrics with bootstrap 95% CIs and mean ± SD over 3 seeds.

### Report outline (10–15 pages)

1. Problem, users and data sources, with the grouped-split design
2. Methods: baselines, CNN, transfer learning, calibration
3. Results on lab and field data; error analysis of the most confident mistakes
4. Deployment measurements
5. Limitations and ethics (wrong advice can cost a harvest; consent for photos and farm locations), AI Disclosure Statement, individual contributions

### Stretch goals

- Add a segmentation head estimating the share of leaf area infected (severity).
- Extend to other crops (coffee leaf rust in the Central Highlands, durian, dragon fruit).
- Run a small field trial with farmers and measure how often they agree with the app.
