# Core 1 — Mathematics of Computing: Lectures, Exercises, Project

Oct 7, 2026 · @Dang Tu

## How to use this pack

This pack covers the 10 handbook topics of Core 1 as one lecture each, with exercises and a real-world project. Plan for about 12 hours per lecture: 4 h reading, 4 h exercises, 4 h Python. That matches the module's 180 AH.

**Handbook grading:** assignments 20%, group project 30%, closed-book exam 50% (90–120 min). Every assessment needs an AI Disclosure Statement.

**Exercise levels:** (A) by hand, exam style; (B) Python with NumPy/SciPy; (C) stretch. Do every (A); the exam is closed book.

| Book | Use it for |
| --- | --- |
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (free PDF at mml-book.github.io) | Main text, lectures 1–8 |
| Strang, *Introduction to Linear Algebra*, 6th ed. | Lecture 1 depth |
| Boyd & Vandenberghe, *Convex Optimization* (free PDF) | Lecture 4 |
| Golub & Van Loan, *Matrix Computations* | Lecture 6 reference |
| Bishop & Bishop, *Deep Learning: Foundations and Concepts* | Lectures 7–9 |

**Suggested 14-week plan:** weeks 1–10 = one lecture each; weeks 8–13 = the project in parallel; week 14 = revision with all (A) exercises under exam timing.

Python setup: `pip install numpy scipy matplotlib jupyter pandas`.

## Lecture 1 — Linear algebra and matrix factorisations

**Goal:** see a matrix as a map, and use LU, QR and SVD to solve, fit and compress. Reading: MML ch. 2–4; Strang ch. 2, 4, 7.

### 1.1 Vector spaces and the four subspaces

A vector space is closed under addition and scaling. For A of size m×n with rank r there are four fundamental subspaces:

| Subspace | Lives in | Dimension |
| --- | --- | --- |
| Column space C(A) | R^m | r |
| Null space N(A) | R^n | n − r |
| Row space C(Aᵀ) | R^n | r |
| Left null space N(Aᵀ) | R^m | m − r |

The row space is orthogonal to the null space, and the column space to the left null space. Ax = b has a solution exactly when b lies in C(A).

### 1.2 Inner products, norms, projections

The inner product ⟨x, y⟩ = xᵀy gives length ‖x‖ = √(xᵀx) and angle cos θ = xᵀy / (‖x‖‖y‖). Projecting b onto the column space of A (full column rank) gives least squares:

```latex
\hat{x} = (A^\top A)^{-1} A^\top b, \qquad P = A (A^\top A)^{-1} A^\top, \qquad P^2 = P
```

The residual b − Aˆx is orthogonal to every column of A. This is the normal equation Aᵀ(b − Aˆx) = 0.

### 1.3 LU factorisation

Gaussian elimination writes PA = LU: P permutes rows, L is unit lower-triangular, U is upper-triangular. Cost is about (2/3)n³ flops once, then each new right-hand side costs only 2n² (forward then back substitution). Partial pivoting (pick the largest pivot) keeps it stable in practice.

### 1.4 QR factorisation

A = QR with QᵀQ = I and R upper-triangular. Least squares becomes Rˆx = Qᵀb, which avoids forming AᵀA and so avoids squaring the condition number. Gram–Schmidt shows the idea; Householder reflections are what libraries use because they stay orthogonal in floating point.

### 1.5 Eigenvalues and the spectral theorem

Av = λv. A symmetric matrix S has real eigenvalues and orthonormal eigenvectors, so S = QΛQᵀ. S is positive definite when all λ > 0, equivalently xᵀSx > 0 for all x ≠ 0. Covariance matrices and Hessians at a minimum are positive semidefinite.

### 1.6 Singular value decomposition

Every matrix factors as

```latex
A = U \Sigma V^\top = \sum_{i=1}^{r} \sigma_i u_i v_i^\top, \qquad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0
```

U and V are orthogonal; the σᵢ² are the eigenvalues of AᵀA. Eckart–Young: keeping the top k terms gives the best rank-k approximation, with error ‖A − Aₖ‖₂ = σₖ₊₁. Uses: PCA, compression, pseudo-inverse A⁺ = VΣ⁺Uᵀ, and the recommender project at the end of this pack.

### Exercises

1. (A) For A = \[\[1, 2\], \[2, 4\], \[3, 6\]\], find the rank and a basis for each of the four subspaces.
2. (A) Fit y = c + mx to the points (0, 1), (1, 3), (2, 4), (3, 4) by the normal equations. Show the residual is orthogonal to both columns.
3. (A) Compute the LU factorisation of \[\[2, 1, 1\], \[4, −6, 0\], \[−2, 7, 2\]\] and use it to solve Ax = (5, −2, 9).
4. (A) Prove that a projection matrix P = A(AᵀA)⁻¹Aᵀ satisfies P² = P and Pᵀ = P.
5. (A) Find the SVD of \[\[3, 0\], \[4, 5\]\] by hand via the eigenvalues of AᵀA.
6. (B) Implement classical and modified Gram–Schmidt. On a 50×50 Hilbert matrix, plot ‖QᵀQ − I‖ for both and for `np.linalg.qr`. Explain the difference.
7. (B) Load any greyscale photo, take its SVD and display rank k = 5, 20, 50, 100 reconstructions. Plot relative error against k and against the storage ratio k(m + n + 1)/(mn).
8. (C) Show that the least-squares solution of minimum norm is x = A⁺b, and verify numerically on a rank-deficient matrix.

## Lecture 2 — Multivariate calculus

**Goal:** compute and interpret gradients, Jacobians and Hessians, and approximate functions locally with Taylor series. Reading: MML ch. 5.1–5.4, 5.8.

### 2.1 Gradient

For f: Rⁿ → R, the gradient ∇f(x) is the vector of partial derivatives ∂f/∂xᵢ. It points in the direction of steepest increase, and its length is the rate of that increase. The directional derivative along a unit vector u is ∇f(x)ᵀu, which is largest when u is parallel to ∇f. Level sets {f = c} are perpendicular to ∇f.

### 2.2 Jacobian

For f: Rⁿ → Rᵐ, the Jacobian J is the m×n matrix with Jᵢⱼ = ∂fᵢ/∂xⱼ. It is the best linear approximation: f(x + h) ≈ f(x) + Jh. Its determinant (when m = n) gives the local volume change, used in change of variables and normalising flows.

### 2.3 Hessian

For scalar f, the Hessian Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ is symmetric when f is twice continuously differentiable. At a critical point (∇f = 0):

- H positive definite → local minimum
- H negative definite → local maximum
- H has eigenvalues of both signs → saddle point
- H singular → test is inconclusive

In deep learning most critical points in high dimension are saddles, not bad local minima.

### 2.4 Taylor series in many variables

```latex
f(x + h) = f(x) + \nabla f(x)^\top h + \tfrac{1}{2} h^\top H(x) h + O(\lVert h \rVert^3)
```

The first-order model gives gradient descent; the second-order model gives Newton's method (Lecture 5). Minimising the quadratic model in h gives h = −H⁻¹∇f.

### 2.5 Useful identities

| f(x) | ∇f(x) | Hessian |
| --- | --- | --- |
| aᵀx | a | 0 |
| xᵀAx | (A + Aᵀ)x | A + Aᵀ |
| ‖Ax − b‖² | 2Aᵀ(Ax − b) | 2AᵀA |
| log(1 + e^{wᵀx}) in w | σ(wᵀx)x | σ(1 − σ)xxᵀ |

### Exercises

1. (A) For f(x, y) = x²y + 3xy² − y, compute ∇f and H. Find and classify all critical points.
2. (A) Compute the Jacobian of polar-to-Cartesian (r, θ) → (r cos θ, r sin θ) and its determinant. Explain why dx dy = r dr dθ.
3. (A) Derive the gradient and Hessian of f(w) = ‖Xw − y‖² + λ‖w‖². Show the Hessian is positive definite for λ > 0, so the minimum is unique.
4. (A) Write the second-order Taylor expansion of f(x, y) = e^{x} cos y at (0, 0) and use it to estimate f(0.1, 0.2). Compare with the true value.
5. (B) Write a finite-difference gradient checker using central differences (f(x + εeᵢ) − f(x − εeᵢ))/(2ε). Plot the error against ε from 1e−1 to 1e−12 for f = Rosenbrock. Explain the V shape (truncation vs round-off).
6. (B) Plot contour lines and gradient arrows of the Rosenbrock function to show that the gradient is perpendicular to level sets.
7. (C) Show that for logistic loss the Hessian is XᵀDX with D diagonal and positive, so the loss is convex.

## Lecture 3 — Matrix calculus and automatic differentiation

**Goal:** differentiate through matrix expressions and understand exactly what backpropagation computes. Reading: MML ch. 5.5–5.6; Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (JMLR 2018).

### 3.1 Chain rule in matrix form

If y = g(x) and z = f(y), then the Jacobians multiply:

```latex
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \, \frac{\partial y}{\partial x}
```

For a chain of L functions, the full derivative is a product of L Jacobians. The order in which you multiply them is the only difference between forward and reverse mode.

### 3.2 Shapes and the denominator rule

Use the convention that the gradient of a scalar with respect to a matrix W has the same shape as W. Key results:

| Expression | Derivative |
| --- | --- |
| ∂(aᵀXb)/∂X | abᵀ |
| ∂ tr(AX)/∂X | Aᵀ |
| ∂ log det X /∂X | X⁻ᵀ |
| Linear layer y = Wx, upstream g = ∂L/∂y | ∂L/∂W = gxᵀ, ∂L/∂x = Wᵀg |

The last row is the core of every backward pass. Check shapes first: if they don't match, the formula is wrong.

### 3.3 Computational graphs

Write the function as a directed acyclic graph of primitive operations (add, multiply, exp, matmul). Each node stores its value in the forward pass. Every primitive knows its local derivative.

### 3.4 Forward mode

Push a tangent ẋ alongside each value: one pass computes Jẋ, a Jacobian-vector product. Cost: one pass per input direction. Best when there are few inputs and many outputs. Dual numbers (a + bε with ε² = 0) implement it neatly.

### 3.5 Reverse mode (backpropagation)

Run forward, store intermediates, then push an adjoint ȳ = ∂L/∂y backwards from the output. One backward pass gives the gradient with respect to all inputs, a vector-Jacobian product ȳᵀJ. Cost is a small constant (about 2–4×) times the forward pass, regardless of the number of parameters. That is why neural networks with billions of parameters can be trained. The price is memory: every intermediate value is kept until its adjoint is used. Gradient checkpointing trades compute for memory by recomputing some of them.

### 3.6 Worked example: two-layer network

h = ReLU(W₁x + b₁), ŷ = W₂h + b₂, L = ½‖ŷ − y‖². Backward:

1. δ₂ = ŷ − y
2. ∂L/∂W₂ = δ₂hᵀ, ∂L/∂b₂ = δ₂
3. δ₁ = (W₂ᵀδ₂) ⊙ 1\[W₁x + b₁ > 0\]
4. ∂L/∂W₁ = δ₁xᵀ, ∂L/∂b₁ = δ₁

### Exercises

1. (A) Derive ∂L/∂W for L = ‖XW − Y‖²\_F with X (n×d), W (d×k). Check every shape.
2. (A) Draw the computational graph of f(x, y) = (x + y) · sin(xy). Do one forward and one reverse pass by hand at (x, y) = (1, 2).
3. (A) Derive the gradient of softmax cross-entropy with respect to the logits z and show it equals softmax(z) − onehot(y).
4. (A) A network has 1 input and 10⁶ outputs. Which mode is cheaper, and by roughly what factor? Repeat for 10⁶ inputs and 1 output.
5. (B) Implement a tiny reverse-mode autodiff in about 100 lines: a `Value` class with +, ×, exp, tanh, ReLU and `backward()` (in the style of Karpathy's micrograd). Train a 2-layer MLP on the scikit-learn "moons" dataset.
6. (B) Check your autodiff against the finite-difference checker from Lecture 2 and against `torch.autograd`.
7. (C) Implement forward mode with dual numbers and use it to compute a Hessian-vector product by combining it with your reverse mode.

## Lecture 4 — Optimisation theory: convexity, Lagrange, KKT

**Goal:** recognise convex problems, write optimality conditions with constraints, and use duality. Reading: MML ch. 7.2–7.3; Boyd & Vandenberghe ch. 2–5.

### 4.1 Convex sets and functions

A set C is convex if the segment between any two points stays in C. A function f is convex if

```latex
f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y), \quad \theta \in [0, 1]
```

Equivalent tests: f(y) ≥ f(x) + ∇f(x)ᵀ(y − x) (first order), or H(x) positive semidefinite everywhere (second order). For a convex function, every local minimum is global. Convexity is preserved by non-negative sums, affine composition and pointwise maximum.

Examples: norms, ‖Ax − b‖², logistic loss, log-sum-exp, −log det X. Neural network losses are not convex.

### 4.2 Unconstrained optimality

First order: ∇f(x\*) = 0. Second order: H(x\*) ⪰ 0 is necessary, H(x\*) ≻ 0 is sufficient for a strict local minimum.

### 4.3 Equality constraints: Lagrange multipliers

Minimise f(x) subject to h(x) = 0. At the optimum ∇f is a combination of the constraint gradients:

```latex
\mathcal{L}(x, \nu) = f(x) + \nu^\top h(x), \qquad \nabla_x \mathcal{L} = 0, \quad h(x) = 0
```

Geometric picture: at the optimum the level set of f is tangent to the constraint surface.

### 4.4 Inequality constraints: KKT conditions

Minimise f(x) subject to gᵢ(x) ≤ 0 and hⱼ(x) = 0. The Karush–Kuhn–Tucker conditions are:

1. Stationarity: ∇f + Σλᵢ∇gᵢ + Σνⱼ∇hⱼ = 0
2. Primal feasibility: gᵢ(x) ≤ 0, hⱼ(x) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementary slackness: λᵢgᵢ(x) = 0 (a constraint is either active or its multiplier is zero)

For convex problems that satisfy Slater's condition (a strictly feasible point exists), KKT is both necessary and sufficient.

### 4.5 Duality

The dual function q(λ, ν) = inf over x of L(x, λ, ν) is always concave, and gives a lower bound on the optimum (weak duality). For convex problems under Slater, the bound is tight (strong duality). The support vector machine is the classic ML use: its dual depends on data only through inner products xᵢᵀxⱼ, which enables kernels.

### 4.6 Semidefinite programming (overview)

An SDP minimises a linear function over matrices X ⪰ 0 with linear constraints. It generalises linear programming and is used for relaxations such as Max-Cut (Goemans–Williamson) and for robust control. Solve with CVXPY.

### Exercises

1. (A) Prove that f(x) = ‖x‖ is convex using the triangle inequality. Is ‖x‖² strictly convex?
2. (A) Maximise xy subject to x + 2y = 10 with Lagrange multipliers. Interpret the multiplier.
3. (A) Minimise x² + y² subject to x + y ≥ 1. Write the KKT conditions and solve. Which constraints are active?
4. (A) Show that projection onto the probability simplex {x ≥ 0, Σxᵢ = 1} has KKT solution xᵢ = max(vᵢ − τ, 0).
5. (A) Derive the dual of the hard-margin SVM from its primal.
6. (B) Use CVXPY to solve a portfolio problem: minimise risk wᵀΣw subject to expected return μᵀw ≥ r, Σw = 1, w ≥ 0, on real stock returns (e.g. 10 tickers via `yfinance`). Plot the efficient frontier as r varies.
7. (C) Solve the Max-Cut SDP relaxation for a 20-node random graph in CVXPY and round it with random hyperplanes. Compare to the best cut by brute force.

## Lecture 5 — Optimisation algorithms: GD, SGD, Adam, Newton

**Goal:** know how first- and second-order methods work, how fast they converge, and when to use each. Reading: MML ch. 7.1; Goodfellow et al. ch. 8; Nocedal & Wright ch. 2–3 (optional).

### 5.1 Gradient descent

xₖ₊₁ = xₖ − η∇f(xₖ). Assume f is L-smooth (gradient is L-Lipschitz) and μ-strongly convex. With step η = 1/L:

```latex
f(x_k) - f^* \le \left(1 - \frac{\mu}{L}\right)^k \big(f(x_0) - f^*\big)
```

The condition number κ = L/μ controls speed: large κ means zig-zagging down narrow valleys. For convex but not strongly convex f the rate slows to O(1/k). Steps above 2/L diverge.

### 5.2 Momentum and Nesterov

Heavy-ball momentum vₖ₊₁ = βvₖ − η∇f(xₖ), xₖ₊₁ = xₖ + vₖ₊₁ damps zig-zags and speeds progress along valleys. Nesterov's accelerated gradient evaluates the gradient at the look-ahead point and improves the rate to depend on √κ instead of κ.

### 5.3 Stochastic gradient descent

For f(x) = (1/n)Σfᵢ(x), use a mini-batch gradient gₖ that is unbiased: E\[gₖ\] = ∇f(xₖ). Each step is n/B times cheaper. The noise means a constant step only reaches a neighbourhood of the optimum; convergence needs a decaying step (Σηₖ = ∞, Σηₖ² < ∞) or a schedule such as cosine decay. In deep learning the noise also helps escape saddles and tends to find flatter minima.

### 5.4 Adaptive methods: AdaGrad, RMSProp, Adam

Adam keeps running averages of the gradient and its square:

```latex
m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k, \quad v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2, \quad x_{k+1} = x_k - \eta \frac{\hat m_k}{\sqrt{\hat v_k} + \epsilon}
```

with bias corrections m̂ = m/(1 − β₁ᵏ), v̂ = v/(1 − β₂ᵏ). Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e−8. Each coordinate gets its own step size. AdamW decouples weight decay from the adaptive step and is the default for transformers.

### 5.5 Newton and quasi-Newton

Newton: xₖ₊₁ = xₖ − H⁻¹∇f. Near a minimum with H positive definite it converges quadratically (the number of correct digits doubles each step) and is unaffected by scaling. Costs: forming H is O(n²) memory and solving is O(n³), so it is impractical for millions of parameters. Far from the optimum it needs a line search or trust region. Newton–Raphson for root-finding (g(x) = 0) is the same update applied to g with its Jacobian.

BFGS builds an approximation of H⁻¹ from successive gradient differences; L-BFGS stores only the last m pairs (memory O(mn)). It is the standard choice for medium-size smooth problems such as logistic regression.

### 5.6 Which method when

| Situation | Method |
| --- | --- |
| Small, smooth, convex | Newton or L-BFGS |
| Large data, convex | SGD with schedule, or L-BFGS on full batch |
| Deep networks | SGD + momentum (vision), AdamW (transformers) |
| Non-smooth (L1) | Proximal gradient / coordinate descent |

### Exercises

1. (A) For f(x) = ½xᵀAx with A = diag(1, 100), find L, μ, κ and the largest stable step. How many GD steps reduce the error by 10⁶?
2. (A) Show that one Newton step solves any strictly convex quadratic exactly.
3. (A) Apply Newton–Raphson to g(x) = x² − 2 from x₀ = 1 for three steps. Count correct digits of √2 at each step.
4. (A) Show that the Adam bias correction makes E\[m̂ₖ\] = E\[g\] when gradients are stationary.
5. (B) Implement GD, momentum, Nesterov, Adam and Newton on the Rosenbrock function. Plot paths on the contour plot and loss vs iteration on a log scale.
6. (B) Train logistic regression on the UCI breast-cancer dataset with full-batch GD, SGD (B = 1, 32) and L-BFGS (`scipy.optimize.minimize`). Compare loss vs wall-clock time.
7. (C) Show empirically that SGD with a constant step stalls at a noise floor proportional to η, then fix it with a decaying schedule.

## Lecture 6 — Numerical linear algebra at scale

**Goal:** judge when a computed answer can be trusted, and solve or approximate huge systems cheaply. Reading: Trefethen & Bau lectures 12–15, 32–38 (or Golub & Van Loan ch. 2, 10); Halko, Martinsson & Tropp, "Finding Structure with Randomness" (SIAM Review 2011).

### 6.1 Floating point

IEEE double has machine epsilon ε ≈ 1.1×10⁻¹⁶ (float32 ≈ 6×10⁻⁸, bfloat16 ≈ 3.9×10⁻³). Every operation is exact up to a relative error of ε. Danger: catastrophic cancellation when subtracting nearly equal numbers, e.g. computing variance as E\[x²\] − E\[x\]².

### 6.2 Conditioning vs stability

Conditioning is a property of the problem; stability is a property of the algorithm. For solving Ax = b:

```latex
\kappa(A) = \lVert A \rVert \, \lVert A^{-1} \rVert = \frac{\sigma_{\max}}{\sigma_{\min}}, \qquad \frac{\lVert \delta x \rVert}{\lVert x \rVert} \lesssim \kappa(A) \cdot \frac{\lVert \delta b \rVert}{\lVert b \rVert}
```

Rule of thumb: you lose about log₁₀ κ digits. A backward-stable algorithm returns the exact answer to a slightly perturbed problem; combined with conditioning this bounds the forward error. Householder QR and LU with partial pivoting are backward stable in practice; forming AᵀA squares κ and should be avoided.

### 6.3 Iterative solvers

For large sparse A, never form A⁻¹. Iterative methods only need products Av.

| Method | Matrix type | Notes |
| --- | --- | --- |
| Jacobi, Gauss–Seidel | Diagonally dominant | Simple, slow |
| Conjugate Gradient (CG) | Symmetric positive definite | Error shrinks like ((√κ − 1)/(√κ + 1))^k |
| GMRES | General square | Minimises the residual over a Krylov subspace |
| LSQR | Least squares | CG-like for AᵀA without forming it |

Preconditioning replaces Ax = b with M⁻¹Ax = M⁻¹b, where M ≈ A is cheap to invert (diagonal, incomplete Cholesky). It lowers κ and the iteration count.

### 6.4 Power iteration and Lanczos

Repeated v ← Av/‖Av‖ converges to the top eigenvector at rate |λ₂/λ₁|. PageRank is power iteration on the web graph. Lanczos (symmetric) and Arnoldi (general) extract several eigenpairs from a Krylov subspace; `scipy.sparse.linalg.eigsh` and `svds` use them.

### 6.5 Randomised low-rank approximation

To approximate a huge A of rank about k:

1. Draw a Gaussian test matrix Ω of size n×(k + p), oversampling p ≈ 10.
2. Y = AΩ, then Q = qr(Y) gives an orthonormal basis for the range.
3. B = QᵀA is small; take its SVD B = ŨΣVᵀ, then U = QŨ.

One or two power iterations (Y = (AAᵀ)^q AΩ) sharpen accuracy when singular values decay slowly. Cost is about O(mnk) instead of O(mn·min(m, n)). Used in `sklearn` PCA with `svd_solver='randomized'` and in LoRA-style compression.

### Exercises

1. (A) Compute κ₂ of \[\[1, 1\], \[1, 1.0001\]\]. If b is known to 6 digits, how many digits of x can you trust?
2. (A) Explain why x = (−b + √(b² − 4ac))/(2a) loses accuracy when b² ≫ 4ac and give a stable alternative.
3. (A) Show that CG solves an n×n SPD system in at most n steps in exact arithmetic.
4. (B) Solve Hilbert systems of size 5 to 15 with known x = ones. Plot the relative error against κ and compare `solve`, normal equations, and QR.
5. (B) Build the 2D Poisson matrix (5-point stencil) for a 200×200 grid with `scipy.sparse`. Solve with Jacobi, CG and preconditioned CG (incomplete Cholesky or `pyamg`). Plot residual vs iteration.
6. (B) Implement randomised SVD and compare time and error to `np.linalg.svd` on a 5000×2000 matrix with decaying spectrum, for k = 10, 50, 100.
7. (C) Implement PageRank by power iteration on the SNAP web-Google graph and report the top 10 nodes.

## Lecture 7 — Probability foundations

**Goal:** work fluently with continuous distributions, joints, marginals and conditional expectation, and know why measure theory is needed. Reading: MML ch. 6; Bishop & Bishop ch. 2–3.

### 7.1 Probability spaces and why measure theory

A probability space is (Ω, F, P): outcomes, a σ-algebra of events, and a measure with P(Ω) = 1 that adds up over countably many disjoint events. Measure theory is needed because on continuous spaces not every subset can be given a probability consistently, and because it unifies sums (discrete) and integrals (continuous) as one expectation E\[X\] = ∫X dP. A random variable is a measurable function X: Ω → R. A density p(x) exists when the distribution is absolutely continuous with respect to Lebesgue measure.

### 7.2 Joint, marginal, conditional

```latex
p(x) = \int p(x, y)\, dy, \qquad p(y \mid x) = \frac{p(x, y)}{p(x)}, \qquad p(y \mid x) = \frac{p(x \mid y)\, p(y)}{p(x)}
```

The sum rule, the product rule and Bayes' rule are all you need for probabilistic ML. Independence: p(x, y) = p(x)p(y). Conditional independence X ⟂ Y | Z is what graphical models and naive Bayes exploit.

### 7.3 Expectation, variance, covariance

E is linear: E\[aX + bY\] = aE\[X\] + bE\[Y\], always. Var(X) = E\[X²\] − E\[X\]². Cov(X) = E\[(X − μ)(X − μ)ᵀ\] is symmetric positive semidefinite. Under a linear map Y = AX + b: E\[Y\] = AE\[X\] + b, Cov(Y) = A Cov(X) Aᵀ.

### 7.4 Conditional expectation

E\[Y | X\] is a random variable (a function of X). Key facts:

- Tower rule: E\[E\[Y | X\]\] = E\[Y\]
- Law of total variance: Var(Y) = E\[Var(Y | X)\] + Var(E\[Y | X\])
- E\[Y | X\] is the best predictor of Y from X in squared error. This is why regression with squared loss estimates a conditional mean.

### 7.5 The multivariate Gaussian

```latex
\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \det \Sigma}} \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
```

Its marginals and conditionals are Gaussian, sums of independent Gaussians are Gaussian, and its level sets are ellipsoids aligned with the eigenvectors of Σ. Sampling: x = μ + Lz with Σ = LLᵀ (Cholesky) and z \~ N(0, I). This is the reparameterisation trick used in VAEs.

### 7.6 Change of variables

If y = g(x) is invertible, p\_Y(y) = p\_X(g⁻¹(y)) |det J\_{g⁻¹}(y)|. Normalising flows are built on this formula.

### Exercises

1. (A) A test for a disease with 1% prevalence has 95% sensitivity and 90% specificity. Compute P(disease | positive). Explain the result to a non-specialist.
2. (A) For p(x, y) = 2 on 0 ≤ y ≤ x ≤ 1, find p(x), p(y | x) and E\[Y | X\].
3. (A) Prove the tower rule and the law of total variance for discrete variables.
4. (A) For a 2D Gaussian with μ = (0, 0), Σ = \[\[2, 1\], \[1, 2\]\], derive p(x₁ | x₂) using the Gaussian conditioning formula.
5. (A) If X \~ Uniform(0, 1), find the density of Y = −log X.
6. (B) Sample 10,000 points from the Gaussian in exercise 4 via Cholesky, plot them with the 1σ and 2σ ellipses from the eigendecomposition, and check empirical vs theoretical covariance.
7. (B) Use Monte Carlo to estimate E\[max(X₁, X₂)\] for independent standard normals. Plot the estimate and its error against sample size on log axes and check the 1/√n rate.
8. (C) Show that minimising E\[(Y − f(X))²\] over all functions f gives f(X) = E\[Y | X\].

## Lecture 8 — Statistical estimation: MLE, MAP, properties of estimators

**Goal:** derive estimators from a model, connect them to loss functions and regularisers, and judge their quality. Reading: MML ch. 8.1–8.4; Murphy, *Probabilistic Machine Learning: An Introduction* ch. 4.

### 8.1 Maximum likelihood

Given i.i.d. data x₁…xₙ from p(x | θ):

```latex
\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^{n} \log p(x_i \mid \theta)
```

The negative log-likelihood (NLL) is the training loss. Gaussian noise gives squared error; Bernoulli gives binary cross-entropy; categorical gives softmax cross-entropy; Laplace gives absolute error. Gaussian MLE: μ̂ = sample mean, σ̂² = (1/n)Σ(xᵢ − μ̂)².

### 8.2 MAP estimation

Add a prior p(θ) and maximise the posterior:

```latex
\hat\theta_{\text{MAP}} = \arg\max_\theta \Big[ \sum_i \log p(x_i \mid \theta) + \log p(\theta) \Big]
```

A Gaussian prior N(0, τ²I) adds an L2 penalty (ridge, weight decay) with λ = σ²/τ². A Laplace prior adds an L1 penalty (lasso, sparsity). MAP is a point estimate; full Bayesian inference keeps the whole posterior and averages predictions over it.

### 8.3 Conjugate priors

With a conjugate prior the posterior has the same family, so updates are closed form.

| Likelihood | Conjugate prior | Posterior update |
| --- | --- | --- |
| Bernoulli | Beta(a, b) | Beta(a + #heads, b + #tails) |
| Categorical | Dirichlet(α) | Dirichlet(α + counts) |
| Gaussian, known σ² | Gaussian on μ | Precision-weighted average of prior mean and data mean |
| Poisson | Gamma(a, b) | Gamma(a + Σx, b + n) |

### 8.4 Properties of estimators

- Bias: E\[θ̂\] − θ. The Gaussian MLE of σ² is biased by factor (n − 1)/n; dividing by n − 1 fixes it.
- Variance: Var(θ̂).
- MSE = bias² + variance. A small bias can buy a large drop in variance (ridge, shrinkage).
- Consistency: θ̂ → θ as n → ∞. MLE is consistent under regularity conditions.
- Efficiency: the Cramér–Rao bound says Var(θ̂) ≥ 1/(n I(θ)) for unbiased estimators, where I(θ) is the Fisher information. MLE reaches it asymptotically and is asymptotically normal: √n(θ̂ − θ) → N(0, I(θ)⁻¹).

### 8.5 Confidence intervals and the bootstrap

Asymptotic normality gives θ̂ ± 1.96·SE. When no formula exists, the bootstrap resamples the data with replacement B times (B ≈ 1000), recomputes θ̂, and uses percentiles as the interval. Use this to report model accuracy with an uncertainty range.

### Exercises

1. (A) Derive the MLE for the rate λ of an exponential distribution and for the parameter p of a Bernoulli.
2. (A) Show that linear regression with Gaussian noise has MLE equal to least squares, and that a Gaussian prior gives ridge regression. Find λ in terms of σ² and τ².
3. (A) A coin lands heads 3 times in 3 flips. Give the MLE and the MAP under a Beta(2, 2) prior. Which is more sensible and why?
4. (A) Prove that E\[σ̂²\_MLE\] = ((n − 1)/n)σ².
5. (A) Compute the Fisher information of a Bernoulli(p) and the Cramér–Rao bound for n flips.
6. (B) Simulate 10,000 datasets of size n = 5, 20, 100 from N(0, 4). Plot the sampling distributions of both variance estimators and report bias, variance and MSE.
7. (B) Fit ridge regression with λ from 1e−4 to 1e3 on the scikit-learn diabetes dataset. Plot training and test error. Use 5-fold cross-validation to pick λ.
8. (B) Bootstrap a 95% confidence interval for the accuracy of a classifier on a 500-example test set.

## Lecture 9 — Concentration inequalities and information theory

**Goal:** bound how far averages stray from their mean, and measure information, surprise and the distance between distributions. Reading: Bishop & Bishop ch. 2.5; Cover & Thomas, *Elements of Information Theory* ch. 2; Vershynin, *High-Dimensional Probability* ch. 2 (free PDF).

### 9.1 Concentration inequalities

| Inequality | Assumption | Bound |
| --- | --- | --- |
| Markov | X ≥ 0 | P(X ≥ a) ≤ E\[X\]/a |
| Chebyshev | Finite variance | P(\|X − μ\| ≥ a) ≤ σ²/a² |
| Hoeffding | Independent Xᵢ in \[a, b\] | P(\|X̄ − μ\| ≥ t) ≤ 2exp(−2nt²/(b − a)²) |
| Chernoff | Sums of independent variables | Exponential tail via the moment generating function |

Hoeffding gives a sample-size rule: to estimate a classifier's accuracy within t = 0.02 with 95% confidence, n ≥ ln(2/0.05)/(2·0.02²) ≈ 4,612 test examples. With a union bound over K models, replace 2 by 2K. This is the basic idea behind generalisation bounds and why test sets must be large enough and used once.

### 9.2 Entropy

```latex
H(X) = -\sum_x p(x) \log_2 p(x)
```

The average number of bits needed to encode X with the best code. It is maximal for the uniform distribution (log₂ K for K outcomes) and zero for a certain outcome. For continuous variables, differential entropy can be negative; the Gaussian has the largest differential entropy for a fixed variance.

### 9.3 Cross-entropy and KL divergence

```latex
H(p, q) = -\sum_x p(x) \log q(x), \qquad D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
```

KL ≥ 0 with equality iff p = q (Gibbs' inequality, from Jensen), and it is not symmetric. Training a classifier with cross-entropy minimises KL from the data distribution to the model, which is the same as maximum likelihood. Forward KL (p‖q) makes q cover all modes of p; reverse KL (q‖p), used in variational inference, makes q lock onto one mode.

### 9.4 Mutual information

I(X; Y) = H(X) − H(X | Y) = D\_KL(p(x, y) ‖ p(x)p(y)). It is zero iff X and Y are independent and, unlike correlation, catches non-linear dependence. Uses: feature selection, decision-tree splits (information gain), contrastive learning (InfoNCE is a lower bound on mutual information).

### 9.5 Perplexity

Perplexity = 2^{H(p, q)} in bits per token. A language model with perplexity 20 is as uncertain as choosing uniformly among 20 tokens. You will use this in NLP.

### Exercises

1. (A) Compute H for a fair die and for a die with probabilities (0.5, 0.1, 0.1, 0.1, 0.1, 0.1). Interpret in bits.
2. (A) Compute D\_KL(p‖q) and D\_KL(q‖p) for p = (0.5, 0.5), q = (0.9, 0.1). Explain the asymmetry.
3. (A) Prove D\_KL ≥ 0 using Jensen's inequality.
4. (A) Using Hoeffding, how many test examples guarantee accuracy within ±1% at 99% confidence? How does the answer change if you pick the best of 100 models on that same test set?
5. (A) Derive the KL divergence between two univariate Gaussians. (It appears in every VAE loss.)
6. (B) Simulate means of n Bernoulli(0.3) variables for n = 10, 100, 1000. Plot the empirical P(|X̄ − 0.3| ≥ t) against the Chebyshev and Hoeffding bounds.
7. (B) Estimate mutual information between each feature and the label on the UCI adult income dataset (`sklearn.feature_selection.mutual_info_classif`) and compare the ranking to absolute correlation.
8. (C) Fit a single Gaussian to a two-component Gaussian mixture by minimising forward KL and reverse KL numerically. Plot both fits.

## Lecture 10 — Discrete mathematics, graphs, combinatorics, recurrences

**Goal:** model problems as graphs, count configurations, and solve the recurrences that describe algorithm cost. Reading: Rosen, *Discrete Mathematics and Its Applications* ch. 6, 8, 10; Cormen et al. ch. 4, 22. This lecture is also the bridge into Core 2.

### 10.1 Graphs and their matrices

A graph G = (V, E) with n nodes has an adjacency matrix A (Aᵢⱼ = 1 if i–j is an edge) and a degree matrix D. Entry (Aᵏ)ᵢⱼ counts walks of length k from i to j. The graph Laplacian L = D − A is positive semidefinite, and the number of zero eigenvalues equals the number of connected components. Its second-smallest eigenvector (Fiedler vector) gives spectral clustering. The normalised adjacency D⁻¹/²AD⁻¹/² is the propagation step of graph neural networks.

### 10.2 Structures to recognise

| Structure | Property | Example |
| --- | --- | --- |
| Tree | Connected, n − 1 edges, no cycles | Decision tree, parse tree |
| DAG | Directed, no cycles, has a topological order | Computational graph (Lecture 3), task scheduler |
| Bipartite | No odd cycles | User–item graph in recommenders |
| Planar | Drawable without crossings; E ≤ 3V − 6 | Road networks (approximately) |

### 10.3 Combinatorics

- Permutations n!, arrangements n!/(n − k)!, combinations C(n, k).
- Stirling's approximation: log n! ≈ n log n − n. So any comparison sort needs log₂ n! ≈ n log₂ n comparisons.
- Inclusion–exclusion: |A ∪ B ∪ C| = Σ|A| − Σ|A ∩ B| + |A ∩ B ∩ C|.
- Pigeonhole principle and its use in hashing (collisions are guaranteed once keys outnumber buckets; birthday bound ≈ √(2N ln 2)).

### 10.4 Recurrences

Linear recurrences such as Fibonacci aₙ = aₙ₋₁ + aₙ₋₂ are solved by the characteristic equation r² = r + 1, giving aₙ ∝ φⁿ with φ ≈ 1.618. The same answer comes from the eigenvalues of \[\[1, 1\], \[1, 0\]\], which links back to Lecture 1.

Divide-and-conquer costs T(n) = aT(n/b) + f(n) are solved by the master theorem. Compare f(n) with n^{log\_b a}:

| Case | Condition | Result | Example |
| --- | --- | --- | --- |
| 1 | f grows slower | Θ(n^{log\_b a}) | Strassen: 7T(n/2) + n² → n^{2.81} |
| 2 | Same order | Θ(n^{log\_b a} log n) | Merge sort: 2T(n/2) + n → n log n |
| 3 | f grows faster (regularity holds) | Θ(f(n)) | T(n) = 2T(n/2) + n² → n² |

Generating functions turn a sequence into a power series and solve harder recurrences and counting problems.

### 10.5 Complexity modelling

Use Big-O (upper), Ω (lower) and Θ (tight). Count the dominant operation, write the recurrence, solve it. Know the reference costs: matrix multiply O(n³) naive, sorting Θ(n log n), self-attention O(L²d) in sequence length L.

### Exercises

1. (A) Prove that a tree with n nodes has n − 1 edges.
2. (A) For the path graph on 4 nodes, write L and find its eigenvalues. Verify the number of zero eigenvalues.
3. (A) Solve aₙ = 5aₙ₋₁ − 6aₙ₋₂ with a₀ = 1, a₁ = 4.
4. (A) Solve with the master theorem: T(n) = 4T(n/2) + n, T(n) = 3T(n/4) + n log n, T(n) = T(n/2) + 1.
5. (A) How many 8-character passwords from 62 symbols contain at least one digit? Use the complement.
6. (B) Run spectral clustering using the Fiedler vector on Zachary's karate club graph (`networkx.karate_club_graph()`). Compare your split to the real split.
7. (B) Time naive recursive, memoised and matrix-power Fibonacci for n up to 10⁶ (use Python ints or mod 10⁹ + 7). Plot time on log axes and relate each curve to its recurrence.
8. (C) Use generating functions to count binary trees with n nodes and show the answer is the Catalan number C(2n, n)/(n + 1).

## Project — A product recommender built from first principles

**What you build:** a "customers like you also bought" engine of the kind Shopee, Tiki, Netflix and Spotify run. You build it from the maths in this module: SVD (L1), gradients (L2–3), regularised optimisation (L4–5), randomised low-rank methods (L6), MAP estimation (L8) and the user–item bipartite graph (L10). It fits the handbook's group project: teams of 3–4, code plus a 10–15 page report and a 15-minute presentation.

**Data:** MovieLens 25M (grouplens.org) or Amazon Reviews 2023 (one category, e.g. Electronics). Both are public. Keep a time-based split: train on older ratings, test on the newest, as a real shop would.

### Stages

1. **Baselines (week 1):** global mean, user/item bias model (r̂ = μ + bᵤ + bᵢ), most popular items. Every later model must beat these.
2. **Truncated SVD (week 2):** impute missing ratings with biases, run `scipy.sparse.linalg.svds` and your own randomised SVD from Lecture 6 exercise 6. Compare time and accuracy for k = 10–200.
3. **Matrix factorisation by optimisation (weeks 3–4):** minimise

```latex
\sum_{(u,i) \in \Omega} \big(r_{ui} - \mu - b_u - b_i - p_u^\top q_i\big)^2 + \lambda \big(\lVert p_u \rVert^2 + \lVert q_i \rVert^2 + b_u^2 + b_i^2\big)
```

Derive the gradients by hand, implement SGD and Adam, and implement Alternating Least Squares (each step is a ridge regression with a closed form). Explain the L2 term as a Gaussian prior (MAP).

4. **Implicit feedback (week 5):** switch from star ratings to clicks/purchases. Implement weighted ALS for implicit data (Hu, Koren & Volinsky, 2008) and evaluate ranking.
5. **Serving demo (week 6):** a small FastAPI or Streamlit app: pick a user, show their history and top-10 recommendations, and show "similar items" by cosine similarity of item vectors.

### Evaluation

| Metric | What it tells you |
| --- | --- |
| RMSE on held-out ratings | Rating accuracy (stages 1–3) |
| Precision@10, Recall@10, NDCG@10 | Quality of the top-10 list a user actually sees |
| Catalogue coverage | Share of items ever recommended; detects popularity bias |
| Training time vs k | Cost of the factorisation |

Report each metric with a bootstrap 95% confidence interval (Lecture 8). Include one plot of the singular value spectrum and one of the learned item vectors projected to 2D, with a few labelled items.

### Report outline (10–15 pages)

1. Problem and data, with the split and its justification
2. Methods with derivations (gradients, ALS closed form, MAP view)
3. Experiments: baselines vs SVD vs SGD/Adam vs ALS, with confidence intervals
4. Analysis: cold start, popularity bias, conditioning of the ALS systems as λ → 0
5. Limitations, ethics (filter bubbles, privacy of purchase data), and the AI Disclosure Statement
6. Individual contribution statement (the handbook grades individuals from it)

### Stretch goals

- Add item side-information (genre, price) and handle cold-start items.
- Compare against a graph method: random-walk recommendations on the user–item bipartite graph.
- Partner with a small Vietnamese online shop or a VGU service (library loans, canteen orders) and test on their real data, with permission and anonymisation.
