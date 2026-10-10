# Core 2 — Advanced Algorithms: Lectures, Exercises, Project

Oct 7, 2026 · @Dang Tu

## How to use this pack

This pack covers the 10 handbook topics of Core 2 as one lecture each, with exercises and a real-world project. Budget about 12 hours per lecture: read, prove, then implement and benchmark.

**Handbook grading:** assignments 20% (proofs plus implementation and benchmarking in Python or C++), group project 30%, closed-book exam 50%. The exam rewards proofs and recurrences done by hand.

**Exercise levels:** (A) by hand, exam style; (B) implement and benchmark; (C) stretch. Practise (B) problems on Codeforces or LeetCode for speed.

| Book | Use it for |
| --- | --- |
| Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms*, 4th ed. (CLRS) | Main text |
| Kleinberg & Tardos, *Algorithm Design* | Greedy, DP, flows, intuition |
| Williamson & Shmoys, *The Design of Approximation Algorithms* (free PDF) | Lecture 8 |
| Motwani & Raghavan, *Randomized Algorithms* | Lecture 3 |
| de Berg et al., *Computational Geometry* | Lecture 6 |

**Prerequisites check:** Big-O, basic data structures (heaps, hash tables, BSTs), BFS/DFS, Dijkstra. Review CLRS ch. 1–4 and 20–22 in week 0 if needed. Core 1 Lecture 10 covers recurrences and the master theorem.

Setup: Python 3.11+ with `networkx`, `numpy`, `matplotlib`, `pytest`; a C++17 compiler for timing-critical code.

## Lecture 1 — Amortised analysis and advanced recurrences

**Goal:** bound the total cost of a sequence of operations when single operations are occasionally expensive. Reading: CLRS ch. 4, 16 (amortised analysis), 19 (disjoint sets).

### 1.1 Why amortised analysis

Worst-case per operation can be pessimistic. A dynamic array (Python `list`, C++ `vector`) occasionally copies all n elements when it doubles, yet n appends cost O(n) in total, so O(1) amortised. Amortised cost is a guarantee over any sequence, not an average over random inputs.

### 1.2 Three methods

| Method | Idea | Dynamic array with doubling |
| --- | --- | --- |
| Aggregate | Total cost T(n) / n | Copies 1 + 2 + 4 + … + n < 2n, so total < 3n |
| Accounting | Overcharge cheap operations and store credit | Charge 3 per append: 1 to insert, 2 saved to pay for future copying |
| Potential | Φ(state) ≥ 0; amortised = actual + ΔΦ | Φ = 2·size − capacity |

Potential method formally:

```latex
\hat c_i = c_i + \Phi(D_i) - \Phi(D_{i-1}), \qquad \sum_i c_i = \sum_i \hat c_i - \Phi(D_n) + \Phi(D_0) \le \sum_i \hat c_i
```

### 1.3 Classic amortised structures

- **Binary counter:** n increments flip fewer than 2n bits.
- **Two-stack queue:** each element is moved at most once, O(1) amortised per operation.
- **Union-Find** with union by rank and path compression: O(α(n)) amortised per operation, where α is the inverse Ackermann function (≤ 4 in practice). Used in Kruskal's algorithm (Lecture 4).
- **Splay trees:** O(log n) amortised per access without storing balance information.
- **Fibonacci heaps:** decrease-key in O(1) amortised, which gives Dijkstra O(E + V log V).

### 1.4 Advanced recurrences

Beyond the master theorem:

- **Recursion tree:** draw levels, sum the cost of each level, sum over levels. Works for unbalanced splits such as T(n) = T(n/3) + T(2n/3) + n = Θ(n log n).
- **Substitution:** guess the bound, prove it by induction; watch for lower-order terms (prove T(n) ≤ cn − d).
- **Akra–Bazzi:** for T(n) = ΣaᵢT(bᵢn) + g(n), find p with Σaᵢbᵢᵖ = 1; then T(n) = Θ(nᵖ(1 + ∫₁ⁿ g(u)/uᵖ⁺¹ du)).
- **Change of variables:** T(n) = 2T(√n) + log n; set m = log n to get S(m) = 2S(m/2) + m, so T(n) = Θ(log n · log log n).

### Exercises

1. (A) Prove that a dynamic array that doubles when full and halves when 1/4 full has O(1) amortised cost per operation, using the potential method.
2. (A) Show that if the array halved at 1/2 full instead, a sequence of n operations can cost Θ(n²).
3. (A) Prove that k increments of a binary counter starting from 0 flip at most 2k bits.
4. (A) Solve T(n) = T(n/4) + T(3n/4) + n with a recursion tree, and T(n) = 2T(√n) + 1 by change of variables.
5. (A) Use Akra–Bazzi on T(n) = T(n/2) + T(n/3) + n.
6. (B) Implement Union-Find with and without path compression and union by rank. Benchmark 10⁶ random unions and finds; plot average find depth.
7. (B) Implement a two-stack queue and measure the maximum and the average time per operation over 10⁷ operations.
8. (C) Prove the O(log n) amortised bound for splay trees using the potential Φ = Σ log(size of subtree).

## Lecture 2 — Advanced dynamic programming

**Goal:** design DP solutions by defining states, transitions and order, and speed them up. Reading: CLRS ch. 14; Kleinberg & Tardos ch. 6.

### 2.1 The recipe

1. Define the **state**: the smallest description of a subproblem.
2. Write the **transition**: how the answer for a state depends on smaller states (optimal substructure).
3. Fix the **order** so dependencies are computed first (bottom-up), or memoise top-down.
4. **Reconstruct** the solution by storing the choice made at each state.

Complexity = (number of states) × (cost per transition). DP works when subproblems overlap; divide-and-conquer works when they don't.

### 2.2 Core patterns

| Pattern | State | Example | Cost |
| --- | --- | --- | --- |
| Prefix | dp\[i\] | Longest increasing subsequence | O(n log n) with patience sorting |
| Two sequences | dp\[i\]\[j\] | Edit distance, LCS, sequence alignment | O(nm) |
| Interval | dp\[i\]\[j\] over a range | Matrix-chain order, optimal BST | O(n³) |
| Knapsack | dp\[i\]\[w\] | 0/1 knapsack (pseudo-polynomial) | O(nW) |
| Tree | dp\[v\]\[state\] | Maximum independent set on a tree | O(n) |
| Bitmask | dp\[mask\]\[v\] | Travelling salesman (Held–Karp) | O(2ⁿ n²) |

### 2.3 Bitmask DP: Held–Karp for TSP

dp\[S\]\[v\] = shortest path that starts at city 0, visits exactly the set S, and ends at v:

```latex
dp[S][v] = \min_{u \in S \setminus \{v\}} \big( dp[S \setminus \{v\}][u] + d(u, v) \big)
```

The answer is the minimum over v of dp\[all\]\[v\] + d(v, 0). O(2ⁿ n²) time beats n! by far but is still exponential: feasible to about n = 20–25. You will use it as the exact baseline in the project.

### 2.4 Speed-ups

- **Space:** keep only the previous row (edit distance in O(min(n, m)) memory). Hirschberg recovers the alignment in linear space.
- **Monotone queue / deque:** sliding-window minimum in transitions.
- **Convex hull trick:** transitions of the form min over j of (mⱼxᵢ + bⱼ) in O(log n) or amortised O(1).
- **Knuth–Yao and divide-and-conquer optimisation:** when the optimal split point is monotone, interval DP drops from O(n³) to O(n²).

### 2.5 Connection to AI

Viterbi decoding of HMMs (NLP tagging), CTC loss (speech), dynamic time warping (time series) and the Bellman equation in reinforcement learning are all DP.

### Exercises

1. (A) Write the DP for edit distance with costs (insert 1, delete 1, substitute 2). Fill the table for "kitten" → "sitting" and recover one optimal alignment.
2. (A) Give an O(n³) interval DP for matrix-chain multiplication with dimensions (10, 30, 5, 60, 10). Compute the optimal cost.
3. (A) Design a tree DP for minimum vertex cover on a tree and prove it correct.
4. (A) Explain why 0/1 knapsack in O(nW) is not polynomial time.
5. (B) Implement Held–Karp for TSP. Find the largest n you can solve in 60 s, and verify on random instances against brute force for n ≤ 9.
6. (B) Implement LIS in O(n²) and O(n log n); benchmark up to n = 10⁶.
7. (B) Implement Viterbi for a 3-state weather HMM and decode a 20-day observation sequence.
8. (C) Apply the convex hull trick to: partition an array into segments minimising Σ(segment sum)² + C per segment.

## Lecture 3 — Probabilistic analysis and randomised algorithms

**Goal:** use randomness to get simple, fast algorithms, and prove their expected cost and error probability. Reading: CLRS ch. 5, 7, 9, 11; Motwani & Raghavan ch. 1–4.

### 3.1 Las Vegas vs Monte Carlo

| Type | Always correct? | Running time | Example |
| --- | --- | --- | --- |
| Las Vegas | Yes | Random | Randomised quicksort |
| Monte Carlo | Small error probability | Fixed | Miller–Rabin primality, Karger min-cut |

A Monte Carlo algorithm with error probability p, repeated k times independently, errs with probability pᵏ.

### 3.2 Indicator random variables

The main tool: write a count as a sum of indicators and use linearity of expectation, which holds even without independence.

**Randomised quicksort.** Elements zᵢ and zⱼ (i < j in sorted order) are compared iff one of them is the first pivot chosen from zᵢ…zⱼ, which has probability 2/(j − i + 1). So

```latex
E[\text{comparisons}] = \sum_{i<j} \frac{2}{j-i+1} \le 2n H_n = O(n \log n)
```

### 3.3 Randomised selection

Quickselect finds the k-th smallest in O(n) expected time: partition around a random pivot, recurse into one side only. The deterministic median-of-medians algorithm also gives O(n) worst case but with larger constants.

### 3.4 Hashing

A universal family H satisfies P(h(x) = h(y)) ≤ 1/m for x ≠ y. With chaining and load factor α = n/m, expected lookup cost is O(1 + α). Perfect hashing gives O(1) worst-case lookup for a static set. Bloom filters answer "is x in the set?" with no false negatives and a tunable false-positive rate (1 − e^{−kn/m})^k.

### 3.5 Karger's min-cut

Contract random edges until 2 nodes remain. One run finds a specific min cut with probability ≥ 2/(n(n − 1)). Repeating O(n² log n) times gives high success probability. Karger–Stein improves this to O(n² log³ n) total.

### 3.6 Tail bounds in algorithms

Use Chernoff bounds (Core 1 Lecture 9) to show results hold with high probability, e.g. quicksort runs in O(n log n) with probability 1 − 1/nᶜ, and the maximum chain length when hashing n keys into n buckets is O(log n / log log n) w.h.p.

### Exercises

1. (A) Compute the expected number of fixed points of a random permutation of n elements using indicators.
2. (A) Hiring problem: candidates arrive in random order; you hire whenever the current one is the best so far. Prove E\[hires\] = Hₙ ≈ ln n.
3. (A) Prove that quickselect runs in expected O(n) time.
4. (A) Show that Karger's algorithm succeeds with probability ≥ 2/(n(n − 1)) in one run, and how many runs give success probability ≥ 1 − 1/n.
5. (A) Choose m and k for a Bloom filter holding 10⁶ URLs with a 1% false-positive rate.
6. (B) Implement deterministic (first-element pivot) and randomised quicksort. Benchmark on random, sorted and many-duplicate arrays; add 3-way partitioning for duplicates.
7. (B) Implement a Bloom filter and verify the false-positive formula empirically.
8. (C) Implement Miller–Rabin and test 64-bit numbers; explain why a fixed set of 12 bases makes it deterministic for 64-bit inputs.

## Lecture 4 — Greedy algorithms and matroid theory

**Goal:** prove when a greedy choice is optimal, and recognise the matroid structure behind those proofs. Reading: CLRS ch. 15, 21; Kleinberg & Tardos ch. 4; Schrijver, *A Course in Combinatorial Optimization* ch. 10 (matroids).

### 4.1 Proving a greedy algorithm correct

- **Greedy stays ahead:** after each step, greedy's partial solution is at least as good as any other.
- **Exchange argument:** take an optimal solution that differs from greedy; swap one element to agree with greedy without making it worse; repeat.
- **Cut property (MSTs):** the lightest edge crossing any cut belongs to some MST.

### 4.2 Classic greedy problems

| Problem | Greedy rule | Cost |
| --- | --- | --- |
| Interval scheduling (max jobs) | Earliest finish time | O(n log n) |
| Scheduling to minimise maximum lateness | Earliest deadline first | O(n log n) |
| Huffman coding | Merge two least frequent symbols | O(n log n) |
| Minimum spanning tree | Kruskal (sort edges + Union-Find) or Prim (heap) | O(E log V) |
| Shortest paths, w ≥ 0 | Dijkstra: settle the closest node | O((V + E) log V) |
| Fractional knapsack | Highest value/weight first | O(n log n) |

Greedy fails for 0/1 knapsack and for shortest paths with negative edges (use Bellman–Ford).

### 4.3 Matroids

A matroid (E, I) is a ground set E with a family I of independent sets such that:

1. ∅ ∈ I
2. Hereditary: subsets of independent sets are independent
3. Exchange: if A, B ∈ I and |A| < |B|, some x ∈ B \\ A has A ∪ {x} ∈ I

**Theorem (Rado–Edmonds):** for every weight function, greedy (sort by weight, add an element if independence is kept) returns a maximum-weight basis iff (E, I) is a matroid.

Examples: graphic matroid (forests, gives Kruskal), linear matroid (linearly independent columns), uniform matroid (any k elements), partition matroid (at most kᵢ from group i).

### 4.4 Beyond one matroid

The intersection of two matroids (e.g. bipartite matching) is solvable in polynomial time but not by plain greedy. Three or more is NP-hard. For maximising a monotone submodular function under a cardinality constraint, greedy achieves a (1 − 1/e) ≈ 63% approximation; this is used for sensor placement, document summarisation and influence maximisation.

### 4.5 Scheduling and flows link

Task scheduling with deadlines and penalties (unit-time jobs) is a matroid problem, so greedy by penalty is optimal. Scheduling with resource constraints usually needs flows (Lecture 7) or approximation (Lecture 8).

### Exercises

1. (A) Prove earliest-finish-time is optimal for interval scheduling with a stays-ahead argument.
2. (A) Prove the cut property and use it to prove Kruskal's algorithm correct.
3. (A) Build the Huffman code for frequencies (a: 45, b: 13, c: 12, d: 16, e: 9, f: 5) and compute the average code length.
4. (A) Show that the graphic matroid satisfies the exchange axiom.
5. (A) Give a counterexample where greedy by value/weight fails for 0/1 knapsack.
6. (B) Implement Kruskal and Prim. Benchmark on random sparse (E = 4V) and dense (E = V²/4) graphs up to V = 10⁵.
7. (B) Implement Dijkstra with a binary heap on the OpenStreetMap road graph of Thu Dau Mot or District 1 (`osmnx`). Time 1,000 random queries.
8. (C) Implement the greedy algorithm for maximum coverage (choose k sets covering the most elements) and compare to the exact optimum on small instances; check the (1 − 1/e) bound.

## Lecture 5 — String matching and suffix structures

**Goal:** search, index and compare large texts in linear or near-linear time. Reading: CLRS ch. 32; Gusfield, *Algorithms on Strings, Trees, and Sequences* ch. 1–7.

### 5.1 Exact pattern matching

Text T of length n, pattern P of length m.

| Algorithm | Idea | Time |
| --- | --- | --- |
| Naive | Try every shift | O(nm) |
| Rabin–Karp | Rolling hash; verify on hash match | O(n + m) expected |
| KMP | Prefix function avoids re-reading the text | O(n + m) worst case |
| Z-algorithm | Z\[i\] = longest prefix match starting at i | O(n + m) |
| Boyer–Moore | Skip using bad-character / good-suffix rules | Sublinear in practice |
| Aho–Corasick | Trie + failure links for many patterns | O(n + Σm + matches) |

### 5.2 KMP prefix function

π\[i\] = length of the longest proper prefix of P\[0..i\] that is also a suffix of it. When a mismatch happens after matching q characters, fall back to π\[q − 1\] instead of restarting. Amortised analysis (Lecture 1) proves O(n) total.

### 5.3 Rolling hash

```latex
h(s_{i+1..i+m}) = \big( (h(s_{i..i+m-1}) - s_i \cdot b^{m-1}) \cdot b + s_{i+m} \big) \bmod p
```

With a random base and a large prime, the collision probability per comparison is about m/p. Use double hashing to make it negligible. Rolling hashes power plagiarism detection (winnowing) and deduplication of LLM training data (MinHash over shingles).

### 5.4 Tries

A trie stores a set of strings with shared prefixes; lookup costs O(m). Used in autocomplete, IP routing (longest-prefix match) and BPE tokenisers.

### 5.5 Suffix arrays and trees

The suffix array SA lists the starting positions of all suffixes in sorted order. With the LCP array (longest common prefix of adjacent suffixes) it answers:

- pattern search in O(m log n) by binary search;
- number of distinct substrings = n(n + 1)/2 − ΣLCP;
- longest repeated substring = max LCP.

Construction: prefix doubling in O(n log n), SA-IS in O(n); Kasai's algorithm builds LCP in O(n). A suffix tree (Ukkonen, O(n)) answers the same queries faster but uses about 20n bytes. The Burrows–Wheeler transform and FM-index compress the text and still allow search; they are how genome aligners (BWA, Bowtie) index 3 billion bases.

### 5.6 Approximate matching

Edit distance (Lecture 2) generalises to finding P in T with at most k errors in O(kn). Practical tools: n-gram indexes, BK-trees, and Levenshtein automata for spell checking.

### Exercises

1. (A) Compute the KMP prefix function for P = "ababaca" and trace the search in T = "bacbababaabcbab".
2. (A) Prove that KMP runs in O(n + m).
3. (A) Build the suffix array and LCP array of "banana$" by hand. Count the distinct substrings.
4. (A) Compute the BWT of "banana$" and invert it.
5. (B) Implement naive, KMP, Rabin–Karp and Z-algorithm. Benchmark on a 100 MB DNA file and on English text (e.g. a Project Gutenberg book) with patterns of length 5, 50, 500.
6. (B) Build a suffix array by prefix doubling and use it to find the longest repeated substring in the full text of *Truyện Kiều* or any long Vietnamese text.
7. (B) Implement Aho–Corasick to flag 10,000 banned words in a stream of comments; measure throughput in MB/s.
8. (C) Implement MinHash with locality-sensitive hashing to find near-duplicate documents in a 100k-document corpus.

## Lecture 6 — Computational geometry

**Goal:** solve geometric problems robustly and index space for fast queries. Reading: CLRS ch. 33; de Berg, Cheong, van Kreveld & Overmars, *Computational Geometry* ch. 1, 2, 5, 7, 9.

### 6.1 Primitives

The orientation test is the building block of almost everything:

```latex
\text{orient}(p, q, r) = (q_x - p_x)(r_y - p_y) - (q_y - p_y)(r_x - p_x)
```

Positive: r is left of p→q (counter-clockwise); negative: right; zero: collinear. Segment intersection, point-in-polygon and convex hull all reduce to it. Use integer or exact arithmetic where possible: floating-point orientation can give inconsistent answers and crash algorithms (robustness).

### 6.2 Convex hull

| Algorithm | Time | Note |
| --- | --- | --- |
| Graham scan | O(n log n) | Sort by angle, then a stack |
| Andrew's monotone chain | O(n log n) | Sort by x; simplest to code correctly |
| Jarvis march | O(nh) | Good when the hull has few points h |
| Chan's algorithm | O(n log h) | Output-sensitive optimum |

Lower bound Ω(n log n), by reduction from sorting.

### 6.3 Sweep line

Move a vertical line across the plane, keeping an ordered structure of objects that cross it and a queue of events. Bentley–Ottmann reports all k intersections among n segments in O((n + k) log n). Closest pair of points is O(n log n) by divide and conquer or sweep.

### 6.4 Voronoi diagrams and Delaunay triangulation

The Voronoi cell of a site is the set of points closer to it than to any other site. Fortune's sweep builds the diagram in O(n log n). Its dual is the Delaunay triangulation, which maximises the minimum angle and contains the Euclidean MST as a subgraph. Uses: nearest-facility regions (which hospital or delivery hub serves which area), mesh generation, interpolation of sensor readings.

### 6.5 Spatial partitioning

| Structure | Best for | Query |
| --- | --- | --- |
| Uniform grid / geohash | Evenly spread points | O(1) cell lookup |
| Quadtree | Clustered 2D points, maps | Adaptive depth |
| k-d tree | Low-dimension nearest neighbour | O(log n) average, degrades for d > 20 |
| R-tree | Rectangles, GIS databases (PostGIS) | Range and overlap queries |
| HNSW / LSH | High-dimensional embeddings | Approximate nearest neighbour |

The curse of dimensionality makes exact nearest-neighbour search close to linear for high d. That is why vector databases for AI embeddings use approximate methods such as HNSW.

### Exercises

1. (A) Use the orientation test to decide whether segments (0,0)–(4,4) and (0,4)–(4,0) intersect. Handle the collinear case.
2. (A) Prove that any convex hull algorithm needs Ω(n log n) comparisons by reducing sorting to it (points (x, x²)).
3. (A) Draw the Voronoi diagram and Delaunay triangulation of 5 given points by hand.
4. (A) Explain why the k-d tree nearest-neighbour query visits few nodes in 2D but almost all nodes in 100 dimensions.
5. (B) Implement monotone chain and Jarvis march; benchmark on uniform points in a disk and on points on a circle (h = n).
6. (B) Using `scipy.spatial.Voronoi` and real coordinates of hospitals in Binh Duong or Ho Chi Minh City (OpenStreetMap), map each hospital's service region and find the area farthest from any hospital.
7. (B) Implement a k-d tree and compare query times to brute force and to `faiss` HNSW on 10⁶ random vectors of dimension 2, 16 and 128.
8. (C) Implement Bentley–Ottmann and verify on random segments against the O(n²) check.

## Lecture 7 — Graph algorithms: SCC, Euler circuits, max-flow min-cut

**Goal:** decompose directed graphs, find Euler circuits, and model allocation problems as flows. Reading: CLRS ch. 20, 24 (flows); Kleinberg & Tardos ch. 7.

### 7.1 Depth-first search and its timestamps

DFS assigns discovery and finish times and classifies edges as tree, back, forward and cross. A directed graph has a cycle iff DFS finds a back edge. Reverse finish order gives a topological sort of a DAG.

### 7.2 Strongly connected components

An SCC is a maximal set where every node reaches every other. Two linear-time algorithms:

- **Kosaraju:** DFS on G for finish order; DFS on the reversed graph in decreasing finish order; each tree is one SCC.
- **Tarjan:** one DFS with low-link values and a stack.

Collapsing SCCs gives the condensation, a DAG. Uses: 2-SAT in linear time, finding dependency cycles in software builds, analysing one-way street networks (can every intersection reach every other?).

### 7.3 Euler circuits

A connected undirected graph has an Euler circuit (uses every edge once) iff every vertex has even degree; an Euler path iff exactly 0 or 2 vertices are odd. Directed: in-degree equals out-degree everywhere. Hierholzer's algorithm finds it in O(E). The Chinese Postman problem (cover every street at least once at minimum cost) adds a minimum-weight matching on odd vertices; it is used to route garbage trucks, street sweepers and mail carriers. Euler paths in de Bruijn graphs assemble genomes.

### 7.4 Maximum flow

Network G with capacities c(u, v), source s, sink t. Find the most flow from s to t respecting capacities and conservation.

**Max-flow min-cut theorem:** the maximum flow value equals the minimum capacity of an s–t cut.

| Algorithm | Idea | Time |
| --- | --- | --- |
| Ford–Fulkerson | Any augmenting path in the residual graph | O(E·f\*) for integer capacities |
| Edmonds–Karp | Shortest augmenting path (BFS) | O(VE²) |
| Dinic | Blocking flows on BFS level graph | O(V²E); O(E√V) for unit capacities |
| Push–relabel | Local pushes with height labels | O(V³) or better; fast in practice |

Integrality theorem: with integer capacities there is an integral max flow, which is why flows solve matching and assignment.

### 7.5 Problems that reduce to flow

- Bipartite matching (students to projects, drivers to orders): add s and t, unit capacities.
- Edge-disjoint paths and network reliability (Menger's theorem).
- Project selection and image segmentation as a minimum cut.
- Min-cost flow: transportation and assignment with costs (Hungarian algorithm for the dense assignment case).

### Exercises

1. (A) Run Kosaraju's algorithm by hand on a given 8-node directed graph and draw the condensation.
2. (A) Prove that a connected graph with all even degrees has an Euler circuit.
3. (A) Run Edmonds–Karp by hand on a 6-node network; show the residual graph after each augmentation and the final minimum cut.
4. (A) Model this as a flow: 5 delivery drivers, 8 orders, each driver can take orders in their district, at most 2 each. How many orders can be served?
5. (A) Show how 2-SAT reduces to SCCs on the implication graph.
6. (B) Implement Tarjan's SCC; run it on the directed road graph of a city from `osmnx` and find any unreachable parts.
7. (B) Implement Dinic's algorithm and compare to `networkx.maximum_flow` on random graphs with 10⁴ nodes.
8. (C) Solve the Chinese Postman route for a small real neighbourhood (OSMnx) and draw the route on a map.

## Lecture 8 — NP-hardness, approximation algorithms and heuristics

**Goal:** recognise NP-hard problems, then get provably good or practically good solutions. Reading: CLRS ch. 34–35; Williamson & Shmoys ch. 1–3; Kleinberg & Tardos ch. 8, 11, 12.

### 8.1 P, NP, NP-complete

NP problems have certificates checkable in polynomial time. A problem is NP-hard if every NP problem reduces to it in polynomial time. To show your problem X is NP-hard, reduce a known NP-hard problem to X (direction matters). Classic chain: SAT → 3-SAT → Independent Set → Vertex Cover; 3-SAT → Hamiltonian cycle → TSP.

### 8.2 Approximation ratio

For minimisation, an α-approximation returns cost ≤ α·OPT in polynomial time. Proofs compare to a lower bound on OPT, since OPT itself is unknown.

| Problem | Algorithm | Ratio | Lower bound used |
| --- | --- | --- | --- |
| Vertex cover | Take both ends of a maximal matching | 2 | Any cover hits every matching edge |
| Metric TSP | Double the MST, shortcut | 2 | MST ≤ OPT |
| Metric TSP | Christofides (MST + matching on odd nodes) | 1.5 | MST and half-tour bounds |
| Set cover | Greedy most-new-elements | H(n) ≈ ln n | LP / pricing argument |
| Knapsack | FPTAS by scaling values | 1 + ε | DP on rounded values |
| Bin packing | First-fit decreasing | 11/9·OPT + 6/9 | Total size |
| Max-cut | SDP + random hyperplane (Core 1 L4) | 0.878 | SDP relaxation |

Some problems resist: general TSP (no triangle inequality) cannot be approximated within any constant unless P = NP, and set cover cannot beat ln n.

### 8.3 LP relaxation and rounding

Write the problem as an integer program, relax x ∈ {0, 1} to 0 ≤ x ≤ 1, solve the LP, round. Vertex cover: take every node with xᵥ ≥ 1/2, which gives a 2-approximation. The LP optimum is a lower bound on OPT, so you can also report how far any heuristic is from optimal.

### 8.4 Heuristics and metaheuristics

When guarantees are weak, practical solvers use:

- **Construction:** nearest neighbour, cheapest insertion, savings (Clarke–Wright) for routing.
- **Local search:** 2-opt, 3-opt, Or-opt; Lin–Kernighan for TSP.
- **Metaheuristics:** simulated annealing, tabu search, genetic algorithms, large neighbourhood search.
- **Exact solvers on modest sizes:** integer programming (OR-Tools, Gurobi), branch and bound with LP bounds.

Always evaluate a heuristic with: the gap to a lower bound or known optimum, multiple random seeds, and run time. TSPLIB and CVRPLIB have instances with known optima.

### Exercises

1. (A) Reduce Independent Set to Vertex Cover and prove the reduction correct.
2. (A) Prove the 2-approximation for vertex cover by maximal matching. Give an instance where it achieves exactly 2.
3. (A) Prove the double-tree algorithm is a 2-approximation for metric TSP. Where is the triangle inequality used?
4. (A) Show that greedy set cover can be a factor Ω(log n) from optimal on a constructed instance.
5. (B) Implement nearest-neighbour + 2-opt for TSP and evaluate on 10 TSPLIB instances. Report the gap to the known optimum.
6. (B) Solve the vertex cover LP with `scipy.optimize.linprog`, round, and compare to the matching algorithm and to the exact ILP on random graphs.
7. (B) Implement simulated annealing for TSP; study the effect of the cooling schedule on quality and time.
8. (C) Implement Christofides with `networkx` minimum-weight matching and compare to 2-opt in quality and time.

## Lecture 9 — Parallel and distributed algorithms

**Goal:** design algorithms for multi-core machines and clusters, and analyse them with work, span and communication. Reading: CLRS ch. 26 (parallel algorithms); Blelloch, "Programming Parallel Algorithms" (CACM 1996); Dean & Ghemawat, "MapReduce" (OSDI 2004). Core 4 HPC goes deeper on hardware.

### 9.1 Work-span model

Model a parallel algorithm as a DAG of tasks. Work T₁ = total operations; span T∞ = longest dependency chain. With P processors and a greedy scheduler (Brent's theorem):

```latex
\max\!\left(\frac{T_1}{P}, T_\infty\right) \le T_P \le \frac{T_1}{P} + T_\infty
```

Parallelism = T₁/T∞: the most speedup you can hope for. A good parallel algorithm is work-efficient (T₁ equals the best sequential cost) with polylogarithmic span.

### 9.2 Core parallel patterns

| Pattern | Work | Span | Use |
| --- | --- | --- | --- |
| Map | O(n) | O(1) | Apply f to every element |
| Reduce (sum, max) | O(n) | O(log n) | Tree of pairwise combines |
| Scan (prefix sum) | O(n) | O(log n) | Blelloch up-sweep / down-sweep; stream compaction, radix sort |
| Parallel merge sort | O(n log n) | O(log² n) or O(log³ n) | With parallel merge |
| Parallel BFS | O(V + E) | O(D log V) | Frontier by frontier; D = diameter |

The scan is the surprising one: an apparently sequential computation becomes logarithmic-span.

### 9.3 Races and synchronisation

A determinacy race occurs when two parallel strands touch the same location and at least one writes. Fixes: give each strand its own output slot and reduce afterwards, use atomics, or use locks (risking contention and deadlock). Prefer designs with no shared writes.

### 9.4 Distributed algorithms and data synchronisation

On a cluster, communication dominates. Cost model: latency α per message plus β per byte.

- **MapReduce / Spark:** map → shuffle by key → reduce. Word count, joins, PageRank iterations. The shuffle is the expensive step.
- **All-reduce:** combine values from P machines and give everyone the result. The ring algorithm sends about 2(P − 1)/P of the data per node, independent of P. This is how gradients are synchronised in distributed deep learning.
- **Sorting at scale:** sample sort picks splitters from a sample so each machine gets a balanced range.
- **Consistency:** distributed agreement (consensus, logical clocks) is covered in Elective 3 Distributed Systems.

### Exercises

1. (A) Compute work, span and parallelism of the recursive sum of n numbers and of naive parallel Fibonacci.
2. (A) Trace Blelloch's exclusive scan on \[3, 1, 7, 0, 4, 1, 6, 3\].
3. (A) Using Brent's bound, estimate T₈ for an algorithm with T₁ = 10⁹ and T∞ = 10⁶.
4. (A) Show that ring all-reduce communicates 2(P − 1)N/P values per node for N values.
5. (B) Implement parallel prefix sum with Python `multiprocessing` or Numba (or C++ with OpenMP). Measure speedup for 1–16 cores on 10⁸ elements; compare to Amdahl's law.
6. (B) Implement parallel merge sort and compare to `std::sort` / `numpy.sort`.
7. (B) Write word count and inverted-index building in PySpark on a Wikipedia dump subset; inspect the shuffle size in the Spark UI.
8. (C) Implement parallel BFS with frontier arrays on a large SNAP graph and measure scaling.

## Lecture 10 — Online algorithms, competitive analysis and streaming

**Goal:** make decisions before the future is known, and compute statistics over data too big to store. Reading: Borodin & El-Yaniv, *Online Computation and Competitive Analysis* ch. 1–3; Kleinberg & Tardos ch. 13.8; Cormode & Yi, *Small Summaries for Big Data* ch. 1–3.

### 10.1 Competitive ratio

An online algorithm ALG is c-competitive if, for every input sequence σ, ALG(σ) ≤ c·OPT(σ) + b, where OPT knows the whole sequence in advance. You compare against an all-knowing adversary, so the bound holds for the worst case.

### 10.2 Ski rental

Renting costs 1 per day; buying costs B. You don't know how many days you will ski. Rent for B − 1 days, then buy: total ≤ 2B − 1, so 2-competitive, and no deterministic strategy does better. A randomised strategy achieves e/(e − 1) ≈ 1.58. The same structure appears in: when to spin up a cloud server, when to stop waiting and take a taxi, when to power down a disk.

### 10.3 Paging and caching

Cache of k pages, unknown request sequence.

| Policy | Competitive ratio |
| --- | --- |
| Belady's MIN (evict farthest future use) | Optimal offline, 1 |
| LRU, FIFO | k (and every deterministic policy is ≥ k) |
| Randomised marking | 2Hₖ ≈ 2 ln k |

In practice LRU does far better than k because real request sequences have locality; competitive analysis is a worst-case lens. Caching appears in CPUs, CDNs, databases and the KV cache of LLM serving.

### 10.4 Other online problems

- **Online bipartite matching** (ads to search queries, riders to drivers): greedy is 1/2-competitive; the RANKING algorithm achieves 1 − 1/e.
- **Secretary problem:** observe the first n/e candidates, then take the first one better than all of them; picks the best with probability 1/e.
- **Online learning / multiplicative weights:** regret O(√(T log n)) against the best expert in hindsight; the basis of boosting and of bandits in RL.

### 10.5 Streaming algorithms

One pass, memory polylogarithmic in the stream length.

| Task | Algorithm | Memory |
| --- | --- | --- |
| Uniform sample of k items | Reservoir sampling | O(k) |
| Majority element | Boyer–Moore voting | O(1) |
| Heavy hitters (frequency > n/k) | Misra–Gries | O(k) |
| Frequency estimates | Count-Min sketch | O((1/ε) log(1/δ)) |
| Number of distinct items | HyperLogLog | \~1.5 KB for 2% error |
| Quantiles (median, p99 latency) | t-digest, KLL | O((1/ε) log) |

These run in production: Redis has HyperLogLog, monitoring systems compute p99 latencies with sketches, and trending hashtags use heavy-hitter algorithms.

### Exercises

1. (A) Prove the break-even ski-rental strategy is (2 − 1/B)-competitive and that no deterministic strategy beats 2 − 1/B.
2. (A) Prove that LRU is k-competitive and construct a sequence on which every deterministic policy faults on every request.
3. (A) Prove that reservoir sampling keeps each of n items with probability k/n.
4. (A) Show that Misra–Gries with k counters underestimates any frequency by at most n/(k + 1).
5. (B) Simulate LRU, FIFO, random and MIN on a real web trace (e.g. a Wikipedia access trace) for cache sizes 10–10⁴ and plot hit rates.
6. (B) Implement Count-Min and HyperLogLog; on a stream of 10⁷ Zipf-distributed items compare estimates with exact counts and memory used.
7. (B) Simulate the secretary rule for n = 100 over 10⁵ trials and confirm the 1/e success rate.
8. (C) Implement multiplicative weights for online portfolio selection on real stock data and compare its regret to the best single stock.

## Project — Last-mile delivery routing on a real city map

**What you build:** a route planner for a fleet of delivery motorbikes, the core problem at Grab, GHN, Giao Hàng Tiết Kiệm and Shopee Express. Given a depot, a day's orders with time windows and bike capacities, it produces routes that minimise total travel time. It uses almost every lecture: Dijkstra (L4), DP (L2), approximation and heuristics (L8), geometry for clustering (L6), flows/matching for assignment (L7), parallelism (L9) and online insertion of new orders (L10).

**Data:** the real road network of Binh Duong or a few districts of Ho Chi Minh City from OpenStreetMap via `osmnx`. Generate orders by sampling real addresses (building footprints) with realistic time windows, or use public CVRPLIB/Solomon instances to validate against known optima first.

### Stages

1. **Road graph and distance matrix (week 1):** download the network, keep the largest strongly connected component (L7, one-way streets), compute the order-to-order travel-time matrix with Dijkstra from each order. Measure the time; parallelise across cores (L9).
2. **Single vehicle, exact (week 2):** Held–Karp DP (L2) for up to \~20 stops. This is your ground truth.
3. **Single vehicle, large (week 2–3):** nearest neighbour + 2-opt/Or-opt, and the MST double-tree bound (L8). Report the gap to Held–Karp for small cases and to the MST lower bound for large ones.
4. **Fleet with capacities and time windows (weeks 3–5):** Clarke–Wright savings, then improve with simulated annealing or large neighbourhood search. Compare against Google OR-Tools as an industrial baseline.
5. **Online orders (week 5):** new orders arrive during the day. Insert each with the cheapest feasible insertion, re-optimise periodically, and compare with the offline plan to estimate an empirical competitive ratio (L10).
6. **Map demo (week 6):** draw the routes on an interactive Folium map with each bike in its own colour and a table of stop times.

### Evaluation

| Metric | Meaning |
| --- | --- |
| Total travel time (min) | Main objective |
| Gap to optimum or lower bound (%) | Quality of each heuristic |
| Time-window violations, capacity violations | Feasibility; should be zero |
| Solve time (s) for 50, 200, 1000 orders | Scalability |
| Online / offline cost ratio | Price of not knowing the future |

Run each heuristic over at least 10 random seeds and report mean ± standard deviation.

### Report outline (10–15 pages)

1. Problem statement and model (graph, VRPTW formulation)
2. Algorithms with complexity analysis and approximation guarantees where they exist
3. Benchmarks: CVRPLIB/Solomon results against known optima, then the real-city results
4. Scaling and parallel speedup
5. Online variant and its competitive behaviour
6. Limitations (traffic data, rain, real road speeds), ethics (driver workload and fairness), AI Disclosure Statement, individual contributions

### Stretch goals

- Use time-dependent travel speeds (rush hour) on major roads.
- Balance workload fairly across drivers (min-max objective).
- Pitch the tool to a local shop or VGU canteen that delivers on campus and test it with their real orders.
