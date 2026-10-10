import random, time, json, itertools, bisect, sys, math
random.seed(42)
out = {}

# ---------- Held-Karp ----------
def held_karp(d):
    n = len(d); INF = float('inf'); FULL = 1 << n
    dp = [[INF]*n for _ in range(FULL)]
    dp[1][0] = 0
    for S in range(1, FULL):
        if not (S & 1): continue
        row = dp[S]
        for v in range(n):
            if row[v] == INF: continue
            dv = d[v]; cur = row[v]
            for w in range(n):
                if S >> w & 1: continue
                nS = S | (1 << w); c = cur + dv[w]
                if c < dp[nS][w]: dp[nS][w] = c
    last = dp[FULL-1]
    return min(last[v] + d[v][0] for v in range(1, n))

def brute(d):
    n = len(d)
    best = float('inf')
    for p in itertools.permutations(range(1, n)):
        c = d[0][p[0]] + sum(d[p[i]][p[i+1]] for i in range(len(p)-1)) + d[p[-1]][0]
        best = min(best, c)
    return best

def rand_metric(n):
    pts = [(random.random(), random.random()) for _ in range(n)]
    return [[math.dist(a, b) for b in pts] for a in pts]

ok = True
for n in range(4, 10):
    for _ in range(5):
        d = rand_metric(n)
        if abs(held_karp(d) - brute(d)) > 1e-9: ok = False
out['hk_verified_n_le_9'] = ok
hk = []
for n in range(8, 30):
    d = rand_metric(n); t = time.perf_counter(); held_karp(d); dt = time.perf_counter() - t
    hk.append((n, round(dt, 3))); print('HK', n, dt, flush=True)
    if dt > 60 or (dt > 15 and dt*2.3 > 70): break
out['held_karp'] = hk

# ---------- LIS ----------
def lis_n2(a):
    n = len(a); dp = [1]*n
    for i in range(n):
        ai = a[i]
        for j in range(i):
            if a[j] < ai and dp[j]+1 > dp[i]: dp[i] = dp[j]+1
    return max(dp) if dp else 0
def lis_nlogn(a):
    tails = []
    for x in a:
        i = bisect.bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
for n in (100, 1000, 3000):
    a = [random.randint(0, 10**9) for _ in range(n)]
    assert lis_n2(a) == lis_nlogn(a)
lis = []
for n in (10**3, 10**4, 2*10**4, 10**5, 10**6):
    a = [random.randint(0, 10**9) for _ in range(n)]
    t = time.perf_counter(); L = lis_nlogn(a); t1 = time.perf_counter() - t
    t2 = None
    if n <= 2*10**4:
        t = time.perf_counter(); lis_n2(a); t2 = time.perf_counter() - t
    lis.append({'n': n, 'nlogn_s': round(t1, 4), 'n2_s': None if t2 is None else round(t2, 3), 'L': L})
    print('LIS', lis[-1], flush=True)
out['lis'] = lis

# ---------- Viterbi ----------
states = ['Sunny', 'Cloudy', 'Rainy']; obs_syms = ['Dry', 'Damp', 'Wet']
pi = [0.6, 0.3, 0.1]
A = [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]
B = [[0.8, 0.15, 0.05], [0.4, 0.4, 0.2], [0.1, 0.3, 0.6]]
# sample a 20-day sequence
s = random.choices(range(3), pi)[0]; true_path = [s]; obs = [random.choices(range(3), B[s])[0]]
for _ in range(19):
    s = random.choices(range(3), A[s])[0]; true_path.append(s); obs.append(random.choices(range(3), B[s])[0])
def viterbi(obs):
    T = len(obs); K = 3
    lp = [[-math.inf]*K for _ in range(T)]; bp = [[0]*K for _ in range(T)]
    for k in range(K): lp[0][k] = math.log(pi[k]) + math.log(B[k][obs[0]])
    for t in range(1, T):
        for k in range(K):
            best, arg = -math.inf, 0
            for j in range(K):
                c = lp[t-1][j] + math.log(A[j][k])
                if c > best: best, arg = c, j
            lp[t][k] = best + math.log(B[k][obs[t]]); bp[t][k] = arg
    k = max(range(K), key=lambda k: lp[T-1][k]); path = [k]
    for t in range(T-1, 0, -1): k = bp[t][k]; path.append(k)
    return path[::-1], max(lp[T-1])
path, lp = viterbi(obs)
out['viterbi'] = {'obs': [obs_syms[o] for o in obs], 'true': [states[s] for s in true_path],
                  'decoded': [states[s] for s in path], 'logp': round(lp, 3),
                  'accuracy': sum(a == b for a, b in zip(path, true_path)) / 20}
print('VIT', out['viterbi'], flush=True)

# ---------- Union-Find (L1 ex 6) ----------
def uf_bench(n, ops, compress, rank):
    parent = list(range(n)); rk = [0]*n
    depth_total = 0; finds = 0
    def find(x):
        nonlocal depth_total, finds
        r = x; d = 0
        while parent[r] != r: r = parent[r]; d += 1
        depth_total += d; finds += 1
        if compress:
            while parent[x] != r: parent[x], x = r, parent[x]
        return r
    t = time.perf_counter()
    for _ in range(ops):
        a = random.randrange(n); b = random.randrange(n)
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank:
                if rk[ra] < rk[rb]: ra, rb = rb, ra
                parent[rb] = ra
                if rk[ra] == rk[rb]: rk[ra] += 1
            else: parent[rb] = ra
    dt = time.perf_counter() - t
    return {'compress': compress, 'rank': rank, 'time_s': round(dt, 2), 'avg_find_depth': round(depth_total / finds, 3)}
uf = []
N = 10**6
for compress, rank in [(False, False), (False, True), (True, False), (True, True)]:
    random.seed(7)
    r = uf_bench(N, N, compress, rank); uf.append(r); print('UF', r, flush=True)
out['union_find'] = uf

# ---------- Two-stack queue (L1 ex 7) ----------
class Q:
    def __init__(self): self.i = []; self.o = []
    def push(self, x): self.i.append(x)
    def pop(self):
        if not self.o:
            while self.i: self.o.append(self.i.pop())
        return self.o.pop()
q = Q(); OPS = 2 * 10**6; mx = 0; tot = 0; size = 0
pc = time.perf_counter_ns
for k in range(OPS):
    if size == 0 or random.random() < 0.55:
        t = pc(); q.push(k); dt = pc() - t; size += 1
    else:
        t = pc(); q.pop(); dt = pc() - t; size -= 1
    tot += dt
    if dt > mx: mx = dt
out['two_stack_queue'] = {'ops': OPS, 'max_ns': mx, 'avg_ns': round(tot / OPS, 1)}
print('TSQ', out['two_stack_queue'], flush=True)
json.dump(out, open('results.json', 'w'), indent=1)
print('DONE')
