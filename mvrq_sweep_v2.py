import numpy as np
import faiss
import matplotlib.pyplot as plt

# ---------------------------
# 0) Config rápida
# ---------------------------
N = 100_000         # docs
Qn = 2_000          # queries
d = 256             # dimensión
M_MAX = 8           # RQ stages
NBITS = 8           # bits por stage
KM = 512            # k-means clusters para predictor
W = 1.0             # peso varianza-cluster (1.0 = solo varianza; 0.0 = solo kNN)
ALPHAS = [0.4, 0.6, 0.8, 1.0]
SEED = 42

np.random.seed(SEED)

# ---------------------------
# 1) Dataset sintético
# ---------------------------
X = np.random.randn(N, d).astype('float32')
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
Q = X[np.random.choice(N, size=Qn, replace=False)] + 0.05*np.random.randn(Qn, d).astype('float32')
Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
print(f"Dataset: X={X.shape}, Q={Q.shape}")

# ---------------------------
# 2) Entrenar RQ (baseline)
# ---------------------------
rq = faiss.ResidualQuantizer(d, M_MAX, NBITS)
perm = np.random.permutation(N)[:min(50_000, N)]
rq.train(X[perm])
codes_fixed = rq.compute_codes(X)
print("RQ-8 entrenado.")

# Ground truth (Flat exacto, Inner Product)
gt_index = faiss.IndexFlatIP(d)
gt_index.add(X)
_, GT = gt_index.search(Q, 10)

def recall_at_k(gt, pred, k=10):
    inter = np.array([len(set(gt[i]) & set(pred[i])) for i in range(len(gt))])
    return (inter / k).mean()

# Baseline RQ-8 reconstrucción
Xhat_rq = rq.decode(codes_fixed)
idx_rq = faiss.IndexFlatIP(d); idx_rq.add(Xhat_rq)
_, pred_rq = idx_rq.search(Q, 10)
recall_rq = recall_at_k(GT, pred_rq)
bytes_rq = M_MAX + 2  # ~1B Mi + 1B misceláneo
print(f"Baseline RQ-8 → Bytes: {bytes_rq}, Recall@10: {recall_rq:.4f}")

# ---------------------------
# 3) Predictores g(x)
#    (a) kNN densidad (sobre subset para costo)
#    (b) Varianza intra-cluster (k-means KM)
#    g = W*var + (1-W)*knn
# ---------------------------
# (a) kNN densidad (10-NN) en subset
subN = min(20_000, N)
hnsw = faiss.IndexHNSWFlat(d, 32)
hnsw.add(X[:subN])
D, _ = hnsw.search(X, 11)
g_knn = D[:,1:].mean(axis=1).astype('float32')  # promedio distancia a 10-NN
# normaliza a [0,1]
g_knn = (g_knn - g_knn.min()) / (np.ptp(g_knn) + 1e-9)

# (b) Varianza intra-cluster
print("Entrenando k-means para predictor de varianza...")
kmeans = faiss.Kmeans(d, KM, niter=10, verbose=False)
kmeans.train(X[perm])
_, assign = kmeans.index.search(X, 1)
assign = assign.ravel()

cluster_var = np.zeros(KM, dtype='float32')
for c in range(KM):
    idx = np.where(assign == c)[0]
    if idx.size > 1:
        Xc = X[idx]
        mu = Xc.mean(axis=0, keepdims=True)
        cluster_var[c] = np.mean(np.linalg.norm(Xc - mu, axis=1))
    else:
        cluster_var[c] = 0.0
# normaliza a [0,1]
cluster_var = (cluster_var - cluster_var.min()) / (np.ptp(cluster_var) + 1e-9)
g_var = cluster_var[assign]

# Predictor final
g = (W * g_var + (1.0 - W) * g_knn).astype('float32')
# Evita ceros extremos
g = np.clip(g, np.percentile(g, 1), np.percentile(g, 99))
g = (g - g.min()) / (np.ptp(g) + 1e-9)

print(f"Predictor listo. Corrientes: W={W} (1.0=solo varianza).")

# ---------------------------
# 4) MVRQ runner (early stop)
# ---------------------------
def run_mvrq(alpha):
    M_i = np.zeros(N, dtype=np.uint8)
    codes_var = []
    # NOTA: no tenemos assign_residual expuesto; aproximamos greedy con compute_codes parcial
    for i in range(N):
        r = X[i].copy()
        codes_i = []
        for m in range(M_MAX):
            # codifica r con RQ como si fuese un vector
            tmp = rq.compute_codes(r.reshape(1,-1))  # usa el pipeline completo
            idx_m = int(tmp[0, m])
            codes_i.append(idx_m)
            # decodifica las m etapas escogidas hasta ahora y recalcula residual
            padded = np.array([codes_i + [0]*(M_MAX - len(codes_i))], dtype='uint8')
            decoded = rq.decode(padded)[0]
            r = X[i] - decoded
            # umbral dinámico
            if np.linalg.norm(r) <= alpha * g[i]:
                M_i[i] = m + 1
                break
        if M_i[i] == 0:
            M_i[i] = M_MAX
        codes_var.append(codes_i[:M_i[i]])

    avg_bytes = float(M_i.mean() + 2.0)  # ~1B Mi + 1B extra
    # Reconstrucción aproximada con longitudes variables (padded)
    approx_codes = np.array([c + [0]*(M_MAX - len(c)) for c in codes_var], dtype='uint8')
    Xhat = rq.decode(approx_codes)
    # Búsqueda aproximada con Flat sobre reconstrucciones
    idx = faiss.IndexFlatIP(d); idx.add(Xhat)
    _, pred = idx.search(Q, 10)
    rec = recall_at_k(GT, pred)
    return avg_bytes, rec, M_i, Xhat

# ---------------------------
# 5) Barrido de alphas
# ---------------------------
results = []
for a in ALPHAS:
    print(f"\n--- MVRQ con α={a} ---")
    b, r, Mi, Xhat = run_mvrq(a)
    print(f"α={a} → Avg Bytes: {b:.2f}, Recall@10: {r:.4f}")
    results.append((a, b, r, Mi, Xhat))

# ---------------------------
# 6) Reporte + Gráficos
# ---------------------------
# Tabla
print("\nResumen:")
print(f"{'alpha':>6} | {'Bytes':>7} | {'Recall@10':>9}")
for a,b,r,_,_ in results:
    print(f"{a:>6.2f} | {b:>7.2f} | {r:>9.4f}")
print(f"{'RQ-8':>6} | {bytes_rq:>7.2f} | {recall_rq:>9.4f}  (baseline)")

# Curva Bytes vs Recall
plt.figure(figsize=(7,5))
plt.plot([x[1] for x in results], [x[2] for x in results], 'o-', label=f'MVRQ (W={W})')
plt.scatter([bytes_rq], [recall_rq], c='r', marker='x', s=120, label='RQ-8 baseline')
plt.xlabel('Avg Bytes / Vector')
plt.ylabel('Recall@10')
plt.title('Bytes vs Recall@10 — MVRQ (variance-based predictor)')
plt.grid(True); plt.legend()
plt.show()

# Correlación predictor vs error (para último alpha)
errors = np.linalg.norm(X - results[-1][4], axis=1)
corr = np.corrcoef(g, errors)[0,1]
print(f"\nCorrelation(g, residual error) = {corr:.4f}")

plt.figure(figsize=(6,5))
sample = np.random.choice(N, size=min(5000, N), replace=False)
plt.scatter(g[sample], errors[sample], s=2, alpha=0.3)
plt.xlabel("g(x) predictor (normalized)"); plt.ylabel("Residual error (last α)")
plt.title("Predictor vs Error")
plt.grid(True)
plt.show()

# Distribución de longitudes M_i (para cada α)
plt.figure(figsize=(7,5))
for a,_,_,Mi,_ in results:
    hist, bins = np.histogram(Mi, bins=np.arange(1, M_MAX+2))
    plt.plot(bins[:-1], hist / hist.sum(), '-o', label=f'α={a}')
plt.xlabel("Mi (stages)"); plt.ylabel("Frequency")
plt.title("Distribution of used stages (M_i)")
plt.legend(); plt.grid(True)
plt.show()
