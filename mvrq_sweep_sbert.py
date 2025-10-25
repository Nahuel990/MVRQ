import numpy as np
import faiss
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================================================================
# 0️⃣ Configuración
# ================================================================
N = 100_000      # número de documentos
Qn = 2_000       # número de queries
M_MAX = 8        # etapas RQ
NBITS = 8        # bits/etapa
KM = 512         # clusters k-means
W = 0.6          # mezcla varianza/knn/norma
ALPHAS = [0.4, 0.6, 0.8, 1.0]
SEED = 42
np.random.seed(SEED)
faiss.omp_set_num_threads(4)

# ================================================================
# 1️⃣ Generar embeddings reales (SBERT)
# ================================================================
print("Cargando modelo SBERT (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [f"This is document number {i}" for i in range(N)]
X = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype('float32')
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
d = X.shape[1]

Q = X[np.random.choice(N, size=Qn, replace=False)] + 0.05 * np.random.randn(Qn, d).astype('float32')
Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
print(f"Embeddings listos: X={X.shape}, Q={Q.shape}")

# ================================================================
# 2️⃣ Entrenar RQ baseline
# ================================================================
rq = faiss.ResidualQuantizer(d, M_MAX, NBITS)
perm = np.random.permutation(len(X))[:min(50_000, len(X))]
rq.train(X[perm])
codes_fixed = rq.compute_codes(X)
print("RQ-8 entrenado.")

centroids = rq.get_train_centroids()
K = centroids.shape[0] // M_MAX
C = centroids.reshape(M_MAX, K, d).copy()
print(f"Codebooks extraídos correctamente: {C.shape}")

# Recall@10 exacto
def recall_at_k(gt, pred, k=10):
    inter = np.array([len(set(gt[i]) & set(pred[i])) for i in range(len(gt))])
    return (inter / k).mean()

# ground truth
gt_index = faiss.IndexFlatIP(d)
gt_index.add(X)
_, GT = gt_index.search(Q, 10)

# baseline RQ-8
Xhat_rq = rq.decode(codes_fixed)
idx_rq = faiss.IndexFlatIP(d); idx_rq.add(Xhat_rq)
_, pred_rq = idx_rq.search(Q, 10)
recall_rq = recall_at_k(GT, pred_rq)
bytes_rq = M_MAX + 2
print(f"Baseline RQ-8 → Bytes: {bytes_rq}, Recall@10: {recall_rq:.4f}")

# ================================================================
# 3️⃣ Predictor combinado g(x): varianza + densidad + norma
# ================================================================
print("Entrenando k-means para predictor de varianza...")
kmeans = faiss.Kmeans(d, KM, niter=10, verbose=False)
kmeans.train(X[perm])
_, assign = kmeans.index.search(X, 1)
assign = assign.ravel()

cluster_var = np.zeros(KM, dtype='float32')
for c in tqdm(range(KM), desc="Calculando varianza intra-cluster"):
    idx = np.where(assign == c)[0]
    if len(idx) > 1:
        Xc = X[idx]
        mu = Xc.mean(axis=0, keepdims=True)
        cluster_var[c] = np.mean(np.linalg.norm(Xc - mu, axis=1))
    else:
        cluster_var[c] = 0.0
cluster_var = (cluster_var - cluster_var.min()) / (np.ptp(cluster_var) + 1e-9)
g_var = cluster_var[assign]

# densidad local (10-NN en subset)
print("Calculando densidad local (10-NN)...")
subN = min(20_000, N)
index_knn = faiss.IndexHNSWFlat(d, 32)
index_knn.add(X[:subN])
D, _ = index_knn.search(X, 11)
g_knn = D[:,1:].mean(axis=1)
g_knn = (g_knn - g_knn.min()) / (np.ptp(g_knn) + 1e-9)

# norma
g_norm = np.linalg.norm(X, axis=1)
g_norm = (g_norm - g_norm.min()) / (np.ptp(g_norm) + 1e-9)

# mezcla
g = (0.6*g_var + 0.3*g_knn + 0.1*g_norm).astype('float32')
g = np.clip(g, np.percentile(g, 1), np.percentile(g, 99))
g = (g - g.min()) / (np.ptp(g) + 1e-9)
print("Predictor g(x) listo.")

# ================================================================
# 4️⃣ Codificación MVRQ
# ================================================================
def run_mvrq(alpha):
    global C
    M_i = np.zeros(N, dtype=np.uint8)
    codes_var = np.zeros((N, M_MAX), dtype='uint8')

    # Reconstruir codebooks a partir del buffer plano de FAISS
    all_cb = np.array(rq.codebooks).reshape(M_MAX, -1, d)  # (M, K, d)
    C = [all_cb[m] for m in range(M_MAX)]

    for i in tqdm(range(N), desc=f"α={alpha} codificando"):
        r = X[i].copy()
        used = 0
        for m in range(M_MAX):
            sims = np.dot(C[m], r)
            idx = int(np.argmax(sims))
            codes_var[i, m] = idx
            r = r - C[m][idx]
            used += 1
            if np.linalg.norm(r) <= alpha * g[i]:
                break
        M_i[i] = used

    avg_bytes = M_i.mean() + 2.0

    # Reconstrucción
    Xhat = np.zeros_like(X)
    for i in range(N):
        for m in range(M_i[i]):
            Xhat[i] += C[m][codes_var[i, m]]

    idx = faiss.IndexFlatIP(d)
    idx.add(Xhat)
    _, pred = idx.search(Q, 10)
    rec = recall_at_k(GT, pred)

    return avg_bytes, rec, M_i, Xhat



# ================================================================
# 5️⃣ Barrido α
# ================================================================
results = []
for a in ALPHAS:
    print(f"\n--- MVRQ α={a} ---")
    b, r, Mi, Xhat = run_mvrq(a)
    print(f"α={a} → Bytes: {b:.2f}, Recall@10: {r:.4f}")
    results.append((a, b, r, Mi, Xhat))

# ================================================================
# 6️⃣ Gráficos y correlaciones
# ================================================================
print("\nResumen:")
print(f"{'alpha':>6} | {'Bytes':>7} | {'Recall@10':>9}")
for a,b,r,_,_ in results:
    print(f"{a:>6.2f} | {b:>7.2f} | {r:>9.4f}")
print(f"{'RQ-8':>6} | {bytes_rq:>7.2f} | {recall_rq:>9.4f}  (baseline)")

plt.figure(figsize=(7,5))
plt.plot([x[1] for x in results], [x[2] for x in results], 'o-', label=f'MVRQ (real embeddings)')
plt.scatter([bytes_rq], [recall_rq], c='r', marker='x', s=120, label='RQ-8 baseline')
plt.xlabel('Avg Bytes / Vector')
plt.ylabel('Recall@10')
plt.title('Bytes vs Recall@10 — MVRQ on SBERT embeddings')
plt.grid(True); plt.legend()
plt.show()

errors = np.linalg.norm(X - results[-1][4], axis=1)
corr = np.corrcoef(g, errors)[0,1]
print(f"\nCorrelation(g, residual error) = {corr:.4f}")

plt.figure(figsize=(6,5))
sample = np.random.choice(N, size=min(5000, N), replace=False)
plt.scatter(g[sample], errors[sample], s=2, alpha=0.3)
plt.xlabel("g(x) predictor (normalized)")
plt.ylabel("Residual error (last α)")
plt.title("Predictor vs Error — SBERT embeddings")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
for a,_,_,Mi,_ in results:
    hist, bins = np.histogram(Mi, bins=np.arange(1, M_MAX+2))
    plt.plot(bins[:-1], hist / hist.sum(), '-o', label=f'α={a}')
plt.xlabel("Mi (stages)")
plt.ylabel("Frequency")
plt.title("Distribution of used stages (M_i)")
plt.legend(); plt.grid(True)
plt.show()
exit