import numpy as np
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================
# 1️⃣ Generar embeddings sintéticos
# =========================================
N = 100_000
d = 256
np.random.seed(42)

X = np.random.randn(N, d).astype('float32')
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
Q = X[np.random.choice(N, size=2000, replace=False)] + 0.05 * np.random.randn(2000, d).astype('float32')
Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
print(f"Dataset listo: X={X.shape}, Q={Q.shape}")

# =========================================
# 2️⃣ Entrenar RQ baseline (M=8)
# =========================================
M_max, nbits = 8, 8
rq = faiss.ResidualQuantizer(d, M_max, nbits)
perm = np.random.permutation(len(X))[:min(50000, len(X))]
rq.train(X[perm])

codes_fixed = rq.compute_codes(X)
print("RQ baseline entrenado y codificado.")

# =========================================
# 3️⃣ Calcular predictor g(x) = densidad local (10-NN)
# =========================================
index = faiss.IndexHNSWFlat(d, 32)
index.add(X[:20000])  # usa subset
D, _ = index.search(X, 11)
g = D[:, 1:].mean(axis=1).astype('float32')
print("Predictor g(x) calculado.")

# =========================================
# 4️⃣ Funciones auxiliares
# =========================================
def recall_at_k(gt, pred, k=10):
    inter = np.array([len(set(gt[i]) & set(pred[i])) for i in range(len(gt))])
    return (inter / k).mean()

# ground truth (Flat exacto)
index_fp32 = faiss.IndexFlatIP(d)
index_fp32.add(X)
_, gt = index_fp32.search(Q, 10)

def run_mvrq(alpha):
    """Codifica con umbral dinámico α*g(x)"""
    M_i = np.zeros(len(X), dtype=np.uint8)
    codes_var = []
    for i in range(len(X)):
        r = X[i].copy()
        codes_i = []
        for m in range(M_max):
            tmp = rq.compute_codes(r.reshape(1, -1))
            idx = int(tmp[0, m])
            codes_i.append(idx)
            decoded = rq.decode(np.array([codes_i + [0]*(M_max-len(codes_i))], dtype='uint8'))[0]
            r = X[i] - decoded
            if np.linalg.norm(r) < alpha * g[i]:
                M_i[i] = m + 1
                break
        if M_i[i] == 0:
            M_i[i] = M_max
        codes_var.append(codes_i[:M_i[i]])
    avg_bytes = M_i.mean() + 2
    # reconstruir aproximado
    approx_codes = np.array([c + [0]*(M_max - len(c)) for c in codes_var], dtype='uint8')
    Xhat = rq.decode(approx_codes)
    index_mvrq = faiss.IndexFlatIP(d)
    index_mvrq.add(Xhat)
    _, pred = index_mvrq.search(Q, 10)
    r = recall_at_k(gt, pred)
    return avg_bytes, r, M_i, Xhat

# =========================================
# 5️⃣ Evaluar varios α
# =========================================
alphas = [0.25, 0.5, 0.75, 1.0]
results = []

for a in alphas:
    print(f"\n--- Ejecutando MVRQ con α={a} ---")
    bytes_avg, recall, M_i, Xhat = run_mvrq(a)
    print(f"α={a} → Bytes promedio: {bytes_avg:.2f}, Recall@10: {recall:.4f}")
    results.append((a, bytes_avg, recall, M_i, Xhat))

# baseline RQ-8 para comparar
Xhat_rq = rq.decode(codes_fixed)
index_rq = faiss.IndexFlatIP(d)
index_rq.add(Xhat_rq)
_, pred_rq = index_rq.search(Q, 10)
recall_rq = recall_at_k(gt, pred_rq)
bytes_rq = M_max + 2
print(f"\nBaseline RQ-8 → Bytes: {bytes_rq}, Recall@10: {recall_rq:.4f}")

# =========================================
# 6️⃣ Graficar resultados
# =========================================
plt.figure(figsize=(7,5))
plt.plot([r[1] for r in results], [r[2] for r in results], 'o-', label='MVRQ')
plt.scatter([bytes_rq], [recall_rq], color='r', marker='x', s=100, label='RQ-8 Baseline')
plt.xlabel("Avg Bytes / Vector")
plt.ylabel("Recall@10")
plt.title("Bytes vs Recall@10 — MVRQ Sweep")
plt.legend()
plt.grid(True)
plt.show()

# =========================================
# 7️⃣ Correlación predictor vs error (α=1.0)
# =========================================
errors = np.linalg.norm(X - results[-1][4], axis=1)
corr = np.corrcoef(g, errors)[0, 1]
print(f"\nCorrelación(g, error residual) = {corr:.4f}")

plt.figure(figsize=(6,5))
plt.scatter(g[:5000], errors[:5000], s=2, alpha=0.3)
plt.xlabel("g(x) (densidad local)")
plt.ylabel("Error residual (α=1.0)")
plt.title("Correlación Predictor vs Error")
plt.show()
