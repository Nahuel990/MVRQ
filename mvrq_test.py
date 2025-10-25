import numpy as np
import faiss
from tqdm import tqdm

# ==============================
# 1️⃣ Generar embeddings sintéticos
# ==============================
N = 100_000   # número de vectores (documentos)
d = 256       # dimensión (más chico que 1024 para test rápido)
np.random.seed(42)

X = np.random.randn(N, d).astype('float32')
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

# queries = subconjunto con leve ruido
Q = X[np.random.choice(N, size=2000, replace=False)] + 0.05 * np.random.randn(2000, d).astype('float32')
Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

print(f"Dataset listo: X={X.shape}, Q={Q.shape}")

# ==============================
# 2️⃣ Entrenar RQ baseline (M=8)
# ==============================
M_max, nbits = 8, 8
rq = faiss.ResidualQuantizer(d, M_max, nbits)
perm = np.random.permutation(len(X))[:min(50000, len(X))]
rq.train(X[perm])

codes_fixed = rq.compute_codes(X)
print("RQ baseline entrenado y codificado.")

# ==============================
# 3️⃣ Calcular predictor g(x): densidad local (10-NN)
# ==============================
index = faiss.IndexHNSWFlat(d, 32)
index.add(X[:20000])  # usa subset para rapidez
D, _ = index.search(X, 11)
g = D[:,1:].mean(axis=1).astype('float32')  # promedio distancia 10-NN
print("Predictor g(x) calculado.")

# ==============================
# 4️⃣ Codificación variable (MVRQ)
# ==============================
alpha = 0.5
M_i = np.zeros(len(X), dtype=np.uint8)
codes_var = []

for i in tqdm(range(len(X)), desc="Codificando MVRQ"):
    r = X[i].copy()
    codes_i = []
    for m in range(M_max):
        # FAISS no tiene assign_residual, aproximamos usando compute_codes
        # (no ideal, pero sirve para probar early stop)
        tmp = rq.compute_codes(r.reshape(1,-1))
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

avg_bytes = M_i.mean() + 2  # overhead aproximado
print(f"Promedio bytes/vector ≈ {avg_bytes:.2f}")

# ==============================
# 5️⃣ Evaluar Recall@10 con FAISS Flat rerank
# ==============================
index_fp32 = faiss.IndexFlatIP(d)
index_fp32.add(X)
_, gt = index_fp32.search(Q, 10)  # ground truth

# reconstruir aproximado con RQ-8
Xhat = rq.decode(codes_fixed)
index_rq = faiss.IndexFlatIP(d)
index_rq.add(Xhat)
_, pred_rq = index_rq.search(Q, 10)

# reconstruir aproximado con MVRQ (usando M_i promedio)
approx_codes = np.array([c + [0]*(M_max - len(c)) for c in codes_var], dtype='uint8')
Xhat_mvrq = rq.decode(approx_codes)
index_mvrq = faiss.IndexFlatIP(d)
index_mvrq.add(Xhat_mvrq)
_, pred_mvrq = index_mvrq.search(Q, 10)

def recall_at_k(gt, pred, k=10):
    inter = np.array([len(set(gt[i]) & set(pred[i])) for i in range(len(gt))])
    return (inter / k).mean()

r_rq = recall_at_k(gt, pred_rq)
r_mvrq = recall_at_k(gt, pred_mvrq)
print(f"Recall@10 baseline RQ-8:  {r_rq:.4f}")
print(f"Recall@10 MVRQ (α={alpha}): {r_mvrq:.4f}")

print("Done ✅")
