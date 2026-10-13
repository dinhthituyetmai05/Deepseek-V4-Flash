# Kiến trúc chi tiết DeepSeek-V4: Layer, sublayer, tensor, ví dụ số cụ thể

**Tài liệu nội bộ — PGS.TS Nguyễn Đức Minh (HUST/SEEE)**
Nguồn: DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026 [1] (file `References/DeepSeek_V4.pdf`).
Cấu hình tham chiếu chính trong tài liệu: **DeepSeek-V4-Flash** (284B tổng / 13B kích hoạt); bảng so sánh với **DeepSeek-V4-Pro** (1.6T / 49B) cho ở cuối mỗi mục.

## Mục lục

1. Tổng quan kiến trúc
2. Bảng hyperparameter
3. Layer 0 — Token Embedding
4. Manifold-Constrained Hyper-Connections (mHC)
5. Khối Transformer thứ `l` — cấu trúc chung
6. Sublayer Attention — CSA (Compressed Sparse Attention)
7. Sublayer Attention — HCA (Heavily Compressed Attention)
8. Sublayer Attention — Sliding Window Attention (SWA) đầu mạng
9. Nhánh phụ Sliding Window trong CSA/HCA, Attention Sink, RoPE bán phần
10. Sublayer FFN — DeepSeekMoE
11. Output: RMSNorm + Prediction Head + MTP module
12. Kiểm chứng tổng số tham số
13. Tóm tắt ví dụ số end-to-end (n=4096)
14. Tài liệu tham khảo

---

## 1. Tổng quan kiến trúc

DeepSeek-V4 vẫn dùng cấu trúc Transformer (Vaswani et al., 2017) [2] + module Multi-Token Prediction (MTP) như V3, nhưng thay đổi ba thành phần chính so với V3 [1, §2]:

1. **mHC** (Manifold-Constrained Hyper-Connections) thay residual chuẩn.
2. **Hybrid attention**: CSA và HCA xen kẽ, thay cho MLA của V3.
3. **Muon optimizer** thay AdamW cho phần lớn module.

Sơ đồ tổng thể (Figure 2 trong [1]):

```
token_ids → Embedding → [mHC residual mở rộng width n_hc]
   ┌────────── L Transformer Blocks ──────────┐
   │  Pre-Block Mixing (mHC)                  │
   │  Attention sublayer (SWA / CSA / HCA)    │
   │  Residual Mixing (mHC)                   │
   │  Post-Block Mixing (mHC)                 │
   │  DeepSeekMoE FFN sublayer                │
   │  Residual Mixing (mHC)                   │
   └──────────────────────────────────────────┘
→ Final RMSNorm → Prediction Head → LM Loss
                ↘ MTP Module(s) → MTP Loss
```

Quy ước ký hiệu xuyên suốt tài liệu:

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | Batch size (token-level) |
| `n` | Sequence length |
| `d` | Hidden size |
| `V` | Vocabulary size = 128 000 [1, §4.1] |
| `L` | Số block Transformer |
| `n_hc` | Hệ số mở rộng residual của mHC |
| `c` | Head dimension attention |
| `d_c` | Query compression dim của low-rank query |
| `n_h` | Số query head |
| `n_I_h` | Số indexer query head |
| `c_I` | Indexer head dim |
| `m, m'` | Compression rate của CSA, HCA |
| `g, d_g` | Số nhóm và dim của grouped output projection |
| `n_win` | Sliding window size |
| `d_ff` | Expert intermediate dim |
| `n_routed`, `n_act` | Số routed expert / số expert kích hoạt mỗi token |

Ví dụ số cụ thể trong các mục dưới đây dùng:

```
Flash:  B = 1,  n = 4096,  d = 4096
Pro:    B = 1,  n = 4096,  d = 7168
```

## 2. Bảng hyperparameter

Trích trực tiếp từ [1, §4.2.1].

| Tham số | V4-Flash | V4-Pro |
|---|---:|---:|
| L (số block) | **43** | **61** |
| d (hidden) | **4096** | **7168** |
| Vocab V | 128 000 | 128 000 |
| Layer 1–2 attention | Pure SWA | Pure HCA |
| Layer ≥ 3 | CSA ↔ HCA xen kẽ | CSA ↔ HCA xen kẽ |
| CSA: m | 4 | 4 |
| CSA: n_I_h (indexer heads) | 64 | 64 |
| CSA: c_I (indexer head dim) | 128 | 128 |
| CSA: top-k | **512** | **1024** |
| HCA: m' | 128 | 128 |
| Cả CSA & HCA: n_h | **64** | **128** |
| Cả CSA & HCA: c (head dim) | 512 | 512 |
| Cả CSA & HCA: d_c (query compress) | 1024 | 1536 |
| Output groups g | 8 | 16 |
| d_g (per-group intermediate) | 1024 | 1024 |
| Sliding window n_win | 128 | 128 |
| MoE per layer | 1 shared + 256 routed | 1 shared + 384 routed |
| Expert d_ff | 2048 | 3072 |
| Routed top-k | 6 | 6 |
| Hash routing (3 layer đầu MoE) | có | có |
| MTP depth | 1 | 1 |
| mHC n_hc | 4 | 4 |
| Sinkhorn–Knopp t_max | 20 | 20 |
| RoPE | 64 dims cuối | 64 dims cuối |
| KV cache: dtype | FP8 (non-RoPE) + BF16 (RoPE 64 dims) | như Flash |
| Tổng param | **284 B** | **1.6 T** |
| Activated param/token | **13 B** | **49 B** |
| Pre-train tokens | 32 T | 33 T |
| Max sequence length | 1 M | 1 M |

## 3. Layer 0 — Token Embedding

**Mô tả:** Tra cứu vector embedding cho `B·n` token. Không có RoPE ở đây; RoPE được áp dụng cục bộ trong từng sublayer attention.

**Tensor**

| Tensor | Shape | Flash (ví dụ) | Pro |
|---|---|---|---|
| Input `token_ids` | `(B, n)` ∈ ℤ | `(1, 4096)` int32 | `(1, 4096)` |
| Weight `W_emb` | `(V, d)` | `(128000, 4096)` → **524 288 000 ≈ 524.3 M** params | `(128000, 7168)` → **917.5 M** |
| Output `H_0` | `(B, n, d)` | `(1, 4096, 4096)`, BF16 → **128 MiB** | `(1, 4096, 7168)` |

DeepSeek-V3 dùng BF16/FP8 hỗn hợp cho activation; V4 giữ pattern này [1, §2.3.4].

## 4. Manifold-Constrained Hyper-Connections (mHC)

mHC mở rộng residual stream từ `(d,)` thành `(n_hc, d)` cho mỗi token rồi giới hạn ma trận trộn `B_l` trên đa tạp Birkhoff (doubly stochastic) [1, §2.2].

### 4.1 Mở rộng residual lần đầu

Sau Embedding ta cần khởi tạo `X_0 ∈ ℝ^(B, n, n_hc, d)`.

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `H_0` | `(B, n, d)` | `(1, 4096, 4096)` | `(1, 4096, 7168)` |
| `X_0` | `(B, n, n_hc, d)` | `(1, 4096, 4, 4096)` → **512 MiB** BF16 | `(1, 4096, 4, 7168)` → **896 MiB** |

Trong cài đặt thực, mỗi vị trí token mang một mini-tensor `(n_hc, d) = (4, 4096)`.

### 4.2 Tham số mHC tại mỗi sublayer

Với mỗi block Transformer có hai sublayer (Attention và MoE) ⇒ hai khối mHC. Tại sublayer thứ `l`:

**Bước 1 — RMSNorm trên residual đã flatten:**

```
X_l ∈ ℝ^(n_hc, d) = ℝ^(4, 4096)  → vec(X_l) ∈ ℝ^(1, n_hc·d) = ℝ^(1, 16384)
X̂_l = RMSNorm(vec(X_l)) ∈ ℝ^(1, 16384)
```

**Bước 2 — Sinh tham số động (Equations 3–5 trong [1]):**

| Weight | Shape | Flash (số phần tử) | Pro |
|---|---|---:|---:|
| `W_pre_l` | `(n_hc·d, n_hc)` | `(16384, 4)` → 65 536 | `(28672, 4)` |
| `W_res_l` | `(n_hc·d, n_hc²)` | `(16384, 16)` → 262 144 | `(28672, 16)` |
| `W_post_l` | `(n_hc·d, n_hc)` | `(16384, 4)` → 65 536 | `(28672, 4)` |
| Static bias `S_pre, S_post, S_res` | `(n_hc,) (n_hc,) (n_hc,n_hc)` | 4+4+16 | tương tự |
| Gate `α_pre, α_res, α_post` | scalar × 3 | 3 | 3 |

Tổng params mHC một sublayer ≈ **393 K** với Flash (rất nhỏ so với attention/MoE).

**Bước 3 — Áp ràng buộc (Eq. 6–8):**

```
A_l = σ(Ã_l) ∈ ℝ^(1, n_hc) = ℝ^(1, 4),   0 < A_l < 1
C_l = 2σ(C̃_l) ∈ ℝ^(n_hc, 1) = ℝ^(4, 1), 0 < C_l < 2
M⁽⁰⁾ = exp(B̃_l) ∈ ℝ^(4, 4)
M⁽ᵗ⁾ = 𝒯_r(𝒯_c(M⁽ᵗ⁻¹⁾)),  t = 1..20      # Sinkhorn–Knopp
B_l = M⁽²⁰⁾ ∈ Birkhoff polytope ℝ^(4, 4)
```

**Bước 4 — Cập nhật residual (Eq. 1):**

```
input_to_sublayer = A_l · X_l ∈ ℝ^(1, d) = ℝ^(1, 4096)         # đầu vào của F_l
output_of_sublayer = F_l(A_l · X_l) ∈ ℝ^(1, d) = ℝ^(1, 4096)
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l) ∈ ℝ^(4, 4096)
```

**Ý nghĩa với phần cứng:** mỗi token cần lưu residual `4 × 4096 × 2 B = 32 KiB`. Với `n=4096`, residual buffer = 128 MiB (BF16). Việc gating `A_l`, `B_l`, `C_l` thay đổi mỗi sublayer ⇒ FPGA cần một small-MAC engine 16×16 cho phép nhân Birkhoff matrix + scalar pipeline cho Sinkhorn–Knopp (20 iter).

## 5. Khối Transformer thứ `l` — cấu trúc chung

Mỗi block có 2 sublayer + 4 lần mHC mixing (Pre/Post cho mỗi sublayer):

```
X_l ─┬── (Pre-Block Mixing mHC_attn)  → input_a ── F_attn(·) → out_a
     │                                                   │
     ├── (Residual Mixing)  X_l ← B_attn X_l + C_attn out_a
     │
     ├── (Post-Block Mixing mHC_moe) → input_m ── F_moe(·) → out_m
     │                                                  │
     └── (Residual Mixing)  X_{l+1} ← B_moe X_l + C_moe out_m
```

Với Flash (L=43): block 1–2 dùng SWA, block 3–43 luân phiên CSA / HCA bắt đầu từ CSA (paper không khoá thứ tự cứng, nhưng tỷ lệ ~50/50). Với Pro (L=61): block 1–2 dùng HCA, các block sau xen kẽ.

## 6. Sublayer Attention — Compressed Sparse Attention (CSA)

CSA = "nén KV với hệ số `m`" + "DeepSeek Sparse Attention chọn top-k block đã nén" + "MQA chia sẻ K/V" + "Grouped Output Projection" [1, §2.3.1].

Ví dụ minh hoạ dưới đây giả định một query token mới `t` với `n` token đã sinh, tức `n=4096` token đã có trong KV cache (prefill xong, decode đang chạy ở vị trí `t`).

### 6.1 Nén Key-Value (Eq. 9–12)

**Input**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `H` (hidden states của n token) | `(n, d)` | `(4096, 4096)` BF16 = 32 MiB | `(4096, 7168)` = 56 MiB |
| Hidden state query token `h_t` | `(d,)` | `(4096,)` | `(7168,)` |

**Weight cho KV-compression (4 ma trận)**

| Weight | Shape | Flash params | Pro params |
|---|---|---:|---:|
| `W_aKV` | `(d, c)` | `(4096, 512)` → 2.10 M | `(7168, 512)` → 3.67 M |
| `W_bKV` | `(d, c)` | 2.10 M | 3.67 M |
| `W_aZ` | `(d, c)` | 2.10 M | 3.67 M |
| `W_bZ` | `(d, c)` | 2.10 M | 3.67 M |
| Positional bias `B_a, B_b` | `(m, c)` | `(4, 512)` → 2 048 mỗi cái | `(4, 512)` |

**Hidden activations sau projection**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `C_a = H · W_aKV` | `(n, c)` | `(4096, 512)` = 4 MiB BF16 | `(4096, 512)` |
| `C_b = H · W_bKV` | `(n, c)` | `(4096, 512)` | `(4096, 512)` |
| `Z_a`, `Z_b` | `(n, c)` | `(4096, 512)` | `(4096, 512)` |

**Phép nén overlapped (m=4)** — softmax trên `2m=8` phần tử của `[Z_a | Z_b]`:

```
S_a_{mi..m(i+1)-1}, S_b_{m(i-1)..mi-1}
   = Softmax_row( [Z_a_{mi..m(i+1)-1} + B_a ;
                   Z_b_{m(i-1)..mi-1} + B_b] )      # ℝ^(8, 512)

C^Comp_i = Σ_{j=mi}^{m(i+1)-1} S_a_j ⊙ C_a_j
         + Σ_{j=m(i-1)}^{mi-1} S_b_j ⊙ C_b_j           # ∈ ℝ^c
```

Mỗi block nén `C^Comp_i` lấy nguyên liệu từ **2m = 8** KV entry. Tổng số block sau nén:

| Đại lượng | Flash | Pro |
|---|---|---|
| Số block nén = `n/m` | `4096/4 = 1024` | `1024` |
| `C^Comp` shape | `(1024, 512)` = 1 MiB BF16 | `(1024, 512)` |
| Tỷ lệ giảm KV | 4× | 4× |

KV-cache **non-RoPE** lưu FP8 (mỗi entry 512 dim, 64 dim cuối là RoPE giữ BF16). Bộ nhớ KV cache CSA cho 1 layer ở `n=1M` (token đã sinh): `(1M/4) × (448 × 1 B + 64 × 2 B) = 250 K × 576 B ≈ 144 MB / layer`.

### 6.2 Lightning Indexer (Eq. 13–17)

**Weight**

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_DQ` (down-proj query) | `(d, d_c)` | `(4096, 1024)` → 4.19 M | `(7168, 1536)` → 11.01 M |
| `W_IUQ` (up-proj indexer query) | `(d_c, c_I·n_I_h)` | `(1024, 128·64)=(1024, 8192)` → 8.39 M | `(1536, 8192)` → 12.58 M |
| `W_w` (head weight) | `(d, n_I_h)` | `(4096, 64)` → 0.26 M | `(7168, 64)` → 0.46 M |

**Activation (cho 1 query token)**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `c_Q_t = h_t · W_DQ` | `(d_c,)` | `(1024,)` | `(1536,)` |
| `q_I_t` | `(n_I_h, c_I)` | `(64, 128)` | `(64, 128)` |
| `w_I_t = h_t · W_w` | `(n_I_h,)` | `(64,)` | `(64,)` |
| `K^IComp` (đã được nén tương tự `C^Comp` nhưng dim `c_I`) | `(n/m, c_I)` | `(1024, 128)` | `(1024, 128)` |
| Index score `I_{t,:}` | `(⌊t/m⌋,)` | tối đa `(1024,)` | `(1024,)` |

**Phép tính score** (Eq. 16):

```
I_{t,s} = Σ_{h=1}^{n_I_h} w_I_{t,h} · ReLU( q_I_{t,h} · K^IComp_s )
        ∈ ℝ                                # 64 head × dot 128 dims × ReLU × weight sum
```

**Precision:** indexer chạy **FP4** ⇒ tăng tốc đáng kể với n lớn [1, §2.3.4].

**Top-k selection** (Flash: k=512, Pro: k=1024):

```
𝒞^SprsComp_t = { C^Comp_s : I_{t,s} ∈ Top-k(I_{t,:}) }
```

| Đại lượng | Flash | Pro |
|---|---|---|
| Số candidate | n/m = 1024 | 1024 |
| Top-k giữ lại | 512 | 1024 (giữ hết, nhưng vẫn dùng cơ chế) |
| Bộ KV sparse | `(512, 512)` BF16+FP8 hỗn hợp | `(1024, 512)` |

### 6.3 Core attention — MQA chia sẻ K/V (Eq. 18–19)

**Weight**

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_UQ` (up-proj query) | `(d_c, c·n_h)` | `(1024, 512·64)=(1024, 32768)` → 33.55 M | `(1536, 512·128)=(1536, 65536)` → 100.66 M |

**Activation**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `q_t = c_Q_t · W_UQ` | `(c·n_h,)` | `(32768,)` | `(65536,)` |
| Reshape `q_t` | `(n_h, c)` | `(64, 512)` | `(128, 512)` |
| Sau RMSNorm trên mỗi head | giữ nguyên | `(64, 512)` | `(128, 512)` |
| RoPE áp 64 dim cuối | partial | dims 448–511 quay | dims 448–511 quay |
| Sink logit `z'_h` | `(n_h,)` | `(64,)` learnable | `(128,)` |

**Attention** (per head h, query t):

```
Cho key/value Kₕ = Vₕ = 𝒞^SprsComp_t  ∈ ℝ^(512, c)   # CHIA SẺ giữa các head (MQA)
logits l_{h,t,j} = (qₕ,t · K_j) / √c                   # √512 ≈ 22.63
scores s_{h,t,j} = exp(l_{h,t,j}) / ( Σⱼ exp(l_{h,t,j}) + exp(z'_h) )   # attention sink
o_{h,t} = Σⱼ s_{h,t,j} · V_j                           # ∈ ℝ^c = ℝ^(512)
```

**Output 64 head:**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `o_t` (concat heads) | `(c·n_h,)` | `(32768,)` | `(65536,)` |

### 6.4 Grouped Output Projection (mục cuối §2.3.1)

Thay vì `W_O ∈ ℝ^(c·n_h, d) = (32768, 4096) ≈ 134 M params`, paper chia `n_h` thành `g` nhóm, mỗi nhóm projection nhỏ rồi cộng.

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_OG_i` (i=1..g) | `(c·n_h/g, d_g)` | `g=8` → mỗi cái `(4096, 1024)` = 4.19 M ⇒ 8×4.19 = 33.55 M | `g=16` → mỗi cái `(4096, 1024)` = 4.19 M ⇒ 16×4.19 = 67.11 M |
| `W_O` (final) | `(g·d_g, d)` | `(8192, 4096)` = 33.55 M | `(16384, 7168)` = 117.44 M |

**Activation theo nhóm:**

```
n_h chia g nhóm: Flash 64/8 = 8 head/group → o^G_t,i ∈ ℝ^(c·n_h/g) = ℝ^4096
Mỗi nhóm:  o^G'_t,i = o^G_t,i · W_OG_i  ∈ ℝ^d_g = ℝ^1024
Concat:    [o^G'_t,1 ; … ; o^G'_t,g] ∈ ℝ^(g·d_g) = ℝ^8192
Final:     ô_t = concat · W_O ∈ ℝ^d = ℝ^4096
```

**Tổng params CSA (Flash, 1 layer):** ≈ **121.9 M** (đối chiếu mục 12).

### 6.5 So sánh tham số CSA per-layer

| Khối | Flash params | Pro params |
|---|---:|---:|
| KV compression (4 W) | 8.39 M | 14.68 M |
| Lightning Indexer (W_DQ + W_IUQ + W_w) | 12.85 M | 24.06 M |
| W_UQ | 33.55 M | 100.66 M |
| Grouped output (W_OG×g + W_O) | 67.11 M | 184.55 M |
| **Tổng CSA / layer** | **≈ 121.9 M** | **≈ 323.9 M** |

## 7. Sublayer Attention — Heavily Compressed Attention (HCA)

HCA giống CSA nhưng (i) nén mạnh hơn `m' = 128`, (ii) **không** dùng indexer + sparse selection, (iii) không có dạng overlapped (chỉ 1 nhánh `C`) [1, §2.3.2].

### 7.1 Nén Key-Value (Eq. 20–23)

**Weight**

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_KV` | `(d, c)` | `(4096, 512)` → 2.10 M | `(7168, 512)` → 3.67 M |
| `W_Z` | `(d, c)` | `(4096, 512)` → 2.10 M | `(7168, 512)` → 3.67 M |
| Pos bias `B` | `(m', c)` | `(128, 512)` → 65 536 | `(128, 512)` |

**Activation**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `C = H · W_KV` | `(n, c)` | `(4096, 512)` | `(4096, 512)` |
| `Z = H · W_Z` | `(n, c)` | `(4096, 512)` | `(4096, 512)` |
| `S = Softmax_row(Z + B)` | `(n, c)` | `(4096, 512)` | `(4096, 512)` |
| `C^Comp` (Eq. 23) | `(n/m', c)` | `(32, 512)` = **8 KiB BF16!** | `(32, 512)` |

**Tỷ lệ nén:** `m' = 128` ⇒ KV-cache HCA của Flash chỉ giữ `32 entries × 512 dim` cho `n=4096`. Ở `n=1M` ⇒ `7813 entries` ≈ **4.5 MB / layer** sau pack FP8 + BF16. Đây là điểm then chốt cho 1M context.

### 7.2 Query + Core Attention (Eq. 24–26)

**Weight**

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_DQ` | `(d, d_c)` | `(4096, 1024)` → 4.19 M | `(7168, 1536)` → 11.01 M |
| `W_UQ` | `(d_c, c·n_h)` | `(1024, 32768)` → 33.55 M | `(1536, 65536)` → 100.66 M |

**Activation**

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `c_Q_t = h_t · W_DQ` | `(d_c,)` | `(1024,)` | `(1536,)` |
| `q_t` reshape | `(n_h, c)` | `(64, 512)` | `(128, 512)` |
| Sau RMSNorm + partial RoPE | giữ nguyên | `(64, 512)` | `(128, 512)` |
| Core attention output `o_t` | `(c·n_h,)` | `(32768,)` | `(65536,)` |

**Core MQA:**

```
key = value = C^Comp ∈ ℝ^(32, c)     # CHỈ 32 entries cho n=4096!
o_{h,t,j} weighted as in CSA, vẫn với attention sink z'_h
```

### 7.3 Grouped Output Projection (giống CSA)

Tổng params HCA / layer (Flash): **≈ 109.05 M** (ít hơn CSA vì không có indexer).

### 7.4 Tóm tắt khác biệt CSA ↔ HCA

| Khía cạnh | CSA | HCA |
|---|---|---|
| Compression rate | m = 4 | m' = 128 |
| Overlapping `C_a, C_b` | Có (2 nhánh) | Không (1 nhánh) |
| Sparse top-k | Có (Lightning Indexer + Top-k) | Không |
| Số KV entry sau nén (n=4096) | 1024 | 32 |
| Số KV entry tham gia attention | 512 (top-k) + 128 (SW) = 640 | 32 + 128 (SW) = 160 |
| Mục đích | local-medium context | long-context (giảm cache cực mạnh) |

## 8. Sublayer Attention — Sliding Window Attention (SWA) ở layer đầu (Flash)

Layer 1–2 của Flash dùng SWA "thuần" (không nén, không sparse). Mặc dù paper không nói chi tiết shape, hệ thống đối xứng các thiết kế V3-style nên:

- Window size `n_win = 128`.
- Query/Key/Value đầy đủ với `n_h = 64`, `c = 512`.
- KV cache trượt (cache vòng) cho 128 token gần nhất.

**Tensor minh hoạ (Flash, n=4096 trong batch, decode token t):**

| Tensor | Shape | Số phần tử |
|---|---|---|
| `Q_t` | `(n_h, c)` | `(64, 512)` |
| `K_t` history | `(n_win, c)` | `(128, 512)` |
| `V_t` history | `(n_win, c)` | `(128, 512)` |
| Attention logits | `(n_h, n_win)` | `(64, 128)` |

V4-Pro KHÔNG dùng SWA cho layer đầu — Pro dùng **pure HCA** cho 2 layer đầu để giữ khả năng nhìn xa ngay từ đầu mạng [1, §4.2.1].

## 9. Nhánh phụ Sliding Window, Attention Sink, RoPE bán phần

### 9.1 Nhánh phụ Sliding Window (cho cả CSA và HCA)

Để bảo toàn quan hệ cục bộ trong block đang xét, mỗi query bổ sung `n_win=128` KV entry KHÔNG nén của các token gần nhất [1, §2.3.3]:

```
Tổng KV entries cho attention CSA (Flash):  512 (top-k nén) + 128 (SW) = 640
Tổng KV entries cho attention HCA (Flash):   32  (compressed)+ 128 (SW) = 160
```

KV bổ sung này dùng cùng `c = 512` head dim, BF16 cho 64 RoPE dims + FP8 cho 448 dims còn lại.

### 9.2 Attention Sink (Eq. 27)

Mỗi head có scalar learnable `z'_h`. Mẫu số softmax cộng thêm `exp(z'_h)` ⇒ score có thể nhỏ hơn 1 hoặc gần 0 (head có thể "im lặng"). Lưu trữ thêm: `n_h` scalar/head/layer.

| Số scalar sink | Flash | Pro |
|---|---|---|
| Per layer | 64 | 128 |
| Toàn mạng | 64 × 43 = 2 752 | 128 × 61 = 7 808 |

### 9.3 Partial RoPE 64 dims

- Áp lên **64 dim cuối** của mỗi `q_t,i`, `k_j` (k ở đây là `C^Comp_j` hoặc KV của SW).
- Áp **RoPE với vị trí âm `−i`** lên 64 dim cuối của `o_{t,i}` để khử bias vị trí tuyệt đối, biến output thành relative-PE.

Hệ quả lưu trữ: 64 dim cuối phải giữ BF16 (lý do FP8 không đủ chính xác cho rotation); 448 dim còn lại có thể FP8.

## 10. Sublayer FFN — DeepSeekMoE

### 10.1 Cấu trúc chung

Mỗi layer Transformer có 1 sublayer FFN = DeepSeekMoE [3]:

- 1 shared expert (luôn tính) + `n_routed` routed expert (Flash: 256, Pro: 384).
- Top-k = 6 routed expert kích hoạt cho mỗi token.
- Mỗi expert là một SwiGLU FFN: `y = (SiLU(x W_gate) ⊙ (x W_up)) · W_down`.
- V4 đổi hàm affinity từ Sigmoid(·) (V3) ⇒ **Sqrt(Softplus(·))** [1, §2.1].
- Load balancing: auxiliary-loss-free + sequence-wise balance loss (weight 0.0001).
- 3 layer MoE đầu tiên dùng **Hash routing** (router theo hash của token_ID) — đơn giản hoá communication ở các layer đầu.

### 10.2 Tham số một expert (SwiGLU)

| Weight | Shape | Flash params | Pro params |
|---|---|---:|---:|
| `W_gate` | `(d, d_ff)` | `(4096, 2048)` → 8 388 608 | `(7168, 3072)` → 22 020 096 |
| `W_up` | `(d, d_ff)` | `(4096, 2048)` → 8 388 608 | `(7168, 3072)` → 22 020 096 |
| `W_down` | `(d_ff, d)` | `(2048, 4096)` → 8 388 608 | `(3072, 7168)` → 22 020 096 |
| **Tổng / expert** | | **25 165 824 ≈ 25.17 M** | **66 060 288 ≈ 66.06 M** |

### 10.3 Tham số MoE per layer

| Đại lượng | Flash | Pro |
|---|---:|---:|
| Routed experts | 256 | 384 |
| Shared expert | 1 | 1 |
| Params / MoE layer | (256+1) × 25.17 M ≈ **6.47 B** | (384+1) × 66.06 M ≈ **25.43 B** |
| Tổng L MoE layer | 43 × 6.47 ≈ **278.1 B** (≈ 98 % tổng) | 61 × 25.43 ≈ **1551 B** (≈ 97 %) |

### 10.4 Router

**Weight router:**

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_router` | `(d, n_routed)` | `(4096, 256)` → 1.05 M | `(7168, 384)` → 2.75 M |
| Bias per-expert b_i (aux-loss-free) | `(n_routed,)` | 256 | 384 |

**Activation (1 token):**

```
g_raw = h · W_router ∈ ℝ^(n_routed)            # affinity raw
g̃ = Sqrt(Softplus(g_raw + b))                  # V4 thay vì Sigmoid V3
top6_idx = TopK(g̃, k=6)
weights = Softmax(g̃[top6_idx])                  # normalized
```

**Anticipatory routing:** routing index ở step `t` được tính từ tham số `θ_{t−Δt}` (cũ hơn 1 step) [1, §4.2.3]. Trick này giúp ổn định train; với inference engine FPGA, anticipatory routing không bắt buộc, nhưng pipeline router-1-step-ahead vẫn áp dụng tốt khi cần overlap dispatch/compute.

### 10.5 Tensor pipeline MoE (1 token)

| Bước | Tensor | Shape | Flash | Pro |
|---|---|---|---|---|
| 1 | Input `h` | `(d,)` | `(4096,)` | `(7168,)` |
| 2 | `g̃` raw | `(n_routed,)` | `(256,)` | `(384,)` |
| 3 | top-6 indices | `(6,)` int | `(6,)` | `(6,)` |
| 4 | top-6 gating weights | `(6,)` | `(6,)` | `(6,)` |
| 5a | Shared expert intermediate | `(d_ff,)` | `(2048,)` | `(3072,)` |
| 5b | Routed expert intermediate (mỗi) | `(d_ff,)` | `(2048,)` | `(3072,)` |
| 6 | SwiGLU `SiLU(h Wg) ⊙ (h Wu)` | `(d_ff,)` | `(2048,)` | `(3072,)` |
| 7 | Sau `W_down` | `(d,)` | `(4096,)` | `(7168,)` |
| 8 | Tổng hợp `y = e_shared(h) + Σ_{i=1..6} w_i · e_{r_i}(h)` | `(d,)` | `(4096,)` | `(7168,)` |

**SwiGLU Clamping** [1, §4.2.3]: linear component clip vào `[-10, 10]`, gate component cap `≤ 10`. Cần thực hiện trong activation pipeline FPGA.

### 10.6 Activated params/token cho MoE

- Per layer: `(1 shared + 6 routed) × 25.17 M = 176.16 M`.
- Toàn mạng: `43 × 176.16 M ≈ 7.58 B` (≈ 58 % tổng activated; còn lại từ attention + mHC + embedding).

Pro: `7 × 66.06 M × 61 = 28.21 B`.

### 10.7 Hash routing cho 3 layer đầu

3 layer đầu (đếm từ layer MoE đầu, tức layer 3–5 trong Flash) dùng `hash(token_ID) mod (n_routed/6)` để chọn 6 expert mặc định. Mục đích: giảm router overhead, ổn định distribute token vì lúc này embedding chưa giàu thông tin. KHÔNG có affinity, không có top-k logits.

## 11. Output: RMSNorm + Prediction Head + MTP

### 11.1 RMSNorm cuối

Sau layer cuối, lấy `h_final = A_{final} · X_L ∈ ℝ^d`, rồi RMSNorm.

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| `h_final` | `(B, n, d)` | `(1, 4096, 4096)` | `(1, 4096, 7168)` |
| Weight RMSNorm γ | `(d,)` | `(4096,)` | `(7168,)` |

### 11.2 Prediction Head

| Weight | Shape | Flash | Pro |
|---|---|---|---|
| `W_lm` | `(d, V)` | `(4096, 128000)` → **524.3 M** | `(7168, 128000)` → **917.5 M** |
| Logits | `(B, n, V)` | `(1, 4096, 128000)` BF16 = **1 GiB activation** | `(1, 4096, 128000)` = 1.75 GiB |

**LM Loss:** Cross-entropy với label-shift 1 (next-token).

### 11.3 MTP Module (depth=1)

MTP depth=1 ⇒ chỉ 1 module dự đoán token `t+2`. Cấu trúc: 1 Transformer block phụ + 1 prediction head (chia sẻ `W_lm` với main head theo phong cách V3) [1, §2.1].

| Tensor | Shape | Flash | Pro |
|---|---|---|---|
| Input MTP | `(B, n, d)` | `(1, 4096, 4096)` | `(1, 4096, 7168)` |
| Output logits MTP | `(B, n, V)` | `(1, 4096, 128000)` | `(1, 4096, 128000)` |

**MTP loss weight:** 0.3 cho phần lớn quá trình train, giảm xuống 0.1 khi bắt đầu decay learning rate [1, §4.2.2]. Inference vẫn dùng MTP head cho speculative decoding (giống V3).

## 12. Kiểm chứng tổng số tham số

Cross-check chính xác bằng Python (đã chạy trong session). Công thức từng phần đã list ở mục 4–11.

**DeepSeek-V4-Flash:**

| Module | Param | % |
|---|---:|---:|
| Embedding | 524.3 M | 0.18 |
| LM head | 524.3 M | 0.18 |
| MoE (43 × 6.47 B) | 278.1 B | 97.96 |
| Attention (20 CSA × 122 M + 21 HCA × 109 M) | ≈ 4.73 B | 1.67 |
| mHC (86 sublayer × 393 K) | ≈ 33.8 M | 0.01 |
| Router + mHC bias + RMSNorm + Sink | < 100 M | < 0.04 |
| **Tổng** | **≈ 283.9 B** | **100** |

Sát với 284 B paper báo (chênh < 0.1 B do MTP module phụ + một số bias/scalar nhỏ).

**Activated/token (Flash):** ≈ 12.3 B (paper báo 13 B; chênh do MTP block + một số overhead nhỏ tham gia activation path).

**DeepSeek-V4-Pro** (ước tính tương tự, dùng công thức trên):

- MoE all layers ≈ 1 551 B (≈ 97 % tổng 1.6 T) ✓
- Activated ≈ 49 B ✓

## 13. Tóm tắt ví dụ số end-to-end (Flash, n=4096, batch 1, decode)

Giả sử model Flash đã prefill xong `n=4096` token và đang decode token `t = 4097`. Một forward pass cho token decode:

| Bước | Tensor / Compute | Shape / số liệu |
|---|---|---|
| 0 | `token_id` → embed | `(1, 4096)` → `h ∈ ℝ^(4096)` |
| 1 | Khởi tạo residual mHC | `X ∈ ℝ^(4, 4096)` |
| 2 | Block 1 (SWA): nhân W_QKV, softmax 128 KV, output proj | dot product 128 × 4096-vector |
| 3 | mHC update sau attention | `X ← B_1 X + C_1 · out_attn` |
| 4 | DeepSeekMoE block 1 (Hash routing) | 1 shared + 6 routed = 7 expert × `(4096→2048→4096)` |
| 5 | mHC update sau MoE | tiếp tục `X ∈ ℝ^(4, 4096)` |
| 6–7 | Block 2 (SWA + MoE Hash) | tương tự |
| 8 | Block 3 (CSA): | |
| 8a | KV-compress 4 token cuối cùng → 1 entry `(512,)` | cache CSA giờ có `⌈4097/4⌉ = 1025` entries |
| 8b | Lightning Indexer FP4: chấm điểm 1025 candidate | score `(1025,)` |
| 8c | Top-k 512 + 128 SW = **640 entries** | `K, V ∈ ℝ^(640, 512)` |
| 8d | MQA 64 head: `q ∈ ℝ^(64, 512)` × `K ∈ ℝ^(640, 512)` | logits `(64, 640)`, sink scalar 64 |
| 8e | Output: `o ∈ ℝ^(64, 512)` → grouped proj 8×(4096→1024) → final `(4096,)` | |
| 9 | mHC + DeepSeekMoE block 3 | 1 + 6 expert |
| 10 | Block 4 (HCA): | |
| 10a | KV-compress 128-token block → 1 entry | cache HCA có `⌈4097/128⌉ = 33` entries |
| 10b | Core attn: `q (64, 512)` × `K (33+128, 512) = (161, 512)` | logits `(64, 161)` |
| 11 | mHC + MoE block 4 | ... |
| ... | Lặp tới block 43 | xen kẽ CSA, HCA |
| 44 | Final RMSNorm + LM head | `h_final (4096,) → logits (128000,)` |
| 45 | MTP block + MTP head | logits `(128000,)` cho token t+2 |

**Bộ nhớ KV cache (1 token mới sinh thêm):**

| Loại layer | Delta KV / token | KV / layer tại n=1M |
|---|---|---|
| SWA (layer 1–2) | 0 thêm (cache vòng 128) | 128 × 512 × ~1.25 B/dim = 80 KB |
| CSA (≈ 20 layer) | 1/4 entry per token | 250 K entry × 576 B = 144 MB |
| HCA (≈ 21 layer) | 1/128 entry per token | 7813 entry × 576 B = 4.5 MB |
| **Tổng / token (44 layer)** | ~`20×0.25 + 21×(1/128) ≈ 5.16` entry equiv. | **~3.0 GB / 1M ctx model-wide** |

So với MLA của V3 (~70 GB cho 1M ctx) ⇒ V4 giảm KV cache **~20×**, đúng tinh thần Figure 1 paper.

## 14. Tài liệu tham khảo

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026. Available: `References/DeepSeek_V4.pdf` (project).

[2] A. Vaswani et al., "Attention Is All You Need," in *Proc. NeurIPS*, 2017.

[3] D. Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models," 2024.

[4] DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024.

[5] DeepSeek-AI, "DeepSeek-V3.2 (DSA — DeepSeek Sparse Attention)," 2025.

[6] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," *Neurocomputing*, 2024.

[7] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.

[8] K. Jordan et al., "Muon Optimizer," 2024; J. Liu et al., "Muon Refinements," 2025.

[9] J. Xie et al., "Manifold-Constrained Hyper-Connections," 2026.

[10] Y. Zhu et al., "Hyper-Connections," 2025.

[11] G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," 2024.

[12] OpenAI, "GPT-OSS / Sink Logits," 2025.

[13] J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," 2023.

---

**Bước tiếp theo gợi ý (cho project DeepSeek-FPGA):**

1. Dùng file `DeepSeek_V32_microarch_roofline.xlsx` đã có sẵn để cập nhật sheet "V4_Flash" với các shape ở Mục 6–10; bổ sung cột FLOPs/token cho CSA, HCA, MoE để so sánh roofline với V3.2.
2. Vẽ lại `DeepSeek_V3_Datapath_Architecture.md` thành phiên bản V4 (thêm CSA path với Lightning Indexer FP4 + Top-k selector và HCA path với compression ratio 128).
3. Xác định mapping HBM↔URAM cho FPGA flagship dựa trên KV cache estimate Mục 13 (≈ 3 GB/1M ctx).
