# Bảng giải thích biến số kiến trúc DeepSeek-V4

Tham chiếu: DeepSeek-V4 Technical Report, §2 và §4.2.1 [1].
Quy ước: **scalar** = đại lượng vô hướng (cấu hình); **tensor** = mảng nhiều chiều.

## 1. Biến cấu hình (scalar) — hyperparameter mô hình

| Ký hiệu | Ý nghĩa | Phân loại | V4-Flash | V4-Pro | Ghi chú |
|---|---|---|---:|---:|---|
| `B` | Batch size (số sequence chạy song song) | Runtime | 1 – 64 (decode) / 1 (prefill 1M) | tương tự | Do hệ thống chọn, không cố định trong paper |
| `n` | Sequence length (số token trong context) | Runtime | 4 K → 16 K → 64 K → **1 M** | 4 K → 16 K → 64 K → **1 M** | Train progressive; max context 1 M tokens |
| `V` | Vocabulary size | Cố định | **128 000** | 128 000 | Kế thừa tokenizer V3 [1, §4.1] |
| `L` | Số block Transformer | Cố định | **43** | **61** | |
| `d` | Hidden size của residual stream "thực" | Cố định | **4096** | **7168** | Sau RMSNorm vào sublayer |
| `n_hc` | Hệ số mở rộng residual của mHC | Cố định | **4** | **4** | Residual width = n_hc × d |
| `t_max` | Số iteration Sinkhorn–Knopp cho `B_l` | Cố định | 20 | 20 | mHC stability |
| `n_h` | Số query head (Multi-Query Attention) | Cố định | **64** | **128** | Heads dùng chung 1 K,V |
| `c` | Head dimension attention | Cố định | **512** | 512 | Mỗi query head dim 512 |
| `d_c` | Query compression dimension (low-rank Q) | Cố định | **1024** | **1536** | `c_Q = h · W_DQ ∈ ℝ^d_c` |
| `n_I_h` | Số indexer query head (chỉ trong CSA) | Cố định | 64 | 64 | Cho Lightning Indexer |
| `c_I` | Indexer head dim | Cố định | 128 | 128 | Indexer chạy FP4 |
| `m` | Compression rate CSA (số KV gốc / 1 entry nén) | Cố định | **4** | 4 | CSA nén "vừa" |
| `m'` | Compression rate HCA | Cố định | **128** | 128 | HCA nén "mạnh" |
| `k` (top-k) | Số block nén CSA giữ lại sau Lightning Indexer | Cố định | **512** | **1024** | Sparse attention budget |
| `g` | Số nhóm trong Grouped Output Projection | Cố định | **8** | **16** | Giảm cost projection lớn |
| `d_g` | Dim của intermediate output mỗi nhóm | Cố định | 1024 | 1024 | |
| `n_win` | Sliding window size (nhánh phụ CSA/HCA + SWA layer đầu) | Cố định | **128** | 128 | Bắt local dependency |
| `d_RoPE` | Số dim cuối được áp RoPE | Cố định | 64 | 64 | 64 dims giữ BF16 |
| `n_routed` | Số routed expert trong DeepSeekMoE | Cố định | **256** | **384** | |
| `n_shared` | Số shared expert | Cố định | 1 | 1 | Luôn kích hoạt |
| `n_act` | Số routed expert kích hoạt/token (top-k MoE) | Cố định | **6** | **6** | |
| `d_ff` | Expert intermediate dimension (SwiGLU) | Cố định | **2048** | **3072** | |
| `n_hash` | Số layer MoE đầu dùng Hash routing | Cố định | 3 | 3 | Đơn giản hoá router đầu mạng |
| `MTP_depth` | Số module Multi-Token Prediction | Cố định | 1 | 1 | Dự đoán token t+2 |
| `t` | Index của query token đang xử lý (decode/prefill) | Runtime | 0 ≤ t < n | 0 ≤ t < n | |
| `Δt` | Khoảng cách "anticipatory routing" | Train-only | 1 step | 1 step | Routing dùng `θ_{t−Δt}` |

## 2. Trọng số (Weight tensors) — đếm theo MỘT layer

Đơn vị: **shape** = kích thước tensor; **# params** = tích các chiều.

### 2.1 Embedding & LM head (toàn mạng)

| Weight | Ý nghĩa | Shape | Flash # params | Pro # params |
|---|---|---|---:|---:|
| `W_emb` | Ma trận embedding token → vector | `(V, d)` | (128000, 4096) = **524.3 M** | (128000, 7168) = 917.5 M |
| `W_lm` | Prediction head: hidden → logits vocab | `(d, V)` | (4096, 128000) = 524.3 M | (7168, 128000) = 917.5 M |

### 2.2 mHC (mỗi sublayer; 1 block = 2 sublayer)

| Weight | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---:|---:|
| `W_pre_l` | Sinh tham số động `Ã_l` | `(n_hc·d, n_hc)` | (16384, 4) = 65 536 | (28672, 4) = 114 688 |
| `W_res_l` | Sinh tham số động `B̃_l` (residual mixing) | `(n_hc·d, n_hc²)` | (16384, 16) = 262 144 | (28672, 16) = 458 752 |
| `W_post_l` | Sinh tham số động `C̃_l` (output mixing) | `(n_hc·d, n_hc)` | (16384, 4) = 65 536 | (28672, 4) = 114 688 |
| `S_pre, S_post, S_res` | Bias tĩnh | `(n_hc), (n_hc), (n_hc, n_hc)` | 4+4+16 | 4+4+16 |
| `α_pre, α_res, α_post` | Gating scalar khởi tạo nhỏ | scalar × 3 | 3 | 3 |

### 2.3 CSA — Compressed Sparse Attention (per layer)

| Weight | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---:|---:|
| `W_aKV` | Sinh KV nhánh a | `(d, c)` | (4096, 512) = 2.10 M | (7168, 512) = 3.67 M |
| `W_bKV` | Sinh KV nhánh b (overlapped) | `(d, c)` | 2.10 M | 3.67 M |
| `W_aZ` | Sinh trọng số nén nhánh a | `(d, c)` | 2.10 M | 3.67 M |
| `W_bZ` | Sinh trọng số nén nhánh b | `(d, c)` | 2.10 M | 3.67 M |
| `B_a, B_b` | Positional bias trong khối nén | `(m, c)` | (4, 512) = 2 048 | (4, 512) = 2 048 |
| `W_DQ` | Down-projection query → latent | `(d, d_c)` | (4096, 1024) = 4.19 M | (7168, 1536) = 11.01 M |
| `W_IUQ` | Up-projection latent → indexer queries | `(d_c, c_I·n_I_h)` | (1024, 8192) = 8.39 M | (1536, 8192) = 12.58 M |
| `W_w` | Trọng số per-head cho indexer score | `(d, n_I_h)` | (4096, 64) = 0.26 M | (7168, 64) = 0.46 M |
| `W_UQ` | Up-projection latent → attention queries | `(d_c, c·n_h)` | (1024, 32768) = 33.55 M | (1536, 65536) = 100.66 M |
| `W_OG_i` (i=1..g) | Per-group output projection | `(c·n_h/g, d_g)` | g=8: (4096, 1024) × 8 = 33.55 M | g=16: (4096, 1024) × 16 = 67.11 M |
| `W_O` | Final output projection | `(g·d_g, d)` | (8192, 4096) = 33.55 M | (16384, 7168) = 117.44 M |
| `z'_h` | Sink logit per head | `(n_h,)` | 64 | 128 |
| `γ_RMSNorm` (Q + KV) | RMSNorm trên Q-heads và KV entries | `(c,) × 2` | 512 × 2 | 512 × 2 |
| **Tổng CSA / layer** | | | **≈ 121.9 M** | **≈ 323.9 M** |

### 2.4 HCA — Heavily Compressed Attention (per layer)

| Weight | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---:|---:|
| `W_KV` | Sinh KV (1 nhánh, không overlapped) | `(d, c)` | (4096, 512) = 2.10 M | (7168, 512) = 3.67 M |
| `W_Z` | Sinh trọng số nén | `(d, c)` | 2.10 M | 3.67 M |
| `B` | Positional bias trong khối nén `m'` | `(m', c)` | (128, 512) = 65 536 | (128, 512) = 65 536 |
| `W_DQ` | Down-projection query | `(d, d_c)` | 4.19 M | 11.01 M |
| `W_UQ` | Up-projection query | `(d_c, c·n_h)` | 33.55 M | 100.66 M |
| `W_OG_i, W_O` | Grouped output (như CSA) | như trên | 33.55 + 33.55 M | 67.11 + 117.44 M |
| `z'_h` | Sink logit | `(n_h,)` | 64 | 128 |
| `γ_RMSNorm` | RMSNorm Q + KV | `(c,) × 2` | 1024 | 1024 |
| **Tổng HCA / layer** | | | **≈ 109.05 M** | **≈ 303.6 M** |

### 2.5 DeepSeekMoE (per layer)

| Weight | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---:|---:|
| `W_router` | Sinh raw affinity g_raw | `(d, n_routed)` | (4096, 256) = 1.05 M | (7168, 384) = 2.75 M |
| `b_router` | Bias per-expert (load balance) | `(n_routed,)` | 256 | 384 |
| `W_gate^(e)` | SwiGLU gate của expert e | `(d, d_ff)` | (4096, 2048) = 8.39 M | (7168, 3072) = 22.02 M |
| `W_up^(e)` | SwiGLU up của expert e | `(d, d_ff)` | 8.39 M | 22.02 M |
| `W_down^(e)` | SwiGLU down của expert e | `(d_ff, d)` | 8.39 M | 22.02 M |
| **Tổng / 1 expert** | | | **25.17 M** | **66.06 M** |
| **Tổng MoE / layer** | (n_shared+n_routed) × per-expert | | (1+256) × 25.17 = **6.47 B** | (1+384) × 66.06 = **25.43 B** |

### 2.6 SWA layer đầu (chỉ Flash, layer 1–2)

| Weight | Ý nghĩa | Shape | Flash |
|---|---|---|---:|
| `W_Q, W_K, W_V` | Multi-head projection (không nén) | `(d, n_h·c)` × 3 | (4096, 32768) × 3 ≈ 402 M |
| `W_O` | Output projection | `(n_h·c, d)` | (32768, 4096) ≈ 134 M |

(V4-Pro **không** dùng SWA layer đầu; layer 1–2 của Pro là HCA thuần.)

## 3. Tensor động (Activation) — trong một forward pass

Quy ước `t` = vị trí query token; `s` = vị trí block KV đã nén.

### 3.1 Backbone

| Tensor | Ý nghĩa | Shape | Flash (B=1, n=4096) | Pro |
|---|---|---|---|---|
| `token_ids` | Chuỗi ID đầu vào | `(B, n)` ∈ ℤ | (1, 4096) | (1, 4096) |
| `H_0` | Embedding đầu ra | `(B, n, d)` | (1, 4096, 4096) | (1, 4096, 7168) |
| `X_l` | Residual stream mở rộng tại block l | `(B, n, n_hc, d)` | (1, 4096, 4, 4096) | (1, 4096, 4, 7168) |
| `X̂_l` | Sau RMSNorm + flatten | `(B, n, n_hc·d)` | (1, 4096, 16384) | (1, 4096, 28672) |
| `A_l` | Input mixing vector mHC | `(n_hc,)` per token | (4,) | (4,) |
| `B_l` | Birkhoff matrix mHC | `(n_hc, n_hc)` | (4, 4) doubly-stochastic | (4, 4) |
| `C_l` | Output mixing vector mHC | `(n_hc,)` per token | (4,) | (4,) |
| `h_final` | Sau layer cuối, sau pre-mix | `(B, n, d)` | (1, 4096, 4096) | (1, 4096, 7168) |
| `logits` | Đầu ra LM head | `(B, n, V)` | (1, 4096, 128000) | (1, 4096, 128000) |

### 3.2 CSA activation (per query token t)

| Tensor | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---|---|
| `H` | Hidden states của n token (KV source) | `(n, d)` | (4096, 4096) | (4096, 7168) |
| `C_a, C_b` | KV nhánh a/b trước nén | `(n, c)` | (4096, 512) | (4096, 512) |
| `Z_a, Z_b` | Trọng số nén nhánh a/b | `(n, c)` | (4096, 512) | (4096, 512) |
| `S_a, S_b` | Softmax weight (2m phần tử/block) | `(n, c)` | (4096, 512) | (4096, 512) |
| `C^Comp` | KV-cache đã nén CSA | `(n/m, c)` | (1024, 512) | (1024, 512) |
| `K^IComp` | Indexer key đã nén | `(n/m, c_I)` | (1024, 128) | (1024, 128) |
| `c_Q_t` | Latent query (chia sẻ giữa indexer & attention) | `(d_c,)` | (1024,) | (1536,) |
| `q_I_t` | Indexer queries (n_I_h heads) | `(n_I_h, c_I)` | (64, 128) | (64, 128) |
| `w_I_t` | Trọng số indexer per head | `(n_I_h,)` | (64,) | (64,) |
| `I_{t,:}` | Index score giữa t và mọi block nén | `(⌊t/m⌋,)` | tối đa (1024,) | (1024,) |
| `𝒞^SprsComp_t` | Top-k KV entry chọn được | `(k, c)` | (512, 512) | (1024, 512) |
| `q_t,i` | Attention queries n_h heads | `(n_h, c)` | (64, 512) | (128, 512) |
| `o_t,i` | Core attention output per head | `(n_h, c)` | (64, 512) | (128, 512) |
| `o^G_t,i` | Output chia g nhóm | `(g, c·n_h/g)` | (8, 4096) | (16, 4096) |
| `o^G'_t,i` | Sau per-group projection | `(g, d_g)` | (8, 1024) | (16, 1024) |
| `ô_t` | Output cuối sublayer attention | `(d,)` | (4096,) | (7168,) |

### 3.3 HCA activation (per query token t)

| Tensor | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---|---|
| `C` | KV trước nén | `(n, c)` | (4096, 512) | (4096, 512) |
| `Z` | Trọng số nén | `(n, c)` | (4096, 512) | (4096, 512) |
| `S` | Softmax weight (m' phần tử/block) | `(n, c)` | (4096, 512) | (4096, 512) |
| `C^Comp` | KV-cache đã nén HCA | `(n/m', c)` | **(32, 512)** | **(32, 512)** |
| `c_Q_t` | Latent query | `(d_c,)` | (1024,) | (1536,) |
| `q_t,i` | Attention queries | `(n_h, c)` | (64, 512) | (128, 512) |
| `o_t,i` | Core attention output | `(n_h, c)` | (64, 512) | (128, 512) |
| `ô_t` | Output sublayer | `(d,)` | (4096,) | (7168,) |

### 3.4 DeepSeekMoE activation (per query token t)

| Tensor | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---|---|
| `h` | Input vào MoE | `(d,)` | (4096,) | (7168,) |
| `g_raw` | Raw affinity (h · W_router) | `(n_routed,)` | (256,) | (384,) |
| `g̃` | Sau Sqrt(Softplus(·)) + bias | `(n_routed,)` | (256,) | (384,) |
| `top_idx` | Index của top-n_act expert | `(n_act,)` ∈ ℤ | (6,) | (6,) |
| `w_top` | Gating weight chuẩn hoá | `(n_act,)` | (6,) | (6,) |
| `inter^(e)` | SwiGLU intermediate (per expert) | `(d_ff,)` | (2048,) | (3072,) |
| `y_shared` | Output shared expert | `(d,)` | (4096,) | (7168,) |
| `y_routed^(i)` | Output routed expert thứ i | `(d,)` | (4096,) | (7168,) |
| `y` | Tổng hợp MoE | `(d,)` | (4096,) | (7168,) |

### 3.5 Sliding window + Attention sink (chung)

| Tensor | Ý nghĩa | Shape | Flash | Pro |
|---|---|---|---|---|
| `KV_win` | KV của n_win token gần nhất (chưa nén) | `(n_win, c)` | (128, 512) | (128, 512) |
| `KV_attn` | KV thực sự dùng cho core attn | CSA: `(k+n_win, c)`; HCA: `(n/m'+n_win, c)` | CSA: (640, 512); HCA: (160, 512) | CSA: (1152, 512); HCA: (160, 512) |
| `z'_h` | Sink logit (1 scalar/head) | `(n_h,)` | (64,) | (128,) |

## 4. KV cache (bộ nhớ trạng thái — quan trọng cho FPGA)

Tính cho **1 layer**, **1 sequence**, theo `n` (số token đã sinh).

| Loại layer | KV cache shape | Dtype mix | Bytes / entry | n = 4 K | n = 64 K | n = 1 M |
|---|---|---|---|---|---|---|
| SWA (Flash layer 1–2) | `(min(n, n_win), c)` × 2 (K & V chung 1 entry MQA) | 64 BF16 + 448 FP8 | 576 | 73.7 KB | 73.7 KB | 73.7 KB |
| CSA | `(⌈n/m⌉, c)` | 64 BF16 + 448 FP8 | 576 | 576 KB | 9 MB | 144 MB |
| HCA | `(⌈n/m'⌉, c)` | 64 BF16 + 448 FP8 | 576 | 18 KB | 288 KB | 4.5 MB |

Toàn mạng (Flash, 43 layer ~ 2 SWA + ~20 CSA + ~21 HCA):

| n | KV total / sequence |
|---|---|
| 4 K | ~ 12 MB |
| 64 K | ~ 192 MB |
| **1 M** | **~ 3.0 GB** |

So với baseline GQA8 BF16 head-dim 128 ở `n=1M` → V4 KV cache ≈ **2%** của baseline [1, §2.3.4].

## 5. Ví dụ ánh xạ ký hiệu paper ↔ ký hiệu code (PyTorch/Triton)

| Ký hiệu paper | Tên biến phổ biến trong code |
|---|---|
| `H`, `h_t` | `hidden_states`, `hidden_states[:, t, :]` |
| `c_Q_t` | `q_lora`, `q_compressed` |
| `q_t,i` | `query_states[..., i, :]` |
| `C^Comp` | `kv_compressed_cache` |
| `K^IComp` | `indexer_key_cache` |
| `q_I_t` | `indexer_query` |
| `I_{t,:}` | `index_scores` |
| `𝒞^SprsComp_t` | `selected_kv` |
| `o_t,i` | `attn_output[..., i, :]` |
| `ô_t` | `attn_output_final` |
| `A_l, B_l, C_l` | `mhc_in_proj`, `mhc_residual_proj`, `mhc_out_proj` |
| `g̃` | `routing_logits` |
| `top_idx, w_top` | `top_k_indices, top_k_weights` |
| `z'_h` | `sink_logits` |

## Tài liệu tham khảo

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
