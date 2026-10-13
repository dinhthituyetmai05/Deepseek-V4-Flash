# Vi kiến trúc DeepSeek-V4-Pro Accelerator — Đặc tả tổng hợp
# DeepSeek-V4-Pro Accelerator Microarchitecture — Consolidated Spec

**Tài liệu thiết kế nội bộ — PGS.TS Nguyễn Đức Minh (HUST/SEEE), project DeepSeek-FPGA.**
Bản trực quan tương tác: `DeepSeek_V4Pro_Accelerator_Explorer.html` (8 tab). Mapping & roofline: `DeepSeek_V4Pro_FU_Mapping.xlsx`. Giao thức liên card: `DeepSeek_V4Pro_InterCard_Protocol.md`.

---

## Mục lục

1. Tổng quan hệ thống
2. Cấu hình mô hình DeepSeek-V4-Pro
3. Ba cấp song song
4. Tập Functional Unit (FU) và phân bổ DSP thống nhất
5. Phân cấp bộ nhớ: HBM ↔ SRAM bank ↔ bus ↔ FU
6. Phân loại dữ liệu (bền vững / transient / vào-ra)
7. Pipeline mức card (8 stage, microbatch = chunk/token của user khác nhau)
8. Tensor/Head Parallel (attention) và Expert Parallel (MoE)
9. Pipeline mức FU trong một chip (Load / Compute / Store)
10. Ví dụ end-to-end: chunk của nhiều user đi qua card / chip / layer
11. Prefill vs Decode (compute-bound vs memory-bound)
12. Lập lịch fused prefill + decode
13. Bảng số liệu tổng hợp
14. Khuyến nghị (action items)
15. Tài liệu tham khảo

---

## 1. Tổng quan hệ thống

| Thành phần | Số lượng | Chi tiết |
|---|---:|---|
| CPU node (host) | 1 | nạp weight lúc boot, tokenize, sampling, điều phối pipeline |
| FPGA card | 8 | mỗi card = 1 pipeline stage |
| FPGA chipset/card | 4 | Intel Agilex 7 M-Series AGM 039 |
| Chipset toàn hệ | 32 | 8 × 4 |
| HBM2e / chipset | 32 GB @ 820 GB/s | tổng **1.024 GB @ 26,24 TB/s** |
| DSP block / chipset | 12.300 (24.600 mult 18×19) | tổng 393.600 |
| M20K / chipset | 370 Mb ≈ 46,25 MB (~18.944 block) | tổng ~1.480 MB |
| Peak compute / chipset | ~37 TFLOPS FP · ~75 TOPS INT8 | tổng ~1.184 TFLOPS · ~2.400 TOPS |

Toàn bộ trọng số (~797 GB FP4-MoE + FP8-attn) thường trú trong 1.024 GB HBM (78%). Đây là ràng buộc nền tảng → chọn pipeline-parallel giữ trọng số tĩnh.

## 2. Cấu hình mô hình DeepSeek-V4-Pro

| Tham số | Giá trị | Tham số | Giá trị |
|---|---:|---|---:|
| L (block) | 61 | n_h (query head, MQA) | 128 |
| d (hidden) | 7168 | c (head dim) | 512 |
| n_hc (mHC width) | 4 | d_c (query latent) | 1536 |
| routed expert/layer | 384 | g (output group) | 16 |
| shared expert/layer | 1 | d_ff (expert) | 3072 |
| top-k routed | 6 | CSA top-k / m | 1024 / 4 |
| HCA m' | 128 | MTP depth | 1 |
| Tổng params | 1,6 T | Activated/token | 49 B |

Attention: L1–L2 = HCA thuần; L≥3 xen kẽ CSA(lẻ)/HCA(chẵn) → 30 CSA + 31 HCA. MoE mọi layer (3 layer đầu Hash routing). Precision: MoE FP4 (0,5 B/param), attention/embed/head FP8, KV cache FP8(448 dim)+BF16(64 dim RoPE), indexer FP4.

## 3. Ba cấp song song

| Cấp | Phạm vi | Cơ chế | Đồng bộ |
|---|---|---|---|
| **Pipeline parallel** | giữa 8 card | mỗi card giữ ~8 layer; token đi C0→C7 | forward residual qua inter-card link |
| **Tensor/Head parallel** | 4 chip trong card (attention) | 128 head ÷ 4 = 32 head/chip; output 16 group ÷ 4 | **all-reduce** ô |
| **Expert parallel** | 4 chip trong card (MoE) | 384 expert ÷ 4 = 96 expert/chip | **all-to-all** dispatch + combine |

Phân bổ block → card: Card0 = L1–8 (+Embedding), Card1 = L9–16, …, Card6 = L49–56, Card7 = L57–61 (+Final RMSNorm + LM Head + MTP).

## 4. Tập Functional Unit (FU) và phân bổ DSP thống nhất

Phân bổ DSP cân bằng theo nguyên tắc **DSP ∝ GOP** để C-time mọi FU ≈ nhau (~3,75 ms/layer/chunk-256) → pipeline liên-FU không nghẽn; ECA khớp thời gian đọc HBM (read-bound 3,87 ms).

| FU | Phép toán | **DSP** | MAC lanes | Peak GFLOPS | GOP/chunk(256) |
|---|---|---:|---:|---:|---:|
| mHCE | RMSNorm + sinh A,B,C + Sinkhorn–Knopp(20) + mix X←B·X+C·F | **36** | 72 | 108 | 0,4 |
| ACU | KV-compress (4×d×c) + softmax → C^Comp | **667** | 1.334 | 2.001 | 7,5 |
| LIU | Lightning Indexer FP4 (W_DQ/W_IUQ/W_w) + Top-k=1024 | **1.094** | 2.188 | 3.282 | 12,3 |
| QCAU | W_UQ (32 head) + RoPE + core MQA + attention sink | **2.864** | 5.728 | 8.592 | 32,2 |
| GOPU | W_OG×4 group + W_O slice → partial ô | **2.099** | 4.198 | 6.297 | 23,6 |
| MRU | W_router + Sqrt(Softplus) + Top-6 + softmax | **125** | 250 | 375 | 1,4 |
| ECA | SwiGLU expert (W_gate/W_up/SiLU/W_down) batched + clamp | **5.265** | 10.530 | 15.795 | 59,2 |
| EWD/ICI | DMA prefetch + all-reduce/all-to-all control | **150** | 300 | (control) | — |
| **TỔNG** | | **12.300** | 24.600 | 36.900 | 136,6 |

(Số DSP trên dùng nhất quán trong toàn tài liệu và trong tab "Pipeline & lập lịch" của file Explorer. ECA chi phối ~43% GOP nhưng chỉ 43% DSP vì là read-bound — cho thêm DSP không nhanh hơn vì HBM giới hạn.)

## 5. Phân cấp bộ nhớ: HBM ↔ SRAM bank ↔ bus ↔ FU

**HBM (DRAM ngoại, in-package, 32 GB/chip):** giữ toàn bộ trọng số (~25 GB/chip cho 8 layer) + KV cache đầy đủ + embed/head. Đường: `HBM2e ↔ HBM Controller (hard) ↔ Hard NoC [AXI4] ↔ AXI4 interconnect ↔ mSGDMA ↔ cổng ghi (port B) các bank M20K`. M20K dual-port: port B = DMA ghi, port A = FU đọc → double-buffer ping-pong.

**On-chip SRAM (M20K 46,25 MB) — buffer working-set (không phải cache trọng số):**

| Buffer | Nội dung | Kích thước | #bank (32-bit) | Bus DATA | ADDR | BW @750MHz |
|---|---|---:|---:|---:|---:|---:|
| WTB | weight tile (double-buffer) | 16 MB | 32 | 1.024-bit | 17-bit | 96 GB/s |
| RAB | activation chunk (X,q,o,ô,g/u,y) | 10 MB | 64 | 2.048-bit | 15-bit | 192 GB/s |
| KVB | KV nóng (top-k + SW) | 12 MB | 16 | 512-bit | 16-bit | 48 GB/s |
| PSA | ∑K partial + top-k/router scratch | 6 MB | 8 | 256-bit | 13-bit | 24 GB/s |
| Ctrl | token queue, packet | 2 MB | 4 | 128-bit | — | 12 GB/s |

**Bus FU↔SRAM:** KHÔNG full crossbar. Trong mỗi buffer: 1 địa chỉ **broadcast** → mọi bank đọc song song (width-slice) → ghép một từ rộng. Giữa các FU: mỗi buffer 1 bus + **arbiter/mux** chọn FU đang active (FU chạy tuần tự nên đủ). PSA accumulator là register file cục bộ trong mảng MAC, ngoài bus. Đường operand dùng banked M20K + AGU + Avalon-ST (không AXI-MM, tránh overhead handshake mỗi chu kỳ); AXI4-Lite chỉ cho thanh ghi cấu hình.

## 6. Phân loại dữ liệu

| Lớp | Nội dung | Vị trí |
|---|---|---|
| **Bền vững (init 1 lần)** | trọng số 25 GB/chip; bias router, sink z'_h, mHC static, γ | HBM (weight) + register nhỏ on-chip (param nhỏ) |
| **Transient (FU↔FU)** | X, c_Q, q, C^Comp, o, ô, g/u, y | M20K (RAB/PSA), sống theo chunk |
| **Vào/ra chip** | X (link), all-to-all h/y + all-reduce ô (ICI), weight/KV stream (HBM) | inter-card link, ICI, HBM controller |

GHI HBM trong một layer: chỉ KV-cache append (~37 KB/layer/chip). Activation không ghi HBM (trừ tràn). ĐỌC HBM: trọng số (expert ~3,17 GB chi phối + attn ~181 MB) + top-k KV history.

## 7. Pipeline mức card (8 stage)

**Đơn vị:** macro-tick τ = thời gian một card xử lý phần layer của nó cho một microbatch. Trong decode τ ≈ 1,33 ms (memory-bound); trong prefill chunk-256, τ ≈ 3,87 ms/layer (ECA read-bound), một card 8 layer.

**Microbatch = chunk/token của user KHÁC nhau.** Quy luật: Card c tại tick t xử lý microbatch m = t − c. Tại một thời điểm steady-state, 8 card xử lý **8 microbatch khác nhau** (8 user):

```
tick t:  Card0←mb(t)   Card1←mb(t-1)  Card2←mb(t-2)  …  Card7←mb(t-7)
```

Ví dụ tick 8: Card0 xử lý chunk của U7 (L1–8), Card1 chunk U6 (L9–16), …, Card7 token U0 (L57–61+Head). Mỗi token (prefill chunk hoặc decode token) đi **full-depth qua cả 61 layer (8 card)** trong một lần forward; **không** decode xen giữa các layer. Phase (prefill/decode) ⟂ layer (độ sâu).

Throughput steady (decode, 8 microbatch lấp đầy) ≈ 1 token/τ; per-token latency = 8τ.

## 8. Tensor/Head Parallel (attention) và Expert Parallel (MoE)

**Tensor/Head Parallel (attention) — trong card, 4 chip:**
- Trọng số: `W_UQ (1536×65536)` chia **CỘT** theo head → mỗi chip 32 head; `W_O` chia **HÀNG**. ACU/LIU/W_DQ replicate (MQA → KV dùng chung).
- Activation X **nhân bản** trên 4 chip; mỗi chip tính 32 head độc lập (không comm); output proj cho **partial ô** → **all-reduce** (sum) → ô đồng nhất. Chỉ 1 all-reduce/attention.
- Phép chia cột (W_UQ): contraction K=1536 không chia → partial cộng cục bộ, không combine. Phép chia hàng (W_O): contraction bị chia → all-reduce.

**Expert Parallel (MoE) — trong card, 4 chip:**
- Trọng số: 384 expert ÷ 4 = **96 expert/chip, khối nguyên** (C{c}.0: e0–95 … C{c}.3: e288–383). Shared expert replicate.
- Router (replicate) → top-6 giống nhau; **all-to-all dispatch** token → chip giữ expert; ECA chạy SwiGLU đầy đủ cho (token,expert) cục bộ (không partial xuyên chip); **all-to-all combine** → cộng-có-trọng-số 6 expert + shared trên home-chip.
- Lý do EP (không TP cho MoE): expert thưa (6/384) và to (33 MB FP4); EP di chuyển token (14 KB) thay vì trọng số (33 MB).

Trong một layer, mỗi chip: replicated (mHCE/ACU/LIU/MRU) — sharded (QCAU/GOPU theo head/group; ECA theo expert).

## 9. Pipeline mức FU trong một chip (Load / Compute / Store)

Mỗi FU xử lý một dòng tile theo 3 pha tuần tự: **R (Load — đọc operand)** → **C (Compute — MAC)** → **W (Store — ghi kết quả)**. Không pipeline → MAC rảnh khi R/W (util ≈ C/(R+C+W) ~50%).

**Pipeline 3 tầng (double-buffer):** R(tile i+1) chồng C(tile i) chồng W(tile i−1) → sau khi lấp đầy, 1 tile/nhịp, MAC bận liên tục; throughput = max(R,C,W) thay vì tổng.

Hai mức pipeline:
- *Trong 1 FU (giữa các tile):* prefetch tile kế khi đang tính — cần WTB/RAB double-buffer 2 bank.
- *Giữa các FU (trong layer):* nạp trọng số FU kế (R từ HBM) khi FU hiện tại đang C — EWD prefetch.

Định lượng (chip C0.1, layer CSA, chunk-256):

| FU | C (ms) | R weight (ms) | Bound |
|---|---:|---:|---|
| mHCE/ACU/LIU/QCAU/GOPU/MRU | ~3,75 | 0,02–0,16 | COMPUTE → pipeline đạt ~100% MAC |
| **ECA** | 3,75 | **3,87** | **READ** → pipeline ẩn C dưới R, MAC ~59% (=C/R) |

ECA read-bound: khắc phục bằng tăng batch N_e (C→R) hoặc nén weight mạnh hơn.

## 10. Ví dụ end-to-end: chunk của nhiều user đi qua card / chip / layer

**8 user vào hệ (continuous batching):**

| User | Prompt (token) | #chunk (s=256) | TTFT (ước tính) |
|---|---:|---:|---:|
| U0 | 6 | 1 | 84 ms |
| U1 | 40 | 1 | 84 ms |
| U2 | 256 | 1 | 84 ms |
| U3 | 512 | 2 | 94 ms |
| U4 | 1.024 | 4 | 115 ms |
| U5 | 4.096 | 16 | 240 ms |
| U6 | 16.384 | 64 | 742 ms |
| U7 | 65.536 | 256 | 2.747 ms |

**Bước 0 — Tokenize + Embedding:** host token hóa, gán SEQ_ID. token_ids vào **C0.0** → EMB (W_emb) → H_0 → mHCE mở rộng → X[u][k]. Một chunk s=256 → activation ~14,7 MB; xử lý theo tile BLOCK_M=64 (4 tile/chunk) để vừa RAB.

**Bước 1 — Chunk đi qua các card (pipeline):** ví dụ `chunk[u6][32]`:
- Card0 (L1–8): chunk vào, qua 8 layer; tại MỖI layer chạy chuỗi FU (mục dưới). Pipeline: khi chunk[u6][32] ở Card1, chunk[u6][33] vào Card0; đồng thời các card khác xử lý chunk/token của user khác.
- → Card1 (L9–16) → … → Card7 (L57–61). Sau Card7 với chunk CUỐI (chunk[u6][63]) + LM Head → token đầu tiên của U6.

**Bước 2 — Trong một card, 4 chip (TP/EP):** với chunk[u6][32] tại Card2 (L17 = CSA):
- C2.0 head 0–31, C2.1 head 32–63, C2.2 head 64–95, C2.3 head 96–127 (TP attention).
- C2.0 expert e0–95, …, C2.3 expert e288–383 (EP MoE).
- replicated: mHCE, ACU (KV-compress), LIU (indexer), MRU (router); all-reduce sau GOPU; all-to-all quanh ECA.

**Bước 3 — Trong một chip, một layer, chuỗi FU (Load/Compute/Store):** cho chunk[u6][32] trên C2.1:

| Bước | FU | ĐỌC (R) | TÍNH (C) | GHI (W) |
|---|---|---|---|---|
| 1 | mHCE | X[u6][32] (RAB) | RMSNorm+Sinkhorn → A,B,C | RAB |
| 2 | ACU | W_aKV/bKV/aZ/bZ (WTB←HBM 14,7MB) + X | KV-compress 256 tok | KVB + **HBM KV append U6 (64 entry ~37KB)** |
| 3 | LIU | W_DQ/IUQ/w (WTB 24MB) + idx-key (HBM) | index-score + Top-k=1024 (FP4) | PSA top-k; fetch top-k KV → KVB |
| 4 | QCAU | W_UQ slice head32–63 (WTB 25MB) + KV chọn (KVB) | q + RoPE + core MQA + sink | RAB o |
| 5 | GOPU | o (RAB) + W_OG+W_O (WTB) | grouped proj → partial ô | RAB → **all-reduce** ô |
| 6 | mHCE | X+ô | X←B·X+C·ô | RAB |
| 7 | MRU | X + W_router (WTB) | top-6 + gating | PSA |
| 8 | ICI | h (RAB) | **all-to-all dispatch** | RAB recv |
| 9 | **ECA** | **expert weight (WTB←HBM ~3,17GB)** + token | SwiGLU 256 tok | RAB y |
| 10 | ICI | y_e | **all-to-all combine** | RAB y |
| 11 | mHCE | X+y | X←B·X+C·y = X' | RAB → layer kế |

Mỗi FU bên trong dùng pipeline Load/Compute/Store cho các tile BLOCK_M; EWD prefetch trọng số FU kế khi FU hiện tại đang C.

**Bước 4 — Decode:** sau khi prefill U6 xong (64 chunk × 61 layer), token0 của U6 vào lại L1 → … → L61 → token1; lặp. Mỗi decode token đi full-depth.

## 11. Prefill vs Decode

| | Prefill (chunk 256) | Decode (1 token) |
|---|---|---|
| GOP/layer/chip | 136,6 | 0,50 (256× ít) |
| Nút cổ chai | compute (DSP) | **HBM bandwidth** |
| Stage-time/layer | 3,87 ms (ECA read) | 0,199 ms |
| MAC util (batch=1) | ~97% | **~7%** |
| Đòn bẩy | phân bổ DSP cân bằng | batch + HBM BW |

Decode đọc 163 MB weight/layer/chip cho 1 token → memory-bound; DSP rảnh ~93%. Đó là lý do phải fused.

## 12. Lập lịch fused prefill + decode

Ghép decode token của các user KHÁC vào cùng batch với prefill chunk (256 prefill U6 + 8 decode U0…U9 = 264 row). **Weight đọc 1 lần từ HBM dùng chung** cho cả prefill và decode (decode không tốn HBM thêm; expert decode ⊆ expert prefill đã bão hoà). MAC chạy tile prefill rồi đuôi decode trên cùng weight tile.

| Chế độ | MAC util | Thời gian/layer | Ghi chú |
|---|---:|---:|---|
| Decode-alone | ~7% | — | tránh |
| Tách rời | ~83% | 4,62 ms | decode tốn 1 lần đọc HBM riêng |
| **Fused** | **~96%** | **4,00 ms** | decode đi ké, +16% thông lượng |

Riêng: KV (theo từng user) và activation row khác nhau; chia sẻ: weight tile (layer L, slice chip). Scheduler luôn kèm 1 prefill chunk khi còn prefill chờ → DSP ~96%; tỉ lệ decode/chunk cân TTFT vs TPOT.

## 13. Bảng số liệu tổng hợp

| Đại lượng | Giá trị |
|---|---:|
| Weight resident toàn hệ | ~797 GB / 1.024 GB (78%) |
| Weight / card (8 block) | ~104 GB / 128 GB |
| KV / chip / seq @1M ctx | ~658 MB (worst card) |
| Trần B×n (KV) / card | ~31,6 M token |
| Context tối đa | 1 M (model limit) |
| Prefill throughput (compute) | ~24.500 tok/s/chip |
| Decode throughput (HBM) | ~629 tok/s/chip (batch=1) |
| Aggregate HBM BW | 26,24 TB/s |
| Roofline decode (1 user, hệ) | ~640–750 tok/s |
| Inter-card residual | 56 KB/token (BF16) |

## 14. Khuyến nghị (action items)

1. **Căn cứ:** ECA read-bound, decode 7% util. **Hành động:** triển khai scheduler fused (luôn kèm prefill chunk). **Ai:** nhóm firmware/lập lịch. **Khi nào:** trước RTL freeze.
2. **Căn cứ:** GOPU/QCAU/LIU từng thiếu DSP. **Hành động:** chốt phân bổ DSP cân bằng (mục 4) vào floorplan. **Ai:** nhóm RTL. **Khi nào:** giai đoạn micro-arch.
3. **Căn cứ:** FP4 trên Agilex cần soft-pack. **Hành động:** benchmark MAC FP4 1 DSP block AGM 039, xác nhận ~75 TOPS/chip. **Ai:** nhóm DSP. **Khi nào:** song song bước 2.
4. **Căn cứ:** WTB double-buffer ẩn 4 ms nạp HBM. **Hành động:** RTL prototype EWD prefetch + verify HBM efficiency thực. **Ai:** nhóm RTL. **Khi nào:** sau bước 2.

## 15. Tài liệu tham khảo

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
[2] A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
[3] D. Dai et al., "DeepSeekMoE," 2024.
[4] M. Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019.
[5] A. Agrawal et al., "Sarathi-Serve: chunked-prefill + decode piggyback," 2024.
[6] Intel, "Agilex 7 M-Series Product Table & NoC/HBM2e User Guide," 2025.
[7] S. Williams et al., "Roofline," CACM 2009.
