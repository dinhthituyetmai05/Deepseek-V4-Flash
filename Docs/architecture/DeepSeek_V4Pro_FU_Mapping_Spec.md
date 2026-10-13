# Phương án ánh xạ DeepSeek-V4-Pro lên hệ 32 FPGA chipset
# DeepSeek-V4-Pro → 32-FPGA-Chipset Functional-Unit Mapping

**Tài liệu thiết kế vi kiến trúc — PGS.TS Nguyễn Đức Minh (HUST/SEEE), project DeepSeek-FPGA**
Nguồn mô hình: `DeepSeek_V4_Architecture_Detailed.md`, `DeepSeek_V4_Variables_Glossary.md`. Nguồn phần cứng: `FPGA_flagship_comparison.xlsx`.
Bảng tra cứu định lượng kèm theo: `DeepSeek_V4Pro_FU_Mapping.xlsx` (7 sheet).

---

## Mục lục / Table of Contents

1. Phạm vi & cấu hình / Scope & configuration
2. Tổng quan hệ phần cứng / Hardware overview
3. Sơ đồ song song hai cấp / Two-level parallelism scheme
4. Ánh xạ block → card / Block-to-card map
5. Tập Functional Unit per chipset / Per-chipset functional units
6. Ánh xạ chi tiết sublayer/sub-block → FU → chipset
7. Ngân sách tài nguyên / Resource budget
8. Đường truyền dữ liệu & dataflow / Datapath & dataflow
9. Roofline & throughput
10. Giả định, giới hạn, bước tiếp theo / Assumptions, limits, next steps
11. Tài liệu tham khảo / References

---

## 1. Phạm vi & cấu hình / Scope & configuration

Mục tiêu: ánh xạ toàn bộ layer/sublayer/sub-block của **DeepSeek-V4-Pro** (1.6 T tổng / 49 B kích hoạt) lên các functional unit (FU) phần cứng, chỉ rõ FU nào nằm trên FPGA chipset nào.

Cấu hình mô hình (từ spec project):

| Tham số | Giá trị | Tham số | Giá trị |
|---|---:|---|---:|
| L (số block) | 61 | n_h (query head, MQA) | 128 |
| d (hidden) | 7168 | c (head dim) | 512 |
| n_hc (mHC width) | 4 | g (output groups) | 16 |
| routed experts/layer | 384 | d_ff (expert) | 3072 |
| shared expert/layer | 1 | top-k routed | 6 |
| CSA top-k / m | 1024 / 4 | HCA m' | 128 |
| MTP depth | 1 | Vocab | 128 000 |
| Tổng params | 1.6 T | Activated/token | 49 B |

Mỗi block = 1 sublayer Attention (CSA hoặc HCA) + 1 sublayer DeepSeekMoE, bao bởi 2 cặp mHC mixing (Pre/Res cho attention, Post/Res cho MoE).

---

## 2. Tổng quan hệ phần cứng / Hardware overview

1 CPU node điều phối **8 FPGA card × 4 chipset = 32 chipset**. Mỗi chipset là **Intel Agilex 7 M-Series AGM 039** — flagship hiện hành (đối chiếu file so sánh project, khớp tên `AGM039` trong các file V4 trước).

| Tài nguyên | / chipset | / card (×4) | / hệ (×32) |
|---|---:|---:|---:|
| DSP block (24,600 mult 18×19) | 12,300 | 49,200 | 393,600 |
| M20K on-chip SRAM | 370 Mb (46.25 MB) | 1,480 Mb | 11,840 Mb (1,480 MB) |
| HBM2e | 32 GB | 128 GB | **1,024 GB** |
| HBM2e bandwidth | 820 GB/s | 3,280 GB/s | **26,240 GB/s** |
| Logic cells | 3.85 M | 15.4 M | 123.2 M |

Ràng buộc quyết định thiết kế: tổng weight resident của V4-Pro ≈ **797 GB** (xem §7) khớp **78%** của 1,024 GB HBM. Toàn bộ trọng số phải nằm thường trú trong 32 HBM stack — host không thể nạp lại trong critical path. Đây là lý do chọn sơ đồ song song giữ-trọng-số-tĩnh ở §3.

---

## 3. Sơ đồ song song hai cấp / Two-level parallelism scheme

**Cấp 1 — Pipeline Parallel (giữa các card):** 8 card = 8 pipeline stage. Mỗi card giữ thường trú ~8 block liên tiếp. Token đi tuần tự Card 0 → Card 7. Khi pipeline đầy (8 micro-batch / 8 user đồng thời), cả 8 card chạy song song.

**Cấp 2 — Expert + Tensor Parallel (trong card, 4 chipset):**
- **Expert-Parallel (MoE):** 384 routed expert/layer chia đều 4 chipset → **96 expert/chipset**. Shared expert replicated trên cả 4. Token được dispatch tới chipset chứa top-6 expert qua interconnect nội card (all-to-all).
- **Tensor-Parallel (Attention):** 128 query head chia 4 chipset → **32 head/chipset**; 16 output group → **4 group/chipset**, reduce cuối qua interconnect. KV nén (MQA dùng chung) được replicate.

Lý do: workload chi phối là MoE (97% params, memory-bound). Expert-parallel giữ mỗi expert thường trú đúng 1 chipset, tránh di chuyển trọng số. Pipeline-parallel chia 797 GB thành ~25 GB/chipset, vừa 32 GB.

---

## 4. Ánh xạ block → card / Block-to-card map

| Card (stage) | Chipsets | Block | Attention | Ghi chú |
|---|---|---|---|---|
| Card 0 | C0.0–C0.3 | **L1–L8** + Token Embedding | L1–2 HCA, L3 CSA, … | L1–3 Hash routing; Embedding ở C0.0 |
| Card 1 | C1.0–C1.3 | L9–L16 | xen kẽ CSA/HCA | |
| Card 2 | C2.0–C2.3 | L17–L24 | xen kẽ | |
| Card 3 | C3.0–C3.3 | L25–L32 | xen kẽ | |
| Card 4 | C4.0–C4.3 | L33–L40 | xen kẽ | |
| Card 5 | C5.0–C5.3 | L41–L48 | xen kẽ | |
| Card 6 | C6.0–C6.3 | L49–L56 | xen kẽ | |
| Card 7 | C7.0–C7.3 | **L57–L61** + đầu ra | xen kẽ | Final RMSNorm + LM Head (vocab-shard) + MTP |

Quy luật loại attention: **L1, L2 = HCA thuần; L≥3 xen kẽ CSA (lẻ) / HCA (chẵn)** → tổng **30 CSA + 31 HCA = 61**. Hash routing áp cho 3 block MoE đầu (L1–L3).

---

## 5. Tập Functional Unit per chipset / Per-chipset functional units

Mỗi chipset chứa **đủ bộ FU** vì nó xử lý một lát (slice) của mọi block thuộc card của nó.

| FU | Tên đầy đủ | Xử lý sublayer/sub-block | DSP block (của 12,300) |
|---|---|---|---:|
| **mHCE** | mHC Engine | RMSNorm flatten residual, sinh W_pre/W_res/W_post, Sinkhorn–Knopp 20 iter (Birkhoff B_l), mixing X←B·X+C·F | 120 |
| **ACU** | Attn Compression Unit | KV-compression CSA (4 nhánh overlapped) & HCA (1 nhánh m'=128) → C^Comp | 750 |
| **LIU** | Lightning Indexer Unit | (chỉ CSA) W_DQ, W_IUQ, W_w; score 64h×128 ReLU; Top-k=1024 — FP4 | 500 |
| **QCAU** | Query & Core-Attention Unit | W_UQ up-proj query, RMSNorm/head, partial RoPE 64d, core MQA + attention sink | 1,500 |
| **GOPU** | Grouped Output Projection Unit | W_OG×16 (4/chipset) + W_O → ô | 600 |
| **MRU** | MoE Router Unit | W_router → Sqrt(Softplus) → top-6 → gating softmax; hoặc Hash routing | 150 |
| **ECA** | Expert Compute Array | SwiGLU expert (W_gate/W_up/SiLU/W_down + clamp[±10]); shared + 96 routed/chipset — **chi phối** | 8,600 |
| **EWD** | Expert Weight DMA | prefetch trọng số expert từ HBM channel, double-buffer | (chung 80) |
| **ICI** | Inter-Chipset Interconnect | all-to-all dispatch token↔expert trong card, forward pipeline sang card kế | (chung 80) |

FU đặc biệt (chỉ một số chipset):
- **EMB** (Embedding) — chỉ **C0.0**: gather W_emb (128k×7168).
- **FRN** (Final RMSNorm), **LMH** (LM Head, vocab-shard 32k/chipset), **MTPU** (MTP block) — chỉ **Card 7 (C7.0–C7.3)**.

Tổng DSP: 8,600+1,500+750+600+500+150+120+80 = **12,300** (100%).

---

## 6. Ánh xạ chi tiết sublayer/sub-block → FU → chipset

Bảng đầy đủ 61 block ở sheet `FU_Mapping` của file Excel. Dưới đây là **mẫu cho 3 block đại diện**; mọi block khác theo cùng khuôn (chỉ khác card và loại attention).

### 6.1 L0 — Embedding
| Sub-block | FU | Chipset | Precision |
|---|---|---|---|
| Token embedding gather W_emb | EMB | C0.0 | FP8 |

### 6.2 L1 (HCA, Hash routing) — trên Card 0
| Sublayer | Sub-block | FU | Chipset | Parallel | Precision |
|---|---|---|---|---|---|
| mHC (attn) | RMSNorm + W_pre/res/post + Sinkhorn → A,B,C; mixing | mHCE | C0.0–.3 | replicated+reduce | BF16 |
| Attn HCA | KV-compress 1-branch (W_KV,W_Z, m'=128) → C^Comp(32×512) | ACU | C0.0–.3 | replicated KV | FP8 |
| Attn HCA | Query W_DQ,W_UQ → q(128,512), RMSNorm/head, RoPE 64d | QCAU | C0.0–.3 | TP 32 head/chip | FP8 |
| Attn HCA | Core MQA q×K(32+128 SW) + sink → o | QCAU | C0.0–.3 | TP 32 head/chip | FP8/BF16 |
| Attn HCA | Grouped output W_OG×16 + W_O → ô(7168) | GOPU | C0.0–.3 | TP 4 group/chip+reduce | FP8 |
| mHC (moe) | mixing X←B·X+C·F_moe | mHCE | C0.0–.3 | replicated+reduce | BF16 |
| MoE | **Hash routing** hash(token_ID) mod 64 → 6 expert | MRU | C0.0–.3 | replicated | INT |
| MoE | Shared expert SwiGLU | ECA | C0.0–.3 | replicated | FP4 |
| MoE | Routed top-6 SwiGLU (96 exp/chipset) | ECA | C0.0:e0–95 · C0.1:96–191 · C0.2:192–287 · C0.3:288–383 | EP | FP4 |
| MoE | All-to-all dispatch + combine y | ICI | C0.0–.3 | intra-card | FP8 act |

### 6.3 L3 (CSA, Hash routing) — trên Card 0
Khác L1 ở sublayer attention (có thêm Lightning Indexer):
| Sub-block | FU | Chipset | Precision |
|---|---|---|---|
| KV-compress overlapped (W_aKV,W_bKV,W_aZ,W_bZ) → C^Comp(1024×512) | ACU | C0.0–.3 | FP8 |
| **Lightning Indexer** W_DQ,W_IUQ,W_w → score + Top-k=1024 | LIU | C0.0–.3 | FP4 |
| Query W_UQ → q, RMSNorm, RoPE | QCAU | C0.0–.3 | FP8 |
| Core MQA q×K(top1024+128 SW) + sink | QCAU | C0.0–.3 | FP8/BF16 |
| Grouped output | GOPU | C0.0–.3 | FP8 |
| (MoE giống L1: MRU/ECA/ICI, expert index như trên) | | | |

### 6.4 Đầu ra (sau L61) — trên Card 7
| Sub-block | FU | Chipset | Precision |
|---|---|---|---|
| Final pre-mix A_final·X_L + RMSNorm → h_final | FRN | C7.0–.3 | BF16 |
| LM Head W_lm(7168×128000), vocab-shard | LMH | C7.0:0–31999 · C7.1:32k–63999 · C7.2:64k–95999 · C7.3:96k–127999 | FP8 |
| MTP block + head (share W_lm) → token t+2 | MTPU | C7.0–.3 | FP4/FP8 |

**Quy luật tổng quát:** với block L bất kỳ, card = `⌈L/8⌉−1` theo §4; trong card, expert chia `C{card}.j` giữ expert `[96j, 96j+95]`, head chia `[32j, 32j+31]`, output group `[4j, 4j+3]`.

---

## 7. Ngân sách tài nguyên / Resource budget

### 7.1 HBM/chipset (card 8-block điển hình, 32 GB)

| Hạng mục | GB | % |
|---|---:|---:|
| Routed expert weights (8 block × 96 exp × 66.06M × 0.5B FP4) | 24.77 | 77% |
| Shared expert (8 block, replicated) | 0.26 | 1% |
| Attention weights (8 layer, 1/4 TP, FP8) | 0.61 | 2% |
| mHC + router | 0.10 | <1% |
| KV cache (per-seq, FP8+BF16) | ~3.00 | 9% |
| Activation/residual + double-buffer | ~1.50 | 5% |
| **Tổng** | **~30.2** | **~94%** |

Card 7 (5 block) nhẹ hơn (~16 GB expert) → dư chỗ cho LM Head logits (1.75 GiB BF16) + MTP.

### 7.2 Tổng weight toàn hệ

| Khối | Params | Precision | Bytes |
|---|---:|---|---:|
| MoE (61×25.43 B) | 1,551 B | FP4 | 776 GB |
| Attention (30 CSA + 31 HCA) | 19.1 B | FP8 | 19.1 GB |
| Embedding + LM Head | 1.84 B | FP8 | 1.8 GB |
| MTP + mHC | ~1.5 B | FP4/BF16 | ~0.8 GB |
| **Tổng resident** | **~1.574 T** | | **~797 GB (78% của 1,024 GB)** |

---

## 8. Đường truyền dữ liệu & dataflow / Datapath & dataflow

1. **Boot:** CPU node DMA 797 GB weight vào 32 HBM stack qua PCIe Gen5 (một lần).
2. **Per token, per card:** residual X (4×7168 BF16, 57 KB/token) vào → mHCE → attention (ACU→[LIU]→QCAU→GOPU, TP reduce qua ICI) → mHCE → MoE (MRU chọn top-6 → ICI all-to-all dispatch tới chipset chứa expert → ECA SwiGLU → ICI combine) → X' forward sang card kế qua interconnect liên card.
3. **Card 7 tail:** FRN → LMH (vocab-shard, gather logits về host) → MTPU. CPU node sampling token.
4. **Bottleneck:** đọc trọng số expert từ HBM (memory-bound). EWD double-buffer trọng số expert kế tiếp trong khi ECA tính expert hiện tại.

---

## 9. Roofline & throughput

| Đại lượng | Giá trị |
|---|---:|
| Aggregate HBM BW | 26,240 GB/s |
| Active weight/token (FP4) | 24.5 GB |
| Trần decode (1 user) | ~1,071 tok/s |
| Thực tế @60% HBM eff. | ~640 tok/s |
| Per-card stage latency | ~0.56 ms (1.85 GB / 3.28 TB/s) |
| Regime | Memory-bound (AI ≈ 2 FLOPs/B, batch=1) |
| Ngưỡng compute-bound | batch ≥ ~37 |

Decode batch=1 hoàn toàn memory-bound trên mọi nền tảng FPGA/GPU. Để tận dụng 393,600 DSP, cần chạy ≥8 micro-batch (lấp pipeline) hoặc batch lớn để chuyển sang compute-bound.

---

## 10. Giả định, giới hạn, bước tiếp theo / Assumptions, limits, next steps

**Giả định:** số DSP/M20K/HBM theo vendor product brief (chưa benchmark). FP4 trên Agilex cần soft-pack INT4 + scaling trên variable-precision DSP. Efficiency 60% là giả định thiết kế.

**Giới hạn:** (i) HBM/chipset ~94% → dư mỏng cho batch/KV growth ở context dài; có thể spill KV sang host hoặc giảm số block/card. (ii) Pipeline 8-stage cần ≥8 request đồng thời để không bubble. (iii) Cross-card interconnect (residual 57 KB/token × ~640 tok/s ≈ 36 MB/s/hướng) — nhẹ, không phải bottleneck.

**Bước tiếp theo (action items):**
1. **Căn cứ:** HBM dư mỏng (§7.1). **Hành động:** dựng RTL prototype 1 card (4 chipset, 8 block) đo HBM efficiency thực. **Ai:** nhóm RTL. **Khi nào:** trước khi khoá floorplan toàn hệ.
2. **Căn cứ:** FP4 cần soft-pack. **Hành động:** benchmark MAC FP4 trên 1 DSP block AGM 039, xác nhận ~75 TOPS/chipset. **Ai:** nhóm DSP. **Khi nào:** song song bước 1.
3. **Căn cứ:** Lightning Indexer là workload mới. **Hành động:** thiết kế Top-k=1024 selector pipeline trên LIU, đo throughput vs n=1M. **Ai:** nhóm kiến trúc. **Khi nào:** sau bước 1.

---

## 11. Tài liệu tham khảo / References

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
[2] A. Vaswani et al., "Attention Is All You Need," *Proc. NeurIPS*, 2017.
[3] D. Dai et al., "DeepSeekMoE," 2024.
[4] Intel, "Agilex 7 M-Series Product Table," 2025.09.30.
[5] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model," *CACM*, vol. 52, no. 4, 2009.
[6] J. Xie et al., "Manifold-Constrained Hyper-Connections," 2026.
