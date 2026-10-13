# Định nghĩa giao diện vào/ra các Functional Unit — 1 chip Alveo U250
# FU Input/Output Interface Specification — Single-Chip U250

**Tài liệu thiết kế nội bộ.** Định nghĩa hợp đồng giao diện (interface contract) mức **tensor + bus/port** cho 8 FU của vi kiến trúc một chip U250 (slice 1/4 card: 32/128 head, 4/16 group, 96/384 expert, 8 layer). Ký hiệu tensor theo `01_Background/DeepSeek_V4_Variables_Glossary.md`; kiến trúc và phân bổ DSP theo `DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md`.

---

## Mục lục

1. Phạm vi và quy ước chung
2. Bảng tóm tắt dòng dữ liệu một layer
3. mHCE — Hyper-Connection Engine
4. ACU — Attention Compression Unit
5. LIU/HCA — Lightning Indexer Unit
6. QCAU — Query-Core Attention Unit
7. GOPU — Grouped Output Projection Unit
8. MRU — MoE Router Unit
9. ICI — Inter-Card Interconnect
10. ECA — Expert Compute Array
11. Giao diện điều khiển chung (Ctrl / EWD / AXI4-Lite)
12. Ghi chú nhất quán số liệu
13. Tài liệu tham khảo

---

## 1. Phạm vi và quy ước chung

**Đơn vị công việc:** chunk prefill `s = 256` token; batch `B = 8` chunk (2.048 token); xử lý nội bộ theo tile `BLOCK_M = 64` row (4 tile/chunk). Kích thước tensor ghi **/chunk-256**; nhân 8 cho batch. Decode: `s = 1`, cùng giao diện, khác kích thước.

**Slice trên chip (TP/EP):** `n_h^chip = 32` head, `g^chip = 4` group, `n_e^chip = 96` routed expert + 1 shared (replicate). FU replicated (mHCE, ACU, LIU, MRU) tính giống nhau trên 4 card của stage; FU sharded (QCAU, GOPU, ECA) mỗi card một slice.

**Bus operand (cổng A buffer → FU, @500 MHz, bank URAM 72b = 64b dữ liệu + 8b phụ trợ):**

| Bus | Buffer nguồn | Độ rộng | BW | Ghi chú |
|---|---|---:|---:|---|
| WT-bus | WTB 8,0 MiB (4 lane × 2 × 1 MiB, 256 URAM) | 2.048b dữ liệu/lane (32 bank × 64b) | 128 GB/s/lane (512 tổng) @500 MHz | qua dequant FP4→INT8 khi tensor FP4; 8b phụ trợ = scale E8M0 khối 32 |
| RA-bus | RAB 16,0 MiB (4 lane × 16 bank × 8 URAM, 512 URAM) | 1.024b/lane (4.096b tổng) | 64 GB/s/lane (256 tổng) @500 MHz | width-slice 16 bank/lane, broadcast địa chỉ; giữ h thường trú 14,0 MiB |
| KV-bus | KVB 8,0 MiB (4 lane × 8 bank × 8 URAM, 256 URAM) | 512b/lane | 32 GB/s/lane (128 tổng) @500 MHz | phân vùng theo mục KV (idx mod 4), hợp nhất partial qua SLL |
| PS-bus | PSA 6,0 MiB (4 lane × 8 bank × 6 URAM, 192 URAM) | 512b/lane | 32 GB/s/lane (128 tổng) @500 MHz | top-k đóng gói 18b, router, partial softmax |

**Handshake:** mọi cổng dữ liệu là **AXI4-Stream**: `TDATA` (rộng như bảng trên), `TVALID/TREADY`, `TLAST` = hết một tile BLOCK_M, `TUSER = {layer_id[2:0], chunk_id[2:0], tile_id[1:0], phase}`. Cổng ghi (FU → buffer, port B) cùng quy ước. Weight nạp DDR4 → WTB do mSGDMA (AXI4-MM 512b @300 MHz/kênh), không thuộc giao diện FU.

**Precision:** FP4 (weight MoE, indexer) theo MXFP4 khối 32 (scale E8M0), dequant thành INT8 trước MAC bằng LUT phân tán 16 mục — scale áp bằng phép dịch trên tổng tích luỹ INT32; FP8 (weight attention, activation); BF16 (residual, accumulate output); INT32→FP32 trong accumulator DSP. Từ nhớ URAM 72 bit = 64 bit dữ liệu + 8 bit phụ trợ; miền buffer + MAC 500 MHz, AXI/MIG 300 MHz.

**Điều khiển:** mỗi FU có một cổng AXI4-Lite 32b (thanh ghi cấu hình) và cặp tín hiệu `START/DONE` với Ctrl sequencer (§11).

## 2. Bảng tóm tắt dòng dữ liệu một layer

| # | FU | Vào chính (nguồn) | Ra chính (đích) |
|---|---|---|---|
| 1 | mHCE p1 | `X_l` (RAB) | `X̂` (RAB) |
| 2 | ACU | `X̂` (RAB) + `W_aKV/bKV/aZ/bZ` (WTB) | `C^Comp`,`K^IComp` (KVB + DDR append) |
| 3 | LIU | `X̂` (RAB) + `W_DQ/IUQ/w` (WTB) + `K^IComp` (KVB/DDR) | `c_Q` (RAB), top-k idx (PSA) |
| 4 | QCAU | `c_Q` (RAB) + `W_UQ` (WTB) + `𝒞^SprsComp`+`KV_win` (KVB) | `o` (RAB) |
| 5 | GOPU | `o` (RAB) + `W_OG/W_O` (WTB) | partial `ô` (RAB) |
| 6 | ICI | partial `ô` (RAB) | `ô` all-reduced (RAB) |
| 7 | mHCE p2 | `X_l` + `ô` (RAB) | `X'` (RAB) |
| 8 | MRU | `X̂'` (RAB) + `W_router` (WTB) | `top_idx`,`w_top` (PSA) |
| 9 | ICI | `h` home-token (RAB) | `h` remote-token (RAB) |
| 10 | ECA | `h` (RAB) + `W_gate/up/down^(e)` (WTB) | `y_e` (RAB) |
| 11 | ICI | `y_e` (RAB) | `y_e` về home-card (RAB) |
| 12 | mHCE p3 | `X'` + `y` (RAB) | `X_{l+1}` (RAB) |

## 3. mHCE — Hyper-Connection Engine (40 DSP · 0,10 TOPS)

Chuẩn hoá RMSNorm, sinh hệ số động mHC (`A_l`, `B_l`, `C_l`, Sinkhorn–Knopp 20 vòng), trộn residual mở rộng `n_hc = 4`. Chạy 3 pass/layer trên cùng datapath.

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus | Pha |
|---|---|---|---|---:|---|---|
| `i_x` | `X_l` residual mở rộng | (s, n_hc, d) = (256, 4, 7168) | BF16 | 14,68 MB | RAB · RA-bus | p1/p2/p3 |
| `i_f` | `ô` (p2) / `y` (p3) | (s, d) = (256, 7168) | BF16 | 3,67 MB | RAB · RA-bus | p2, p3 |
| `i_w` | `W_pre/res/post_l` ×2 sublayer | (n_hc·d, n_hc), (n_hc·d, n_hc²), (n_hc·d, n_hc) | FP8 | 1,38 MB/layer | WTB · WT-bus | p1 |
| `i_cfg` | `S_*`, `α_*`, `γ` RMSNorm | scalar/vector nhỏ | FP32 | <1 KB | AXI4-Lite | init |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus | Pha |
|---|---|---|---|---:|---|---|
| `o_xn` | `X̂` (flatten sau norm + pre-mix) | (s, d) | FP8 | 1,84 MB | RAB · RA-bus | p1 |
| `o_x` | `X'` / `X_{l+1}` | (s, n_hc, d) | BF16 | 14,68 MB | RAB · RA-bus | p2, p3 |

Ghi chú: `A_l/B_l/C_l` (4,)·(4,4)·(4,) mỗi token là tensor nội bộ, sống trong register file, không ra bus. Sinkhorn 20 vòng trên ma trận 4×4 — tải MAC không đáng kể, nằm trong 40 DSP.

## 4. ACU — Attention Compression Unit (768 DSP · 1,84 TOPS)

Sinh KV nén: chiếu `X̂` qua các nhánh KV/Z, softmax nén khối `m = 4` (CSA) hoặc `m' = 128` (HCA), append vào KV cache.

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus | Điều kiện |
|---|---|---|---|---:|---|---|
| `i_xn` | `X̂` | (256, 7168) | FP8 | 1,84 MB | RAB · RA-bus | luôn |
| `i_w` | `W_aKV, W_bKV, W_aZ, W_bZ` | 4 × (d, c) = 4 × (7168, 512) | FP8 | 14,68 MB/layer | WTB · WT-bus | CSA |
| `i_w` | `W_KV, W_Z` | 2 × (7168, 512) | FP8 | 7,34 MB/layer | WTB · WT-bus | HCA |
| `i_cfg` | `B_a, B_b` bias vị trí, `γ_KV` | (m, c) ×2 | FP8 | 4 KB | AXI4-Lite/BRAM | init |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus |
|---|---|---|---|---:|---|
| `o_kv` | `C^Comp` mới (CSA) | (s/m, c) = (64, 512) | 448 FP8 + 64 BF16 = 576 B/entry | 36,9 KB | KVB · KV-bus **và** DDR append (mSGDMA) |
| `o_kv` | `C^Comp` mới (HCA) | (s/m', c) = (2, 512) | 576 B/entry | 1,2 KB | KVB + DDR |
| `o_ik` | `K^IComp` indexer key mới | (s/m, c_I) = (64, 128) | FP4 | 4,1 KB | KVB + DDR (CSA) |
| `o_win` | `KV_win` sliding window | (n_win, c) = (128, 512) | 576 B/entry | 73,7 KB (vòng) | KVB (ring buffer) |

Ghi chú: tensor trung gian `C_a/C_b/Z_a/Z_b/S_a/S_b` (s, c) sống trong PSA cục bộ, không ra RAB. Ghi DDR = giao dịch duy nhất ghi bộ nhớ ngoài của FRONT (~41 KB/layer/chunk CSA).

## 5. LIU/HCA — Lightning Indexer Unit (1.280 DSP · 3,07 TOPS)

CSA (layer lẻ): sinh latent query `c_Q`, indexer query `q_I`, chấm điểm `I_{t,:}` với `K^IComp` lịch sử, chọn **top-k = 1024** entry/query. HCA (layer chẵn): định vị cluster `m' = 128` (không top-k).

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus | Điều kiện |
|---|---|---|---|---:|---|---|
| `i_xn` | `X̂` | (256, 7168) | FP8 | 1,84 MB | RAB · RA-bus | luôn |
| `i_w` | `W_DQ` | (d, d_c) = (7168, 1536) | FP8 | 11,01 MB/layer | WTB · WT-bus | luôn |
| `i_w` | `W_IUQ` | (d_c, n_I_h·c_I) = (1536, 8192) | FP4 | 6,29 MB/layer | WTB · WT-bus | CSA |
| `i_w` | `W_w` | (d, n_I_h) = (7168, 64) | FP4 | 0,23 MB/layer | WTB · WT-bus | CSA |
| `i_ik` | `K^IComp` lịch sử | (⌈n/m⌉, c_I), FP4 | FP4 | theo context (n=64K → 2 MB) | KVB ← DDR (DMA prefetch) | CSA |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus |
|---|---|---|---|---:|---|
| `o_cq` | `c_Q` | (s, d_c) = (256, 1536) | FP8 | 0,39 MB | RAB · RA-bus |
| `o_idx` | top-k index (`I` đã sort) | (s, k) = (256, 1024) | INT32 | 1,05 MB | PSA · PS-bus |
| `o_fetch` | descriptor fetch KV top-k | danh sách entry unique | — | ≤ 0,7 MB desc | mSGDMA (nạp `𝒞^SprsComp` → KVB) |

Ghi chú: `q_I` (s, n_I_h, c_I) FP4 1,05 MB và `I_{t,:}` là tensor nội bộ (PSA). Top-k dùng bitonic sort trên LUT, không tốn DSP. HCA: chỉ `o_cq` hoạt động.

## 6. QCAU — Query-Core Attention Unit (3.328 DSP · 7,99 TOPS)

Sinh query 32 head slice: `q = c_Q · W_UQ`, áp RoPE (64 dim cuối), core MQA trên KV được chọn + sliding window + attention sink.

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus |
|---|---|---|---|---:|---|
| `i_cq` | `c_Q` | (256, 1536) | FP8 | 0,39 MB | RAB · RA-bus |
| `i_w` | `W_UQ` slice cột 32 head | (d_c, c·32) = (1536, 16384) | FP8 | 25,17 MB/layer | WTB · WT-bus |
| `i_kv` | `𝒞^SprsComp` + `KV_win` | (k + n_win, c) = (1152, 512)/query | 576 B/entry | 664 KB working set/query-tile | KVB · KV-bus |
| `i_cfg` | `z'_h` slice (32,), `γ_Q`, bảng RoPE | nhỏ | FP32/ROM | <8 KB | AXI4-Lite/BRAM |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus |
|---|---|---|---|---:|---|
| `o_o` | `o` attention output 32 head | (s, 32, c) = (256, 32, 512) | BF16 | 8,39 MB | RAB · RA-bus |

Ghi chú: `q` (s, 32, c) FP8 4,19 MB là tensor nội bộ (register/PSA theo tile). Softmax hàng thực hiện online (2-pass, FP32 accumulate). MQA: 32 head dùng chung một dòng KV — KV-bus đọc 1 lần/entry/tile, broadcast 32 head.

## 7. GOPU — Grouped Output Projection Unit (2.432 DSP · 5,84 TOPS)

Chiếu output theo nhóm: `o^G' = o^G · W_OG_i` (4/16 group slice) rồi `ô_partial = o^G' · W_O` (W_O chia **hàng** 1/4). Partial cần all-reduce (§9).

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus |
|---|---|---|---|---:|---|
| `i_o` | `o` (nhóm lại g^chip = 4) | (s, 4, c·n_h/g) = (256, 4, 4096) | BF16 | 8,39 MB | RAB · RA-bus |
| `i_w` | `W_OG_i`, i ∈ slice 4 group | 4 × (4096, d_g) = 4 × (4096, 1024) | FP8 | 16,78 MB/layer | WTB · WT-bus |
| `i_w` | `W_O` slice hàng 1/4 | (g·d_g/4, d) = (4096, 7168) | FP8 | 29,36 MB/layer | WTB · WT-bus |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus |
|---|---|---|---|---:|---|
| `o_op` | partial `ô` | (s, d) = (256, 7168) | BF16 | 3,67 MB | RAB · RA-bus → ICI all-reduce |

Ghi chú: `o^G'` (s, 4, 1024) BF16 2,10 MB nội bộ. Vì W_UQ chia **cột** nên partial theo head cộng cục bộ trong QCAU; vì W_O chia **hàng** (contraction bị chia) nên partial `ô` phải all-reduce 4 card.

## 8. MRU — MoE Router Unit (128 DSP · 0,31 TOPS)

Router: `g_raw = X̂' · W_router` → `g̃ = Sqrt(Softplus(g_raw)) + b_router` → top-6 + chuẩn hoá gating.

**Cổng vào:**

| Port | Tensor | Shape | Prec | /chunk-256 | Nguồn · bus |
|---|---|---|---|---:|---|
| `i_xn` | `X̂'` (sau p2, chuẩn hoá) | (256, 7168) | FP8 | 1,84 MB | RAB · RA-bus |
| `i_w` | `W_router` | (d, n_routed) = (7168, 384) | FP8 | 2,75 MB/layer | WTB · WT-bus |
| `i_cfg` | `b_router` | (384,) | FP32 | 1,5 KB | AXI4-Lite |

**Cổng ra:**

| Port | Tensor | Shape | Prec | /chunk-256 | Đích · bus |
|---|---|---|---|---:|---|
| `o_route` | `top_idx` + `w_top` + flags | (s, 6) INT16 + (s, 6) BF16 + 4 B | 28 B/token | 7,2 KB | PSA · PS-bus |

Ghi chú: `g_raw/g̃` (s, 384) BF16 0,20 MB nội bộ. 3 layer đầu mạng dùng Hash routing — MRU bypass MAC, chỉ phát `top_idx` từ hash(token_id). Router replicate: 4 card cho kết quả giống nhau, không đồng bộ.

## 9. ICI — Inter-Card Interconnect (216 DSP control)

Ba giao dịch collective trên ring 4 card (2×QSFP28, 12,5 GB/s/hướng), gói theo `DeepSeek_V4Pro_InterCard_Protocol.md` (header 32 B).

| Giao dịch | Vào (RAB) | Ra (RAB) | Payload/chunk | Thời gian B=8 |
|---|---|---|---:|---:|
| all-reduce `ô` | partial `ô` (s, d) BF16 3,67 MB | `ô` tổng (s, d) BF16 | 5,5 MB (ring 2×¾) | 3,5 ms |
| all-to-all dispatch | `h` home-token (d,)/token FP8 + `top_idx` | `h` remote-token vào slice expert cục bộ | ~8,3 MB | 5,3 ms |
| all-to-all combine | `y_e` cục bộ (d,)/token-expert FP8 | `y_e` về home-card; cộng `w_top`-weighted | ~8,3 MB | 5,3 ms |

Ghi chú: mọi payload đọc/ghi RAB qua RA-bus; ICI không chạm WTB/KVB. Cộng có trọng số 6 expert + shared thực hiện tại home-card (trong mHCE p3 hoặc adder tree của ICI-combine — chọn khi RTL). Toàn bộ ba giao dịch chồng lấn dưới ECA stream (ngoài đường găng).

## 10. ECA — Expert Compute Array (4.096 DSP · 8,19 TOPS @500 MHz)

SwiGLU cho 96 routed expert slice + 1 shared: `inter = SiLU(h·W_gate) ⊙ (h·W_up)`; `y_e = inter · W_down`. Weight-stream theo tile, batch B=8 dùng chung một lần stream; 4 sub-array 1.024 DSP, mỗi sub-array một SLR + một kênh DDR (24 expert).

**Cổng vào:**

| Port | Tensor | Shape | Prec | Kích thước | Nguồn · bus |
|---|---|---|---|---:|---|
| `i_h` | `h` token sau dispatch | (d,) = (7168,)/token | FP8 | 7,2 KB/token · ~12 MB/batch-8 | RAB · RA-bus |
| `i_route` | danh sách (token, expert, `w_top`) | tuple/token-expert | INT16+BF16 | ~86 KB/batch | PSA · PS-bus |
| `i_w` | `W_gate^(e), W_up^(e), W_down^(e)` | 2×(d, d_ff) + (d_ff, d) = 66,06 M param/expert | FP4 | 33 MB/expert · **3,20 GB/layer** (96+1 shared) | WTB · WT-bus (dequant FP4→INT8) |

**Cổng ra:**

| Port | Tensor | Shape | Prec | Kích thước | Đích · bus |
|---|---|---|---|---:|---|
| `o_y` | `y_e` | (d,)/cặp token-expert | FP8 | 7,2 KB/cặp | RAB · RA-bus → ICI combine |

Ghi chú: `inter^(e)` (d_ff,) = (3072,) INT32→FP32 nội bộ (register file, không ra bus). Thứ tự stream: expert-major (ôm toàn bộ token của batch chạm expert e, rồi sang e+1) — mỗi weight tile đọc DDR đúng 1 lần. Decode/batch nhỏ: chỉ stream expert unique (`N_exp/chip ≈ 96(1−(1−6/384)^N)`).

## 11. Giao diện điều khiển chung

- **Ctrl sequencer** (BRAM 2 MB): phát descriptor thứ tự FU theo §2, quản lý `chunk_id/tile_id`, fence giữa pha (ví dụ ECA chỉ START khi dispatch xong tile đầu). Tín hiệu: `START(fu, layer, chunk_mask)` / `DONE(fu, status)`.
- **EWD prefetch:** trong khi FU k đang C, mSGDMA nạp weight FU k+1 (descriptor do Ctrl phát) — WTB ping-pong 2×1 MB/SLR, tile ≤ 4 MB toàn chip.
- **AXI4-Lite map:** mỗi FU 4 KB register space (base addr weight/DDR, tham số layer CSA/HCA, `s`, `B`, precision mode, γ/bias nhỏ).
- **Đồng hồ:** MAC array 500 MHz; bus buffer cổng A 500 MHz (trần f_max URAM của XCU250-2L ở 0,72 V); fabric control + DMA AXI4-MM 300 MHz; CDC qua async FIFO tại ranh giới control/data và tại cổng B của WTB.

## 12. Ghi chú nhất quán số liệu

1. Tổng weight FRONT/layer/chip tính từ shape danh định: **CSA 107,6 MB · HCA 93,8 MB** — spec vi kiến trúc dùng số làm tròn ~113 MB (gồm bảng scale FP4 per-block, `z'_h`, `γ`, bias; +5%). Roofline không đổi.
2. ECA đọc **3,20 GB/layer** trọng số FP4 thuần; cộng scale E8M0 khối 32 thành **3,40 GB/layer**. Tổng lưu lượng DDR4 một layer (kèm residual X/X′ 470 MB và tích luỹ y 353 MB) là 4,37 GB → chu kỳ layer 66,9 ms.
3. Kích thước C^Comp entry 576 B (448 FP8 + 64 BF16) thống nhất với spec §4 (KV 576 B/token/layer).
4. Số trong tài liệu này là **hợp đồng interface**; nếu RTL đổi banking/width, cập nhật tài liệu này trước, các tài liệu trình bày theo sau.

## 13. Tài liệu tham khảo

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
[2] Nội bộ: `DeepSeek_V4_Variables_Glossary.md` (shape/ký hiệu); `DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md` (DSP, buffer, bus); `DeepSeek_V4Pro_InterCard_Protocol.md` (gói ICI); `DeepSeek_V4Pro_Microarchitecture_Spec.md` (chuỗi FU §10).
[3] AMD, "UltraScale Architecture Memory Resources," UG573; "AXI4-Stream Protocol Specification," ARM IHI 0051.
