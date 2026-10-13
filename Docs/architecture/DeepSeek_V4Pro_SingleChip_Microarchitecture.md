# Vi kiến trúc một chip — DeepSeek-V4-Pro Accelerator
# Single-Chip Microarchitecture — DeepSeek-V4-Pro Accelerator

**Tài liệu thiết kế nội bộ.** Mô tả specification và vi kiến trúc của **một chip** (1 trong 4 chip của một card; 1 trong 32 chip toàn hệ). Bản trực quan: `DeepSeek_V4Pro_SingleChip.html`. Spec hệ thống: `DeepSeek_V4Pro_Microarchitecture_Spec.md`.

---

## 1. Tổng quan & vai trò của chip

| Hạng mục | Giá trị |
|---|---|
| Linh kiện | Intel Agilex 7 M-Series **AGM 039** |
| DSP block | 12.300 (24.600 nhân 18×19) |
| Peak compute | ~37 TFLOPS FP · ~75 TOPS INT8 (≈ 36,9 Gops/ms) |
| SRAM on-chip (M20K) | 370 Mb ≈ **46,25 MB** (~18.944 block) |
| HBM2e (in-package) | **32 GB @ 820 GB/s** |
| Tần số tham chiếu | 750 MHz |

**Vai trò trong card (TP/EP):** một card có **4 chip** chạy song song. Chip giữ **một slice** của *tất cả các layer của card*:
- **Tensor-Parallel (attention):** 32/128 head (vd chip C.1 → head 32–63); output proj 1/4 group.
- **Expert-Parallel (MoE):** 96/384 expert (vd chip C.1 → e96–191).
- FU "replicated" (mHCE, ACU, LIU/HCA, MRU) tính giống nhau trên 4 chip; FU "sharded" (QCAU, GOPU, ECA) mỗi chip một slice.
- Đồng bộ giữa 4 chip qua **ICI**: all-reduce (sau attention) và all-to-all (dispatch/combine cho MoE).

Chip giữ thường trú ~25 GB weight (slice của ~8 layer) trong HBM 32 GB.

## 2. Tập Functional Unit (FU) và phân bổ DSP

Phân bổ DSP **∝ GOP** để C-time các FU cân bằng và để **tầng FRONT (attention) + ECA cùng nằm vừa chip** (xem §9).

| FU | Phép toán | DSP | GOP/chunk(256) | Buffer R | Buffer W |
|---|---|---:|---:|---|---|
| mHCE | RMSNorm + sinh A,B,C + Sinkhorn + mix | 36 | 0,4 | RAB | RAB |
| ACU | KV-compress (nén KV) | 667 | 7,5 | WTB+RAB | KVB (+HBM append) |
| LIU (CSA) / HCA | Lightning indexer Top-k=1024 / cluster m=128 | 1.094 | 12,3 | WTB | PSA |
| QCAU | q·W_UQ + RoPE + core MQA (head slice) | 2.864 | 32,2 | KVB+WTB | RAB |
| GOPU | grouped output proj (group slice) | 2.099 | 23,6 | RAB+WTB | RAB |
| MRU | router → Top-6 + gating | 125 | 1,4 | WTB | PSA |
| ICI | all-reduce / all-to-all (mạng giữa chip) | (ctrl) | — | RAB | RAB |
| ECA | SwiGLU expert (gate/up/SiLU/down, expert slice) | 5.266 | 59,2 | WTB←HBM | RAB |
| EWD/ctrl | DMA prefetch + điều khiển | 150 | — | — | — |
| **Tổng** | | **12.300** | 136,6 | | |

## 3. Bộ nhớ on-chip (M20K 46,25 MB) — buffer working-set

Weight **không** được cache; M20K dùng làm buffer luồng (working-set), không phải cache trọng số.

| Buffer | Nội dung | Dung lượng | #bank (32-bit) | Bus DATA | BW @750MHz |
|---|---|---:|---:|---:|---:|
| **WTB** | weight tile (double-buffer) | 16 MB | 32 | 1.024-bit | 96 GB/s |
| **RAB** | activation (X, q, o, ô, y) | 10 MB | 64 | 2.048-bit | 192 GB/s |
| **KVB** | KV nóng (top-k + sliding window) | 12 MB | 16 | 512-bit | 48 GB/s |
| **PSA** | ∑K partial + top-k/router scratch | 6 MB | 8 | 256-bit | 24 GB/s |
| Ctrl | token queue, packet | 2 MB | 4 | 128-bit | 12 GB/s |

**Banking width-slice:** trong mỗi buffer, một địa chỉ **broadcast** tới mọi bank → đọc song song một từ rộng. M20K **dual-port**: port A = FU đọc (qua AGU), port B = DMA ghi → ping-pong double-buffer.

## 4. Đường HBM ↔ buffer ↔ FU

**Ghi (HBM → buffer), do mSGDMA:**
```
HBM2e ↔ HBM Controller (hard) ↔ Hard NoC [AXI4] ↔ AXI4 interconnect ↔ mSGDMA ↔ port B các bank M20K
```
**Đọc operand (buffer → FU):** banked M20K → **AGU** (sinh địa chỉ) → **arbiter/mux** (chọn FU active) → bus Avalon-ST → FU. KHÔNG dùng AXI-MM cho operand (tránh handshake mỗi chu kỳ); AXI4-Lite chỉ cho thanh ghi cấu hình.

**Đường weight chung & MAC pool dùng chung** là tài nguyên nối tiếp: trong một chip, các FU weight-nặng phải **time-share** đường mSGDMA→WTB; đây là gốc của việc xử lý tuần tự (xem §8–9).

## 5. Phân loại dữ liệu

| Lớp | Nội dung | Vị trí |
|---|---|---|
| Bền vững (init 1 lần) | weight ~25 GB (slice ~8 layer); bias router, sink z'_h, γ | HBM (+ register nhỏ) |
| Transient (FU↔FU) | X, c_Q, q, C^Comp, o, ô, y | M20K (RAB/PSA), sống theo chunk |
| Vào/ra chip | residual X (inter-card), all-to-all h/y + all-reduce ô (ICI), weight/KV stream (HBM) | link / ICI / HBM |

Ghi HBM/layer: chỉ KV-cache append (~37 KB). Đọc HBM/layer: weight (expert ~3,17 GB chi phối + attn ~113 MB) + top-k KV.

## 6. Chia tile weight — ràng buộc WTB 16 MB

Weight một layer ≫ WTB ⇒ phải **stream theo tile**, double-buffer 2 × 8 MB ping-pong (tile ≤ 8 MB).

| Khối | Weight/chip | Số tile ≤8MB | Reuse (row/tile, chunk 256) |
|---|---:|---:|---:|
| ACU/LIU/QCAU/GOPU/MRU (attention) | ~113 MB | ~15 | 256 |
| **ECA — 96 expert** | ~3,17 GB | **~400** | ~4 (rải trên 96 expert) |

Cơ chế: `for weight_tile: nạp HBM→WTB nửa B ∥ MAC đọc nửa A (×256 row) → đổi A/B`. MAC không chờ giữa các tile nếu compute(tile) ≥ load(tile); ECA thì **load > compute → read-bound**.

## 7. Đọc weight expert & độ phủ theo batch (ECA)

ECA đọc theo **số expert UNIQUE mà cả batch chạm tới**, không nhân theo từng token. Top-6, 384 expert, 96/chip. Số expert unique/chip với batch N token:

$$N_{exp/chip} \approx 96\left(1-\left(1-\tfrac{6}{384}\right)^{N}\right)$$

| Batch N | Expert unique/chip | Weight đọc/chip |
|---:|---:|---:|
| 1 | 1,5 | ~50 MB |
| 8 (decode) | ~12 | ~0,40 GB |
| 64 | ~60 | ~2,0 GB |
| **256 (chunk prefill)** | **~94 → 96** | **~3,17 GB** |
| ≥512 | ~96 (bão hoà) | 3,17 GB (trần) |

Mỗi expert (66 M tham số, FP4 0,5 B) = **33 MB**; 96 × 33 = 3,17 GB. Với chunk 256, ~98% expert bị chạm → đọc gần như toàn bộ slice. Weight đọc 1 lần, reuse ~4 token/expert ⇒ MoE prefill **memory-bound** (arithmetic intensity ≈ 4×reuse FLOP/byte < điểm cân bằng 45).

## 8. Luồng dữ liệu một layer (R → C → W theo FU)

Một chunk đi qua chuỗi 11 bước; mỗi FU: **đọc (R) → tính (C) → ghi (W)** tuần tự (một mảng MAC dùng chung).

| Bước | FU | ĐỌC | TÍNH | GHI |
|---|---|---|---|---|
| 1 | mHCE | X (RAB) | RMSNorm+mix | RAB |
| 2 | ACU | W (WTB) + X | KV-compress | KVB + **HBM KV append** |
| 3 | LIU/HCA | W (WTB) | indexer Top-k / cluster | PSA |
| 4 | QCAU | KV (KVB) + W (WTB) | MQA+RoPE (head slice) | RAB (o) |
| 5 | GOPU | o (RAB) + W (WTB) | grouped proj (group slice) | RAB (ô) → **all-reduce** |
| 6 | mHCE | X + ô | cộng residual | RAB |
| 7 | MRU | W (WTB) | router Top-6 | PSA |
| 8 | ICI | h (RAB) | **all-to-all dispatch** | RAB |
| 9 | ECA | **expert W (WTB←HBM, ~3,17 GB)** + token | SwiGLU (expert slice) | RAB (y) |
| 10 | ICI | y | **all-to-all combine** | RAB |
| 11 | mHCE | X + y | cộng → X' | RAB |

Một chip xử lý cả ~8 layer của card **tuần tự** (cùng tập FU, weight mỗi layer khác nhau) ⇒ ~50 ms/chunk; đây là **một stage của pipeline mức card**.

## 9. Pipeline 2 tầng trên chip: FRONT ∥ ECA

Gộp FU 1–8 thành tầng **FRONT** (attention+router+dispatch) và **ECA** là tầng sau. Vì DSP riêng và **tổng vừa chip**, hai tầng chạy đồng thời cho các chunk/user khác nhau:

| Tầng | DSP | Thời gian/chunk | Bound |
|---|---:|---:|---|
| FRONT (FU1–8) | ~7.000 | ~2,2 ms (8 FU tuần tự) | compute |
| ECA (FU9) | 5.266 | ~3,86 ms (stream 3,17 GB) | **HBM (read)** |
| **Tổng DSP** | **≈ 12.300** | chu kỳ = max = **3,86 ms/chunk** | ECA-bound |

Steady-state: `FRONT(U_{i+1}) ∥ ECA(U_i)` ⇒ thông lượng ~**×1,5** so với tuần tự; FRONT lấp DSP attention vào khoảng ECA chờ HBM. ECA mỗi chunk vẫn chờ ~2,3 ms (read-bound) → **batch nhiều chunk tại ECA** (dùng chung một lần stream weight) để lấp nốt. Song song "user khác nhau" ở quy mô lớn đặt ở **mức card** (8 card, 8 microbatch).

## 10. Số liệu tổng hợp (1 chip)

| Đại lượng | Giá trị |
|---|---:|
| DSP / MAC | 12.300 / 24.600 (36,9 Gops/ms) |
| SRAM on-chip | 46,25 MB (WTB16/RAB10/KVB12/PSA6/Ctrl2) |
| HBM | 32 GB @ 820 GB/s |
| Weight thường trú | ~25 GB (slice ~8 layer) |
| Weight đọc/layer (prefill) | ~3,3 GB (~415 tile 8 MB) |
| ECA: load / compute | 3,86 / 1,60 ms (read-bound) |
| FRONT compute | ~2,2 ms |
| Pipeline 2 tầng | ~3,86 ms/chunk (×1,5) |
| Layer/chip (Card 0) | 8 (~50 ms/chunk) |

## Tài liệu tham khảo
[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," 2026. [2] Intel, "Agilex 7 M-Series NoC/HBM2e User Guide," 2025. [3] D. Dai et al., "DeepSeekMoE," 2024. [4] S. Williams et al., "Roofline," CACM 2009.
