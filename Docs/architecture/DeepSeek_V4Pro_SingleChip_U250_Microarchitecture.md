# Vi kiến trúc một chip trên AMD Alveo U250 — DeepSeek-V4-Pro Accelerator
# Single-Chip Microarchitecture on AMD Alveo U250 — DeepSeek-V4-Pro Accelerator

**Tài liệu thiết kế nội bộ.** Đặc tả nền tảng Alveo U250 và vi kiến trúc **một chip** thực thi **8 layer** DeepSeek-V4-Pro theo vai trò **slice 1/4 card** (32/128 head, 96/384 expert) — tương đương vai trò một chip AGM 039 trong `DeepSeek_V4Pro_SingleChip_Microarchitecture.md`. Một pipeline-stage (8 layer) gồm **4 card U250** ghép TP/EP qua QSFP28.

---

## Mục lục

1. Đặc tả nền tảng Alveo U250 (đã kiểm chứng nguồn AMD)
2. So sánh nền tảng U250 vs AGM 039 — hệ quả kiến trúc
3. Vai trò chip và ánh xạ mô hình
4. Ngân sách bộ nhớ DDR4 (weight thường trú + KV)
5. Roofline: nút cổ chai DDR4 và quyết định B = 8
6. Tập FU và phân bổ DSP (đảo ngược so với Agilex)
7. Bộ nhớ on-chip: URAM/BRAM banking (định dạng từ · ràng buộc lưu trú · phân bổ)
8. Đường DDR4 ↔ buffer ↔ FU; stripe 4 kênh × 4 SLR
9. Floorplan SLR
10. Pipeline FRONT ∥ ECA trên U250
11. ICI (QSFP28) và liên stage (PCIe)
12. Prefill / Decode — số liệu định lượng
13. Bảng số liệu tổng hợp và so sánh
14. Rủi ro và phương án dự phòng
15. Khuyến nghị (action items)
16. Tài liệu tham khảo

---

## 1. Đặc tả nền tảng Alveo U250 (đã kiểm chứng nguồn AMD)

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| FPGA | Virtex UltraScale+ **XCU250**-2LFIGD2104E, 16 nm, **4 SLR** | [2] |
| LUT / Register | 1.728K / 3.456K | [1] |
| DSP | **12.288** DSP48E2 (27×18 mult + ALU 48-bit) | [1] |
| Block RAM | 2.688 × 36 Kb = 94,5 Mb ≈ **11,8 MB** | [3] |
| UltraRAM | 1.280 × 288 Kb = 360 Mb = **45,0 MiB** (320/SLR) | [1], [3] |
| URAM288 | 4.096 từ × 72 bit; 2 cổng dùng chung clock; cascade trong cột | [3] |
| f_max URAM, −2 @0,90/0,85 V, OREG_B = True | 600 MHz | [10] |
| f_max URAM, −2 @**0,72 V** (linh kiện 2L của U250), OREG_B = True | **500 MHz** | [10] |
| f_max URAM, OREG_B = False (không ECC / có ECC) | 404–528 / 312–435 MHz | [10] |
| Internal SRAM (AMD công bố) | 54 MB @ 38 TB/s (khả dụng sau shell) | [4] |
| DDR4 | **4 kênh × 16 GB** DIMM, 2400 MT/s, 64-bit+ECC → **64 GB @ 77 GB/s** (4 × 19,2) | [1], [2] |
| PCIe | Gen3 ×16 (~15,8 GB/s raw, ~13 GB/s hiệu dụng) | [1] |
| Network | **2 × QSFP28 (100 GbE)** = 2 × 12,5 GB/s | [4] |
| Peak INT8 | **33,3 TOPS** (= 24.576 mult × 2 op × 679 MHz) | [4] |
| Công suất tối đa | 225 W (passive) / 215 W tải điện | [1], [4] |

Điểm khác biệt quyết định so với AGM 039: **không có HBM**. DDR4 77 GB/s = **9,4%** băng thông HBM2e 820 GB/s; SRAM on-chip 54 MB ≈ 46,25 MB M20K (tương đương); DSP 12.288 ≈ 12.300 (tương đương).

**Tần số thiết kế:** miền compute + buffer **500 MHz**, fabric/AXI/MIG 300 MHz. Trần 500 MHz do URAM quyết định: XCU250-**2L** trên Alveo U250 vận hành V_CCINT = 0,72 V, ở đó f_max URAM với OREG_B = True là 500 MHz [10]; DSP48E2 còn biên nhưng chạy cùng miền để tránh CDC giữa buffer và MAC. Peak INT8 tại 500 MHz = 24.576 × 2 op × 0,5 GHz = **24,6 TOPS**. INT8 dùng packing 2 mult/DSP (2 weight FP4→INT8 dồn cổng 27-bit, chia sẻ activation cổng 18-bit) [5].

## 2. So sánh nền tảng — hệ quả kiến trúc

| Đại lượng | AGM 039 (1 chip) | U250 (1 card) | Tỉ lệ |
|---|---:|---:|---:|
| Peak INT8 | 75 TOPS | 24,6 TOPS (@500 MHz) | 0,33× |
| BW bộ nhớ ngoài | 820 GB/s (HBM2e) | 77 GB/s (65,3 GB/s hiệu dụng @85%) | **0,094×** |
| Dung lượng bộ nhớ ngoài | 32 GB | 64 GB | 2,0× |
| SRAM on-chip | 46,25 MB | 45,0 MiB URAM + 11,8 MiB BRAM; **40,0 MiB dữ liệu** sau định dạng từ 64 b (§7.1) | ~0,9× |
| DSP | 12.300 | 12.288 | 1,0× |

Hệ quả: compute giảm 2,5× nhưng băng thông giảm **10,6×** → mọi khâu weight-streaming (ECA) trở thành nút cổ chai tuyệt đối. Ba quyết định kiến trúc bắt buộc:

1. **Đảo phân bổ DSP**: ECA không cần 5.266 DSP như Agilex (compute chỉ cần theo kịp 65 GB/s); DSP dồn về FRONT và về việc **batch nhiều chunk tại ECA**.
2. **Stripe weight trên cả 4 kênh DDR4** (mỗi kênh 19,2 GB/s): ECA chia thành 4 sub-array, mỗi sub-array bám một kênh DDR + một SLR — luồng weight không bao giờ vượt ranh giới SLR.
3. **Batching B=8 chunk tại ECA** là đòn bẩy thông lượng chính: một lần stream 3,40 GB (trọng số FP4 kèm scale E8M0) phục vụ 8 chunk (2.048 token).

## 3. Vai trò chip và ánh xạ mô hình

Giữ nguyên kiến trúc hệ thống 3 cấp song song của `DeepSeek_V4Pro_Microarchitecture_Spec.md`; thay 1 chip AGM 039 bằng 1 card U250:

- **Pipeline parallel:** 8 stage × 8 layer; stage s giữ layer 8s+1…8s+8.
- **TP (attention):** U250 giữ **32/128 head** (W_UQ chia cột), 4/16 group W_O (chia hàng); all-reduce ô qua QSFP28.
- **EP (MoE):** U250 giữ **96/384 expert** khối nguyên; all-to-all dispatch/combine qua QSFP28.
- FU replicated: mHCE, ACU, LIU/HCA, MRU. FU sharded: QCAU, GOPU, ECA.
- Toàn hệ: 8 stage × 4 card = **32 card U250**.

## 4. Ngân sách bộ nhớ DDR4 (64 GB)

| Nội dung | Dung lượng | Ghi chú |
|---|---:|---|
| Weight slice 8 layer (FP4 MoE + FP8 attn) | **~25 GB** | 96 expert × 33 MB × 8 layer ≈ 24,3 GB + attn ~0,9 GB |
| KV cache (8 layer, 576 B/token/layer) | ~4,6 GB / 1M-token seq | 448 B FP8 + 64×2 B BF16 |
| Trần KV | ~39 GB → **~8,5 M token** (B×ctx) | so 31,6 M/card Agilex |
| Embed/head (chỉ stage 0/7) | ~1,6 GB | FP8 |

Weight thường trú chiếm 39% — **vừa thoải mái** (Agilex: 78%). Dung lượng không phải ràng buộc; băng thông mới là ràng buộc.

**Bố trí kênh:** weight expert và attention **stripe đều 4 kênh** (interleave theo tile 1 MB/kênh); KV cache phân vùng theo kênh cùng SLR với QCAU/ACU truy cập.

## 5. Roofline: nút cổ chai DDR4 và quyết định B = 8

Lưu lượng DDR4 một layer, một chip, B = 8 (2.048 token), tính đủ scale MXFP4 khối 32 [11], residual X/X′ và tích luỹ y (§7.2):

| Thành phần | Traffic DDR/layer | Ghi chú |
|---|---:|---|
| Trọng số expert FP4 (97 × 33 MB) | 3.201 MB | 96 routed + 1 shared |
| Scale E8M0 khối 32 | 200 MB | +6,25% trên phần trọng số |
| Trọng số attention + router FP8 | 108 MB | W_aKV…W_bZ 14,7 · W_DQ 11,0 · W_IUQ 6,3 · W_w 0,2 · W_UQ 25,2 · W_OG 16,8 · W_O 29,4 · W_router 2,8 · W_mHC 1,4 |
| Residual X/X′ BF16 (2 đọc + 2 ghi × 117,4 MB) | 470 MB | X_l = (2.048, 4, 7168) = 112,0 MiB |
| Tích luỹ y qua DDR4 (6 lượt đọc–sửa–ghi × 29,4 MB) | 353 MB | quyết định ở §7.2 |
| KV top-k (query-tile M = 8) + append | 40 MB | hợp 1.264 mục/tile |
| **Tổng** | **4.372 MB** | |
| **Chu kỳ layer @65,3 GB/s** | **66,9 ms** | |

| Khâu | GOP (B=8) | Peak @500 MHz | Thời gian compute |
|---|---:|---:|---:|
| ECA (4 sub-array, 4.096 DSP) | 473,6 | 8,19 TOPS | **57,8 ms** < 66,9 ms |
| FRONT (FU1–8, 7.976 DSP) | 619,0 | 15,95 TOPS | **38,8 ms** < 66,9 ms |

- ECA vẫn read-bound sau khi hạ clock: compute 57,8 ms nằm dưới chu kỳ DDR 66,9 ms, MAC util của ECA đạt 86% chu kỳ layer.
- Điểm cân bằng compute–read của ECA ở 500 MHz là B ≤ 9,3. Ràng buộc quyết định B không còn là MAC mà là **dung lượng RAB**: h thường trú tỉ lệ thuận với B (1,75 MiB mỗi chunk), xem §7.2.
- Giữ **B = 8** (2.048 token): h = 14,0 MiB vừa quỹ RAB 16,0 MiB sau khi đưa tích luỹ y xuống DDR4.

## 6. Tập FU và phân bổ DSP (U250)

Nguyên tắc mới: **DSP ∝ GOP chỉ trong FRONT** (để chuỗi FU pipeline tile cân bằng); ECA lấy đúng mức bão hoà B=8; tổng vừa 12.288 và chia đều được 4 SLR (3.072 DSP/SLR).

| FU | Phép toán | DSP (U250) | DSP (Agilex) | Peak @500 MHz | C-time B=8 |
|---|---|---:|---:|---:|---:|
| mHCE | RMSNorm + A,B,C + Sinkhorn + mix | 40 | 36 | 0,08 TOPS | 40,0 ms |
| ACU | KV-compress | 768 | 667 | 1,54 TOPS | 39,1 ms |
| LIU/HCA | Lightning indexer Top-k / cluster | 1.280 | 1.094 | 2,56 TOPS | 38,4 ms |
| QCAU | q·W_UQ + RoPE + MQA (32 head) | 3.328 | 2.864 | 6,66 TOPS | 38,8 ms |
| GOPU | grouped output proj (4 group) | 2.432 | 2.099 | 4,86 TOPS | 38,8 ms |
| MRU | router Top-6 + gating | 128 | 125 | 0,26 TOPS | 43,8 ms* |
| **ECA** | SwiGLU 96 expert, **4 sub-array × 1.024** | **4.096** | 5.266 | 8,19 TOPS | 57,8 ms |
| EWD/ICI/DMA | prefetch + all-reduce/all-to-all ctrl | 216 | 150 | (control) | — |
| **Tổng** | | **12.288** | 12.300 | 24,6 TOPS | chu kỳ 66,9 ms |

(*) MRU chồng lấn với ICI dispatch, không nằm trên đường găng. FRONT (FU1–8) tổng 7.976 DSP = 1.994/SLR; ECA 1.024/SLR; EWD 54/SLR → 3.072/3.072 DSP (phân bổ danh định 100%; RTL giữ dự phòng bằng cách dồn mHCE/MRU sang LUT logic khi cần).

Precision: weight FP4 lưu DDR theo MXFP4 khối 32 (scale E8M0) [11], **dequant FP4→INT8 trong fabric** (LUT phân tán 16 mục, sau WTB, trước MAC; scale áp bằng phép dịch trên tổng tích luỹ INT32, không nhân vào từng trọng số); MAC INT8 packing 2/DSP; accumulate INT32→FP32 tại ALU DSP; softmax/SiLU/RMSNorm bằng piecewise-LUT trong BRAM.

## 7. Bộ nhớ on-chip: URAM/BRAM banking (định dạng từ · ràng buộc lưu trú · phân bổ)

### 7.1. Định dạng từ nhớ 72 bit

Từ URAM rộng 72 bit, chia **64 bit dữ liệu + 8 bit phụ trợ**: 16 giá trị FP4, hoặc 8 giá trị FP8, hoặc 4 giá trị BF16. Tám bit phụ trợ mang scale E8M0 của khối 32 (ghi ở từ chẵn) trong WTB, và parity tính trong fabric ở các buffer còn lại; ECC nội URAM bị loại vì hạ f_max xuống 312–435 MHz [10].

Hệ quả:

- Mọi tensor có kích thước luỹ thừa 2 ánh xạ khít vào địa chỉ; một nhịp AXI 512 bit ghi đúng 8 bank × 64 bit; một mục KV nén 576 B = 9 nhịp 512 bit; 32 chỉ số top-k 18 bit = 9 từ khít.
- Chi phí 11,1% dung lượng thô: 45,0 MiB URAM → **40,0 MiB dữ liệu**.
- 32 URAM chứa đúng 1 MiB dữ liệu (32 × 4.096 × 64 bit), nên một nửa tile WTB của một lane là đúng 32 khối.

### 7.2. Ràng buộc lưu trú tại B = 8

| Tensor | Hình dạng | Toàn batch-8 | Nơi lưu | Căn cứ |
|---|---|---:|---|---|
| X_l | (2.048, 4, 7168) BF16 | 112,0 MiB | DDR4 | vượt toàn bộ URAM của linh kiện |
| o | (2.048, 32, 512) FP8 | 32,0 MiB | RAB theo tile | fuse QCAU→GOPU, không hiện thực hoá cả batch |
| ô | (2.048, 7168) BF16 | 28,0 MiB | RAB theo tile | tiêu thụ ngay trong chuỗi FRONT |
| y tích luỹ | (2.048, 7168) BF16 | 28,0 MiB | **DDR4** | xem đoạn dưới |
| h | (2.048, 7168) FP8 | 14,0 MiB | **RAB, thường trú** | ECA duyệt 97 expert, thứ tự chạm token do routing quyết định |
| chỉ số top-k | 2.048 × 1.024 × 18 bit | 4,5 MiB | PSA | QCAU đọc lại theo query-tile |
| X̂ · c_Q | | 14,0 · 3,0 MiB | RAB theo tile | |

h và y đều bị truy cập rải rác suốt pha expert, nhưng h + y ở BF16 = **42,0 MiB vượt toàn bộ 40,0 MiB dữ liệu** của XCU250. Giữ h trên chip (14,0 MiB) và đưa tích luỹ y xuống DDR4 là phương án duy nhất còn chỗ cho WTB, KVB và PSA; chi phí là 353 MB/layer, tức 8,1% lưu lượng và 5,4 ms trong chu kỳ 66,9 ms. Hai phương án thay thế: hạ B xuống 4 (h + y = 21,0 MiB vừa RAB 24,0 MiB, nhưng thông lượng giảm một nửa vì chu kỳ layer vẫn do stream weight quyết định), hoặc tích luỹ y ở FP8 trên chip (h + y = 28,0 MiB, không còn chỗ cho PSA 6,0 MiB và KVB).

### 7.3. Bảng phân bổ

| Buffer | Bank/lane | URAM/bank | URAM/lane | URAM tổng | Độ sâu bank | Độ rộng dữ liệu/lane | Dung lượng dữ liệu | BW @500 MHz | Yêu cầu đỉnh | Hệ số dư |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTB | 2 × 32 | 1 | 64 | 256 | 4.096 | 2.048 b | 8,0 MiB | 512 GB/s | 42,7 GB/s | 12,0× |
| RAB | 16 | 8 | 128 | 512 | 32.768 | 1.024 b | 16,0 MiB | 256 GB/s | 14,0 GB/s | 18,3× |
| KVB | 8 | 8 | 64 | 256 | 32.768 | 512 b | 8,0 MiB | 128 GB/s | 22,0 GB/s | 5,8× |
| PSA | 8 | 6 | 48 | 192 | 24.576 | 512 b | 6,0 MiB | 128 GB/s | 14,0 GB/s | 9,1× |
| **Cộng** | | | **304** | **1.216** | | | **38,0 MiB** | | | |
| Dự phòng | | | 16 | 64 | | | 2,0 MiB | | | |
| Quỹ XCU250 | | | 320 | 1.280 | | | 40,0 MiB | | | |

Mức dùng URAM 1.216/1.280 = **95,0%**, đồng đều 304/320 trên cả 4 SLR.

Kiểm tra chiếm chỗ:

- **RAB**: h 14,0 MiB + tile FRONT M = 8 ping-pong (X/X′ tại chỗ 0,875 · X̂ 0,109 · c_Q 0,023 · o FP8 0,250 · ô 0,219 = 1,48 MiB) = **15,48 MiB** trên 16,0 MiB.
- **PSA**: chỉ số top-k 4,50 MiB + partial softmax chờ hợp nhất (8 query × 32 head × 512 × 2 B × 2) 0,50 MiB + router 0,055 MiB = **5,06 MiB** trên 6,0 MiB.
- **KVB**: 8,0 MiB ÷ 576 B = 14.563 mục; ping-pong → 7.281 mục/tile, tức 910 mục phân biệt trên 1.152 mục (top-k 1.024 + cửa sổ 128) mỗi query ở M = 8. Mô hình hợp top-k với δ = 16 cho 1.264 mục/tile, dư 5,8×.
- **WTB**: tile 4 MiB dữ liệu = 4 lane × 1 MiB; ở mật độ DDR 4,25 bit/trọng số, một nửa tile là 1,114 MB DDR, nạp **68,3 µs** @16,3 GB/s/kênh — vẫn ẩn hoàn toàn dưới thời gian compute tile.

KVB phân vùng **theo mục KV** (idx mod 4 = lane) thay vì đặt ở SLR2/SLR3: MQA dùng chung một mục KV cho cả 32 head, nên chia head theo SLR buộc nhân bản KVB hoặc mở bus KV 2.048 bit vượt SLR. Chia theo mục giữ KV cục bộ trong lane; mỗi lane tính phần tổng softmax trên 1/4 số mục cho cả 32 head, hợp nhất kiểu flash-attention qua SLL (q 16 KiB/query phát đi, partial 32 KiB/query trả về, 1,2 GB/s mỗi ranh giới).

BRAM: Ctrl/CtxQ 2 MiB (466 RAMB36), FIFO đàn hồi vượt SLR 96, ROM hàm phi tuyến 12, shell ~250 → **~824 / 2.688 (30,7%)**. Dequant FP4→INT8 là LUT phân tán, không chiếm BRAM.

Giữ nguyên nguyên tắc: một địa chỉ **broadcast** mọi bank trong lane (width-slice); cổng A = FU đọc qua AGU, cổng B = DMA ghi; **không cache weight** — chỉ buffer luồng.

## 8. Đường DDR4 ↔ buffer ↔ FU

**Ghi (DDR4 → buffer):**
```
DDR4 DIMM ↔ DDR4 Controller (MIG, 1/SLR) ↔ AXI4 512-bit @300 MHz ↔ XDMA/mSGDMA descriptor ↔ CDC FIFO 300→500 MHz ↔ cổng B bank URAM (WTB/KVB/RAB)
```
Mỗi kênh 512-bit × 300 MHz = 19,2 GB/s — khớp đúng tốc độ DIMM; **4 DMA engine độc lập, mỗi engine phục vụ một SLR**, không đi qua SLR khác (U250 không có NoC cứng như Agilex — mọi đường dữ liệu rộng phải quy hoạch tay).

**Đọc operand (buffer → FU):** bank URAM → AGU → arbiter/mux (FU active) → AXI4-Stream nội SLR → dequant FP4→INT8 → MAC. AXI4-Lite chỉ cho thanh ghi cấu hình. Activation vượt SLR qua **elastic FIFO trên SLL** (17.280 SLL/ranh giới, dùng ~1.200 bit + handshake, pipelined 2–3 tầng). Độ rộng bus nội SLR sau định dạng từ 64 bit: WT-bus 2.048 b, RA-bus 1.024 b, KV-bus 512 b, PS-bus 512 b, tất cả @500 MHz.

**Quy tắc vàng SLR:** weight (rộng, 512+ bit) **không bao giờ vượt SLR**; chỉ activation/token (hẹp, ≤2 KB/token FP8) và tín hiệu điều khiển vượt SLR.

## 9. Floorplan SLR

| SLR | DDR | FU / khối | DSP | Ghi chú |
|---|---|---|---:|---:|
| SLR0 | CH0 | ECA-sub0 (expert e0–23) + ¼ FRONT (QCAU head 0–7 …) + KVB lane 0 | ~3.028 | QSFP28-0 GTY |
| SLR1 | CH1 | **Shell tĩnh (XDMA PCIe)** + Ctrl + MRU + mHCE + ECA-sub1 (e24–47) + ¼ FRONT + KVB lane 1 | ~3.028 | vùng tĩnh chiếm một phần |
| SLR2 | CH2 | ECA-sub2 (e48–71) + ¼ FRONT + KVB lane 2 | ~3.028 | QSFP28-1 GTY |
| SLR3 | CH3 | ECA-sub3 (e72–95) + ¼ FRONT + KVB lane 3 | ~3.028 | |

FRONT chia ¼ theo head/group (QCAU 8 head/SLR, GOPU 1 group/SLR) — partial cộng nội chip qua cây cộng hẹp vượt SLR (BF16, chấp nhận được). Expert **khối nguyên trong một SLR** (24 expert/SLR): token dispatch tới SLR giữ expert, không bao giờ chia một expert qua hai SLR.

## 10. Pipeline FRONT ∥ ECA trên U250

Giữ pipeline 2 tầng nhưng ngữ nghĩa đổi: đơn vị công việc là **batch 8 chunk** (2.048 token, có thể của nhiều user — continuous batching).

| Tầng | DSP | Thời gian/batch-8/layer | Bound |
|---|---:|---:|---|
| FRONT (FU1–8) | 7.976 | 38,8 ms | compute |
| ECA (4 sub-array) | 4.096 | 57,8 ms compute · 52,4 ms đọc weight | **DDR** |
| **Chu kỳ layer** | 12.288 | **66,9 ms** (= tổng traffic 4,37 GB ÷ 65,3 GB/s) | DDR |

Steady-state: `FRONT(batch i+1) ∥ ECA(batch i)`; cả hai cùng kéo DDR nên chu kỳ tính theo **tổng traffic**, không theo max. MAC util: ECA 57,8/66,9 = **86%**; toàn chip (619 + 474) GOP ÷ (24,6 TOPS × 66,9 ms) = **66%**.

Một stage (8 layer) xử lý một batch-8 trong **~535 ms**; 4 card trong stage chạy đồng thời trên slice của mình.

## 11. ICI (QSFP28) và liên stage (PCIe)

**Nội stage — ring 4 card, dùng cả 2 cổng QSFP28** (mỗi card nối 2 láng giềng, 12,5 GB/s/hướng):

| Giao dịch | Kích thước/chunk | Thời gian (ring, B=8) | % chu kỳ layer |
|---|---:|---:|---:|
| all-reduce ô (sau GOPU) | 5,5 MB | 8 × 0,44 = 3,5 ms | 5,2%* |
| all-to-all dispatch + combine | 2 × 8,3 MB | 8 × 1,3 = 10,6 ms | 15,8%* |

(*) Chồng lấn hoàn toàn với ECA stream 52,4 ms (khác pha trong chuỗi FU) → không nằm trên đường găng; yêu cầu lịch ICI xen kẽ tile. Giao thức gói giữ nguyên `DeepSeek_V4Pro_InterCard_Protocol.md` (header 32 B), đóng trên 100 GbE (Aurora/UDP-RoCE tùy shell).

**Liên stage:** residual X (56 KB/token BF16 mHC) — batch-8 = 115 MB → **8,8 ms qua PCIe Gen3 ×16 (13 GB/s) qua host**, chồng lấn với 535 ms/stage (1,6%). Không cần link quang riêng giữa stage.

**Boot:** nạp 25 GB weight qua PCIe ≈ 1,9 s/card (song song 32 card).

## 12. Prefill / Decode — số liệu định lượng

| Chế độ | Cấu hình | Thời gian/layer | Stage 8 layer | Thông lượng | Ghi chú |
|---|---|---:|---:|---:|---|
| Prefill B=8 | 2.048 token | 66,9 ms | 535 ms | **3.828 tok/s/chip** | DDR-bound, MAC 66% |
| Prefill B=1 | 256 token | 55,4 ms | 443 ms | 578 tok/s | traffic X/y/KV co theo B |
| Decode alone, b=1 | 1 token | 2,63 ms (172 MB) | 21,1 ms | 47 tok/s hệ; **TPOT 169 ms** | memory-bound, MAC <1% |
| Decode batch 32 | 32 token | 22,3 ms (1,46 GB) | 178 ms | 180 tok/s hệ; TPOT 1,42 s | expert coverage 38/96 |
| **Fused** (khuyến nghị) | 8 chunk + D decode | 66,9 ms | 535 ms | 3.828 prefill + D/0,535 decode | decode đi ké stream weight |

- **TTFT** (prompt ≤256 token): ~8 stage × 535 ms ≈ **4,3 s** (Agilex: 84 ms). U250 là nền tảng **thông lượng/prototype**, không phải latency.
- Lịch khuyến nghị: khi còn prefill chờ → fused; khi hàng đợi chỉ còn decode → chuyển **decode-alone b=1..8** để kéo TPOT xuống 169–510 ms.

## 13. Bảng số liệu tổng hợp và so sánh (1 chip, slice 8 layer)

| Đại lượng | AGM 039 | **U250** | Tỉ lệ |
|---|---:|---:|---:|
| Weight thường trú | 25 GB / 32 GB (78%) | 25 GB / 64 GB (39%) | — |
| Trần KV (B×ctx) | ~7,9 M tok/chip | ~8,5 M tok | 1,1× |
| Chu kỳ layer (chunk/batch) | 3,87 ms | 66,9 ms | 17,3× |
| Prefill throughput | ~24.500 tok/s | **~3.828 tok/s** (B=8) | 0,16× |
| Decode alone (b=1, hệ) | ~629 tok/s | ~47 tok/s | 0,07× |
| TTFT (chunk 1) | 84 ms | ~4,3 s | 51× |
| MAC util prefill | ~97% | ~66% | — |
| SRAM dùng | 46 MB | 38,0 MiB dữ liệu (1.216/1.280 URAM) + ~3,6 MiB BRAM | — |
| DSP dùng | 12.300 | 12.288 | — |
| Công suất card | (HBM package) | ≤225 W | — |

## 14. Rủi ro và phương án dự phòng

1. **Miền 500 MHz không đóng timing 4-SLR** → hạ 450 MHz: peak 22,1 TOPS, ECA compute 64,2 ms vẫn dưới chu kỳ DDR 66,9 ms, chu kỳ layer không đổi; băng thông mọi buffer giảm 10%, hệ số dư nhỏ nhất (KVB) còn 5,2×. Căn cứ: f_max URAM ở −2 @0,72 V là 500 MHz với OREG_B = True, biên bằng 0 [10].
2. **DDR efficiency <85%** (refresh, bank conflict khi stream weight xen kẽ KV và y read-modify-write) → dành CH riêng cho y trong pha combine; đo bằng microbenchmark MIG trước khi chốt. Mỗi 1% hiệu suất DDR đổi 0,67 ms chu kỳ layer.
3. **Shell chiếm URAM/BRAM SLR1** → dự phòng 64 khối URAM chưa phân bổ (5,0%); nếu thiếu, WTB lane 1 giảm 2 × 24 URAM (0,75 MiB/nửa, tile 3,5 MiB, nạp 51,2 µs — vẫn ẩn được).
4. **Tích luỹ y qua DDR4 làm tăng 353 MB/layer** (8,1% lưu lượng). Nếu đo được độ phủ expert thấp hơn giả định (một token chạm <6 expert trên toàn hệ), số lượt đọc–sửa–ghi giảm tương ứng. Phương án dự phòng: hạ B xuống 4 để y 14,0 MiB + h 7,0 MiB vừa RAB 16,0 MiB, đổi lấy thông lượng giảm một nửa.
5. **Phân tán top-k cao hơn mô hình δ = 16** → hợp mục KV ở M = 8 vượt 7.281 mục thì bộ điều khiển chia sub-tile 4 query; lưu lượng KV tăng từ 40 MB lên tối đa 340 MB/layer (+6,9% chu kỳ layer).
6. **100 GbE all-to-all jitter** → tăng độ sâu elastic FIFO RAB recv; ICI đã ngoài đường găng với biên 21%.

## 15. Khuyến nghị (action items)

1. **Căn cứ:** trần f_max URAM ở V_CCINT = 0,72 V là 500 MHz [10], mâu thuẫn với giả định 600 MHz của bản trước. **Hành động:** tổng hợp thử một lane WTB + RAB (192 URAM) ở 500 MHz trên XCU250-2L, OREG_B = True, báo cáo slack. **Ai:** nhóm RTL. **Khi nào:** 2 tuần, trước 05/10/2026.
2. **Căn cứ:** chu kỳ layer phụ thuộc hoàn toàn 65,3 GB/s giả định, nay thêm ba nguồn lưu lượng mới (scale MXFP4, residual X/X′, tích luỹ y). **Hành động:** microbenchmark 4 × MIG DDR4 stream weight + KV interleave + y read-modify-write trên U250 thực, xác nhận ≥85%. **Ai:** nhóm RTL/board. **Khi nào:** 3 tuần, trước 12/10/2026.
3. **Căn cứ:** h + y ở BF16 là 42,0 MiB, vượt 40,0 MiB dữ liệu của linh kiện; lựa chọn giữa y-qua-DDR (B = 8, 3.828 tok/s) và B = 4 (y trên chip, ~1.900 tok/s) chưa được chốt. **Hành động:** mô phỏng cycle-accurate hai phương án, so thông lượng và độ trễ combine. **Ai:** nhóm vi kiến trúc. **Khi nào:** 4 tuần, trước 19/10/2026.
4. **Căn cứ:** packing 2 mult/DSP + dequant FP4 quyết định peak 24,6 TOPS. **Hành động:** prototype MAC tile 1.024 DSP INT8 packing + LUT dequant + áp scale E8M0 trên tổng tích luỹ, đo f_max −2L. **Ai:** nhóm DSP. **Khi nào:** 3 tuần, trước 12/10/2026.
5. **Căn cứ:** phân vùng KVB theo mục KV đổi cấu trúc QCAU sang hợp nhất kiểu flash-attention, kéo theo 1,2 GB/s mỗi ranh giới SLL. **Hành động:** mô hình hoá QCAU 4 lane với hợp nhất partial, xác nhận độ trễ che được dưới thời gian tile. **Ai:** nhóm vi kiến trúc. **Khi nào:** 4 tuần, trước 19/10/2026.
6. **Căn cứ:** kích thước khối scale MXFP4 đổi 1,6–12,3% lưu lượng trọng số. **Hành động:** đo perplexity ở khối 32, 64, 128; chốt kích thước. **Ai:** nhóm mô hình. **Khi nào:** 3 tuần, trước 12/10/2026.

## 16. Tài liệu tham khảo

[1] AMD, "Alveo U200 and U250 Data Center Accelerator Cards Data Sheet," DS962 (v1.4), 2023. [Online]. Available: https://docs.amd.com/r/en-US/ds962-u200-u250
[2] AMD, "Alveo U200 and U250 Accelerator Cards User Guide," UG1289, 2023.
[3] AMD, "Virtex UltraScale+ FPGA Product Table (VU13P-class XCU250)," 2023; UltraScale Architecture Memory Resources, UG573.
[4] AMD, "Alveo U250 Data Center Accelerator Card — Product Page/Brief," 2024. [Online]. Available: https://www.amd.com/en/products/accelerators/alveo/u250
[5] Xilinx, "WP486: Deep Learning with INT8 Optimization on Xilinx Devices," 2017.
[6] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
[7] A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024.
[8] S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," CACM, vol. 52, no. 4, 2009.
[9] Nội bộ: `DeepSeek_V4Pro_Microarchitecture_Spec.md`, `DeepSeek_V4Pro_SingleChip_Microarchitecture.md`, `DeepSeek_V4Pro_InterCard_Protocol.md`, `DeepSeek_V4Flash_U250_Buffer_Spec.md`.

[10] AMD, "Virtex UltraScale+ FPGA Data Sheet: DC and AC Switching Characteristics — UltraRAM Switching Characteristics," DS923. [Online]. Available: https://docs.amd.com/r/en-US/ds923-virtex-ultrascale-plus/UltraRAM-Switching-Characteristics

[11] Open Compute Project, "OCP Microscaling Formats (MX) Specification, Version 1.0," 2023.
