# Đặc tả khối buffer on-chip — DeepSeek-V4-Flash trên Alveo U250

Nhánh: V4-Flash, 2 card U250, 1 chip = trọn layer, 16 layer/chip, B = 4 chunk (1.024 token).
Đối tượng: WTB, RAB, KVB, PSA, Ctrl. Nền tảng: XCU250-2LFIGD2104E (Virtex UltraScale+, 4 SLR).

---

## Mục lục

1. Nguyên thuỷ bộ nhớ và tần số tối đa của XCU250
2. Định dạng từ nhớ 72 bit
3. Ngân sách tensor cho B = 4
4. Ràng buộc lưu trú: tensor nào phải thường trú, tensor nào theo tile
5. WTB — Weight Tile Buffer
6. RAB — Row Activation Buffer
7. KVB — KV Buffer
8. PSA — Partial & Scratch Array
9. Ctrl và các khối BRAM
10. Tổng hợp phân bổ URAM/BRAM theo SLR
11. Hệ quả với chu kỳ layer và thông lượng
12. Chênh lệch so với tài liệu hiện hành
13. Rủi ro
14. Kiến nghị
15. Tài liệu tham khảo

---

## 1. Nguyên thuỷ bộ nhớ và tần số tối đa của XCU250

| Tham số | Giá trị | Nguồn |
|---|---:|---|
| URAM288 (UltraRAM) | 4.096 từ × 72 bit = 288 Kb = 36 KiB/khối | [3] |
| Số khối URAM (XCU250) | 1.280 (320/SLR) = 45,0 MiB thô | [1], [3] |
| RAMB36 (Block RAM) | 36 Kb = 4,5 KiB/khối; chế độ SDP 512 × 72 bit | [3] |
| Số khối RAMB36 | 2.688 (672/SLR) = 11,81 MiB | [1] |
| Cổng URAM | 2 cổng (A, B), dùng chung một clock; mỗi cổng đọc hoặc ghi | [3] |
| Cascade URAM | Giải mã 11 bit MSB địa chỉ trong cột, định tuyến chuyên dụng, +1 chu kỳ/tầng | [3] |
| f_max URAM, speed grade −2, V_CCINT = 0,72 V, OREG_B = True | **500 MHz** | [4] |
| f_max URAM, −2, 0,85/0,90 V, OREG_B = True | 600 MHz | [4] |
| f_max URAM, OREG_B = False, không ECC | 404–528 MHz | [4] |
| f_max URAM, OREG_B = False, bật ECC | 312–435 MHz | [4] |

Alveo U250 mang linh kiện XCU250-**2L**, vận hành V_CCINT = 0,72 V [1], [2]. Trần tần số URAM tương ứng là **500 MHz**, không phải 600 MHz như giả định trong `DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md` §7 và trong bảng bus của `DeepSeek_V4Flash_2xU250_Explorer.html`. Bật ECC nội URAM hạ trần xuống 312–435 MHz, do đó ECC nội bị loại; bảo vệ dữ liệu dùng parity tính trong fabric (§2).

Toàn bộ số liệu băng thông dưới đây lấy mốc **500 MHz** cho miền buffer + MAC, **300 MHz** cho miền AXI/MIG.

---

## 2. Định dạng từ nhớ 72 bit

Từ URAM rộng 72 bit. Phương án chia: **64 bit dữ liệu + 8 bit phụ trợ**.

| Kiểu dữ liệu | Đóng gói trong 64 bit dữ liệu | 8 bit phụ trợ |
|---|---|---|
| FP4 (E2M1) trọng số expert | 16 giá trị | scale E8M0 dùng chung khối 32 (ghi ở từ chẵn), parity ở từ lẻ |
| FP8 activation | 8 giá trị | parity |
| BF16 activation | 4 giá trị | parity |
| Mục KV nén (576 B) | 72 từ = 9 nhịp bus 512 bit | parity |
| Chỉ số top-k 18 bit | 32 chỉ số đóng gói khít trong 9 từ (576 bit) | parity |

Hệ quả của lựa chọn 64 bit dữ liệu:

- Mọi tensor có kích thước luỹ thừa 2 ánh xạ khít vào địa chỉ, không có byte đệm. Nếu dùng đủ 72 bit dữ liệu, một hàng h (4.096 B) chiếm 455,1 từ, phải đệm lên 456 từ (0,2% lãng phí) và địa chỉ hàng không còn là phép dịch bit.
- Một nhịp AXI 512 bit từ mSGDMA ghi đúng **8 bank** × 64 bit, không cần bộ chuyển độ rộng phân số.
- Một mục KV 576 B = 4.608 bit = 9 × 512 bit, khớp bus KV 512 bit/lane.
- Chi phí: 11,1% dung lượng URAM thô không dùng cho dữ liệu (45,0 MiB thô → 40,0 MiB dữ liệu).

Scale MXFP4 [6]: khối 32 phần tử dùng chung một scale E8M0 8 bit. Trong URAM, scale nằm ở 8 bit phụ trợ của từ chẵn; hai từ liên tiếp (32 giá trị FP4) chia nhau scale đó. Trong DDR4, mật độ lưu trữ là 4,25 bit/trọng số.

Dequant FP4 → INT8: bảng 16 mục, 4 bit → 8 bit, hiện thực bằng LUT phân tán (không dùng BRAM). Scale E8M0 không nhân vào từng trọng số mà áp dụng bằng phép dịch trên tổng tích luỹ INT32 của khối 32, do đó đường dữ liệu MAC giữ nguyên INT8.

---

## 3. Ngân sách tensor cho B = 4

Tham số mô hình: d = 4.096; n_hc = 4; 64 head; d_head = 512; mục KV nén 576 B; top-k = 512; cửa sổ trượt 128; 257 expert (256 routed + 1 shared), top-6, d_ff = 2.048; chunk 256 token; B = 4 → 1.024 token/batch.

| Tensor | Hình dạng/token | Độ chính xác | B/token | Toàn batch-4 |
|---|---|---|---:|---:|
| X_l (residual mở rộng) | 4 × 4.096 | BF16 | 32.768 | **32,0 MiB** |
| X̂ | 4.096 | FP8 | 4.096 | 4,0 MiB |
| c_Q | 1.024 | FP8 | 1.024 | 1,0 MiB |
| chỉ số top-k | 512 | 18 bit | 1.152 | 1,125 MiB |
| o (đầu ra core attention) | 64 × 512 | BF16 | 65.536 | **64,0 MiB** |
| o (phương án FP8) | 64 × 512 | FP8 | 32.768 | 32,0 MiB |
| ô | 4.096 | BF16 | 8.192 | 8,0 MiB |
| h (đầu vào expert) | 4.096 | FP8 | 4.096 | 4,0 MiB |
| y (tích luỹ expert) | 4.096 | BF16 | 8.192 | 8,0 MiB |
| route (idx + gate) | — | — | 28 | 28,7 KiB |

Hai tensor vượt xa tổng URAM của linh kiện: o ở 64,0 MiB bằng 1,42× toàn bộ 45,0 MiB URAM, X_l ở 32,0 MiB bằng 71%.

Trong `DeepSeek_V4Flash_2xU250_Explorer.html`, mục `eca` ghi `h ... 16,8 MB/batch`. Con số 16,8 MB ứng với h giữ cả n_hc = 4 luồng (16 KiB/token), hoặc với h được nhân bản trên cả 4 SLR. Đặc tả này lấy h = một luồng d = 4.096 FP8, phân vùng theo token trên 4 lane, không nhân bản: **4,0 MiB/batch**. Token do SLR khác sở hữu được đọc qua SLL với lưu lượng 25,2 MB/layer (0,7% tổng lưu lượng nội chip), nằm trong nhóm activation hẹp được phép vượt SLR.

---

## 4. Ràng buộc lưu trú

Lịch pipeline hai tầng đòi hai batch cùng sống: slot t chạy `FRONT_l(A) ∥ ECA_{l−1}(B)`, slot t+1 chạy `FRONT_l(B) ∥ ECA_l(A)`.

| Tensor | Phải thường trú cả batch? | Căn cứ |
|---|---|---|
| h | Có | ECA duyệt tuần tự 257 expert; expert e chạm ~24 token bất kỳ trong batch, thứ tự truy cập do routing quyết định |
| y | Có | mỗi token tích luỹ qua 6 expert routed + 1 shared, phân tán trên toàn bộ 52,6 ms của pha expert |
| X_l, X′ | Không — đưa xuống DDR4 | 32,0 MiB/batch × 2 batch sống; op8 (X_{l+1} = B·X′ + C·y) chỉ chạy sau khi ECA kết thúc, X′ phải sống 52,6 ms |
| o | Không — tile, fuse QCAU→GOPU | 64,0 MiB/batch; o không bao giờ được hiện thực hoá cho cả batch |
| X̂, c_Q, ô | Không — tile | tiêu thụ ngay trong chuỗi FRONT |
| chỉ số top-k | Có (PSA) | QCAU đọc lại theo từng query-tile |

Kích thước tile FRONT: M = 16 token. Chọn M từ ràng buộc dung lượng RAB (§6) và từ ràng buộc hợp nhất của KVB (§7).

Lưu lượng DDR4 phát sinh do đưa X/X′ xuống bộ nhớ ngoài, mỗi layer: đọc X_l 33,55 MB + ghi X′ 33,55 MB + đọc X′ 33,55 MB + ghi X_{l+1} 33,55 MB = **134,2 MB**. Phương án lưu X′ ở FP8 giảm còn 100,7 MB nhưng đặt residual stream vào FP8 giữa hai lần đọc, chưa có số liệu sai số kèm theo.

---

## 5. WTB — Weight Tile Buffer

| Hạng mục | Giá trị |
|---|---|
| Vai trò | Double-buffer tile trọng số; DMA ghi nửa B trong khi FU đọc nửa A |
| Phân vùng | 4 lane, mỗi lane cố định trong một SLR và bám một kênh DDR4 |
| Tổ chức/lane | 2 nửa × 32 bank; mỗi bank = 1 URAM288 (4.096 × 72) |
| URAM | 64/lane · **256 tổng** (20,0% quỹ URAM) |
| Dung lượng dữ liệu | 1,000 MiB/nửa/lane · 2,0 MiB/lane · **8,0 MiB tổng** (thô 9,0 MiB) |
| Độ rộng đọc | 32 × 64 bit = **2.048 bit dữ liệu/lane** (2.304 bit thô) |
| BW đọc @500 MHz | 128 GB/s/lane · **512 GB/s tổng** |
| Độ rộng ghi | 512 bit @300 MHz (1 nhịp AXI = 8 bank × 64 bit) |
| BW ghi | 19,2 GB/s/kênh danh định; 16,3 GB/s hiệu dụng |
| Độ sâu bank | 4.096 từ (không cascade) |
| Độ trễ đọc | 2 chu kỳ (OREG_B = True) |

Dung lượng một nửa khớp chính xác: 32 × 4.096 × 64 bit = 8.388.608 bit = **1 MiB**.

Thời gian nạp một nửa tile: 1,0 MiB dữ liệu chứa 2,097M trọng số FP4; ở mật độ DDR 4,25 bit/trọng số, lưu lượng DDR là 1,114 MB, nạp trong 1,114 MB ÷ 16,3 GB/s = **68,3 µs**. Tile đầy đủ (4 lane) = 8,389M trọng số, 4,46 MB DDR, 68,3 µs.

Nhu cầu đọc của ECA: một sub-array 1.024 DSP × 2 MAC/DSP @500 MHz sinh 1.024 MAC-INT8/chu kỳ. Mỗi trọng số được dùng lại bằng số token định tuyến vào expert đó (1.024 token × 6/256 ≈ 24). Nhu cầu trọng số = 1.024 ÷ 24 = 42,7 trọng số/chu kỳ = 21,3 B/chu kỳ = 10,7 GB/s. Nguồn cấp WTB 128 GB/s/lane, **hệ số dư 12,0×**. Nút cổ chai nằm ở kênh DDR4 (16,3 GB/s), không ở WTB.

Số bank ở đây là hệ quả của dung lượng (1 MiB cần đúng 32 URAM), không phải của băng thông. Giữ 32 bank song song thay vì cascade để tránh 32 tầng trễ và để cấp đủ operand cho chuỗi FU FRONT khi FRONT đọc W_UQ, W_OG, W_O theo tile.

Xung đột cổng: nửa đang đọc dùng cổng A, nửa đang ghi dùng cổng B; hai nửa không bao giờ cùng vai trò, nên không có tranh chấp cổng trong một khối URAM.

Rủi ro dung lượng đã nêu ở §14.3 của tài liệu U250: nếu shell chiếm URAM của SLR1, giảm WTB lane 1 còn 2 × 24 URAM (0,75 MiB/nửa), thời gian nạp 51,2 µs, vẫn ẩn được dưới thời gian compute tile.

---

## 6. RAB — Row Activation Buffer

| Hạng mục | Giá trị |
|---|---|
| Vai trò | Giữ h và y của batch đang ở ECA, cộng tập tile của batch đang ở FRONT |
| Phân vùng | 4 lane theo token (256 token/lane ở B = 4) |
| Tổ chức/lane | 16 bank × 10 URAM cascade (độ sâu 40.960 từ) |
| URAM | 160/lane · **640 tổng** (50,0% quỹ URAM) |
| Dung lượng dữ liệu | 5,0 MiB/lane · **20,0 MiB tổng** (thô 22,5 MiB) |
| Độ rộng | 16 × 64 bit = **1.024 bit dữ liệu/lane** (1.152 bit thô) |
| BW @500 MHz | 64 GB/s/lane · **256 GB/s tổng**, mỗi cổng |
| Độ trễ đọc | 12 chu kỳ (10 tầng cascade + 2 OREG) |
| Cổng | A = FU đọc qua AGU; B = FU/DMA ghi |

Kiểm tra dung lượng:

| Thành phần | Dung lượng |
|---|---:|
| h(A) + h(B) — hai batch cùng sống | 8,0 MiB |
| y của batch đang ở ECA | 8,0 MiB |
| Tile FRONT M = 16, ping-pong ×2: X/X′ (tại chỗ) 0,5 · X̂ 0,0625 · c_Q 0,0156 · o FP8 0,5 · ô 0,125 | 2,41 MiB |
| **Tổng yêu cầu** | **18,41 MiB** |
| Cấp phát | 20,0 MiB (dư 1,59 MiB = 8,6%) |

X và X′ dùng chung vùng tile (cập nhật tại chỗ, cùng hình dạng BF16), tiết kiệm 1,0 MiB so với cấp phát riêng.

Nhu cầu băng thông đỉnh:

| Tác nhân | Lưu lượng/layer | Cửa sổ | BW |
|---|---:|---:|---:|
| QCAU ghi o (tile FP8) | 32,0 MiB | 21,6 ms | 1,55 GB/s |
| GOPU đọc o | 32,0 MiB | 21,6 ms | 1,55 GB/s |
| ECA đọc h (24 token × 257 expert + shared 1.024) | 29,4 MB | 52,6 ms | 0,56 GB/s |
| ECA đọc–sửa–ghi y (7 lượt chạm/token) | 117,4 MB | 52,6 ms | 2,23 GB/s |
| mHCE 3 pass (X, ô, y) | 88,1 MB | 21,6 ms | 4,08 GB/s |
| **Đỉnh cộng dồn** | | | **~10 GB/s** |

So với 256 GB/s cấp phát: hệ số dư 25×. RAB là khối bị chi phối bởi dung lượng, không bởi băng thông. Độ rộng 1.024 bit/lane là hệ quả của việc chia 160 URAM thành 16 bank; chọn 16 bank thay vì 160 bank × 1 URAM để giảm số dây định tuyến, đổi lấy 10 tầng cascade.

Chế độ địa chỉ: một địa chỉ phát broadcast tới 16 bank trong lane (width-slice), một truy cập trả 128 B. Hàng h (4.096 B) = 32 địa chỉ khít; hàng y (8.192 B) = 64 địa chỉ khít; hàng X (32.768 B) = 256 địa chỉ khít.

---

## 7. KVB — KV Buffer

| Hạng mục | Giá trị |
|---|---|
| Vai trò | Giữ hợp (union) các mục KV mà một query-tile chạm tới: top-k 512 + cửa sổ 128 mỗi query |
| Phân vùng | 4 lane **theo mục KV** (mục idx thuộc lane idx mod 4), không theo head |
| Tổ chức/lane | 8 bank × 6 URAM cascade (độ sâu 24.576 từ) |
| URAM | 48/lane · **192 tổng** (15,0% quỹ URAM) |
| Dung lượng dữ liệu | 1,5 MiB/lane · **6,0 MiB tổng** (thô 6,75 MiB) |
| Độ rộng | 8 × 64 bit = **512 bit dữ liệu/lane** (576 bit thô) |
| BW @500 MHz | 32 GB/s/lane · **128 GB/s tổng** |
| Sức chứa theo mục | 6,0 MiB ÷ 576 B = **10.922 mục**; ping-pong → 5.461 mục/tile |

Phân vùng theo mục KV thay cho phân vùng theo head: MQA dùng chung một mục KV cho cả 64 head, nên nếu chia head theo SLR thì mỗi SLR cần toàn bộ KV, buộc phải nhân bản KVB ×4 hoặc mở một bus KV 2.304 bit vượt SLR — vi phạm quy tắc chỉ cho activation hẹp vượt SLR. Chia theo mục KV giữ KV cục bộ trong lane; mỗi lane tính phần tổng softmax trên 1/4 số mục cho cả 64 head, sau đó hợp nhất kiểu flash-attention qua SLL.

Lưu lượng vượt SLR do hợp nhất: q (64 head × 512 FP8 = 32 KiB/query) phát tới 4 lane, và phần tổng (64 head × 512 BF16 + 2 vô hướng = 64 KiB/query) trả về. Mỗi ranh giới SLR: 1.024 query × 96 KiB ÷ 3 = 32,8 MB trong 21,6 ms = **1,5 GB/s**, chiếm ~1.024 trong 17.280 dây SLL sẵn có mỗi ranh giới.

Kích thước query-tile M và sức chứa: với M query liên tiếp trong cùng chuỗi, hợp các tập top-512 xấp xỉ 512 + (M − 1)·δ, δ là số mục mới trên mỗi bước.

| M | Hợp ước tính (δ = 16) | Dung lượng cần (ping-pong) | Lưu lượng KV từ DDR/layer |
|---:|---:|---:|---:|
| 8 | 624 + 128 | 0,83 MiB | 46,0 MB |
| **16** | **752 + 128** | **0,97 MiB** | **27,7 MB** |
| 64 | 1.520 + 128 | 1,81 MiB | 14,0 MB |
| 16, trường hợp xấu nhất δ = 512 (không chồng lấn) | 10.240 | 11,25 MiB | 377 MB |

Cấp phát 6,0 MiB phủ M = 16 tới hợp 5.461 mục/tile, tức 341 mục phân biệt/query trên tổng 640 — gấp 6,2× mô hình δ = 16, nhưng không phủ trường hợp không chồng lấn. Cơ chế dự phòng: LIU đã tính xong tập top-k trước khi QCAU chạy, nên bộ điều khiển đo kích thước hợp và tự chia query-tile thành các sub-tile 8 hoặc 4 query khi hợp vượt sức chứa; chi phí là lưu lượng KV từ DDR tăng, không phải tràn buffer.

Cấp phát 16 MB trong tài liệu hiện hành chỉ cần thiết nếu top-k phân tán gần như hoàn toàn giữa các query liền kề; ở mức đó KVB chiếm 456 URAM (35,6% quỹ) và buộc RAB phải xuống 12 MB, dưới mức đủ cho h + y (§6).

---

## 8. PSA — Partial & Scratch Array

| Hạng mục | Giá trị |
|---|---|
| Vai trò | Chỉ số top-k của LIU, phần tổng softmax chờ hợp nhất, kết quả router |
| Phân vùng | 4 lane |
| Tổ chức/lane | 8 bank × 4 URAM cascade (độ sâu 16.384 từ) |
| URAM | 32/lane · **128 tổng** (10,0% quỹ URAM) |
| Dung lượng dữ liệu | 1,0 MiB/lane · **4,0 MiB tổng** (thô 4,5 MiB) |
| Độ rộng | 8 × 64 bit = **512 bit dữ liệu/lane** |
| BW @500 MHz | 32 GB/s/lane · 128 GB/s tổng |

Kiểm tra dung lượng:

| Thành phần | Tính toán | Dung lượng |
|---|---|---:|
| Chỉ số top-k, đóng gói 18 bit | 1.024 query × 512 × 18 bit | 1,125 MiB |
| Phần tổng softmax chờ hợp nhất, ping-pong | 16 query × 64 head × 512 × 2 B × 2 | 2,0 MiB |
| Router (idx + gate) | 1.024 × 28 B | 0,027 MiB |
| **Tổng** | | **3,15 MiB** |
| Cấp phát | | 4,0 MiB (dư 27%) |

Đóng gói chỉ số 18 bit thay cho INT32: 32 chỉ số chiếm 576 bit = 9 từ khít, phủ không gian 262.144 hàng KV nén, tương đương ngữ cảnh 1,05 M token ở tỉ lệ nén m = 4. So với INT32, tiết kiệm 0,875 MiB (43,8%).

---

## 9. Ctrl và các khối BRAM

| Khối | Nội dung | Tổ chức | RAMB36 |
|---|---|---|---:|
| Ctrl / CtxQ | hàng đợi token, descriptor DMA, trạng thái vòng lặp 16 layer × 8 op | SDP 512 × 72, 16 bank | 466 |
| FIFO đàn hồi vượt SLR | 3 ranh giới × 2 hướng × 1.152 bit × 512 từ | SDP 512 × 72 | 96 |
| ROM hàm phi tuyến | SiLU, exp, rsqrt — xấp xỉ tuyến tính từng đoạn, 1.024 đoạn × (hệ số góc 16 bit + chặn 16 bit) | 3 hàm × 4 lane | 12 |
| Shell (XDMA + 4 × MIG) | — | — | ~250 |
| **Tổng** | | | **~824 / 2.688 (30,7%)** |

Bảng dequant FP4 → INT8 không chiếm BRAM: 16 mục × 8 bit hiện thực bằng LUT phân tán tại mỗi lane MAC; scale E8M0 áp dụng bằng phép dịch trên tổng tích luỹ INT32. Con số ~3 MB cho "dequant-LUT + FIFO" trong tài liệu hiện hành do đó giảm còn 432 KiB FIFO + 48 KiB ROM.

---

## 10. Tổng hợp phân bổ URAM/BRAM theo SLR

| Buffer | Bank/lane | URAM/bank | URAM/lane | URAM tổng | Độ rộng dữ liệu/lane | Dung lượng dữ liệu | BW @500 MHz | Yêu cầu đỉnh | Hệ số dư |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTB | 2 × 32 | 1 | 64 | 256 | 2.048 bit | 8,0 MiB | 512 GB/s | 42,7 GB/s | 12,0× |
| RAB | 16 | 10 | 160 | 640 | 1.024 bit | 20,0 MiB | 256 GB/s | 10,0 GB/s | 25,6× |
| KVB | 8 | 6 | 48 | 192 | 512 bit | 6,0 MiB | 128 GB/s | 17,5 GB/s | 7,3× |
| PSA | 8 | 4 | 32 | 128 | 512 bit | 4,0 MiB | 128 GB/s | 12,4 GB/s | 10,3× |
| **Cộng** | | | **304** | **1.216** | | **38,0 MiB** | | | |
| Dự phòng | | | 16 | 64 | | 2,0 MiB | | | |
| **Quỹ XCU250** | | | **320** | **1.280** | | 40,0 MiB dữ liệu | | | |

Mức dùng URAM: 1.216/1.280 = **95,0%**, đồng đều 304/320 trên cả 4 SLR. Mức dùng BRAM: ~824/2.688 = **30,7%**.

Bảng bus tương ứng, thay cho bảng trong `DeepSeek_V4Flash_2xU250_Explorer.html`:

| Đường | Giao thức | Độ rộng | Clock | BW cung cấp | BW yêu cầu |
|---|---|---:|---:|---:|---:|
| DDR4 ↔ MIG (×4 kênh) | AXI4-MM burst | 512 bit | 300 MHz | 19,2 GB/s/kênh (65,3 hiệu dụng tổng) | 65,3 — đường găng |
| mSGDMA → WTB cổng B | AXI4-Stream, CDC 300→500 MHz | 512 bit | 300 → 500 MHz | 19,2 GB/s/kênh | 16,3 GB/s/kênh |
| WTB cổng A → dequant → FU | AXI4-Stream, TLAST = tile | 2.048 bit/lane | 500 MHz | 128 GB/s/lane | 10,7 GB/s/lane |
| RAB ↔ FU (RA-bus) | AXI4-Stream + AGU broadcast | 1.024 bit/lane | 500 MHz | 64 GB/s/lane | 2,5 GB/s/lane |
| KVB → QCAU (KV-bus) | AXI4-Stream | 512 bit/lane | 500 MHz | 32 GB/s/lane | 4,4 GB/s/lane |
| PSA ↔ FU (PS-bus) | AXI4-Stream | 512 bit/lane | 500 MHz | 32 GB/s/lane | 3,1 GB/s/lane |
| Hợp nhất flash-attention qua SLL | FIFO đàn hồi | 1.152 bit | 500 MHz | 72 GB/s/ranh giới | 1,5 GB/s/ranh giới |
| h vượt SLR | FIFO đàn hồi | 512 bit | 500 MHz | 32 GB/s | 0,56 GB/s |
| AXI4-Lite cấu hình | AXI4-Lite | 32 bit | 300 MHz | — | — |

---

## 11. Hệ quả với chu kỳ layer và thông lượng

Lưu lượng DDR4 mỗi layer, sau khi tính scale MXFP4 và residual X/X′:

| Thành phần | Lưu lượng | Ghi chú |
|---|---:|---|
| Trọng số expert FP4 thuần | 3.232 MB | 257 × 25,166M trọng số × 4 bit |
| Scale E8M0, khối 32 | 202 MB | +6,25% trên phần trọng số |
| Trọng số attention + router FP8 | 119 MB | W_mHC 0,8 + W_ACU 8,4 + W_LIU 8,5 + W_UQ 33,6 + W_OG·W_O 67,1 + W_rt 1,05 |
| Residual X/X′ (2 đọc + 2 ghi, BF16) | 134 MB | §4 |
| KV top-k (M = 16, δ = 16) + append | 28 MB | §7 |
| **Tổng** | **3.715 MB** | |
| **Chu kỳ layer @65,3 GB/s** | **56,9 ms** | tài liệu hiện hành: 51,3 ms |

Ảnh hưởng của mật độ scale:

| Kích thước khối scale | Mật độ | Lưu lượng expert | Chu kỳ layer |
|---:|---:|---:|---:|
| 16 | 4,500 bit/w | 3.636 MB | 60,1 ms |
| **32 (MXFP4 chuẩn [6])** | **4,250 bit/w** | **3.434 MB** | **56,9 ms** |
| 64 | 4,125 bit/w | 3.333 MB | 55,4 ms |
| 128 | 4,063 bit/w | 3.283 MB | 54,6 ms |
| không scale (giả định của tài liệu hiện hành) | 4,000 bit/w | 3.232 MB | 53,8 ms |

Ở 500 MHz thay vì 600 MHz:

| Đại lượng | 600 MHz (giả định cũ) | 500 MHz (trần URAM −2 @0,72 V) |
|---|---:|---:|
| Peak INT8 | 29,5 TOPS | 24,6 TOPS |
| ECA compute/layer (361 GOP) | 36,7 ms | 44,1 ms |
| ECA read/layer | 49,5 ms | 52,6 ms |
| FRONT compute/layer (340 GOP) | 18,0 ms | 21,6 ms |
| Chu kỳ layer | 51,3 ms | **56,9 ms** (DDR quyết định) |
| MAC util của ECA trong pha expert | 74% | 84% |
| MAC util toàn chip trên chu kỳ layer | 46% | 50% |

Hạ clock từ 600 xuống 500 MHz không làm giảm thông lượng, vì ECA vẫn read-bound (44,1 ms compute < 52,6 ms read) và FRONT vẫn nằm xa đường găng. Toàn bộ mức giảm thông lượng 51,3 → 56,9 ms (−9,8%) đến từ scale MXFP4 và residual X/X′, không từ tần số.

Thông lượng hệ 2 card ở điểm vận hành mới: 1.226 × 51,3/56,9 = **1.105 token/s** prefill; một stage 16 layer = 911 ms cho 1.024 token.

---

## 12. Chênh lệch so với tài liệu hiện hành

| Mục | Tài liệu hiện hành | Đặc tả này | Nguyên nhân |
|---|---|---|---|
| Clock buffer | 600 MHz | 500 MHz | XCU250-2L chạy 0,72 V; trần URAM 500 MHz [4] |
| WTB | 8 MB, 224 URAM, 32 bank × 72 bit | 8,0 MiB, 256 URAM, 4 × 2 × 32 bank × 64 bit | 224 URAM không chia hết cho 32 bank × 4 lane; 1 MiB/nửa cần đúng 32 URAM |
| RAB | 12 MB, 340 URAM, 64 bank | 20,0 MiB, 640 URAM, 4 × 16 bank | 340 không chia hết cho 64 bank; 12 MB không chứa nổi h×2 + y = 16,0 MiB |
| KVB | 16 MB, 456 URAM, 32 bank | 6,0 MiB, 192 URAM, 4 × 8 bank | phân vùng theo mục KV thay vì nhân bản theo head; sức chứa neo vào M = 16 và mô hình chồng lấn top-k |
| PSA | 6 MB, 172 URAM, 16 bank | 4,0 MiB, 128 URAM, 4 × 8 bank | 172 không chia hết cho 16 bank; đóng gói chỉ số 18 bit |
| Dequant LUT + FIFO | ~3 MB, ~680 BRAM | 0,47 MiB, 108 BRAM | dequant bằng LUT phân tán; scale áp trên tổng tích luỹ |
| h | 16,8 MB/batch | 4,0 MiB/batch | không nhân bản trên 4 SLR; một luồng d = 4.096 |
| Lưu lượng DDR/layer | 3,35 GB | 3,715 GB | thêm scale MXFP4 và residual X/X′ |
| Chu kỳ layer | 51,3 ms | 56,9 ms | như trên |
| Tổng URAM | 1.192 (93%) | 1.216 (95%) | — |

---

## 13. Rủi ro

1. Mô hình chồng lấn top-k (δ = 16) chưa có số đo. Nếu δ thực ≥ 128, hợp ở M = 16 vượt 2.000 mục và lưu lượng KV từ DDR tăng từ 28 MB lên 118 MB/layer (+2,4% chu kỳ layer); nếu top-k phân tán hoàn toàn, KVB 6,0 MiB buộc chia sub-tile 4 query và lưu lượng lên 377 MB/layer (+10% chu kỳ layer).
2. URAM ở 500 MHz là trần datasheet, biên bằng 0. Nếu timing không đóng, hạ xuống 450 MHz: ECA compute 49,0 ms vẫn < 52,6 ms read, chu kỳ layer không đổi; BW mọi buffer giảm 10%, hệ số dư nhỏ nhất (KVB) còn 6,6×.
3. Shell tĩnh của Alveo U250 chiếm tài nguyên SLR1. Cấp phát URAM ở mức 95% để lại 16 khối/SLR; nếu shell chiếm URAM, giảm WTB lane 1 xuống 2 × 24 URAM (§5).
4. Cascade 10 tầng của RAB làm độ trễ đọc lên 12 chu kỳ. Nếu vòng đọc–sửa–ghi y của ECA không che được độ trễ này, tách y thành 32 bank × 5 URAM (độ trễ 7 chu kỳ, độ rộng 2.048 bit/lane) và giữ h ở cấu hình 16 bank.
5. Tích luỹ y ở BF16 qua 7 lượt chạm chưa có số liệu sai số. Phương án FP32 nâng y từ 8,0 lên 16,0 MiB, vượt quỹ RAB 20,0 MiB khi cộng h×2 và tile.

---

## 14. Kiến nghị

1. **Căn cứ:** trần URAM 500 MHz ở V_CCINT = 0,72 V [4] mâu thuẫn với giả định 600 MHz đang dùng trong cả hai tài liệu vi kiến trúc. **Hành động:** tổng hợp thử một lane WTB + RAB (224 URAM) ở 500 MHz trên XCU250-2L, OREG_B = True, báo cáo slack; xác nhận hoặc bác bỏ mốc 500 MHz. **Ai:** nhóm RTL. **Khi nào:** 2 tuần, trước 05/10/2026.
2. **Căn cứ:** sức chứa KVB quyết định trực tiếp lưu lượng KV từ DDR, dải biến thiên 28–377 MB/layer, tương ứng 0,8–10% chu kỳ layer. **Hành động:** đo phân bố hợp top-k theo M ∈ {8, 16, 64} trên trace DeepSeek-V4 thực, chốt δ và sức chứa KVB. **Ai:** nhóm mô hình. **Khi nào:** 3 tuần, trước 12/10/2026.
3. **Căn cứ:** scale MXFP4 khối 32 thêm 6,25% lưu lượng trọng số; khối 64 và 128 giảm chi phí còn 3,1% và 1,6% nhưng thay đổi sai số lượng tử. **Hành động:** đo perplexity của DeepSeek-V4 MoE ở FP4 với khối 32, 64, 128; chọn kích thước khối. **Ai:** nhóm mô hình. **Khi nào:** 3 tuần, trước 12/10/2026.
4. **Căn cứ:** phân vùng KVB theo mục KV thay đổi cấu trúc QCAU từ chia-theo-head sang hợp nhất kiểu flash-attention, kéo theo 1,5 GB/s qua mỗi ranh giới SLL. **Hành động:** mô hình hoá cycle-accurate QCAU 4 lane với hợp nhất partial, xác nhận độ trễ hợp nhất che được dưới thời gian tile. **Ai:** nhóm vi kiến trúc. **Khi nào:** 4 tuần, trước 19/10/2026.
5. **Căn cứ:** residual X/X′ ở DDR thêm 134 MB/layer (3,6% chu kỳ layer); phương án FP8 giảm còn 100,7 MB. **Hành động:** đo sai số end-to-end khi lưu X′ ở FP8 giữa hai lần đọc. **Ai:** nhóm mô hình. **Khi nào:** 4 tuần, trước 19/10/2026.
6. **Căn cứ:** các bảng buffer trong `DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md` §7 và `DeepSeek_V4Flash_2xU250_Explorer.html` tab 3–4 chứa số URAM không chia hết cho số bank. **Hành động:** cập nhật hai tài liệu theo bảng §10. **Ai:** nhóm vi kiến trúc. **Khi nào:** sau khi chốt kiến nghị 1 và 2.

---

## 15. Tài liệu tham khảo

[1] AMD, "Alveo U200 and U250 Data Center Accelerator Cards Data Sheet," DS962 (v1.4), 2023. [Online]. Available: https://docs.amd.com/r/en-US/ds962-u200-u250

[2] AMD, "Alveo U200 and U250 Accelerator Cards User Guide," UG1289, 2023.

[3] AMD, "UltraScale Architecture Memory Resources User Guide," UG573. [Online]. Available: https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources

[4] AMD, "Virtex UltraScale+ FPGA Data Sheet: DC and AC Switching Characteristics — UltraRAM Switching Characteristics," DS923. [Online]. Available: https://docs.amd.com/r/en-US/ds923-virtex-ultrascale-plus/UltraRAM-Switching-Characteristics

[5] Xilinx, "Deep Learning with INT8 Optimization on Xilinx Devices," WP486, 2017.

[6] Open Compute Project, "OCP Microscaling Formats (MX) Specification, Version 1.0," 2023.

[7] AMD, "UltraScale Architecture DSP Slice User Guide," UG579 (v1.7), 2018.

[8] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.

[9] Nội bộ: `DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md`, `DeepSeek_V4Pro_U250_FU_Interface_Spec.md`, `DeepSeek_V4Flash_2xU250_Explorer.html`.
