# DeepSeek-V4-Pro Accelerator — Dòng thiết kế vi kiến trúc phần cứng
# Hardware Microarchitecture Design Flow

Tài liệu này mô tả **hành trình thiết kế** (các bước đối thoại) đã dẫn tới vi kiến trúc của bộ tăng tốc DeepSeek-V4-Pro trên hệ đa-FPGA, làm rõ **pipeline**, **song song Tensor (TP) / Expert (EP)**, và các **pipeline diagram theo thời gian** ở ba mức: trong chip, trong card, trong hệ thống. Đây là điểm vào (entry point) — đọc file này trước, rồi mở các deliverable theo dòng.

---

## 0. Bản đồ thư mục (theo dòng thiết kế)

```
DeepSeek-FPGA/
├── 00_README_Design_Flow.md            ← (file này) dòng thiết kế + chỉ mục
├── 01_Background/                       ← đầu vào: mô hình DeepSeek-V4, glossary, so sánh FPGA, literature, proposal
│   ├── DeepSeek_V4Pro_Dataflow_Software_Model.html  ← trực quan dataflow phần mềm (prefill 256 / decode 1 token)
│   └── DeepSeek_V4Pro_Block_Dataflow_Diagram.html   ← sơ đồ khối dataflow 1 block (click khối → phép toán + I/O)
├── 02_System_Architecture/             ← cấp hệ thống: spec, mapping FU, giao thức liên card
│   ├── DeepSeek_V4Pro_Microarchitecture_Spec.md
│   ├── DeepSeek_V4Pro_FU_Mapping.xlsx  (+ _Spec.md)
│   └── DeepSeek_V4Pro_InterCard_Protocol.md
├── 03_SingleChip_Microarchitecture/    ← cấp chip: spec + sơ đồ khối + pipeline FU
│   ├── DeepSeek_V4Pro_SingleChip_Microarchitecture.md
│   ├── DeepSeek_V4Pro_SingleChip.html  (Tab "Vi kiến trúc khối" + Tab "Pipeline FRONT∥ECA")
│   ├── DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md   ← nhánh U250: spec 1 chip Alveo U250 (slice 1/4 card, 8 layer)
│   ├── DeepSeek_V4Pro_SingleChip_U250.html                   ← nhánh U250: sơ đồ khối 4 lane SLR × 4 kênh DDR4
│   ├── DeepSeek_V4Pro_U250_Interactive_Explainer.html        ← nhánh U250: giải thích tương tác (click FU, step-through 8 layer, time–space)
│   ├── DeepSeek_V4Pro_U250_FU_Interface_Spec.md              ← nhánh U250: định nghĩa I/O từng FU (tensor + bus/port, hợp đồng interface)
│   └── DeepSeek_V4Flash_U250_Buffer_Spec.md                  ← đặc tả buffer on-chip: dung lượng · số bank · độ rộng bit (WTB/RAB/KVB/PSA/Ctrl)
├── 04_Pipeline_Diagrams/               ← pipeline theo thời gian
│   └── DeepSeek_V4Pro_Card_4Chip_Pipeline.html   (cấp card: 4 chip TP/EP song song)
├── 05_Explorer_Interactive/            ← bản trực quan tổng hợp 8 tab (hệ thống + chip)
│   ├── DeepSeek_V4Pro_Accelerator_Explorer.html
│   └── DeepSeek_V4Flash_2xU250_Explorer.html     ← nhánh V4-Flash: 4 tab (dataflow · hệ 2 card · vi kiến trúc chip · spec khối)
├── 06_Presentation/                    ← deck trình bày (15 slide, EN)
└── _archive/                           ← bản cũ/đã thay thế
```

Ba mức **pipeline diagram theo thời gian** (xem §4):
- **Trong chip** → `03/DeepSeek_V4Pro_SingleChip.html` (Tab Pipeline FRONT∥ECA) và `05/...Explorer.html` (Tab "Pipeline & lập lịch").
- **Trong card** → `04/DeepSeek_V4Pro_Card_4Chip_Pipeline.html`.
- **Trong hệ thống** → `05/...Explorer.html` (Tab "Lịch thực thi pipeline" — 8 card).

---

## 1. Bài toán & ràng buộc đầu vào

Thiết kế vi kiến trúc tăng tốc suy luận **DeepSeek-V4-Pro** (L=61, d=7168, MoE 384+1 expert/layer, top-6, ~1,6 T tham số, 49 B kích hoạt/token) trên hệ:
- **1 node CPU · 8 card FPGA · 4 chip/card = 32 chip** (Intel Agilex 7 AGM 039).
- Mỗi chip: **12.300 DSP · 46 MB M20K · 32 GB HBM2e @820 GB/s** → tổng **1.024 GB HBM**.

Ràng buộc nền tảng: toàn bộ trọng số phải **thường trú trong HBM** (không nạp lại mỗi token) → quyết định cách phân vùng song song và độ chính xác.

---

## 2. Hành trình thiết kế (các bước đối thoại)

| # | Câu hỏi/quyết định trong đối thoại | Kết quả | Deliverable |
|---|---|---|---|
| 1 | Ánh xạ layer/sublayer/block của DeepSeek-V4 vào FU phần cứng và vào 32 chip? | Chọn **pipeline-parallel** 8 card (giữ weight tĩnh/card) + TP/EP trong card; định nghĩa tập FU + ngân sách DSP | 02: Spec, FU_Mapping.xlsx |
| 2 | 128 GB/card không chứa hết weight — làm sao? | **FP4 cho MoE bắt buộc** (FP8 không vừa); weight ~797 GB / 1.024 GB (78%) thường trú HBM | 02: Spec §precision |
| 3 | Giao thức liên card: dữ liệu nào truyền? | Chỉ **residual X (56 KB/token)** qua card; packet header 32B, command set | 02: InterCard_Protocol.md |
| 4 | KV cache lưu bao nhiêu? Context & batch tối đa? | Tính KV/chip, trần B×ctx, context 1 M | 02: Spec §KV |
| 5 | 8 pipeline stage chạy thế nào theo thời gian? | Card c, tick t ⇒ microbatch (t−c) = **chunk/token của user khác nhau** | 05: Explorer Tab "Lịch thực thi" |
| 6 | Phân tích TP (attention head) và EP (MoE expert) | TP chia 128 head ÷ 4 chip; EP chia 384 expert ÷ 4 chip; all-reduce / all-to-all | 05: Explorer Tab TP↔EP, MoE routing |
| 7 | Vi kiến trúc 1 chip: FU + buffer + đường HBM↔buffer | mSGDMA, AGU, banked M20K (WTB/RAB/KVB/PSA), port A/B | 03: SingleChip (Tab khối), Spec |
| 8 | WTB chỉ 16 MB — chia tile weight thế nào? | Stream theo tile 8 MB double-buffer; ECA **weight-stationary** | 03: Spec §6 |
| 9 | EP top-6 thì 1 chip đọc bao nhiêu weight expert? | Theo **độ phủ batch**: 1 token→1,5 expert; **chunk 256→~96 expert→3,17 GB** | 03: Spec §7 |
| 10 | Pipeline R/C/W theo tile; prefill vs decode; fused | ECA read-bound; decode memory-bound; **fused/batching** lấp MAC | 05: Explorer Tab "Pipeline & lập lịch" |
| 11 | Tại sao FU xử lý 2 user lần lượt? Pipeline FU được không? | Một FU = một mảng MAC dùng chung → tuần tự; gộp để chia sẻ weight | §4, §5 |
| 12 | Gộp FU 1–9 thành FRONT, pipeline với ECA? | **Pipeline 2 tầng FRONT∥ECA** (DSP 7.000+5.266 ≈ 12.300) → ×1,5 | 03: SingleChip Tab Pipeline |
| 13 | Mở rộng ra 4 chip/card, giữ FU theo hàng | 4 chip song song TP/EP, wavefront, ICI đồng bộ, DMA HBM | 04: Card_4Chip_Pipeline |
| 14 | Tổng hợp & trình bày | Spec md, slide deck, Explorer 8 tab | 02, 05, 06 |

---

## 3. Ba cấp song song (Parallelism)

| Cấp | Phạm vi | Cơ chế | Đồng bộ | Vì sao chọn |
|---|---|---|---|---|
| **Pipeline parallel** | 8 card | Card c giữ ~8 layer cố định; token chảy C0→C7 | residual X qua link | Giữ weight tĩnh/card (không nạp lại); chỉ truyền 56 KB/token |
| **Tensor parallel (TP)** | 4 chip/card (attention) | 128 head ÷ 4 = 32 head/chip; output proj ÷ 4 | **all-reduce** ô | Attention dày đặc; chia head → 4 mảng MAC song song |
| **Expert parallel (EP)** | 4 chip/card (MoE) | 384 expert ÷ 4 = 96 expert/chip | **all-to-all** dispatch/combine | Expert thưa & to (33 MB FP4); di chuyển token (14 KB) thay vì weight |

Điểm phân biệt: **card ≠ chip**. Card = một pipeline-stage (dải layer). Chip = 1/4 của card, giữ một **slice** (head/expert) của *tất cả layer của card*; 4 chip chạy đồng thời, trao đổi qua **ICI**.

---

## 4. Pipeline theo thời gian — ba mức

Quy ước chung: **trục dọc = FU** (mỗi FU một hàng, hàng đầu mSGDMA = DMA HBM↔buffer); **trục ngang = thời gian (ms)**; ô màu = hành động **R đọc / C tính / W ghi**; băng dọc tím = **ICI** đồng bộ; ô đỏ nhạt = MAC chờ weight (read-bound).

### 4.1. Trong chip — `03/SingleChip.html` (Tab Pipeline) · `05/Explorer` (Tab Pipeline & lập lịch)
- Mỗi FU làm **R→C→W tuần tự** (một mảng MAC dùng chung).
- **ECA weight-stationary**: ôm 256 row, stream ~400 tile weight (3,17 GB) ⇒ load 3,86 ms > compute 1,6 ms ⇒ **read-bound**.
- **Pipeline 2 tầng FRONT∥ECA**: FRONT (FU1–9, ~7.000 DSP attention) chạy chunk kế trong khi ECA (5.266 DSP) grind chunk trước ⇒ ~×1,5; nút cổ chai cuối là HBM của ECA → khắc phục bằng **batching tại ECA**.

### 4.2. Trong card — `04/Card_4Chip_Pipeline.html`
- **4 chip** thẳng hàng thời gian = song song; cùng FU nhưng **slice khác nhau** (TP head / EP expert).
- Mỗi chip có hàng **mSGDMA** stream slice expert riêng (3,17 GB) từ HBM riêng.
- **ICI** (all-reduce sau attention; all-to-all dispatch/combine quanh MoE) = băng dọc đồng bộ 4 chip.
- Wavefront FRONT∥ECA cho nhiều user; tooltip tensor/size/GOP; zoom thời gian.

### 4.3. Trong hệ thống — `05/Explorer.html` (Tab "Lịch thực thi pipeline")
- **8 card = 8 pipeline-stage**; tại mỗi tick, 8 card xử lý **8 microbatch (8 user) khác nhau**.
- Card c, tick t ⇒ microbatch (t−c); mở rộng card→chip→FU; prefill/decode/fused.

---

## 5. Các quyết định & insight chính (rút từ đối thoại)

1. **FP4 cho MoE là bắt buộc** — FP8 (1.551 GB) vượt 1.024 GB; FP4 (~797 GB) vừa.
2. **Pipeline-parallel mức card** giữ weight tĩnh; chỉ residual qua card → băng thông liên card nhỏ.
3. **WTB 16 MB ≪ weight/layer** → weight **không cache**, phải **stream theo tile** (double-buffer 2×8 MB).
4. **ECA đọc theo độ phủ batch, không theo token**: chunk 256 phủ ~96 expert → 3,17 GB; mỗi expert reuse ~4 token ⇒ MoE prefill **memory-bound**.
5. **Phân bổ DSP ∝ GOP** (mHCE 36 … ECA 5.266, tổng 12.300) để C-time cân bằng và để **FRONT (7.000) + ECA (5.266) cùng vừa chip** → pipeline 2 tầng.
6. **Một FU = một mảng MAC dùng chung** → trong một chip các FU/2 user chạy tuần tự; song song thật ở **mức card (TP/EP)** và **mức hệ (8 card)**.
7. **Fused / continuous batching**: gom token để vừa phủ expert vừa tăng reuse → MAC từ ~7% (decode) lên ~96%.

---

## 6. Bản đồ file theo dòng thiết kế

| Dòng | Cấp | File | Vai trò |
|---|---|---|---|
| Đầu vào | — | `01_Background/*` | Mô hình DeepSeek-V4, glossary, so sánh FPGA, literature, proposal |
| Đầu vào | model | `01_.../DeepSeek_V4Pro_Dataflow_Software_Model.html` | Trực quan dataflow phần mềm: 1 chunk/1 user qua 61 block (CSA/HCA + MoE), thay đổi shape/byte tensor; toggle prefill 256 / decode 1 token |
| Đầu vào | model | `01_.../DeepSeek_V4Pro_Block_Dataflow_Diagram.html` | Sơ đồ khối dataflow 1 Transformer block (định tuyến góc vuông chuẩn hw-diagram); click từng khối tính toán → phép toán + tensor I/O; toggle CSA/HCA, prefill/decode |
| Hệ thống | system | `02_.../DeepSeek_V4Pro_Microarchitecture_Spec.md` | Spec tổng: 3 cấp song song, FU+DSP, bộ nhớ, roofline |
| Hệ thống | system | `02_.../DeepSeek_V4Pro_FU_Mapping.xlsx` (+Spec.md) | Mapping layer→sublayer→FU→chip, ngân sách DSP |
| Hệ thống | system | `02_.../DeepSeek_V4Pro_InterCard_Protocol.md` | Giao thức liên card (packet, command) |
| Chip | chip | `03_.../DeepSeek_V4Pro_SingleChip_Microarchitecture.md` | Spec vi kiến trúc 1 chip |
| Chip | chip | `03_.../DeepSeek_V4Pro_SingleChip.html` | Sơ đồ khối + pipeline FRONT∥ECA (interactive) |
| Chip | chip (U250) | `03_.../DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md` | Spec 1 chip trên Alveo U250 (slice 1/4 card, 8 layer): spec U250 đã kiểm chứng, roofline DDR4 65,3 GB/s, đảo phân bổ DSP (ECA 4.096), batching B=8, stripe 4 kênh × 4 SLR, so sánh AGM 039 |
| Chip | chip (U250) | `03_.../DeepSeek_V4Pro_SingleChip_U250.html` | Sơ đồ khối U250: 4 lane SLR (DDR4→DMA→WTB→FRONT∥ECA) → RAB; khối dùng chung PCIe/Ctrl/PSA/KVB/QSFP28 |
| Chip | chip (U250) | `03_.../DeepSeek_V4Pro_U250_Interactive_Explainer.html` | Tương tác: click từng FU (chức năng, I/O, DSP, TOPS, GOP, throughput, bảng tín hiệu bus AXI4-Stream: tên/độ rộng/clock/GB·s⁻¹); step-through 14 bước × 8 layer; waveform full-sequence từng khối; Gantt FRONT∥ECA; time–space 2 mức (layer 66,9 ms · tile 68 µs) |
| Chip | chip (U250) | `03_.../DeepSeek_V4Pro_U250_FU_Interface_Spec.md` | Hợp đồng interface 8 FU: cổng vào/ra (tensor theo glossary, shape, precision, bytes/chunk), buffer/bus (WT/RA/KV/PS-bus), handshake AXI4-Stream, điều khiển Ctrl/EWD/AXI4-Lite |
| Chip | buffer | `03_.../DeepSeek_V4Flash_U250_Buffer_Spec.md` | Đặc tả buffer on-chip nhánh V4-Flash: nguyên thuỷ URAM288/RAMB36 và trần f_max 500 MHz của XCU250-2L; định dạng từ 72 b = 64 b dữ liệu + 8 b phụ trợ; ngân sách tensor B=4; dung lượng · số bank · URAM/bank · độ rộng bit · băng thông cho WTB/RAB/KVB/PSA/Ctrl; lưu lượng DDR4 và chu kỳ layer 56,9 ms |
| Pipeline | card | `04_.../DeepSeek_V4Pro_Card_4Chip_Pipeline.html` | 4 chip song song TP/EP, DMA HBM, wavefront |
| Tổng hợp | system+chip | `05_.../DeepSeek_V4Pro_Accelerator_Explorer.html` | 8 tab: pipeline hệ, TP/EP, MoE, vi kiến trúc, pipeline & lập lịch |
| Tổng hợp | system+chip (V4-Flash) | `05_.../DeepSeek_V4Flash_2xU250_Explorer.html` | 4 tab V4-Flash trên 2×U250: dataflow đầy đủ; hệ 2 card (146>128 GB → chạy 32/43 layer, mapping L→chip, waveform 2 chunk, ~1.125 tok/s prefill); vi kiến trúc chip (block diagram + bảng bus + waveform op→HW có expand micro-op từng khối); spec 16 khối |
| Trình bày | — | `06_.../DeepSeek_V4Pro_Microarchitecture_Deck.pptx` | 15 slide (EN, executive) |

---

## 7. Trạng thái đồng bộ số liệu (cập nhật 21/09/2026)

Toàn bộ tài liệu nhánh U250 (V4-Pro và V4-Flash) đã chuyển sang bộ tham số nền tảng sau:

| Tham số | Giá trị cũ | Giá trị hiện hành | Căn cứ |
|---|---|---|---|
| Clock miền buffer + MAC | 600 MHz | **500 MHz** | f_max URAM ở −2 @0,72 V (XCU250-2L), OREG_B = True — DS923 |
| Peak INT8 / chip | 29,5 TOPS | **24,6 TOPS** | 24.576 mult × 2 op × 0,5 GHz |
| Từ nhớ URAM | 72 b dữ liệu | **64 b dữ liệu + 8 b phụ trợ** | ánh xạ khít tensor luỹ thừa 2; scale E8M0 / parity |
| Độ rộng bus WT / RA / KV / PS | 2.304 / 1.152 / 2.304 / 1.152 b | **2.048 / 1.024 / 512 / 512 b** | 32 / 16 / 8 / 8 bank × 64 b mỗi lane |
| Lưu lượng trọng số FP4 | không tính scale | **+6,25%** (MXFP4 khối 32) | OCP MX v1.0 |
| Residual X/X′ và tích luỹ y | không tính | **tính vào lưu lượng DDR4** | không đủ URAM để thường trú |
| Chu kỳ layer — V4-Pro U250 (B=8) | 50,3 ms | **66,9 ms** | tổng traffic 4,37 GB ÷ 65,3 GB/s |
| Chu kỳ layer — V4-Flash 2×U250 (B=4) | 51,3 ms | **56,9 ms** | tổng traffic 3,72 GB ÷ 65,3 GB/s |
| Phân bổ URAM | 1.192 khối, số URAM không chia hết cho số bank | **1.216 / 1.280 (95,0%)**, 304 mỗi SLR | §7.3 spec U250 · §10 spec V4-Flash |

File đã cập nhật: `03_.../DeepSeek_V4Pro_SingleChip_U250_Microarchitecture.md`, `03_.../DeepSeek_V4Pro_U250_FU_Interface_Spec.md`, `03_.../DeepSeek_V4Pro_SingleChip_U250.html`, `03_.../DeepSeek_V4Pro_U250_Interactive_Explainer.html`, `05_.../DeepSeek_V4Flash_2xU250_Explorer.html`. Bản trước khi sửa lưu ở `_archive/superseded-2026-09-21/`.

## 8. Bước tiếp theo (gợi ý)

1. Chốt vị trí tích luỹ `y` của nhánh V4-Pro U250: qua DDR4 (B=8, 3.828 tok/s) hay trên chip với B=4 (~1.900 tok/s) — căn cứ: h + y ở BF16 là 42,0 MiB, vượt 40,0 MiB dữ liệu của XCU250; ai: nhóm vi kiến trúc; khi: trước 19/10/2026.
2. Tổng hợp thử một lane WTB + RAB ở 500 MHz trên XCU250-2L để xác nhận slack — ai: nhóm RTL; khi: trước 05/10/2026.
3. Đo phân bố hợp top-k theo query-tile M trên trace thực để chốt sức chứa KVB — ai: nhóm mô hình; khi: trước 12/10/2026.
4. Bổ sung diagram pipeline **mức hệ 8 card** dạng standalone (hiện nằm trong tab Explorer).
5. Xuất `02`/`03` spec sang **docx song ngữ** cho deliverable cấp Trường.
