# Protocol giao tiếp liên card — DeepSeek-V4-Pro Accelerator
# Inter-Card Communication Protocol

**Tài liệu thiết kế — project DeepSeek-FPGA, PGS.TS Nguyễn Đức Minh (HUST/SEEE)**
Bổ sung cho `DeepSeek_V4Pro_FU_Mapping_Spec.md` (sơ đồ 8 card pipeline × 4 chipset EP/TP).

---

## Mục lục

1. Nguyên tắc & dữ liệu nào được truyền
2. Topology vật lý & lớp giao thức
3. Bảng link card i → card j (layer nào → layer nào)
4. Định dạng packet (header bit-level + payload)
5. Tập lệnh (command set)
6. Cờ điều khiển (control flags)
7. Flow control, ordering, error handling
8. Vòng lặp decode (tail → head) & prefill vs decode
9. Ngân sách băng thông & độ trễ
10. Trường hợp đặc biệt (KV alloc, MTP, boot)

---

## 1. Nguyên tắc & dữ liệu nào được truyền / What crosses the link

Quyết định nền tảng: **pipeline-parallel giữ trọng số TĨNH**.

| Loại dữ liệu | Có truyền liên card? | Thời điểm | Đường |
|---|---|---|---|
| **Weight** (expert, attention, embedding, LM head) | **KHÔNG** trong inference | Một lần lúc boot | Host → HBM mỗi chipset (PCIe Gen5) |
| **KV cache** | **KHÔNG** | — | Thường trú HBM của chipset sở hữu layer đó |
| **Activation — residual mHC `X`** | **CÓ** (dữ liệu chính) | Mỗi token/micro-batch | Card i → Card i+1 (forward chain) |
| **token_ids** (đầu vào) | CÓ | Mỗi step | Host → Card 0; và Card 7 → Card 0 (decode loop) |
| **logits** (đầu ra) | CÓ | Mỗi step | Card 7 → Host |
| **Control/command** (credit, seq-open, sync) | CÓ | Liên tục | Reverse channel mỗi link + ring |

Hệ quả của việc gán block liên tiếp cho từng card: **toàn bộ 384 expert của một layer nằm trong cùng 1 card**, nên all-to-all dispatch của MoE là **nội card (ICI)**, KHÔNG bao giờ qua liên card. Liên card chỉ tải duy nhất residual `X` — tensor mỏng `(n_hc, d) = (4, 7168)` = **56 KB/token (BF16)**.

> Vì sao là `X` mở rộng `(4, 7168)` chứ không phải `h (7168)`: mHC (Manifold-Constrained Hyper-Connections) mang trạng thái residual ở `n_hc=4` luồng song song xuyên suốt mạng. Phải forward đủ 4 luồng để card kế tiếp tiếp tục mixing `X ← B·X + C·F`. Giữ **BF16** (không lượng tử) để tránh tích lũy sai số qua 61 layer; băng thông dư thừa nên không cần nén.

---

## 2. Topology vật lý & lớp giao thức / Physical topology & stack

**Topology: 1-D bidirectional ring (chuỗi pipeline khép kín).**

```
        token_ids                                              logits
 Host ─────────────▶ Card0 ─▶ Card1 ─▶ … ─▶ Card6 ─▶ Card7 ──────────▶ Host
   ▲   (PCIe Gen5)    │L1-8    │L9-16        │L49-56   │L57-61    (PCIe)
   │                  ◀────────◀────────…────◀─────────◀  reverse: credit + control
   └──────────────────────────────────────────────────┘  C7→C0 fast path: next token_id (decode loop)
```

- **Forward chain (C i → C i+1):** tải residual `X`. 7 link liên card.
- **Reverse channel (C i+1 → C i):** credit flow-control + NACK + control. Cùng cặp transceiver song công.
- **Ring closure (C7 → C0):** token_id của step kế tiếp trong vòng decode (bỏ qua host để giảm latency); tùy chọn, mặc định host-in-the-loop.

**Lớp giao thức (4 lớp):**

| Lớp | Nội dung | Hiện thực |
|---|---|---|
| L1 Physical | 4–8 lane × 112 Gbps PAM4 → 400–800 Gbps/hướng | Agilex 7 F-tile/E-tile transceiver (≤116G/lane) |
| L2 Link | Framing 64b/66b, RS-FEC, CRC32, credit flow-control, replay buffer | Aurora-like / custom RTL |
| L3 Transport | Packet header (§4) + payload, phân mảnh prefill | RTL packetizer/DMA |
| L4 Semantic | Pipeline handshake, micro-batch scheduling, phase prefill/decode | Control FSM + host orchestrator |

---

## 3. Bảng link card i → card j (layer → layer) / Per-link map

Pipeline tuyến tính: **j = i+1**, không skip. Mỗi link tải output residual của block cuối card i sang input của block đầu card j.

| Link | Nguồn (block cuối) | Đích (block đầu) | Payload | dtype | Bytes/token |
|---|---|---|---|---|---|
| Host → C0 | — | C0: **L1** (sau Embedding ở C0.0) | token_ids | INT32 | 4 B |
| **C0 → C1** | C0: out của **L8** (MoE-mHC) | C1: in của **L9** | residual `X(4,7168)` | BF16 | 56 KB |
| **C1 → C2** | out **L16** | in **L17** | `X` | BF16 | 56 KB |
| **C2 → C3** | out **L24** | in **L25** | `X` | BF16 | 56 KB |
| **C3 → C4** | out **L32** | in **L33** | `X` | BF16 | 56 KB |
| **C4 → C5** | out **L40** | in **L41** | `X` | BF16 | 56 KB |
| **C5 → C6** | out **L48** | in **L49** | `X` | BF16 | 56 KB |
| **C6 → C7** | out **L56** | in **L57** | `X` | BF16 | 56 KB |
| C7 → Host | out **L61** + LM Head | sampler | logits (hoặc top-k) | BF16 | 256 KB (full) / ~0.5 KB (top-64) |
| C7 → C0 | sampler kết quả | C0 Embedding | token_id step t+1 | INT32 | 4 B |

**Điểm nối nội card (KHÔNG qua liên card, để tham chiếu):** trong card, residual `X` đi tuần tự qua 8 block; tại mỗi MoE, token dispatch all-to-all giữa 4 chipset (`C i.0..C i.3`) qua ICI; TP-reduce attention cũng nội card.

---

## 4. Định dạng packet / Packet format

**Header cố định 32 byte (256-bit), little-endian:**

| Field | Bits | Mô tả |
|---|---:|---|
| `SOF` | 8 | Start-of-frame magic (0xA5) |
| `CMD` | 4 | Mã lệnh (§5) |
| `PHASE` | 1 | 0=prefill, 1=decode |
| `VALID` | 1 | Payload hợp lệ (0 = bubble/idle slot) |
| `FIRST` | 1 | Token đầu của sequence → trigger KV alloc |
| `LAST` | 1 | Token cuối (EOS / last micro-batch) |
| `MTP_EN` | 1 | Yêu cầu chạy MTP ở tail (Card 7) |
| `PRIO` | 2 | Lớp ưu tiên (multi-tenant) |
| `rsv0` | 1 | dự phòng |
| `SEQ_ID` | 16 | ID sequence/request (64K đồng thời) |
| `MICROBATCH` | 4 | Slot pipeline (0–15) |
| `TOKEN_POS` | 24 | Vị trí tuyệt đối `t` (≤16M, đủ cho ctx 1M) |
| `SRC_CARD` | 3 | Card nguồn (0–7) |
| `DST_CARD` | 3 | Card đích (0–7) |
| `SRC_BLOCK` | 8 | Block cuối đã xử lý (vd 8) |
| `DST_BLOCK` | 8 | Block đích (vd 9) |
| `N_TOKENS` | 16 | Số token trong payload (prefill burst) |
| `PAYLOAD_LEN` | 24 | Số byte payload |
| `DTYPE` | 4 | 0=BF16,1=FP8E4M3,2=FP8E5M2,3=INT32,4=INT8 |
| `SCALE` | 16 | Shared-exponent/scale (khi FP8); =0 với BF16 |
| `CREDIT` | 16 | Số credit trả về (piggyback trên reverse) |
| `rsv1` | 28 | dự phòng/alignment |
| `HDR_CRC` | 32 | CRC32 của header |

**Payload:** `N_TOKENS × n_hc × d × sizeof(DTYPE)`. Decode (`N_TOKENS=1`, BF16): `1×4×7168×2 = 57,344 B`. Theo sau là **`PAYLOAD_CRC` (32-bit)**.

Overhead header+CRC = 36 B / 57,344 B = **0.06%** (không đáng kể).

---

## 5. Tập lệnh / Command set (`CMD`)

| Mã | Lệnh | Hướng | Payload | Tác dụng |
|---|---|---|---|---|
| 0x0 | `IDLE` | bất kỳ | — | Bubble/keep-alive (VALID=0) |
| 0x1 | `DATA_FWD` | forward | residual `X` | Forward activation sang card kế; lệnh chính |
| 0x2 | `CREDIT` | reverse | CREDIT field | Trả credit flow-control |
| 0x3 | `SEQ_OPEN` | forward (lan truyền) | seq_id, max_len, phase | Mỗi card cấp vùng KV cache cho layer của mình |
| 0x4 | `SEQ_CLOSE` | forward | seq_id | Giải phóng/đưa KV vào `_archive` slot, free buffer |
| 0x5 | `SYNC/BARRIER` | ring | epoch_id | Flush pipeline; chuyển prefill→decode; checkpoint |
| 0x6 | `TOKEN_FB` | C7→C0 | token_id t+1 | Vòng decode: nạp token mới vào Embedding |
| 0x7 | `NACK` | reverse | seq_id, microbatch | Lỗi CRC → yêu cầu replay từ replay buffer |
| 0x8 | `WEIGHT_LD` | host→card | block_id, addr, data | Chỉ lúc boot: nạp trọng số vào HBM |
| 0x9 | `STATUS/HB` | reverse | counters | Heartbeat, telemetry (HBM util, queue depth) |
| 0xF | `EOS` | forward | seq_id | Kết thúc sinh; trigger SEQ_CLOSE downstream |

---

## 6. Cờ điều khiển / Control flags

- `VALID`: phân biệt slot có dữ liệu vs bong bóng pipeline. Card nhận VALID=0 thì chuyển tiếp nhịp mà không tính toán.
- `PHASE`: prefill (xử lý cả `n` token, ghi đầy KV) vs decode (1 token, append KV). Quyết định cấu hình QCAU/ACU và kích thước payload.
- `FIRST`: token đầu sequence → mỗi card cấp KV region (kết hợp `SEQ_OPEN`).
- `LAST`/`EOS`: token cuối → flush + `SEQ_CLOSE` lan truyền.
- `MTP_EN`: bật MTP block ở Card 7 cho speculative decoding; tắt khi không dùng để tiết kiệm.
- `PRIO`: 4 lớp ưu tiên cho lập lịch nhiều người dùng trên cùng pipeline.
- `DTYPE`+`SCALE`: cho phép hạ FP8 nếu cần băng thông (mặc định BF16 cho residual).
- `MICROBATCH`+`SEQ_ID`: định danh luồng để giữ đúng thứ tự token theo từng sequence.

---

## 7. Flow control, ordering, error / Điều khiển luồng, thứ tự, lỗi

**Flow control — credit-based.** Mỗi card có ingress FIFO độ sâu `D` slot. Khi drain 1 slot, gửi `CREDIT` ngược lên upstream. Card nguồn chỉ gửi `DATA_FWD` khi còn credit → chống tràn (backpressure) khi một stage chậm (vd MoE nặng).

**Ordering.** Pipeline decode được lấp bằng **nhiều sequence khác nhau** (tối đa 16 micro-batch slot), KHÔNG phải nhiều token cùng sequence (vì autoregressive). Mỗi sequence chỉ có 1 token "in-flight" tại một thời điểm → thứ tự token trong sequence tự bảo toàn. `SEQ_ID`+`MICROBATCH` định tuyến đúng slot.

**Error handling.** RS-FEC ở transceiver sửa lỗi bit nền; CRC32 (header + payload) phát hiện lỗi còn sót → `NACK` → replay từ **replay buffer** (giữ `D` packet gần nhất, Go-Back-N). `SYNC/BARRIER` cho phép checkpoint và phục hồi pipeline khi lỗi nặng.

---

## 8. Vòng lặp decode (tail → head) & prefill vs decode

**Prefill (1 lần/request, ctx `n`):** Host gửi `n` token_ids → C0. Mỗi card xử lý cả `n` vị trí, ghi đầy KV cache của các layer mình giữ, rồi forward `n × 56 KB` residual sang card kế (phân mảnh/streaming theo chunk để overlap compute). `PHASE=0`.

**Decode (lặp mỗi token):** `PHASE=1`. Token đi C0→…→C7 (~3.9 ms qua 8 stage). Card 7: LM Head → logits → host (hoặc on-card) sampling → token t+1 → **`TOKEN_FB` C7→C0** → bước kế. Pipeline lấp bằng 8–16 sequence song song để cả 8 card luôn bận.

**Tại sao cần C7→C0:** decode tuần tự, token mới phải vào lại đầu mạng. Đường ring trực tiếp tránh round-trip qua host (tiết kiệm vài µs–chục µs/token).

---

## 9. Ngân sách băng thông & độ trễ / Bandwidth & latency budget

| Tình huống | Tải/link | Trên 400G | Trên 800G | Nhận xét |
|---|---|---|---|---|
| Decode steady (sys ~640 tok/s) | 36.7 MB/s | 0.7% | 0.4% | dư thừa lớn |
| Prefill n=4K | 0.23 GB | 4.7 ms | 2.3 ms | overlap compute |
| Prefill n=64K | 3.76 GB | 75 ms | 38 ms | streaming chunk |
| Prefill n=1M | 57.3 GB | 1.15 s | 0.57 s | dùng 800G + overlap |
| token_id feedback | 4 B/token | ~0% | ~0% | trivial |
| logits (top-64) | ~0.5 KB/token | ~0% | ~0% | gửi top-k thay full vocab |

**Latency:** per-hop transceiver+framing ~0.5–1 µs; 7 hop ~ **3.5–7 µs** ≪ stage compute ~560 µs/card. → Liên card KHÔNG phải bottleneck cho decode; chỉ cần chú ý prefill ctx rất dài.

**Khuyến nghị link:** 4 lane × 112G = 400 Gbps/hướng đủ cho decode + prefill ≤64K; cấu hình 8 lane = 800 Gbps khi phục vụ prefill 1M context.

---

## 10. Trường hợp đặc biệt / Special cases

1. **Cấp KV cache (`SEQ_OPEN`):** khi request mới vào, lệnh lan truyền C0→C7; mỗi card cấp vùng KV cho các layer của mình (CSA ~144 MB/layer@1M, HCA ~4.5 MB/layer@1M). Host theo dõi tổng để không vượt HBM.
2. **MTP ở Card 7:** MTP block dự đoán token t+2 cần embedding của token t+1. Giải pháp: **replicate `W_emb` (917 MB) trên C7.0** hoặc host cấp embedding qua `TOKEN_FB`. Mặc định replicate để tự chủ trong ring.
3. **Boot/weight load (`WEIGHT_LD`):** host DMA 797 GB vào 32 HBM stack qua PCIe Gen5 (một lần, ~vài phút). Không dùng link liên card.
4. **Multi-tenant:** `PRIO` + `SEQ_ID` cho phép trộn nhiều người dùng; mỗi micro-batch slot là một sequence độc lập.

---

## Tóm tắt một câu / One-line summary

Giữa card i và card i+1 chỉ truyền **residual mHC `X(4,7168)` BF16 = 56 KB/token** (output block cuối card i → input block đầu card i+1), kèm header 32 B mang `CMD=DATA_FWD`, `SEQ_ID`, `TOKEN_POS`, `PHASE`, `VALID/FIRST/LAST/MTP_EN`; weight và KV cache KHÔNG bao giờ rời chipset; vòng decode khép ring qua `TOKEN_FB` C7→C0.

## Tài liệu tham khảo / References

[1] DeepSeek-AI, "DeepSeek-V4 Technical Report," May 2026.
[2] Intel, "Agilex 7 M-Series Transceiver (F-tile/E-tile) User Guide," 2025.
[3] AMD, "Aurora 64B/66B Protocol Specification (SP011)," 2023.
[4] J. Xie et al., "Manifold-Constrained Hyper-Connections," 2026.
[5] `DeepSeek_V4Pro_FU_Mapping_Spec.md` (project, kèm theo).
