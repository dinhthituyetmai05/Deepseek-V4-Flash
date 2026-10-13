# <block_name> — Đặc tả khối / Block specification

| Thuộc tính | Giá trị |
|---|---|
| Phiên bản | v0.1 |
| Tác giả | |
| Ngày | YYYY-MM-DD |
| RTL | `04-Src/hardware/rtl/<block_name>.sv` |
| C / Python model | `04-Src/cmodel/<block_name>.c` |
| Testbench | `05-Experimental/testbench/<block_name>/` |

## 1. Chức năng / Function

## 2. Giao diện / Interface

| Port | Hướng | Độ rộng | Định dạng (Q m.n, signed/unsigned) | Mô tả |
|---|---|---|---|---|
| `clk` | in | 1 | — | Clock |
| `rst_n` | in | 1 | — | Reset bất đồng bộ, active-low |

## 3. Tham số / Parameters

| Tên | Mặc định | Phạm vi | Mô tả |
|---|---|---|---|

## 4. Hoạt động và timing / Operation & timing
Waveform nguồn: `03-Docs/waveform/<block_name>.json`.

## 5. Hiệu năng / Performance

| Chỉ số | Giá trị | Đơn vị | Điều kiện (node, corner, f) | Nguồn |
|---|---|---|---|---|
| Latency | | cycle | | |
| Throughput | | sample/s | | |
| Area | | mm² | | |
| Power | | µW | | |

## 6. Kiểm chứng / Verification
Golden vector, test case, coverage, kết quả `TEST <name> PASS mismatches=0 samples=<N>`.

## 7. Lịch sử thay đổi / Revision history

| Phiên bản | Ngày | Người sửa | Nội dung |
|---|---|---|---|
