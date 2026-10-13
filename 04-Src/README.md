# 04-Src — Mã nguồn / Source code

| Thư mục | Nội dung |
|---|---|
| `software/` | Tiền xử lý, huấn luyện, lượng tử hóa (Python) |
| `cmodel/` | C model bit-accurate, sinh golden vector |
| `hardware/rtl/` | RTL tổng hợp được |
| `hardware/fpga/` | Project FPGA, constraint, host script |

Quy tắc: không đường dẫn tuyệt đối; không output tool; thư viện bên thứ ba cài qua `requirements.txt` hoặc `git submodule` (có `.gitmodules`), không chép repo lồng.
