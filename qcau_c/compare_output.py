import sys
from pathlib import Path
from itertools import zip_longest

def compare_files(file1_path, file2_path, n):
    """
    So sánh 2 file chứa số float theo từng dòng với độ sai lệch 1e-n.
    
    Args:
        file1_path (str): Đường dẫn file thứ nhất.
        file2_path (str): Đường dẫn file thứ hai.
        n (int/float): Số mũ n để xác định ngưỡng sai lệch (epsilon = 10^-n).
    """
    epsilon = 10 ** -n
    print(f"[*] Bắt đầu kiểm tra với ngưỡng sai lệch: {epsilon} (1e-{n})")
    
    is_valid = True
    line_count = 0
    lines = 0
    
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            # zip_longest giúp duyệt qua cả 2 file ngay cả khi số dòng không bằng nhau
            for i, (line1, line2) in enumerate(zip_longest(f1, f2), 1):
                
                # Trường hợp số dòng không khớp (một file hết trước)
                if line1 is None or line2 is None:
                    print(f"[!] Lỗi tại dòng {i}: Số lượng dòng giữa hai file không khớp.")
                    is_valid = False
                    continue # Hoặc break nếu muốn dừng ngay
                
                lines += 1
                try:
                    val1 = float(line1.strip())
                    val2 = float(line2.strip())
                    
                    diff = abs(val1 - val2)
                    
                    if diff > epsilon:
                        line_count += 1
                        print(f"[x] Sai lệch dòng {i}: |{val1} - {val2}| = {diff} > {epsilon}")
                        is_valid = False
                        
                except ValueError:
                    print(f"[!] Lỗi định dạng dữ liệu tại dòng {i}: Không thể chuyển đổi sang số.")
                    is_valid = False

        if is_valid:
            print(f"OK: Hai file khớp nhau hoàn toàn trong ngưỡng sai số 1e-{n}.")
        else:
            print(f"FAIL: Tìm thấy {line_count} sai lệch hoặc lỗi trong {lines} dòng đã kiểm tra.")
            
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")

# --- Ví dụ cách sử dụng ---
if __name__ == "__main__":
    # Giả sử bạn có 2 file tên là 'data1.txt' và 'data2.txt'
    # Bạn muốn kiểm tra sai số 1e-5 (n=5)
    
    # Gọi hàm kiểm tra
    data_dir = Path(__file__).resolve().parent / "data" / "hardware_seq16_layer2"
    file_1 = data_dir / "qcau_py_o_after_inverse_rope.bf16.txt"
    file_2 = data_dir / "test_output_0.txt"
    
    n_param = 2  # Kiểm tra với 1e-n
    
    compare_files(file_1, file_2, n_param)
