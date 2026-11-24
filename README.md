# Tra cứu video tương đồng

Mã trong thư mục này kế thừa quy trình trích rút A–F và giao diện tìm kiếm của dự án. SQLite lưu 324 giá trị và 324 cờ hợp lệ của mỗi video. Bộ tìm kiếm dùng NumPy để chuẩn hóa theo tập ứng viên, tính điểm theo nhóm và trả top 5.

| Thư mục | Chức năng |
|---|---|
| `data/` | Sáu nhóm video KTH trong `dataset/`, cấu hình trích rút và mô hình PP-HumanSeg. `query_videos/` là thư mục dành cho tệp truy vấn khi xây dựng chức năng nhận tệp mới. |
| `database/` | Tệp SQLite `videos.db`. Mã kết nối, lược đồ và truy cập dữ liệu nằm trong `src/database/`. |
| `src/preprocessing/` và `src/features/` | Đọc khung hình, phân đoạn, optical flow và tính riêng sáu nhóm A–F. |
| `src/retrieval/` và `src/services/` | Chuẩn hóa, đo khoảng cách, xếp hạng và điều phối các bước lập chỉ mục hoặc truy vấn. |
| `src/visualization/` và `src/evaluation/` | Tệp giao diện, hình kiểm tra, xuất đặc trưng và kiểm tra tính nhất quán của kết quả trích rút. |
| `scripts/` và `tests/` | Các lệnh thao tác và bộ kiểm thử kế thừa. |
| `outputs/` | Kết quả trích rút mẫu và toàn bộ nằm dưới `intermediate/`. Báo cáo kiểm tra do `scripts.evaluate` ghi vào `evaluation/`. `search_results/` có thể dùng để lưu kết quả truy vấn xuất từ terminal. |

## Chạy trên Windows

Mở PowerShell tại thư mục chứa `app.py`. Dùng Python 3.13 để cài các thư viện:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Đặt Dataset vào `data/dataset/`, với sáu thư mục `boxing`, `handclapping`, `handwaving`, `jogging`, `running` và `walking`. Các tệp cần có định dạng như `data/dataset/boxing/person01_boxing_d1.mp4`. Sau đó trích rút, tạo CSDL và mở ứng dụng:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_database --mode full
.\.venv\Scripts\python.exe app.py
```

Mở [giao diện cục bộ](http://127.0.0.1:8061/). Nếu cổng đang được dùng, chạy `app.py --port 8063`. Nếu đã có kết quả trích rút, chạy `-m scripts.build_database` để nhập từ `outputs/intermediate/full` hoặc `../results/full`. Dashboard có thể sử dụng lại `database/videos.db` sau khi nhập.

Các lệnh sau gọi lại các chức năng đã có:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_database --mode pilot
.\.venv\Scripts\python.exe -m scripts.query_video boxing/person01_boxing_d1.mp4
.\.venv\Scripts\python.exe -m scripts.fit_normalizer boxing/person01_boxing_d1.mp4
.\.venv\Scripts\python.exe -m scripts.evaluate --mode full
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--mode pilot` trích rút 12 video và lưu ảnh chẩn đoán. `--mode full` trích rút toàn bộ rồi nhập kết quả vào SQLite. Khi cần tạo môi trường mới, cài các gói trong [requirements.txt](requirements.txt). Mô hình đi kèm có [giấy phép](data/PPHumanSeg.LICENSE).

Truy vấn hiện dùng mã video đã có trong kho. Bộ trích rút kiểm tra video có kích thước 160 × 120 và tốc độ 25 fps, rồi chuyển các khung hình sang ảnh xám.
