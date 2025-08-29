# Phân loại Clickbait Tiếng Việt Đa phương thức

Dự án này tập trung vào việc phân loại clickbait trong các bài đăng trên mạng xã hội tiếng Việt bằng cách sử dụng các mô hình đa phương thức, kết hợp thông tin từ cả văn bản và hình ảnh.

## Cài đặt

1.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

## Dữ liệu

Dự án sử dụng bộ dữ liệu ViClickbait. Đảm bảo rằng tệp `data_key_viclickbait.csv` và thư mục `images` được đặt trong thư mục `data/ViClickbait-2025`.

## Cách chạy thử nghiệm

Dưới đây là các lệnh để chạy các thử nghiệm đã được thực hiện. Các tập lệnh này sử dụng cờ `--testing` để chạy trên một mẫu nhỏ của dữ liệu. Để huấn luyện trên toàn bộ bộ dữ liệu, hãy xóa cờ `--testing`.

### 1. PhoBERT-base + ViT + ITC + ITM

```bash
python3 models/run_mm_late.py \
  --txt_model_name phobert \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --use_tim_loss --beta_itm 0.1 \
  --task 7 --epochs 15 \
  --save_model --save_preds
```

### 2. PhoBERT-base + ViT + ITC

```bash
python3 models/run_mm_late.py \
  --txt_model_name phobert \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --task 7 --epochs 15 \
  --save_model --save_preds
```

### 3. PhoBERT-large + ViT + ITC + ITM

```bash
python3 models/run_mm_late.py \
  --txt_model_name phobert-large \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --use_tim_loss --beta_itm 0.1 \
  --task 7 --epochs 15 \
  --save_model --save_preds
```

## Kết quả

Bảng dưới đây tóm tắt hiệu suất của các cấu hình mô hình khác nhau trên tập dữ liệu thử nghiệm, dựa trên epoch có điểm F1-macro cao nhất.

| Cấu hình mô hình | Epoch tốt nhất | F1-macro | F1-weighted | Precision-macro | Recall-macro | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PhoBERT-base` + `ViT` + `ITC` + `ITM` | 6 | 0.8360 | 0.8575 | 0.8309 | 0.8420 | 0.6540 |
| `PhoBERT-base` + `ViT` + `ITC` | 4 | 0.8427 | 0.8633 | 0.8375 | 0.8488 | 0.5467 |
| `PhoBERT-base` + `ViT` + `ITM` | 4 | 0.8231 | 0.8504 | 0.8387 | 0.8120 | 0.5417 |
| `PhoBERT-large` + `ViT` + `ITC` + `ITM` | 3 | 0.8382 | 0.8584 | 0.8301 | 0.8496 | 0.5736 |
| `PhoBERT-large` + `ViT` + `ITC` | 4 | 0.8444 | 0.8683 | 0.8609 | 0.8324 | 0.5908 |

Để biết thêm chi tiết, vui lòng tham khảo `report_detailed.md`.