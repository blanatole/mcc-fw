# Phân loại Clickbait Đa phương thức (Tiếng Việt & Tiếng Anh)

Dự án này tập trung vào việc phân loại clickbait trong các bài đăng trên mạng xã hội bằng cách sử dụng các mô hình đa phương thức, kết hợp thông tin từ cả văn bản và hình ảnh. Dự án hỗ trợ cả bộ dữ liệu tiếng Việt (ViClickbait) và tiếng Anh (WebisClickbaitCorpus2017).

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

Để chạy các thử nghiệm, bạn cần tải xuống và xử lý các bộ dữ liệu. Các thư mục dữ liệu thô (`ViClickbait-2025` và `WebisClickbaitCorpus2017`) đã được thêm vào `.gitignore`.

### 1. ViClickbait (Tiếng Việt)

1.  **Cấu trúc thư mục:** Tạo cấu trúc thư mục sau:
    ```
    data/
    └── ViClickbait-2025/
        ├── clickbait_dataset_vietnamese.csv
        └── images/
            ├── image1.jpg
            └── ...
    ```

2.  **Tạo tệp Key:** Chạy kịch bản sau để tạo tệp `data_key_viclickbait.csv`:
    ```bash
    python3 preprocessing/create_viclickbait.py
    ```

### 2. WebisClickbaitCorpus2017 (Tiếng Anh)

1.  **Tải dữ liệu:** Tải các tệp `train` và `test` từ [Zenodo](https://zenodo.org/records/5530410) và lưu chúng vào thư mục `data`.
    *   Test: `wget -O data/clickbait17-test-170720.zip 'https://zenodo.org/records/5530410/files/clickbait17-test-170720.zip?download=1'`
    *   Train: `wget -O data/clickbait17-train-170630.zip 'https://zenodo.org/records/5530410/files/clickbait17-train-170630.zip?download=1'`

2.  **Giải nén:** Giải nén cả hai tệp vào thư mục `data/WebisClickbaitCorpus2017`.
    ```bash
    unzip data/clickbait17-train-170630.zip -d data/WebisClickbaitCorpus2017
    unzip data/clickbait17-test-170720.zip -d data/WebisClickbaitCorpus2017
    ```

3.  **Tạo tệp Key:** Chạy kịch bản sau để gộp, chia lại dữ liệu (80/10/10) và tạo tệp `data_key_webis2017.csv`:
    ```bash
    python3 preprocessing/create_webis2017.py
    ```

## Cách chạy thử nghiệm

Dưới đây là các lệnh để chạy các thử nghiệm đã được thực hiện. Các tập lệnh này sử dụng cờ `--testing` để chạy trên một mẫu nhỏ của dữ liệu. Để huấn luyện trên toàn bộ bộ dữ liệu, hãy xóa cờ `--testing`.

### Thử nghiệm trên bộ dữ liệu ViClickbait (Tiếng Việt)

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

### Thử nghiệm trên bộ dữ liệu WebisClickbaitCorpus2017 (Tiếng Anh)

#### 1. Bertweet + ViT + ITC + ITM

```bash
python3 models/run_mm_late.py \
  --txt_model_name bertweet \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --use_tim_loss --beta_itm 0.1 \
  --task 8 --epochs 3 \
  --save_model --save_preds
```

```

## Kết quả

### Kết quả trên bộ dữ liệu ViClickbait

Bảng dưới đây tóm tắt hiệu suất của các cấu hình mô hình khác nhau trên tập dữ liệu thử nghiệm, dựa trên epoch có điểm F1-macro cao nhất.

| Cấu hình mô hình | Epoch tốt nhất | F1-macro | F1-weighted | Precision-macro | Recall-macro | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PhoBERT-base` + `ViT` + `ITC` + `ITM` | 6 | 0.8360 | 0.8575 | 0.8309 | 0.8420 | 0.6540 |
| `PhoBERT-base` + `ViT` + `ITC` | 4 | 0.8427 | 0.8633 | 0.8375 | 0.8488 | 0.5467 |
| `PhoBERT-base` + `ViT` + `ITM` | 4 | 0.8231 | 0.8504 | 0.8387 | 0.8120 | 0.5417 |
| `PhoBERT-large` + `ViT` + `ITC` + `ITM` | 3 | 0.8382 | 0.8584 | 0.8301 | 0.8496 | 0.5736 |
| `PhoBERT-large` + `ViT` + `ITC` | 4 | 0.8444 | 0.8683 | 0.8609 | 0.8324 | 0.5908 |

Để biết thêm chi tiết, vui lòng tham khảo `report_detailed_29082025.md`.


### Kết quả trên bộ dữ liệu WebisClickbaitCorpus2017

Bảng dưới đây tóm tắt hiệu suất của các cấu hình mô hình khác nhau trên tập dữ liệu thử nghiệm, dựa trên epoch có điểm F1-macro cao nhất.

| Cấu hình mô hình | Epoch tốt nhất | F1-macro | F1-weighted | Precision-macro | Recall-macro | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bertweet` + `ViT` + `ITC` + `ITM` | 1 | 0.8195 | 0.8664 | 0.8123 | 0.8277 | 0.6657 |
| `bertweet` + `ViT` + `ITC` | 1 | 0.8210 | 0.8674 | 0.8136 | 0.8295 | 0.6341 |
| `bertweet` + `ViT` + `ITM` | 1 | 0.8187 | 0.8652 | 0.8093 | 0.8302 | 0.4243 |

Để biết thêm chi tiết, vui lòng tham khảo `report_detailed_02092025.md`.
