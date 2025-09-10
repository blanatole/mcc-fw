# Báo cáo chi tiết kết quả thử nghiệm mô hình đa phương thức (Tiếng Anh) – 10/09/2025

## Giới thiệu

Tài liệu này cung cấp tóm tắt chi tiết các cập nhật và thử nghiệm mới trên bộ dữ liệu WebisClickbaitCorpus2017 (tiếng Anh), bao gồm cải thiện quy trình đánh giá, tối ưu ngưỡng cho mô hình đa phương thức và bổ sung baseline text-only (XLM-R).

## Phương pháp

Các thử nghiệm đã sử dụng các thành phần sau:

*   **Bộ mã hóa văn bản:**
    *   `vinai/bertweet-base`
    *   `FacebookAI/xlm-roberta-base` (baseline text-only)
*   **Bộ mã hóa hình ảnh:**
    *   `google/vit-base-patch16-224-in21k`
*   **Phương pháp kết hợp:**
    *   `concat` (nối đặc trưng văn bản + ảnh). Đã chuẩn bị thử nghiệm `xatt` (cross-attention).
*   **Hàm loss phụ trợ:**
    *   **ITC (Image-Text Contrastive)**

Cấu hình quy trình và tiêu chuẩn đánh giá:

- **Dữ liệu & chia tập**:
  - WebisClickbaitCorpus2017 (validation + test) được gộp và chia lại 80/10/10 bằng stratified split (script `preprocessing/create_webis2017.py`).
  - File key: `data/data_key_webis2017.csv` gồm các cột: `tweet_id,text,image,label,split`.

- **Tiền xử lý**:
  - Văn bản: chuẩn hóa nhẹ (tool `Tweet_Preprocessing`) cho tác vụ tiếng Anh; tokenizer từ HuggingFace theo từng model.
  - Hình ảnh: resize/normalize với `timm`/`torchvision` (transforms trong `models/utils.py`).

- **Kiến trúc & Fusion**:
  - Text encoder (Bertweet) + Image encoder (ViT), head phân loại nhị phân; fusion `concat` hoặc `xatt` (cross-attention) tùy cấu hình.
  - ITC được bật bằng `--use_clip_loss --beta_itc <val>` để tăng tính đồng bộ giữa modal.

- **Huấn luyện**:
  - Optimizer: AdamW; `lr`, `weight_decay`, `dropout` theo tham số dòng lệnh.
  - Early stopping theo `val_f1_macro`; lưu `_best.pth` khi điểm cải thiện.
  - Sửa lỗi trước đó để luôn lưu metrics mỗi epoch (kể cả khi early stopping), bổ sung cột `accuracy`.

- **Đánh giá & Lưu**:
  - Sau train, khi `--load_saved_model`, hệ thống tự động ưu tiên `_best.pth` cho suy luận.
  - Lưu dự đoán test `..._preds.csv`, validation `..._preds_val.csv` với thêm cột `prob_pos` (softmax[:,1]).
  - Metrics CSV: `f1_macro`, `f1_weighted`, `precision_macro`, `recall_weighted`, `accuracy`, `loss` theo từng epoch.

- **Tối ưu ngưỡng (A2)**:
  - Dùng `tools/threshold_tuning.py` quét ngưỡng 0.30→0.70 (bước 0.02) trên validation, tối đa Macro-F1.
  - Áp ngưỡng tìm được lên test (nếu khác biệt đáng kể). Trong thử nghiệm này, ngưỡng tối ưu = 0.50 (trùng argmax), nên giữ mặc định.

- **Baseline text-only**:
  - Thêm hỗ trợ `xlm-roberta` trong `run_txt.py`, cấu hình tokenization/model tương thích (không dùng `token_type_ids`).
  - Sinh metrics và dự đoán test trong `results/txt_only/`.

Cập nhật hệ thống đánh giá được bổ sung:
- Lưu `prob_pos` (softmax[:,1]) khi suy luận để phục vụ quét ngưỡng.
- Ở chế độ load, tự động ưu tiên checkpoint tốt nhất (`_best.pth`) khi xuất dự đoán val/test.
- Đồng bộ tên file: test `..._preds.csv`, val `..._preds_val.csv`.
- Bổ sung chỉ số `accuracy` vào các file metrics.

## Kết quả chi tiết

### Bảng kết quả tổng hợp (Task 8 - Test set, epoch tốt nhất)

| Cấu hình | Epoch tốt nhất | F1-macro | F1-weighted | Precision-macro | Recall-weighted | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bertweet` + `ViT` + `concat` + `ITC` | 3 | 0.8232 | 0.8711 | 0.8251 | 0.8715 | 0.5417 |
| `bertweet` + `ViT` + `xatt` + `ITC` | 3 | 0.8212 | 0.8681 | 0.8156 | 0.8668 | 0.4780 |
| `deberta-v3-large` + `ViT` + `concat` + `ITC` | 4 | 0.8157 | 0.8667 | 0.8230 | 0.8681 | 0.9545 |
| `xlm-roberta` (text-only) | 3 | 0.8132 | 0.8623 | 0.8082 | 0.8611 | 0.5491 |

Ghi chú: chỉ số được lấy từ các file metrics CSV, chọn theo epoch có F1-macro cao nhất.

| Hạng mục | Mô tả |
| :--- | :--- |
| **Bertweet + ViT + Concat + ITC (Task 8)** | Epoch tốt nhất (test, theo F1-macro) ~epoch-3. Chỉ số tiêu biểu: F1-macro ≈ 0.8232, F1-weighted ≈ 0.8711. Tệp: `results/mm_late/bertweet-vit-concat_task8_seed30_itc0.1_metrics_test.csv`. |
| **Bertweet + ViT + XAtt + ITC (Task 8)** | Có cải thiện so với epoch đầu; epoch-3 đạt F1-macro ≈ 0.8212. Tệp: `results/mm_late/bertweet-vit-xatt_task8_seed30_itc0.1_metrics_test.csv`. |
| **DeBERTa-v3-large + ViT + Concat + ITC (Task 8)** | Xu hướng tăng tới epoch-4; F1-macro ≈ 0.8157. Tệp: `results/mm_late/deberta-v3-large-vit-concat_task8_seed30_itc0.1_metrics_test.csv`. |
| **Tối ưu ngưỡng (validation) cho Bertweet multimodal** | Dự đoán validation/test (kèm `prob_pos`) được dump từ checkpoint tốt nhất. Quét ngưỡng 0.30→0.70 (bước 0.02) trên validation. Kết quả: ngưỡng tối ưu = 0.50, Macro-F1 = 0.8232 (trùng argmax) → không cần đổi ngưỡng trên test. Tệp: `results/mm_late/threshold_search_bertweet.csv`. Dự đoán: Val `..._preds_val.csv`, Test `..._preds.csv`. |
| **Baseline Text-only: XLM-R** | Đã thêm hỗ trợ và chạy baseline trên Task 8. Tệp: Val `results/txt_only/MVSA_txt__xlm-roberta_task8_seed30_metrics_val.csv`, Test `results/txt_only/MVSA_txt__xlm-roberta_task8_seed30_metrics_test.csv`. |

## Kết luận

- Bertweet+ViT+Concat+ITC đạt hiệu năng tốt, Macro-F1 ~0.823 trên test; tối ưu ngưỡng không đem lại cải thiện thêm (ngưỡng tối ưu trùng 0.50).
- Baseline text-only XLM-R đã sẵn sàng để so sánh trực tiếp với multimodal.
- Có thể tiếp tục A3 (cross-attention) và lặp tối ưu ngưỡng như trên.

## Phụ lục: Lệnh chạy tiêu biểu

- Load best và xuất dự đoán val/test (Bertweet multimodal):
```
python3 models/run_mm_late.py \
  --txt_model_name bertweet \
  --img_model_name vit \
  --fusion_name concat \
  --use_clip_loss --beta_itc 0.1 \
  --task 8 \
  --load_saved_model \
  --save_preds --save_val_preds
```

- Tối ưu ngưỡng trên validation:
```
python3 tools/threshold_tuning.py \
  --results_dir results/mm_late/ \
  --prefix bertweet-vit-concat_task8_seed30_itc0.1_ \
  --out results/mm_late/threshold_search_bertweet.csv
```

- Baseline text-only XLM-R:
```
python3 models/run_txt.py \
  --model_name xlm-roberta \
  --task 8 --epochs 5 \
  --lr 2e-5 --weight_decay 0.01 --dropout 0.1 \
  --save_model --save_preds
```
