# Báo cáo chi tiết kết quả thử nghiệm mô hình đa phương thức (WebisClickbaitCorpus2017) — 08/09/2025

## 1) Mục tiêu
Tổng kết các thí nghiệm trong ngày với dữ liệu tiếng Anh WebisClickbaitCorpus2017, so sánh các cấu hình mô hình và chọn cấu hình tốt nhất theo F1-macro (test).

## 2) Dữ liệu và tiền xử lý
- Bộ dữ liệu: WebisClickbaitCorpus2017 (validation + test hợp nhất, chia lại 80/10/10).
- File key: `data/data_key_webis2017.csv` với các cột `tweet_id, text, image, label, split`.
- Ảnh được resolve đường dẫn nội bộ, có fallback khi thiếu ảnh.

## 3) Cấu hình và phương pháp
- Mã chạy: `models/run_mm_late.py` (late fusion), `fusion_name ∈ {concat, attention}`.
- Ảnh: ViT `google/vit-base-patch16-224-in21k`.
- Loss phụ trợ: ITC (Image-Text Contrastive) — bật `--use_clip_loss --beta_itc 0.1`.
- Siêu tham số mặc định: `lr=1e-5`, `weight_decay=2.5e-4`, `dropout=0.05`.
- Batch size: điều chỉnh phù hợp VRAM (task 8 mặc định 8 khi dùng backbone lớn).

## 4) Các thí nghiệm trong ngày
1. RoBERTa + ViT + Attention + ITC
   - Lệnh: `--txt_model_name roberta --img_model_name vit --fusion_name attention --use_clip_loss --beta_itc 0.1 --task 8`
   - Kết quả test (tốt nhất theo epoch):
     - F1-macro: 0.8151 (epoch-1)
     - F1-weighted: 0.8590 (epoch-1)
     - File: `results/mm_late/roberta-vit-attention_task8_seed30_itc0.1_metrics_test.csv`

2. DeBERTa-v3-large + ViT + Concat + ITC
   - Điều chỉnh: hỗ trợ `deberta-v3-large` (tự encode, pool thủ công; ITC dựa trên cosine-sim giữa đặc trưng text/img đã chiếu về `fixed_feat_size`).
   - Lệnh: `--txt_model_name deberta-v3-large --img_model_name vit --fusion_name concat --use_clip_loss --beta_itc 0.1 --task 8`
   - Kết quả test (tốt nhất theo epoch):
     - F1-macro: 0.8051 (epoch-2)
     - F1-weighted: 0.8516 (epoch-2)
     - File: `results/mm_late/deberta-v3-large-vit-concat_task8_seed30_itc0.1_metrics_test.csv`

3. DeBERTa-v3-large + ViT + Attention + ITC (testing quick)
   - Mục đích: xác nhận pipeline sau chỉnh sửa attention và ITC cho DeBERTa.
   - Kết quả test (mẫu nhỏ): F1-macro 0.4074 (không phản ánh hiệu năng thật sự; chỉ để kiểm thử luồng).
   - File: `results/mm_late/testing/deberta-v3-large-vit-attention_task8_seed30_itc0.1_metrics_test.csv`

## 5) Sự cố và khắc phục
- VisionTextDualEncoder không tương thích pooler_output với DeBERTa-v3 → chuyển sang AutoModel (text) + ViTModel (vision), pooling thủ công, ITC bằng cosine-sim sau chiếu về `fixed_feat_size`.
- Lỗi shape mismatch ở attention (xatt/attention) → bỏ reshape cưỡng bức, dùng trực tiếp context từ scaled dot-product attention, ghép CLS-text với context-CLS.
- OOM với backbone lớn → giảm batch size mặc định task 8 xuống 8; gợi ý `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Loại bỏ các package hệ thống/GUI khỏi requirements để tránh lỗi cài đặt (dbus-python, PyGObject, python-apt, iotop,...).

## 6) Kết quả tổng hợp
- Tốt nhất hôm nay (test F1-macro):
  - RoBERTa + ViT + Attention + ITC: **0.8151**
- DeBERTa-v3-large + ViT + Concat + ITC đạt: **0.8051** (test), nặng VRAM hơn.

## 7) Đề xuất bước tiếp theo
- Train đầy đủ DeBERTa-v3-large + Attention + ITC (8–10 epochs) khi tài nguyên VRAM ổn định để so sánh trực tiếp với RoBERTa.
- Thử `beta_itc ∈ {0.05, 0.2}` và tăng max_length nếu VRAM cho phép.
- Theo dõi thêm precision/recall trade-off để cân bằng theo yêu cầu ứng dụng.
