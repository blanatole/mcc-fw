# Báo cáo chi tiết kết quả thử nghiệm mô hình đa phương thức

## Giới thiệu

Tài liệu này cung cấp một bản tóm tắt chi tiết về các phương pháp và kết quả của năm thử nghiệm được thực hiện trên bộ dữ liệu ViClickbait. Mục tiêu là để đánh giá và so sánh hiệu suất của các mô hình đa phương thức khác nhau, kết hợp các bộ mã hóa văn bản và hình ảnh khác nhau với các hàm loss phụ trợ.

## Phương pháp

Các thử nghiệm đã sử dụng các thành phần sau:

*   **Bộ mã hóa văn bản:**
    *   `vinai/phobert-base`: Một mô hình Transformer được đào tạo trước cho tiếng Việt.
    *   `vinai/phobert-large`: Một phiên bản lớn hơn của PhoBERT với nhiều tham số hơn.
*   **Bộ mã hóa hình ảnh:**
    *   `google/vit-base-patch16-224-in2k`: Một mô hình Vision Transformer (ViT) được đào tạo trước trên ImageNet.
*   **Phương pháp kết hợp:**
    *   `concat`: Nối các đặc trưng từ văn bản và hình ảnh.
*   **Hàm loss phụ trợ:**
    *   **ITC (Image-Text Contrastive):** Một hàm loss tương phản nhằm mục đích kéo các cặp văn bản và hình ảnh phù hợp lại gần nhau và đẩy các cặp không phù hợp ra xa nhau trong không gian đặc trưng.
    *   **ITM (Image-Text Matching):** Một nhiệm vụ phân loại nhị phân để dự đoán xem một cặp văn bản và hình ảnh có khớp nhau hay không.

## Kết quả chi tiết

Bảng dưới đây trình bày chi tiết các chỉ số hiệu suất cho từng thử nghiệm, được lấy từ epoch có điểm F1-macro cao nhất trên tập dữ liệu thử nghiệm.

| Cấu hình mô hình | Epoch tốt nhất | F1-macro | F1-weighted | Precision-macro | Recall-macro | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PhoBERT-base` + `ViT` + `ITC` + `ITM` | 6 | 0.8360 | 0.8575 | 0.8309 | 0.8420 | 0.6540 |
| `PhoBERT-base` + `ViT` + `ITC` | 4 | 0.8427 | 0.8633 | 0.8375 | 0.8488 | 0.5467 |
| `PhoBERT-base` + `ViT` + `ITM` | 4 | 0.8231 | 0.8504 | 0.8387 | 0.8120 | 0.5417 |
| `PhoBERT-large` + `ViT` + `ITC` + `ITM` | 3 | 0.8382 | 0.8584 | 0.8301 | 0.8496 | 0.5736 |
| `PhoBERT-large` + `ViT` + `ITC` | 4 | 0.8444 | 0.8683 | 0.8609 | 0.8324 | 0.5908 |

## Kết luận

Các thử nghiệm cho thấy rằng tất cả các cấu hình đều đạt được kết quả tốt trên bộ dữ liệu ViClickbait. Mô hình sử dụng `PhoBERT-large` với chỉ loss ITC đạt được điểm F1-macro cao nhất, cho thấy rằng việc chỉ sử dụng loss tương phản có thể hiệu quả hơn và mô hình lớn hơn mang lại một chút cải thiện.

Việc kết hợp cả hai loss phụ trợ (ITC và ITM) không nhất thiết dẫn đến hiệu suất tốt hơn so với việc chỉ sử dụng ITC. Điều này cho thấy rằng loss tương phản có thể đã đủ mạnh để học các biểu diễn đa phương thức hiệu quả cho nhiệm vụ này.
