# Báo cáo chi tiết kết quả thử nghiệm mô hình đa phương thức (Tiếng Anh)

## Giới thiệu

Tài liệu này cung cấp một bản tóm tắt chi tiết về các phương pháp và kết quả của ba thử nghiệm được thực hiện trên bộ dữ liệu WebisClickbaitCorpus2017 (tiếng Anh). Mục tiêu là để đánh giá hiệu suất của mô hình đa phương thức sử dụng bộ mã hóa văn bản chuyên biệt cho mạng xã hội (`bertweet`) với các hàm loss phụ trợ khác nhau.

## Phương pháp

Các thử nghiệm đã sử dụng các thành phần sau:

*   **Bộ mã hóa văn bản:**
    *   `vinai/bertweet-base`: Một mô hình Transformer được đào tạo trước trên dữ liệu Twitter tiếng Anh.
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
| `bertweet` + `ViT` + `ITC` + `ITM` | 1 | 0.8195 | 0.8664 | 0.8123 | 0.8277 | 0.6657 |
| `bertweet` + `ViT` + `ITC` | 1 | 0.8210 | 0.8674 | 0.8136 | 0.8295 | 0.6341 |
| `bertweet` + `ViT` + `ITM` | 1 | 0.8187 | 0.8652 | 0.8093 | 0.8302 | 0.4243 |

## Kết luận

Các thử nghiệm trên bộ dữ liệu WebisClickbaitCorpus2017 cho thấy rằng cấu hình `bertweet` + `ViT` đạt được hiệu suất rất tốt. Tương tự như kết quả trên bộ dữ liệu tiếng Việt, mô hình chỉ sử dụng loss ITC đạt được điểm F1-macro cao nhất, cho thấy đây là một chiến lược hiệu quả để học các biểu diễn đa phương thức cho tác vụ phân loại clickbait.

Việc kết hợp cả hai loss phụ trợ (ITC và ITM) không mang lại sự cải thiện đáng kể so với việc chỉ sử dụng một trong hai loss.
