# ViClickbait-2025: Bộ Dữ Liệu Phân Loại Clickbait Tiếng Việt

## Giới thiệu

ViClickbait-2025 là một bộ dữ liệu toàn diện để phân loại tiêu đề clickbait trong các bài báo tiếng Việt. Dataset này được thu thập từ các nguồn tin tức uy tín và được gán nhãn thủ công để phục vụ cho việc nghiên cứu và phát triển các mô hình học máy phát hiện clickbait.

## Thông tin Dataset

- **Tổng số mẫu**: 3,408 bài báo
- **Ngôn ngữ**: Tiếng Việt
- **Định dạng**: CSV và JSONL
- **Thời gian thu thập**: 2025
- **Nguồn chính**: VnExpress

## Cấu trúc Dữ liệu

### Các trường dữ liệu

| Trường | Mô tả | Kiểu dữ liệu |
|--------|-------|-------------|
| `id` | Mã định danh duy nhất của bài báo | String |
| `url` | Đường dẫn gốc của bài báo | String |
| `title` | Tiêu đề bài báo | String |
| `lead_paragraph` | Đoạn mở đầu/tóm tắt của bài báo | String |
| `category` | Chuyên mục của bài báo | String |
| `publish_datetime` | Thời gian xuất bản | DateTime (ISO 8601) |
| `source` | Nguồn tin tức | String |
| `thumbnail_url` | Đường dẫn đến hình ảnh minh họa | String |
| `label` | Nhãn phân loại (clickbait/non-clickbait) | String |

### Ví dụ dữ liệu

```json
{
  "id": "article_0001",
  "url": "https://vnexpress.net/san-bay-vinh-dong-cua-6-thang-de-nang-cap-4905048.html",
  "title": "Sân bay Vinh đóng cửa 6 tháng để nâng cấp",
  "lead_paragraph": "Nghệ An Cảng hàng không Vinh sẽ ngừng hoạt động từ 1/7 đến hết 31/12/2025 để phục vụ thi công mở rộng, cải tạo nhà ga, sân đỗ và sửa chữa đường cất hạ cánh.",
  "category": "Tin tức tổng hợp",
  "publish_datetime": "2025-06-23T12:26:00+07:00",
  "source": "VnExpress",
  "thumbnail_url": "data/images/article_0001_image.png",
  "label": "non-clickbait"
}
```

## Phân bố Dữ liệu

### Theo nhãn phân loại
- **Non-clickbait**: Các tiêu đề tin tức thông thường, mang tính thông tin
- **Clickbait**: Các tiêu đề được thiết kế để thu hút click, thường mơ hồ hoặc phóng đại

### Theo chuyên mục
Dataset bao gồm các chuyên mục đa dạng:
- Tin tức tổng hợp
- Thể thao
- Kinh doanh
- Công nghệ
- Giải trí
- Và nhiều chuyên mục khác

## Cách sử dụng

### Đọc dữ liệu CSV
```python
import pandas as pd

# Đọc file CSV
df = pd.read_csv('clickbait_dataset_vietnamese.csv')
print(f"Tổng số mẫu: {len(df)}")
print(f"Phân bố nhãn:\n{df['label'].value_counts()}")
```

### Đọc dữ liệu JSONL
```python
import json

# Đọc file JSONL
data = []
with open('clickbait_dataset_vietnamese.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Tổng số mẫu: {len(data)}")
```

## Hình ảnh minh họa

Dataset bao gồm hình ảnh minh họa cho mỗi bài báo, được lưu trữ trong thư mục `images/`. Tên file ảnh có định dạng: `article_{id}_image.png`

## Ứng dụng

Dataset này có thể được sử dụng cho:

1. **Phân loại văn bản**: Xây dựng mô hình phân loại tiêu đề clickbait/non-clickbait
2. **Xử lý ngôn ngữ tự nhiên**: Nghiên cứu đặc điểm ngôn ngữ của tiêu đề clickbait tiếng Việt
3. **Phân tích truyền thông**: Nghiên cứu xu hướng sử dụng clickbait trong báo chí Việt Nam
4. **Học máy**: Huấn luyện các mô hình transformer, LSTM, hay các thuật toán ML khác
5. **Nghiên cứu học thuật**: Cơ sở dữ liệu cho các nghiên cứu về truyền thông và ngôn ngữ học

## Yêu cầu kỹ thuật

- **Python**: >= 3.7
- **Pandas**: Để xử lý dữ liệu CSV
- **JSON**: Để xử lý dữ liệu JSONL
- **PIL/Pillow**: Để xử lý hình ảnh (nếu cần)

## Lưu ý quan trọng

- Dữ liệu được thu thập từ các nguồn công khai và tuân thủ các quy định về bản quyền
- Việc sử dụng dataset này cần tuân thủ các điều khoản sử dụng của các nguồn tin gốc
- Dataset được cung cấp cho mục đích nghiên cứu và giáo dục

## Định dạng file

- **CSV**: `clickbait_dataset_vietnamese.csv` - Định dạng bảng, dễ đọc với Excel/Pandas
- **JSONL**: `clickbait_dataset_vietnamese.jsonl` - Định dạng JSON Lines, mỗi dòng là một object JSON
- **Images**: Thư mục `images/` chứa hình ảnh minh họa theo định dạng PNG

## Liên hệ

Nếu có thắc mắc về dataset hoặc cần hỗ trợ sử dụng, vui lòng liên hệ qua các kênh chính thức của dự án.

---

*Dataset ViClickbait-2025 được phát triển nhằm hỗ trợ cộng đồng nghiên cứu trong lĩnh vực xử lý ngôn ngữ tự nhiên và phân tích truyền thông tại Việt Nam.* 