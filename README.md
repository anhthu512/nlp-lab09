# LAB 09 — Named Entity Recognition and Information Extraction


## Môi trường

Các thư viện cần có trong kernel Jupyter gồm `torch`, `transformers`, `huggingface_hub`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `ipython` và `ipykernel`.

## Cách chạy

1. Mở thư mục `lab09` trong Jupyter và chọn kernel đã cài các thư viện trên.
2. Chạy toàn bộ [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb) để xem căn chỉnh nhãn và thống kê tokenization.
3. Chạy toàn bộ [ner_finetuning.ipynb](ner_finetuning.ipynb) để fine-tune, đánh giá, tạo bảng lỗi, xuất CSV, biểu đồ và checkpoint.
4. Lưu notebook sau khi chạy để giữ output trong tệp.

Baseline và Information Extraction có thể chạy từ terminal tại thư mục `lab09`.

```powershell
py -3.13 baseline_ner.py
py -3.13 information_extraction.py
py -3.13 information_extraction.py --evaluate
py -3.13 information_extraction.py --text "Barack Obama spoke at Harvard University in Boston."
```

`information_extraction.py` đọc checkpoint tại `outputs/ner_checkpoint/`. Thư mục này đã chứa model và tokenizer được lưu bởi ô đầu tiên của `ner_finetuning.ipynb`. Có thể chọn checkpoint khác bằng tham số `--checkpoint`.

## Tệp và output

| Tệp | Nội dung và output |
| --- | --- |
| [baseline_ner.py](baseline_ner.py) | Token features, DictVectorizer và SGDClassifier với logistic loss. Khi chạy, in số câu train/test, Precision, Recall, F1 và bảng nhãn dự đoán của một câu mẫu. |
| [tokenizer_alignment.ipynb](tokenizer_alignment.ipynb) | Hai ô code cho mục 16 và 17. Hiển thị token, input ID, attention mask, word ID, nhãn đã căn chỉnh và thống kê 20 câu, cùng bảng so sánh câu ngắn/dài và entity/non-entity. |
| [ner_finetuning.ipynb](ner_finetuning.ipynb) | Sáu ô code cho mục 16, 18, 21, 22, 23 và 25–27. Lưu log train, bảng alignment/logits/predictions, bảng so sánh mô hình, metric theo loại và độ dài entity, hai biểu đồ, bảng 15 ví dụ lỗi, ví dụ JSON và đánh giá IE trên 30 câu. |
| [results.csv](results.csv) | Hai hàng Baseline và Pretrained Transformer, với các cột Model, Precision, Recall, F1. |
| [entity_metrics.csv](entity_metrics.csv) | Bốn hàng PER, ORG, LOC, MISC, với các cột Entity, Precision, Recall, F1 của Transformer. |
| [entity_length_analysis.png](entity_length_analysis.png) | Biểu đồ entity-level F1 cho các nhóm độ dài 1, 2, 3 và 4+ từ theo annotation CoNLL-2003. |
| [confusion_matrix.png](confusion_matrix.png) | Ma trận nhầm lẫn của Transformer cho 9 nhãn BIO ở cấp từ. Số trong ô là số lượng dự đoán, màu biểu diễn tỷ lệ trong từng nhãn gold. |
| [information_extraction.py](information_extraction.py) | Trích xuất raw text thành JSON với PERSON, ORGANIZATION, LOCATION. Tham số --evaluate in thêm Precision, Recall, F1 trên 30 câu test. |
| `outputs/ner_checkpoint/` | Trọng số BERT-Tiny đã fine-tune, cấu hình nhãn và tokenizer phục vụ suy luận độc lập. |

## Thiết lập thực nghiệm

- Dữ liệu lấy từ `eriktks/conll2003`, revision `e8237f8875e4ecf7f60b07ae49691b85cf893410`, dưới dạng Parquet. Baseline và Transformer cùng dùng 14.041 câu train và 3.453 câu test chính thức.
- Transformer dùng `google/bert_uncased_L-2_H-128_A-2`, fine-tune toàn bộ tham số trong 5 epoch, batch size 32, learning rate 0,0002 và seed 42. Số epoch được đặt cố định.
- Alignment dùng Strategy A. Chỉ subword đầu tiên nhận nhãn của từ, các subword tiếp theo, special token và padding nhận `-100` trong loss.
- Các bảng Precision, Recall, F1 của NER chấm khớp chính xác cả ranh giới và loại entity. Bảng so sánh dùng micro average. Tỷ lệ metric nằm trong khoảng 0–1.
- Phân tích tokenization chọn ngẫu nhiên 10 câu có tối đa 10 từ và 10 câu có ít nhất 20 từ bằng seed 42. Số token loại special token. Subword splits là tổng số subword vượt quá một token trên mỗi từ. R bằng tổng token chia tổng từ trong nhóm.
- Ô mục 23 chọn 15 câu khác nhau theo thứ tự tập test, gồm 5 spurious false positives, 5 missing false negatives và 5 lỗi ranh giới có cùng loại entity. Bảng có các cột Text, Gold, Prediction, Error type, Explanation. Dấu `—` biểu diễn phía không có entity tương ứng. Explanation mô tả khác biệt quan sát được giữa gold và prediction.
- Ứng dụng IE dùng phương án Academic với PERSON, ORGANIZATION, LOCATION. Đánh giá chọn 30 câu từ test bằng seed 42, ghép các từ thành raw text và đối chiếu với gold có sẵn của CoNLL-2003 theo vị trí ký tự và loại entity.

## Kết quả đã lưu

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Baseline | 0.6628 | 0.7203 | 0.6903 |
| Pretrained Transformer | 0.7803 | 0.8107 | 0.7952 |

Các giá trị chi tiết nằm trong `results.csv` và `entity_metrics.csv`. Các notebook chứa output của lần chạy tương ứng. Khi chạy lại notebook fine-tuning, CSV, biểu đồ và checkpoint được cập nhật theo lần chạy mới.

## AI-assisted code

- What was generated — Hai script Python, hai notebook, mã tính metric, xuất bảng/biểu đồ, chọn ví dụ lỗi, pipeline IE và hướng dẫn chạy này được hỗ trợ bởi AI.
- What was verified — Hai script đã chạy độc lập. Hai notebook đã chạy từ đầu đến cuối và lưu output. CSV đã được đối chiếu với bảng trong notebook. Hai PNG đã được đối chiếu với hình lưu trong notebook. Bảng lỗi đã được kiểm tra đủ 5 ví dụ cho mỗi nhóm.

