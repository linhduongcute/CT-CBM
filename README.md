# CT-CBM

CT-CBM là mã nguồn thí nghiệm cho bài toán giải thích mô hình phân loại văn bản bằng Concept Bottleneck Model (CBM). Dự án xây dựng một pipeline để phát hiện concept từ dữ liệu văn bản, gán nhãn concept, tạo Concept Activation Vectors (CAVs), xếp hạng concept bằng TCAV/LIG/coverage, rồi huấn luyện CBM theo chiến lược joint hoặc projection với residual fitting.

Repo này phù hợp để chạy lại các thí nghiệm nghiên cứu trên nhiều bộ dữ liệu văn bản như AG News, DBpedia, Movies, Medical và một số biến thể khác.

## Tính năng chính

- Chuẩn bị dữ liệu cho nhiều dataset phân loại văn bản.
- Phát hiện và gom cụm concept từ attribution hoặc annotation.
- Hỗ trợ nhiều kiểu annotation: `C3M`, `our_annotation`, `combined_annotation`.
- Tạo CAV bằng mean pooling hoặc SVM.
- Tính điểm và xếp hạng concept bằng TCAV, LIG, tần suất và coverage heuristic.
- Huấn luyện các biến thể CBM:
  - baseline model;
  - joint CBM;
  - projection CBM;
  - joint CBM có residual layer;
  - biến thể dành cho Gemma.
- Lưu checkpoint, metric và kết quả trung gian theo từng mô hình/dataset.

## Cấu trúc thư mục

```text
CT-CBM/
├── notebooks/
│   ├── 1. annotation/
│   ├── 2. clustering of concepts columns/
│   ├── 3. Black box training/
│   ├── 4. cav creation/
│   ├── 5. lig score of concepts/
│   ├── 6. computation of combined ranking score/
│   ├── 7. new heuristique CT-CBM/
│   └── 8. Comparison and ablation study/
├── run_experiments/
│   ├── config_*.py
│   ├── data/
│   ├── models/
│   └── scripts/
└── requirements.txt
```

Trong đó:

- `notebooks/`: các notebook chạy thí nghiệm theo từng bước.
- `run_experiments/config_*.py`: cấu hình cho từng cặp dataset/model.
- `run_experiments/data/`: hàm chuẩn bị DataLoader cho từng dataset.
- `run_experiments/models/`: định nghĩa baseline, joint CBM, projection CBM và các layer phụ trợ.
- `run_experiments/scripts/`: utility cho annotation, clustering, attribution, CAV, TCAV, LIG, ranking và full pipeline.

## Cài đặt

Khuyến nghị dùng Python 3.10 hoặc 3.11 và môi trường ảo riêng.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` đang dùng PyTorch CUDA 11.8:

```text
torch==2.2.0+cu118
torchvision==0.17.0+cu118
torchaudio==2.2.0+cu118
```

Nếu máy không có GPU CUDA 11.8, cần thay phiên bản PyTorch phù hợp với môi trường local.

## Cấu hình trước khi chạy

Các file config hiện có hard-code đường dẫn cho môi trường A100 hoặc Databricks, ví dụ:

```python
DATASET_PATH = "/home/bhan/Yann_CBM/Launch/dbfs/dataset/..."
SAVE_PATH = "/home/bhan/Yann_CBM/Launch/dbfs/results_..."
```

Trước khi chạy local, hãy sửa trong file config tương ứng, ví dụ `run_experiments/config_agnews.py`:

```python
DATASET_PATH = "./data/agnews/"
SAVE_PATH = "./outputs/results_agnews/"
SAVE_PATH_CONCEPTS = f"{SAVE_PATH}concepts_discovery"
```

Một số dataset như AG News và DBpedia có thể được tải qua Hugging Face datasets. Các dataset nội bộ như Movies, Medical hoặc LEDGAR có thể yêu cầu file dữ liệu đặt đúng vị trí mà config/data loader đang trỏ tới.

Nếu dùng Gemma, cần cấu hình Hugging Face token trong file `.env` ở thư mục gốc:

```env
HF_TOKEN=your_huggingface_token_here
```

File `.env` đã được đưa vào `.gitignore` để tránh commit token lên Git.

## Luồng thí nghiệm đề xuất

Các notebook trong thư mục `notebooks/` thể hiện thứ tự chạy chính:

1. Annotation concept:
   - `notebooks/1. annotation/annotation_Ours.ipynb`
   - `notebooks/1. annotation/annotation_CB_LLM.ipynb`
   - `notebooks/1. annotation/Annotation_C3M.ipynb`
2. Clustering concept:
   - `notebooks/2. clustering of concepts columns/Clustering_of_Concepts.ipynb`
3. Huấn luyện black-box/baseline:
   - `notebooks/3. Black box training/black_box.ipynb`
4. Tạo CAV:
   - `notebooks/4. cav creation/cavs_creation_mean.ipynb`
5. Tính LIG score:
   - `notebooks/5. lig score of concepts/LIG_ranking.ipynb`
6. Tính combined ranking score:
   - `notebooks/6. computation of combined ranking score/computation_of_combined_score.ipynb`
7. Chạy heuristic CT-CBM:
   - `notebooks/7. new heuristique CT-CBM/CT_CBM_notebook.ipynb`
8. So sánh và ablation:
   - `notebooks/8. Comparison and ablation study/*.ipynb`

## Ví dụ dùng script

Các script trong `run_experiments/scripts/` chủ yếu được import từ notebook. Ví dụ khởi tạo config và DataLoader:

```python
import sys

sys.path.append("./run_experiments")
sys.path.append("./run_experiments/scripts")
sys.path.append("./run_experiments/models")
sys.path.append("./run_experiments/data")

from load_config import load_config
from prepare_data import load_fc_prepare_data

config = load_config(model_name="bert-base-uncased", dataset="agnews")
prepare_data = load_fc_prepare_data("agnews")

train_loader, test_loader, val_loader, train_df, val_df, test_df = prepare_data(config)
```

Ví dụ tạo model/tokenizer:

```python
from models.utils import load_model_and_tokenizer

embedder_model, tokenizer, cbm_layer, classifier = load_model_and_tokenizer(config)
```

## Kết quả đầu ra

Pipeline lưu kết quả vào `config.SAVE_PATH`, thường theo dạng:

```text
results_<dataset>/
├── concepts_discovery/
│   ├── train_data.pkl
│   ├── val_data.pkl
│   ├── test_data.pkl
│   ├── df_with_topics_v4.csv
│   └── ...
└── blue_checkpoints/
    └── <model_name>/
        ├── cavs/
        ├── jointCBM/
        ├── ProjectionCBM/
        └── Our_CBM_joint/
```

Các file thường gặp:

- CAVs: `cavs_mean_<annotation>.json`, `cavs_svm*.json`
- metric concept detection: `detection_concept_*.json`
- ranking concept: `sorted_macro_concepts*.json` hoặc `.pkl`
- checkpoint CBM: `.pth`
- performance theo strategy: `<model_name>_performance_<strategy>.json`

## Dataset và model được hỗ trợ

Dataset chính trong code:

- `agnews`
- `dbpedia`
- `movies`
- `medical`
- `ledgar`
- `n24`

Model chính:

- `bert-base-uncased`
- `deberta-large`
- `gemma`

Một số model khác như RoBERTa, GPT-2 hoặc LSTM có nhánh code trong utility, nhưng chưa phải luồng chính trong config hiện tại.

## Lưu ý

- Dự án thiên về notebook nghiên cứu, chưa có command-line entrypoint thống nhất.
- Một số comment và thông báo trong code đang bằng tiếng Pháp.
- Đường dẫn dữ liệu/kết quả cần được chỉnh lại nếu chạy ngoài hạ tầng ban đầu.
- Các thí nghiệm với transformer lớn hoặc Gemma cần GPU có đủ VRAM.
- Các file notebook có thể phụ thuộc vào kết quả trung gian từ bước trước, vì vậy nên chạy đúng thứ tự trong phần luồng thí nghiệm.
