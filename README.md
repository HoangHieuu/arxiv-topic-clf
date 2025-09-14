# arxiv-topic-clf

Phân loại chủ đề paper từ abstract (arXiv) bằng các phương pháp ML truyền thống + Sentence Embeddings (E5/SBERT).

## 1. Yêu cầu
- Python 3.11+
- macOS (đã test), chạy tốt trên Linux/Windows
- (Tuỳ chọn) GPU MPS trên Mac (Apple Silicon) cho bước embedding

## 2. Cài đặt
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Cài theo từng bước
pip install -r requirements-preprocess.txt
pip install -r requirements-embed.txt
pip install -r requirements-train.txt

preprocess:
	python preprocess.py

embed:
	python vectorize.py

train:
	python train_models.py

clean:
	rm -rf cache/* hf_cache/*