# vectorize.py
import os, json
from typing import List, Literal, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CACHE_DIR = "./cache"

class EmbeddingVectorizer:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        normalize: bool = True,
        device: str | None = None,
        batch_size: int = 32,
    ):
        # Ưu tiên MPS trên Mac, rồi tới CUDA, cuối cùng là CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size
        print(f"[EmbeddingVectorizer] model={model_name}, device={device}, normalize={normalize}")

    def _format_inputs(self, texts: List[str], mode: Literal["query", "passage"]) -> List[str]:
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        # E5 yêu cầu tiền tố "query: " hoặc "passage: "
        prefix = "query: " if mode == "query" else "passage: "
        return [prefix + (t.strip() if t else "") for t in texts]

    def transform(self, texts: List[str], mode: Literal["query", "passage"] = "passage") -> np.ndarray:
        inputs = self._format_inputs(texts, mode)
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,   # trả về np.ndarray float32
        )
        return embeddings

    def transform_numpy(self, texts: List[str], mode: Literal["query", "passage"] = "passage") -> np.ndarray:
        return self.transform(texts, mode)

def _load_jsonl_texts(path: str) -> Tuple[list, np.ndarray]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label_id"])
    return texts, np.array(labels, dtype=np.int64)

def main():
    # 1) Đọc dữ liệu đã tiền xử lý
    train_path = os.path.join(CACHE_DIR, "train.jsonl")
    test_path  = os.path.join(CACHE_DIR, "test.jsonl")
    assert os.path.exists(train_path) and os.path.exists(test_path), "Chưa thấy file train/test trong ./cache/. Hãy chạy preprocess trước."

    X_train_texts, y_train = _load_jsonl_texts(train_path)
    X_test_texts,  y_test  = _load_jsonl_texts(test_path)
    print(f"Loaded texts — train: {len(X_train_texts)}, test: {len(X_test_texts)}")

    # 2) Mã hoá văn bản bằng E5 (SBERT-style)
    vectorizer = EmbeddingVectorizer()  # mặc định: e5-base + normalize=True
    # với bài toán phân loại document, dùng mode='passage'
    X_train_emb = vectorizer.transform_numpy(X_train_texts, mode="passage")
    X_test_emb  = vectorizer.transform_numpy(X_test_texts,  mode="passage")

    print(f"Shape of X_train_embeddings: {X_train_emb.shape}")
    print(f"Shape of X_test_embeddings:  {X_test_emb.shape}")

    # 3) Lưu để các bước classifier dùng tiếp
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, "X_train_emb.npy"), X_train_emb)
    np.save(os.path.join(CACHE_DIR, "X_test_emb.npy"),  X_test_emb)
    np.save(os.path.join(CACHE_DIR, "y_train.npy"),     y_train)
    np.save(os.path.join(CACHE_DIR, "y_test.npy"),      y_test)
    print(f"Saved embeddings & labels to {CACHE_DIR}/")

if __name__ == "__main__":
    main()
