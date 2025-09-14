# preprocess.py
import os, re, json, random
from typing import List, Dict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)
random.seed(42)

CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
TARGET_NUM_SAMPLES = 1000  # số mẫu cần lấy

def clean_text(abstract: str) -> str:
    abstract = abstract.strip().replace("\n", " ")
    abstract = re.sub(r"[^\w\s]", "", abstract)   # bỏ ký tự đặc biệt
    abstract = re.sub(r"\d+", "", abstract)       # bỏ chữ số
    abstract = re.sub(r"\s+", " ", abstract).strip()
    abstract = abstract.lower()
    return abstract

def iter_stream_samples():
    return load_dataset("UniverseTBD/arxiv-abstracts-large",
                        split="train", streaming=True)

def take_filtered_samples() -> List[Dict]:
    picked = []
    pbar = tqdm(iter_stream_samples(), desc="Scanning arXiv stream…", unit="rec")
    for s in pbar:
        cats = s.get("categories", "")
        if not cats or len(cats.split(" ")) != 1:
            continue
        primary = cats.strip().split(".")[0]
        if primary not in CATEGORIES_TO_SELECT:
            continue
        abstract = s.get("abstract", "")
        if not abstract.strip():
            continue
        picked.append({"text": clean_text(abstract), "label": primary})
        if len(picked) % 50 == 0:
            pbar.set_postfix(found=len(picked))
        if len(picked) >= TARGET_NUM_SAMPLES:
            break
    return picked

def main():
    print(">>> Lấy 1000 mẫu theo 5 primary categories:", CATEGORIES_TO_SELECT)
    samples = take_filtered_samples()
    print(f">>> Đã lấy được {len(samples)} mẫu.")

    sorted_labels = sorted({s["label"] for s in samples}, key=str.lower)
    label_to_id = {label: i for i, label in enumerate(sorted_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    print(">>> Label -> ID:", label_to_id)

    X_full = [s["text"] for s in samples]
    y_full = [label_to_id[s["label"]] for s in samples]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "label_to_id.json"), "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CACHE_DIR, "id_to_label.json"), "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, ensure_ascii=False, indent=2)

    def dump_jsonl(path, texts, labels):
        with open(path, "w", encoding="utf-8") as f:
            for t, lab in zip(texts, labels):
                f.write(json.dumps({"text": t, "label_id": lab}) + "\n")

    dump_jsonl(os.path.join(CACHE_DIR, "train.jsonl"), X_train, y_train)
    dump_jsonl(os.path.join(CACHE_DIR, "test.jsonl"),  X_test,  y_test)

    print(f">>> Saved: {CACHE_DIR}/train.jsonl, {CACHE_DIR}/test.jsonl")
    print(f">>> Saved: {CACHE_DIR}/label_to_id.json, {CACHE_DIR}/id_to_label.json")
    print(">>> Preprocessing DONE.")

if __name__ == "__main__":
    main()
