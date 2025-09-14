# train_models.py
import os, json
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CACHE_DIR = "./cache"
PLOT_DIR = os.path.join(CACHE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------- Utils ----------
def load_embeddings(cache_dir: str = CACHE_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Đọc embeddings & nhãn, trả về (X_train, y_train, X_test, y_test, sorted_labels)."""
    X_train = np.load(os.path.join(cache_dir, "X_train_emb.npy"))
    X_test  = np.load(os.path.join(cache_dir, "X_test_emb.npy"))
    y_train = np.load(os.path.join(cache_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(cache_dir, "y_test.npy"))

    # Lấy danh sách tên nhãn theo id tăng dần
    id_to_label_path = os.path.join(cache_dir, "id_to_label.json")
    if os.path.exists(id_to_label_path):
        with open(id_to_label_path, "r", encoding="utf-8") as f:
            id_to_label: Dict[str, str] = json.load(f)
        # id trong file là string → chuyển về int để sort theo id
        sorted_labels = [id_to_label[str(i)] for i in range(len(id_to_label))]
    else:
        # fallback nếu thiếu file: suy từ y_train/y_test
        ids = sorted(set(np.concatenate([y_train, y_test]).tolist()))
        sorted_labels = [f"class_{i}" for i in ids]

    return X_train, y_train, X_test, y_test, sorted_labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[str],
    figure_name: str = "Confusion Matrix",
    save_path: str | None = None,
):
    """
    Vẽ confusion matrix có cả số đếm & tỉ lệ (chuẩn hoá theo hàng).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Chú thích mỗi ô: "raw\n(perc%)"
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            raw = cm[i, j]
            perc = cm_norm[i, j]
            annotations[i, j] = f"{raw}\n({perc:.2%})"

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=label_list,
        yticklabels=label_list,
        cbar=False,
        linewidths=1,
        linecolor="black",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(figure_name)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"[Saved] {save_path}")
    plt.close()


# ---------- Models ----------
def train_and_test_kmeans(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    target_names: List[str],
):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)

    # Gán cụm cho train để tạo mapping cụm→nhãn theo majority
    cluster_ids = kmeans.fit_predict(X_train)

    cluster_to_label: Dict[int, int] = {}
    for cid in set(cluster_ids):
        labels_in_cluster = [int(y_train[i]) for i in range(len(y_train)) if cluster_ids[i] == cid]
        if len(labels_in_cluster) == 0:
            # Cụm rỗng (hiếm) → gán nhãn phổ biến toàn cục
            most_common_label = Counter(y_train).most_common(1)[0][0]
        else:
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_label[cid] = int(most_common_label)

    # Dự đoán cụm cho test → suy nhãn
    test_cluster_ids = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_label[cid] for cid in test_cluster_ids], dtype=int)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return y_pred, acc, report


def train_and_test_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
    target_names: List[str] | None = None,
):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return y_pred, acc, report


def train_and_test_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str] | None = None,
):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return y_pred, acc, report


def train_and_test_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str] | None = None,
):
    nb = GaussianNB()
    # Embeddings là dense → dùng trực tiếp
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    return y_pred, acc, report


# ---------- Main ----------
def main():
    from tabulate import tabulate

    X_train, y_train, X_test, y_test, sorted_labels = load_embeddings(CACHE_DIR)
    n_classes = len(sorted_labels)
    print(f"[Data] X_train: {X_train.shape}, X_test: {X_test.shape}, classes: {n_classes} → {sorted_labels}")

    results = []

    # 1) K-Means
    km_labels, km_acc, km_report = train_and_test_kmeans(
        X_train, y_train, X_test, y_test, n_clusters=n_classes, target_names=sorted_labels
    )
    plot_confusion_matrix(
        y_test, km_labels, sorted_labels, "KMeans Confusion Matrix (Embeddings)",
        save_path=os.path.join(PLOT_DIR, "cm_kmeans.png"),
    )
    results.append(("KMeans", km_acc, km_report))

    # 2) KNN
    knn_labels, knn_acc, knn_report = train_and_test_knn(
        X_train, y_train, X_test, y_test, n_neighbors=5, target_names=sorted_labels
    )
    plot_confusion_matrix(
        y_test, knn_labels, sorted_labels, "KNN Confusion Matrix (Embeddings)",
        save_path=os.path.join(PLOT_DIR, "cm_knn.png"),
    )
    results.append(("KNN(k=5)", knn_acc, knn_report))

    # 3) Decision Tree
    dt_labels, dt_acc, dt_report = train_and_test_decision_tree(
        X_train, y_train, X_test, y_test, target_names=sorted_labels
    )
    plot_confusion_matrix(
        y_test, dt_labels, sorted_labels, "Decision Tree Confusion Matrix (Embeddings)",
        save_path=os.path.join(PLOT_DIR, "cm_decision_tree.png"),
    )
    results.append(("DecisionTree", dt_acc, dt_report))

    # 4) Naive Bayes (Gaussian)
    nb_labels, nb_acc, nb_report = train_and_test_naive_bayes(
        X_train, y_train, X_test, y_test, target_names=sorted_labels
    )
    plot_confusion_matrix(
        y_test, nb_labels, sorted_labels, "Naive Bayes Confusion Matrix (Embeddings)",
        save_path=os.path.join(PLOT_DIR, "cm_naive_bayes.png"),
    )
    results.append(("GaussianNB", nb_acc, nb_report))

    # Bảng tổng hợp
    table = [(name, f"{acc:.4f}") for name, acc, _ in results]
    print("\n=== Accuracies (Embeddings) ===")
    print(tabulate(table, headers=["Model", "Accuracy"], tablefmt="github"))

    # Lưu báo cáo chi tiết
    report_path = os.path.join(CACHE_DIR, "reports.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": sorted_labels,
                "kmeans": {"accuracy": km_acc, "report": km_report},
                "knn": {"accuracy": knn_acc, "report": knn_report},
                "decision_tree": {"accuracy": dt_acc, "report": dt_report},
                "naive_bayes": {"accuracy": nb_acc, "report": nb_report},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Saved] detailed reports → {report_path}")


if __name__ == "__main__":
    main()
