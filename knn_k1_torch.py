"""
KNN (k=1) classifier for COMP 432 project.

This script:
1. Expects `train.csv` and `test.csv` in the SAME directory.
2. Loads them and converts feature columns to PyTorch tensors.
3. Computes nearest neighbors using batched GPU/CPU tensor math.
4. Outputs a deterministic `submission.csv`.

There is no randomness in this implementation.
Running this script with the same train/test files always produces the same result.
"""

import pandas as pd
import torch
import os


# ===============================
# 0. CONFIG
# ===============================

K = 1
BATCH_SIZE = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# 1. LOAD DATA
# ===============================

def load_data():
    """
    Loads train.csv and test.csv from the current directory.
    Expects the following structure:

    train.csv:
        id, feature_0 ... feature_499, label

    test.csv:
        id, feature_0 ... feature_499
    """

    # Required files
    required_train = "train.csv"
    required_test = "test.csv"

    if not os.path.exists(required_train):
        raise FileNotFoundError("train.csv not found in project directory.")
    if not os.path.exists(required_test):
        raise FileNotFoundError("test.csv not found in project directory.")

    train_df = pd.read_csv(required_train)
    test_df = pd.read_csv(required_test)

    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    if len(feature_cols) != 500:
        raise ValueError(
            f"Expected 500 feature columns, found {len(feature_cols)}."
        )

    X_train_np = train_df[feature_cols].values.astype("float32")
    y_train_np = train_df["label"].values.astype("int64")
    X_test_np = test_df[feature_cols].values.astype("float32")
    test_ids = test_df["id"].values

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.int64, device=device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

    return X_train, y_train, X_test, test_ids


# ===============================
# 2. KNN PREDICTION (BATCHED)
# ===============================

def knn_k1_predict(X_train, y_train, X_test, k=1, batch_size=500):
    """
    Efficient KNN (k=1) prediction using Euclidean distance.
    """
    N_test = X_test.shape[0]
    y_pred = torch.empty(N_test, dtype=torch.int64, device=device)

    # Precompute ||x||^2 for training samples
    train_norm = (X_train ** 2).sum(dim=1, keepdims=True)  # (N_train, 1)

    for start in range(0, N_test, batch_size):
        end = min(start + batch_size, N_test)
        X_batch = X_test[start:end]

        batch_norm = (X_batch ** 2).sum(dim=1).unsqueeze(0)

        # Distance matrix: (N_train, B)
        dists_sq = train_norm + batch_norm - 2.0 * (X_train @ X_batch.T)
        dists_sq = torch.clamp(dists_sq, min=0.0)

        # Get nearest neighbor
        nearest_idx = torch.topk(dists_sq, k=k, dim=0, largest=False).indices

        for i in range(nearest_idx.shape[1]):
            neighbor_label = y_train[nearest_idx[0, i]]
            y_pred[start + i] = neighbor_label

    return y_pred


# ===============================
# 3. MAIN PIPELINE
# ===============================

def main():
    X_train, y_train, X_test, test_ids = load_data()

    print("\nRunning KNN (k = 1)...")
    y_pred = knn_k1_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        k=K,
        batch_size=BATCH_SIZE,
    )

    submission = pd.DataFrame({"id": test_ids, "label": y_pred.cpu().numpy()})
    submission.to_csv("submission.csv", index=False)

    print("\nsubmission.csv created successfully!")


if __name__ == "__main__":
    main()
