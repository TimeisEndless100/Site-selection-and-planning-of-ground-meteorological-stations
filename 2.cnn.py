"""
Lightweight CNN training and inference script for remote-sensing patch classification under positive-unlabeled (PU) learning.
Implements:
- PatchDataset: loads 3-channel raster patches and corresponding binary labels.
- LightCNN: three-convolutional-layer network with global pooling and embedding output.
- train(): model training using weighted sampling and BCEWithLogitsLoss.
- inference(): loads trained weights and performs patch-level probability predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import rasterio
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# ====================== Configuration ======================
PATCH_DIR = "patches--64"  # Directory containing 3-channel TIFF patches
CSV_PATH = os.path.join(PATCH_DIR, "patches_info_binary.csv")  # CSV listing patch filenames and labels
BATCH_SIZE = 64               # Number of samples per mini-batch
LR = 1e-4                     # Learning rate for Adam optimizer
EPOCHS = 50                   # Number of training epochs
SEED = 42                     # Random seed for reproducibility
EMBEDDING_DIM = 64            # Dimensionality of the embedding vector


class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading raster patches and binary labels.

    Args:
        csv_path (str): Path to CSV file with columns [filename, label].
        patch_dir (str): Directory containing the TIFF patch files.
    """
    def __init__(self, csv_path: str, patch_dir: str):
        self.data = pd.read_csv(csv_path)
        self.patch_dir = patch_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Reads a single patch and its label.

        Returns:
            patch_tensor (torch.FloatTensor): Tensor of shape (3, H, W).
            label (torch.FloatTensor): Scalar tensor (0 or 1).
        """
        row = self.data.iloc[idx]
        patch_file = os.path.join(self.patch_dir, row["filename"])
        label = torch.tensor(row["label"], dtype=torch.float32)

        with rasterio.open(patch_file) as src:
            # Read all bands as numpy float32
            arr = src.read().astype(np.float32)
            # Normalize second channel (slope) to [0,1]
            arr[1] = np.clip(arr[1], 0, 85) / 85.0
            # Normalize third channel (elevation) to [0,1]
            arr[2] = (np.clip(arr[2], 40.0, 2790.0) - 40.0) / (2790.0 - 40.0)

        patch_tensor = torch.from_numpy(arr)
        return patch_tensor, label


class LightCNN(nn.Module):
    """
    Lightweight convolutional network for patch classification.

    Architecture:
    - Conv2d(3->32) + BN + ReLU + MaxPool
    - Conv2d(32->64) + BN + ReLU + MaxPool
    - Conv2d(64->128) + BN + ReLU + MaxPool
    - GlobalAvgPool -> FC (to embedding) -> classifier
    """
    def __init__(self, num_features: int = 128):
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        # Head: global pooling + embedding + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_features)
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor):
        # Forward through convolutional blocks
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # Global average pooling and embedding
        x = self.global_pool(x).view(x.size(0), -1)
        embedding = torch.relu(self.fc(x))
        # Logits for binary classification
        logits = self.classifier(embedding).squeeze(1)
        return logits, embedding


def train():
    """
    Train the LightCNN model using weighted sampling to address class imbalance.
    """
    # Reproducibility
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data splits
    dataset = PatchDataset(CSV_PATH, PATCH_DIR)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Compute sample weights for balanced sampling
    labels = [lbl.item() for _, lbl in train_ds]
    counts = np.bincount(labels)
    weights = 1.0 / counts
    sample_weights = [weights[int(l)] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    model = LightCNN(num_features=EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for patches, lbls in train_loader:
            patches, lbls = patches.to(device), lbls.to(device)
            logits, _ = model(patches)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} - avg train loss: {avg_loss:.4f}")

    # Final evaluation on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for patches, lbls in val_loader:
            patches, lbls = patches.to(device), lbls.to(device)
            logits, _ = model(patches)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation - F1: {f1:.4f}, Acc: {acc:.4f}")

    # Save best model
    torch.save(model.state_dict(), "best_bce_model.pth")


def inference(model_path: str, input_csv: str, patch_dir: str, output_csv: str = "patches_predictions.csv", threshold: float = 0.5):
    """
    Perform inference on new patches using a trained model.

    Args:
        model_path: Path to saved model weights (.pth).
        input_csv: CSV with patch filenames to predict.
        patch_dir: Directory containing patch TIFF files.
        output_csv: Path to save predictions (filename, probability, label).
        threshold: Probability cutoff for positive label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightCNN(num_features=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        patch_file = os.path.join(patch_dir, row["filename"])
        with rasterio.open(patch_file) as src:
            arr = src.read().astype(np.float32)
            arr[1] = np.clip(arr[1], 0, 85) / 85.0
            arr[2] = (np.clip(arr[2], 40.0, 2790.0) - 40.0) / (2790.0 - 40.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            logit, _ = model(tensor)
            prob = torch.sigmoid(logit).item()
            label = int(prob >= threshold)
        results.append({"filename": row["filename"], "probability": prob, "predicted_label": label})
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}.")


if __name__ == "__main__":
    train()
    inference("best_bce_model.pth", CSV_PATH, PATCH_DIR)
