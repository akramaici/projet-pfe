# ══════════════════════════════════════════════════════════════
# src/dataset.py — YOLODataset + letterbox + collate_fn
# ══════════════════════════════════════════════════════════════

import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def letterbox(img, new_size=640):
    """Resize avec padding — conserve le ratio d'aspect."""
    h, w    = img.shape[:2]
    scale   = min(new_size / w, new_size / h)
    nw, nh  = int(w * scale), int(h * scale)
    img_r   = cv2.resize(img, (nw, nh))
    canvas  = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top     = (new_size - nh) // 2
    left    = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = img_r
    return canvas, scale, left, top


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size   = img_size
        self.augment    = augment
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )
        print(f"📂 Dataset : {len(self.image_files)} images ({images_dir})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img      = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Image invalide : {img_path}")

        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img, scale, pad_x, pad_y = letterbox(img, self.img_size)
        img    = img.astype(np.float32) / 255.0

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append(list(map(float, parts)))

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * w0 * scale / self.img_size + pad_x / self.img_size
            labels[:, 2] = labels[:, 2] * h0 * scale / self.img_size + pad_y / self.img_size
            labels[:, 3] = labels[:, 3] * w0 * scale / self.img_size
            labels[:, 4] = labels[:, 4] * h0 * scale / self.img_size

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                if len(labels) > 0:
                    labels[:, 1] = 1 - labels[:, 1]
            if np.random.rand() < 0.8:
                img = np.clip(img * np.random.uniform(0.6, 1.4), 0, 1)
            if np.random.rand() < 0.5:
                mean = img.mean()
                img  = np.clip((img - mean) * np.random.uniform(0.8, 1.2) + mean, 0, 1)
            if np.random.rand() < 0.3:
                img  = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)

        img    = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        labels = torch.from_numpy(labels).float()
        return img, labels


def yolo_collate_fn(batch):
    images, targets = [], []
    for img, labels in batch:
        if img is None:
            continue
        images.append(img)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        targets.append(labels.float())
    return torch.stack(images, dim=0), targets
