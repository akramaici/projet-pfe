# ══════════════════════════════════════════════════════════════
# src/dataset.py — YOLODataset + letterbox + collate_fn
# ══════════════════════════════════════════════════════════════

import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def letterbox(img, new_size=640):
    h, w   = img.shape[:2]
    scale  = min(new_size/w, new_size/h)
    nw, nh = int(w*scale), int(h*scale)
    img_r  = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top    = (new_size-nh)//2;  left = (new_size-nw)//2
    canvas[top:top+nh, left:left+nw] = img_r
    return canvas, scale, left, top


def simulate_night(img):
    img = img * np.random.uniform(0.1, 0.35)
    img[:,:,0] *= np.random.uniform(0.7, 0.9)
    img[:,:,1] *= np.random.uniform(0.7, 0.9)
    img[:,:,2] *= np.random.uniform(1.0, 1.2)
    return np.clip(img, 0, 1)

def simulate_rain(img):
    img  = img * np.random.uniform(0.6, 0.8)
    rain = np.zeros_like(img)
    for _ in range(np.random.randint(200, 500)):
        x  = np.random.randint(0, img.shape[1])
        y  = np.random.randint(0, img.shape[0]-20)
        ln = np.random.randint(10, 25)
        rain[y:y+ln, x, :] = np.random.uniform(0.5, 0.9)
    return np.clip(img + rain*0.3, 0, 1)

def simulate_fog(img):
    fog   = np.ones_like(img) * np.random.uniform(0.6, 0.85)
    alpha = np.random.uniform(0.3, 0.6)
    return np.clip(img*(1-alpha) + fog*alpha, 0, 1)


def mosaic_augment(image_files, labels_dir, idx, img_size=640):
    indices = [idx] + [np.random.randint(0, len(image_files)) for _ in range(3)]
    imgs, labels_list = [], []

    for i in indices:
        ip  = image_files[i]
        lp  = labels_dir / f"{ip.stem}.txt"
        img = cv2.imread(str(ip))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size//2, img_size//2))
        img = img.astype(np.float32) / 255.0
        imgs.append(img)
        lbls = []
        if lp.exists():
            for line in open(lp):
                parts = line.strip().split()
                if len(parts) == 5:
                    lbls.append(list(map(float, parts)))
        labels_list.append(np.array(lbls, dtype=np.float32) if lbls else np.zeros((0,5), dtype=np.float32))

    mosaic = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mosaic[:img_size//2, :img_size//2] = imgs[0]
    mosaic[:img_size//2, img_size//2:] = imgs[1]
    mosaic[img_size//2:, :img_size//2] = imgs[2]
    mosaic[img_size//2:, img_size//2:] = imgs[3]

    offsets = [(0,0),(0.5,0),(0,0.5),(0.5,0.5)]
    all_labels = []
    for lbl, (ox,oy) in zip(labels_list, offsets):
        if len(lbl) == 0: continue
        nl = lbl.copy()
        nl[:,1] = lbl[:,1]*0.5 + ox
        nl[:,2] = lbl[:,2]*0.5 + oy
        nl[:,3] = lbl[:,3]*0.5
        nl[:,4] = lbl[:,4]*0.5
        all_labels.append(nl)

    final = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,5), dtype=np.float32)
    return mosaic, final


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False):
        self.images_dir  = Path(images_dir)
        self.labels_dir  = Path(labels_dir)
        self.img_size    = img_size
        self.augment     = augment
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )
        print(f"📂 Dataset : {len(self.image_files)} images ({images_dir})")

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        # Mosaic 40%
        if self.augment and np.random.rand() < 0.4:
            img, labels = mosaic_augment(self.image_files, self.labels_dir, idx, self.img_size)
            img    = torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()
            labels = torch.from_numpy(labels).float()
            return img, labels

        img_path = self.image_files[idx]
        img      = cv2.imread(str(img_path))
        if img is None: raise ValueError(f"Image invalide : {img_path}")
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0   = img.shape[:2]
        img, scale, pad_x, pad_y = letterbox(img, self.img_size)
        img      = img.astype(np.float32) / 255.0

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = []
        if label_path.exists():
            for line in open(label_path):
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(list(map(float, parts)))

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0,5), dtype=np.float32)

        if len(labels) > 0:
            labels[:,1] = labels[:,1]*w0*scale/self.img_size + pad_x/self.img_size
            labels[:,2] = labels[:,2]*h0*scale/self.img_size + pad_y/self.img_size
            labels[:,3] = labels[:,3]*w0*scale/self.img_size
            labels[:,4] = labels[:,4]*h0*scale/self.img_size

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                if len(labels) > 0: labels[:,1] = 1 - labels[:,1]
            if np.random.rand() < 0.8:
                img = np.clip(img * np.random.uniform(0.6, 1.4), 0, 1)
            if np.random.rand() < 0.5:
                mean = img.mean()
                img  = np.clip((img-mean)*np.random.uniform(0.8,1.2)+mean, 0, 1)
            if np.random.rand() < 0.3:
                img = np.clip(img + np.random.normal(0,0.02,img.shape).astype(np.float32), 0, 1)
            r = np.random.rand()
            if   r < 0.15: img = simulate_night(img)
            elif r < 0.25: img = simulate_rain(img)
            elif r < 0.30: img = simulate_fog(img)

        img    = torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()
        labels = torch.from_numpy(labels).float()
        return img, labels


def yolo_collate_fn(batch):
    images, targets = [], []
    for img, labels in batch:
        if img is None: continue
        images.append(img)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        targets.append(labels.float())
    return torch.stack(images, dim=0), targets
