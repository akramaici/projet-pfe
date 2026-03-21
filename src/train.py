# ══════════════════════════════════════════════════════════════
# src/train.py — Entraînement 2 phases
# ══════════════════════════════════════════════════════════════

import sys
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.dataset import YOLODataset, yolo_collate_fn
from src.model   import CustomYOLO, YOLOLossV2


def train_one_phase(model, train_loader, val_loader,
                    epochs, lr, device, img_size, save_path, patience):

    print("\n" + "="*60);  print("  DEBUT ENTRAINEMENT");  print("="*60)

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = YOLOLossV2(num_classes=model.num_classes, img_size=img_size)
    history   = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf');  no_improve = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}\n" + "-"*60)

        # Train
        model.train();  train_loss = 0
        for images, targets in tqdm(train_loader, desc="Train"):
            images = images.to(device);  targets = [t.to(device) for t in targets]
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        # Val
        model.eval();  val_loss = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val"):
                images = images.to(device);  targets = [t.to(device) for t in targets]
                loss   = criterion(model(images), targets)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)
        scheduler.step()

        print(f"\nTrain Loss : {avg_train:.4f} | Val Loss : {avg_val:.4f} | LR : {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_loss:
            best_loss = avg_val;  no_improve = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("✅ BEST MODEL SAVED")
        else:
            no_improve += 1
            print(f"No improve {no_improve}/{patience}")
            if no_improve >= patience:
                print("🛑 EARLY STOPPING");  break

    print(f"\n{'='*60}\nENTRAINEMENT TERMINE — Best Val Loss : {best_loss:.4f}\n{'='*60}")
    return model, history


def plot_history(history, save_path):
    epochs   = range(1, len(history['train_loss']) + 1)
    best_ep  = int(min(range(len(history['val_loss'])), key=lambda i: history['val_loss'][i])) + 1
    best_val = min(history['val_loss'])
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'],   label='Val Loss',   linewidth=2)
    plt.axvline(best_ep, linestyle='--', label=f'Best epoch {best_ep} ({best_val:.4f})')
    plt.xlabel('Epoch');  plt.ylabel('Loss')
    plt.title("Courbes d'entraînement YOLO");  plt.legend();  plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150);  plt.close()
    print(f"📊 Courbes sauvegardées : {save_path}")


def run_training(dataset_path, checkpoint=None):
    dataset_path = Path(dataset_path)
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")
    if device == 'cuda':
        import torch as t;  print(f"GPU    : {t.cuda.get_device_name(0)}")

    with open(dataset_path / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)
    num_classes = data_cfg['nc']
    print(f"Classes ({num_classes}) : {data_cfg['names']}")

    train_dataset = YOLODataset(dataset_path/"train"/"images", dataset_path/"train"/"labels",
                                img_size=IMG_SIZE, augment=True)
    val_dataset   = YOLODataset(dataset_path/"valid"/"images", dataset_path/"valid"/"labels",
                                img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device=='cuda'),
                              collate_fn=yolo_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device=='cuda'),
                              collate_fn=yolo_collate_fn)

    model = CustomYOLO(num_classes=num_classes, base_channels=BASE_CHANNELS, pretrained=True)
    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"✅ Checkpoint chargé : {checkpoint}")
    model.summary()

    # Phase 1
    print("\n" + "="*60);  print("  PHASE 1 : Backbone gelé");  print("="*60)
    model.freeze_backbone()
    model, h1 = train_one_phase(model, train_loader, val_loader,
                                PHASE1_EPOCHS, PHASE1_LR, device, IMG_SIZE,
                                "checkpoints/best_phase1.pth", PHASE1_PATIENCE)

    # Phase 2
    print("\n" + "="*60);  print("  PHASE 2 : Fine-tuning complet");  print("="*60)
    model.unfreeze_backbone()
    model, h2 = train_one_phase(model, train_loader, val_loader,
                                PHASE2_EPOCHS, PHASE2_LR, device, IMG_SIZE,
                                "checkpoints/best_phase2.pth", PHASE2_PATIENCE)

    history = {'train_loss': h1['train_loss']+h2['train_loss'],
               'val_loss'  : h1['val_loss']  +h2['val_loss']}

    print(f"\nTotal : {len(history['train_loss'])} epochs (Phase1: {len(h1['train_loss'])} | Phase2: {len(h2['train_loss'])})")
    plot_history(history, "outputs/training_curves.png")
    return model, history, val_dataset, val_loader, data_cfg['names'], num_classes, device
