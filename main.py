#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════
# main.py — Pipeline YOLO KITTI complet et automatique
#
# Usage :
#   python main.py                          # pipeline complet
#   python main.py --skip-download          # dataset déjà présent
#   python main.py --skip-train             # évaluation seulement
#   python main.py --checkpoint path.pth   # repartir d'un checkpoint
#   python main.py --image photo.jpg        # inférence image
#   python main.py --video video.mp4        # inférence vidéo
# ══════════════════════════════════════════════════════════════

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

from config import *
from src.download   import download_dataset
from src.preprocess import clean_dataset, explore_and_plot
from src.dataset    import YOLODataset, yolo_collate_fn
from src.model      import CustomYOLO
from src.train      import run_training
from src.evaluate   import run_evaluation
from src.inference  import predict_image, predict_video

from torch.utils.data import DataLoader


def main(args):
    print(f"\n{'█'*60}\n  YOLO KITTI — PIPELINE AUTOMATIQUE\n{'█'*60}")

    # ── 1. Téléchargement
    if not args.skip_download:
        print("\n[1/4] 📥 TÉLÉCHARGEMENT DU DATASET")
        download_dataset()
    else:
        print("\n[1/4] ⏭️  Téléchargement ignoré")

    # ── 2. Nettoyage
    print("\n[2/4] 🧹 NETTOYAGE DU DATASET")
    clean_dataset(RAW_DATASET, CLEAN_DATASET, min_area=MIN_AREA)
    if args.explore:
        explore_and_plot(RAW_DATASET, OUTPUTS_DIR/"exploration")

    # ── 3. Entraînement
    if not args.skip_train:
        print("\n[3/4] 🏋️  ENTRAÎNEMENT")
        model, history, val_dataset, val_loader, class_names, num_classes, device = \
            run_training(CLEAN_DATASET, checkpoint=args.checkpoint)
        best_checkpoint = "checkpoints/best_phase2.pth"
    else:
        print("\n[3/4] ⏭️  Entraînement ignoré")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(CLEAN_DATASET/"data.yaml") as f:
            cfg = yaml.safe_load(f)
        num_classes = cfg['nc'];  class_names = cfg['names']
        best_checkpoint = args.checkpoint or "checkpoints/best_phase2.pth"

        if not Path(best_checkpoint).exists():
            print(f"❌ Checkpoint introuvable : {best_checkpoint}")
            sys.exit(1)

        model = CustomYOLO(num_classes=num_classes, base_channels=BASE_CHANNELS, pretrained=False)
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        model = model.to(device);  model.eval()
        print(f"✅ Modèle chargé : {best_checkpoint}")

        val_dataset = YOLODataset(CLEAN_DATASET/"valid"/"images",
                                  CLEAN_DATASET/"valid"/"labels",
                                  img_size=IMG_SIZE, augment=False)
        val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False,
                                 num_workers=2, pin_memory=True, collate_fn=yolo_collate_fn)

    # ── 4. Évaluation complète
    print("\n[4/4] 📊 ÉVALUATION COMPLÈTE")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    map50, map5095 = run_evaluation(
        model, val_loader, val_dataset,
        class_names, num_classes, device,
        img_size=IMG_SIZE,
        save_dir=OUTPUTS_DIR
    )

    # ── Inférence optionnelle
    if args.image:
        print(f"\n🖼️  INFÉRENCE IMAGE : {args.image}")
        predict_image(model, args.image, device, class_names, num_classes,
                      save_path=str(OUTPUTS_DIR/"test_image.png"))

    if args.video:
        print(f"\n🎬 INFÉRENCE VIDÉO : {args.video}")
        predict_video(model, args.video, device, class_names, num_classes,
                      output_path=str(OUTPUTS_DIR/"output_video.mp4"))

    # ── Résumé final
    print(f"\n{'█'*60}")
    print(f"  PIPELINE TERMINÉ")
    print(f"{'█'*60}")
    print(f"  mAP@50         : {map50:.4f}")
    print(f"  mAP@50:95      : {map5095:.4f}")
    print(f"  Checkpoint     : {best_checkpoint}")
    print(f"  Résultats      : {OUTPUTS_DIR}/")
    print(f"{'█'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline YOLO KITTI — automatique")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--checkpoint",    type=str, default=None)
    parser.add_argument("--explore",       action="store_true")
    parser.add_argument("--image",         type=str, default=None)
    parser.add_argument("--video",         type=str, default=None)
    args = parser.parse_args()
    main(args)
