#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════
# main.py — Pipeline YOLO KITTI complet et automatique
#
# Usage :
#   python main.py                        # pipeline complet
#   python main.py --skip-download        # si dataset déjà là
#   python main.py --skip-train           # évaluation seulement
#   python main.py --image photo.jpg      # inférence image
#   python main.py --video video.mp4      # inférence vidéo
# ══════════════════════════════════════════════════════════════

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# ── Imports internes
from config import *
from src.download   import download_dataset
from src.preprocess import clean_dataset, explore_and_plot
from src.dataset    import YOLODataset, yolo_collate_fn
from src.model      import CustomYOLO
from src.train      import run_training
from src.evaluate   import run_evaluation, decode_predictions
from src.inference  import predict_image, predict_video

from torch.utils.data import DataLoader


def main(args):

    print("\n" + "█"*60)
    print("  YOLO KITTI — PIPELINE AUTOMATIQUE")
    print("█"*60)

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 1 — Téléchargement du dataset
    # ══════════════════════════════════════════════════════════
    if not args.skip_download:
        print("\n[1/4] 📥 TÉLÉCHARGEMENT DU DATASET")
        download_dataset()
    else:
        print("\n[1/4] ⏭️  Téléchargement ignoré (--skip-download)")

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 2 — Nettoyage du dataset
    # ══════════════════════════════════════════════════════════
    print("\n[2/4] 🧹 NETTOYAGE DU DATASET")
    clean_dataset(RAW_DATASET, CLEAN_DATASET, min_area=MIN_AREA)

    if args.explore:
        explore_and_plot(RAW_DATASET, OUTPUTS_DIR / "exploration")

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 3 — Entraînement
    # ══════════════════════════════════════════════════════════
    if not args.skip_train:
        print("\n[3/4] 🏋️  ENTRAÎNEMENT")
        model, history, val_dataset, val_loader, class_names, num_classes, device = \
            run_training(CLEAN_DATASET, checkpoint=args.checkpoint)
        best_checkpoint = "checkpoints/best_phase2.pth"
    else:
        print("\n[3/4] ⏭️  Entraînement ignoré (--skip-train)")
        # Charger le modèle existant
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(CLEAN_DATASET / "data.yaml") as f:
            data_cfg = yaml.safe_load(f)
        num_classes = data_cfg['nc']
        class_names = data_cfg['names']

        best_checkpoint = args.checkpoint or "checkpoints/best_phase2.pth"
        if not Path(best_checkpoint).exists():
            print(f"❌ Checkpoint introuvable : {best_checkpoint}")
            print("   Lance d'abord l'entraînement : python main.py")
            sys.exit(1)

        model = CustomYOLO(num_classes=num_classes, base_channels=BASE_CHANNELS, pretrained=False)
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ Modèle chargé : {best_checkpoint}")

        val_dataset = YOLODataset(CLEAN_DATASET/"valid"/"images",
                                  CLEAN_DATASET/"valid"/"labels",
                                  img_size=IMG_SIZE, augment=False)
        val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False,
                                 num_workers=2, pin_memory=True,
                                 collate_fn=yolo_collate_fn)

    # ══════════════════════════════════════════════════════════
    # ÉTAPE 4 — Évaluation
    # ══════════════════════════════════════════════════════════
    print("\n[4/4] 📊 ÉVALUATION")
    map50, aps = run_evaluation(model, val_loader, val_dataset,
                                class_names, num_classes, device, IMG_SIZE)

    # ══════════════════════════════════════════════════════════
    # INFÉRENCE (optionnel)
    # ══════════════════════════════════════════════════════════
    if args.image:
        print(f"\n🖼️  INFÉRENCE IMAGE : {args.image}")
        predict_image(model, args.image, device, class_names, num_classes,
                      save_path="outputs/test_image.png")

    if args.video:
        print(f"\n🎬 INFÉRENCE VIDÉO : {args.video}")
        predict_video(model, args.video, device, class_names, num_classes,
                      output_path="outputs/output_video.mp4")

    # ══════════════════════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ══════════════════════════════════════════════════════════
    print("\n" + "█"*60)
    print("  PIPELINE TERMINÉ")
    print("█"*60)
    print(f"  mAP50 final      : {map50:.4f}")
    print(f"  Checkpoint       : checkpoints/best_phase2.pth")
    print(f"  Courbes          : outputs/training_curves.png")
    print(f"  Visualisations   : outputs/predictions/predictions.png")
    print("█"*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline YOLO KITTI — automatique et reproductible")

    parser.add_argument("--skip-download", action="store_true",
                        help="Ignorer le téléchargement (dataset déjà présent)")
    parser.add_argument("--skip-train",    action="store_true",
                        help="Ignorer l'entraînement (utiliser un checkpoint existant)")
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Chemin vers un checkpoint existant")
    parser.add_argument("--explore",       action="store_true",
                        help="Générer les graphiques d'exploration du dataset")
    parser.add_argument("--image",         type=str, default=None,
                        help="Chemin vers une image pour l'inférence")
    parser.add_argument("--video",         type=str, default=None,
                        help="Chemin vers une vidéo pour l'inférence")

    args = parser.parse_args()
    main(args)
