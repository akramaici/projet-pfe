# ══════════════════════════════════════════════════════════════
# src/preprocess.py
# Nettoyage du dataset KITTI
# ══════════════════════════════════════════════════════════════

import os
import sys
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from config import MIN_AREA, OUTPUTS_DIR


def clean_dataset(raw_dataset: Path, clean_dataset: Path, min_area: float = MIN_AREA):
    """
    Nettoie le dataset :
    - Supprime annotations invalides (mauvais format, coords hors [0,1])
    - Supprime objets trop petits (aire < min_area)
    - Copie images correspondantes
    - Génère nouveau data.yaml
    """

    # Si déjà nettoyé, skip
    if clean_dataset.exists() and (clean_dataset / "data.yaml").exists():
        n = len(list((clean_dataset / "train" / "images").glob("*")))
        if n > 0:
            print(f"✅ Dataset nettoyé déjà présent : {clean_dataset} ({n} images train)")
            return clean_dataset

    print("\n" + "="*60)
    print("  NETTOYAGE DU DATASET")
    print("="*60)

    with open(raw_dataset / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    total_removed_invalid = 0
    total_removed_small   = 0
    total_kept            = 0

    for split in ["train", "valid", "test"]:
        raw_img_dir   = raw_dataset   / split / "images"
        raw_lbl_dir   = raw_dataset   / split / "labels"
        clean_img_dir = clean_dataset / split / "images"
        clean_lbl_dir = clean_dataset / split / "labels"

        if not raw_lbl_dir.exists():
            print(f"  ⚠️  {split} introuvable, ignoré.")
            continue

        clean_img_dir.mkdir(parents=True, exist_ok=True)
        clean_lbl_dir.mkdir(parents=True, exist_ok=True)

        removed_invalid = 0
        removed_small   = 0
        kept            = 0

        for label_file in raw_lbl_dir.glob("*.txt"):
            valid_lines = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        removed_invalid += 1
                        continue
                    cls, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        removed_invalid += 1
                        continue
                    if w * h < min_area:
                        removed_small += 1
                        continue
                    valid_lines.append(line)
                    kept += 1

            if valid_lines:
                with open(clean_lbl_dir / label_file.name, "w") as f:
                    f.writelines(valid_lines)
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_path = raw_img_dir / f"{label_file.stem}{ext}"
                    if img_path.exists():
                        shutil.copy(img_path, clean_img_dir / img_path.name)
                        break

        print(f"  {split:6s} → gardées: {kept:5d} | invalides: {removed_invalid:3d} | trop petites: {removed_small:4d}")
        total_removed_invalid += removed_invalid
        total_removed_small   += removed_small
        total_kept            += kept

    # Générer data.yaml
    new_yaml = {
        "path" : str(clean_dataset.resolve()),
        "train": "train/images",
        "val"  : "valid/images",
        "test" : "test/images",
        "nc"   : data_cfg["nc"],
        "names": data_cfg["names"]
    }
    with open(clean_dataset / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f)

    print(f"\n  Total gardées      : {total_kept}")
    print(f"  Total invalides    : {total_removed_invalid}")
    print(f"  Total trop petites : {total_removed_small}")
    print(f"✅ Dataset nettoyé : {clean_dataset}")
    return clean_dataset


def explore_and_plot(raw_dataset: Path, save_dir: Path = None):
    """Génère les graphiques d'exploration du dataset."""

    save_dir = save_dir or (OUTPUTS_DIR / "exploration")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_dataset / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    label_dir = raw_dataset / "train" / "labels"
    names     = data_cfg["names"]

    class_counts = Counter()
    sizes        = []

    for file in label_dir.glob("*.txt"):
        for line in open(file):
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                cls_name = names[int(cls)] if isinstance(names, list) else names.get(int(cls), str(int(cls)))
                class_counts[cls_name] += 1
                sizes.append(w * h)

    sizes = np.array(sizes)

    # Graphique distribution classes
    fig, ax = plt.subplots(figsize=(12, 5))
    cls_list = [x[0] for x in class_counts.most_common()]
    cnt_list = [x[1] for x in class_counts.most_common()]
    colors   = plt.cm.Reds(np.linspace(0.4, 0.9, len(cls_list)))
    ax.bar(cls_list, cnt_list, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(np.mean(cnt_list), linestyle='--', color='blue', label=f'Moyenne : {np.mean(cnt_list):.0f}')
    ax.set_title("Distribution des instances par classe")
    ax.set_ylabel("Nombre d'instances")
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Graphique tailles
    total      = len(sizes)
    counts_sz  = [(sizes < 0.005).sum(), ((sizes >= 0.005) & (sizes < 0.1)).sum(), (sizes >= 0.1).sum()]
    categories = ['Très petits\n(<0.005)', 'Moyens\n(0.005-0.1)', 'Larges\n(>0.1)']
    colors_sz  = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts_sz, labels=categories, autopct='%1.1f%%',
           colors=colors_sz, startangle=90, explode=(0.05, 0, 0))
    ax.set_title("Distribution des tailles d'objets")
    plt.tight_layout()
    plt.savefig(save_dir / "size_distribution.png", dpi=150)
    plt.close()

    print(f"📊 Graphiques sauvegardés : {save_dir}")
    print(f"   Classes : {dict(class_counts.most_common())}")
