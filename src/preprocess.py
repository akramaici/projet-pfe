# ══════════════════════════════════════════════════════════════
# src/preprocess.py — Nettoyage du dataset KITTI
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
    if clean_dataset.exists() and (clean_dataset / "data.yaml").exists():
        n = len(list((clean_dataset / "train" / "images").glob("*")))
        if n > 0:
            print(f"✅ Dataset nettoyé déjà présent ({n} images train)")
            return clean_dataset

    print("\n" + "="*60)
    print("  NETTOYAGE DU DATASET")
    print("="*60)

    with open(raw_dataset / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    total_invalid = 0
    total_small   = 0
    total_kept    = 0

    for split in ["train", "valid", "test"]:
        raw_img = raw_dataset   / split / "images"
        raw_lbl = raw_dataset   / split / "labels"
        cln_img = clean_dataset / split / "images"
        cln_lbl = clean_dataset / split / "labels"

        if not raw_lbl.exists():
            continue

        cln_img.mkdir(parents=True, exist_ok=True)
        cln_lbl.mkdir(parents=True, exist_ok=True)

        n_invalid = 0;  n_small = 0;  n_kept = 0

        for lf in raw_lbl.glob("*.txt"):
            valid = []
            for line in open(lf):
                parts = line.strip().split()
                if len(parts) != 5:
                    n_invalid += 1; continue
                cls, x, y, w, h = map(float, parts)
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                    n_invalid += 1; continue
                if w*h < min_area:
                    n_small += 1; continue
                valid.append(line);  n_kept += 1

            if valid:
                with open(cln_lbl / lf.name, "w") as f:
                    f.writelines(valid)
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_p = raw_img / f"{lf.stem}{ext}"
                    if img_p.exists():
                        shutil.copy(img_p, cln_img / img_p.name); break

        print(f"  {split:6s} → gardées:{n_kept:5d} | invalides:{n_invalid:3d} | petites:{n_small:4d}")
        total_invalid += n_invalid;  total_small += n_small;  total_kept += n_kept

    new_yaml = {
        "path": str(clean_dataset.resolve()),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": data_cfg["nc"], "names": data_cfg["names"]
    }
    with open(clean_dataset / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f)

    print(f"\n  Total gardées : {total_kept} | invalides : {total_invalid} | petites : {total_small}")
    print(f"✅ Dataset nettoyé : {clean_dataset}")
    return clean_dataset


def explore_and_plot(raw_dataset: Path, save_dir: Path = None):
    save_dir = save_dir or (OUTPUTS_DIR / "exploration")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_dataset / "data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    label_dir = raw_dataset / "train" / "labels"
    names     = data_cfg["names"]
    class_counts = Counter();  sizes = []

    for f in label_dir.glob("*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                name = names[int(cls)] if isinstance(names, list) else names.get(int(cls), str(int(cls)))
                class_counts[name] += 1
                sizes.append(w * h)

    sizes = np.array(sizes)

    # Distribution classes
    fig, ax = plt.subplots(figsize=(12, 5))
    cls_list = [x[0] for x in class_counts.most_common()]
    cnt_list = [x[1] for x in class_counts.most_common()]
    colors   = plt.cm.Reds(np.linspace(0.4, 0.9, len(cls_list)))
    ax.bar(cls_list, cnt_list, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(np.mean(cnt_list), linestyle='--', color='blue', label=f'Moyenne : {np.mean(cnt_list):.0f}')
    ax.set_title("Distribution des instances par classe");  ax.legend()
    plt.xticks(rotation=30, ha='right');  plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=150);  plt.close()

    print(f"📊 Graphiques : {save_dir}")
    print(f"   Classes : {dict(class_counts.most_common())}")
