# ══════════════════════════════════════════════════════════════
# src/download.py — Téléchargement automatique depuis Kaggle
# ══════════════════════════════════════════════════════════════

import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import KAGGLE_DATASET, RAW_DATASET, DATA_DIR


def download_dataset():
    if RAW_DATASET.exists() and (RAW_DATASET / "data.yaml").exists():
        print(f"✅ Dataset déjà présent : {RAW_DATASET}")
        _verify(RAW_DATASET)
        return RAW_DATASET

    print("=" * 60)
    print("  TÉLÉCHARGEMENT DU DATASET DEPUIS KAGGLE")
    print(f"  Source : {KAGGLE_DATASET}")
    print("=" * 60)

    try:
        import kaggle
    except ImportError:
        os.system("pip install kaggle -q")
        import kaggle

    _check_credentials()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("⬇️  Téléchargement en cours...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")

    if not RAW_DATASET.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir() and (item / "data.yaml").exists():
                item.rename(RAW_DATASET)
                break

    if not RAW_DATASET.exists():
        raise RuntimeError("❌ Échec du téléchargement")

    print(f"✅ Dataset téléchargé : {RAW_DATASET}")
    _verify(RAW_DATASET)
    return RAW_DATASET


def _check_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("✅ Credentials Kaggle trouvées")
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if username and key:
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        return

    print("""
⚠️  CREDENTIALS KAGGLE MANQUANTES
  1. Va sur https://www.kaggle.com/settings → API → Create New Token
  2. Place kaggle.json dans ~/.kaggle/kaggle.json
  Ou définis : export KAGGLE_USERNAME=... && export KAGGLE_KEY=...
""")
    sys.exit(1)


def _verify(path: Path):
    print("\n📊 Vérification :")
    for split in ["train", "valid", "test"]:
        img = path / split / "images"
        lbl = path / split / "labels"
        n_img = len(list(img.glob("*"))) if img.exists() else 0
        n_lbl = len(list(lbl.glob("*"))) if lbl.exists() else 0
        print(f"  {split:6s} → {n_img:5d} images | {n_lbl:5d} labels")
