# ══════════════════════════════════════════════════════════════
# src/download.py
# Téléchargement automatique du dataset depuis Kaggle
# ══════════════════════════════════════════════════════════════

import os
import sys
import zipfile
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import KAGGLE_DATASET, RAW_DATASET, DATA_DIR


def download_dataset():
    """
    Télécharge le dataset depuis Kaggle automatiquement.
    Nécessite : kaggle.json dans ~/.kaggle/ ou variables d'env KAGGLE_USERNAME + KAGGLE_KEY
    """

    # Si déjà téléchargé, skip
    if RAW_DATASET.exists() and (RAW_DATASET / "data.yaml").exists():
        print(f"✅ Dataset déjà présent : {RAW_DATASET}")
        _verify(RAW_DATASET)
        return RAW_DATASET

    print("=" * 60)
    print("  TÉLÉCHARGEMENT DU DATASET DEPUIS KAGGLE")
    print(f"  Source : {KAGGLE_DATASET}")
    print("=" * 60)

    # Vérifier que kaggle est installé
    try:
        import kaggle
    except ImportError:
        print("📦 Installation de kaggle...")
        os.system("pip install kaggle -q")
        import kaggle

    # Vérifier les credentials Kaggle
    _check_kaggle_credentials()

    # Téléchargement
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Téléchargement en cours...")

    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")

    # Vérifier que le dossier RAW_DATASET est bien là
    if not RAW_DATASET.exists():
        # Chercher le dossier extrait
        for item in DATA_DIR.iterdir():
            if item.is_dir() and item.name != "dataset_clean":
                print(f"  Dossier trouvé : {item.name}")
                if (item / "data.yaml").exists():
                    item.rename(RAW_DATASET)
                    break

    if not RAW_DATASET.exists():
        raise RuntimeError("❌ Échec du téléchargement — RAW_DATASET introuvable")

    print(f"✅ Dataset téléchargé : {RAW_DATASET}")
    _verify(RAW_DATASET)
    return RAW_DATASET


def _check_kaggle_credentials():
    """Vérifie que les credentials Kaggle sont disponibles."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if kaggle_json.exists():
        print("✅ Credentials Kaggle trouvées")
        return

    # Essayer les variables d'environnement
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    if username and key:
        print("✅ Credentials Kaggle depuis variables d'environnement")
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        return

    # Aucun credential trouvé — guider l'utilisateur
    print("\n" + "="*60)
    print("  ⚠️  CREDENTIALS KAGGLE MANQUANTES")
    print("="*60)
    print("""
Pour télécharger automatiquement le dataset, tu as besoin de ton API Kaggle.

Étapes :
  1. Va sur https://www.kaggle.com/settings
  2. Section 'API' → clique 'Create New Token'
  3. Un fichier 'kaggle.json' sera téléchargé
  4. Place-le dans ~/.kaggle/kaggle.json
     - Linux/Mac : ~/.kaggle/kaggle.json
     - Windows   : C:\\Users\\TON_NOM\\.kaggle\\kaggle.json

Ou définis ces variables d'environnement :
  export KAGGLE_USERNAME=ton_username
  export KAGGLE_KEY=ta_cle_api

Puis relance : python main.py
""")
    sys.exit(1)


def _verify(dataset_path: Path):
    """Vérifie la structure du dataset."""
    print("\n📊 Vérification du dataset :")
    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        n_img   = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl   = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  {split:6s} → {n_img:5d} images | {n_lbl:5d} labels")
