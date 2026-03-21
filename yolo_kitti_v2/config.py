# ══════════════════════════════════════════════════════════════
# config.py — Tous les hyperparamètres centralisés
# ══════════════════════════════════════════════════════════════

from pathlib import Path

# ── Kaggle dataset
KAGGLE_DATASET   = "akramaici/kitty-data"   # username/dataset-name
KAGGLE_DATA_DIR  = "RAW_DATASET"            # dossier extrait depuis Kaggle

# ── Chemins locaux
DATA_DIR         = Path("data")
RAW_DATASET      = DATA_DIR / "RAW_DATASET"
CLEAN_DATASET    = DATA_DIR / "dataset_clean"
CHECKPOINT_DIR   = Path("checkpoints")
OUTPUTS_DIR      = Path("outputs")

# ── Modèle
IMG_SIZE         = 640
BASE_CHANNELS    = 128

# ── Anchors KITTI (calculées par k-means)
ANCHORS_NORM = [
    [(0.020, 0.060), (0.040, 0.090), (0.030, 0.180)],
    [(0.080, 0.150), (0.060, 0.350), (0.150, 0.250)],
    [(0.200, 0.400), (0.350, 0.500), (0.500, 0.800)],
]

# ── Entraînement
BATCH_SIZE   = 16
NUM_WORKERS  = 4

PHASE1_EPOCHS   = 20
PHASE1_LR       = 5e-4
PHASE1_PATIENCE = 10

PHASE2_EPOCHS   = 80
PHASE2_LR       = 5e-5
PHASE2_PATIENCE = 15

# ── Loss
LAMBDA_BOX   = 7.5
LAMBDA_OBJ   = 1.0
LAMBDA_CLS   = 0.5
LAMBDA_NOOBJ = 0.05

# ── Inférence
CONF_THRESH  = 0.5
NMS_THRESH   = 0.20

# ── Nettoyage
MIN_AREA     = 0.0005

# ── Classes KITTI
CLASS_NAMES = {
    0: "car", 1: "van", 2: "truck", 3: "pedestrian",
    4: "Person_sitting", 5: "cyclist", 6: "tram", 7: "misc"
}
