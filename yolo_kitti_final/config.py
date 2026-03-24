# ══════════════════════════════════════════════════════════════
# config.py — Hyperparamètres centralisés
# ══════════════════════════════════════════════════════════════

from pathlib import Path

# ── Kaggle
KAGGLE_DATASET = "akramaici/kitty-data"
KAGGLE_DATA_DIR = "RAW_DATASET"

# ── Chemins
DATA_DIR       = Path("data")
RAW_DATASET    = DATA_DIR / "RAW_DATASET"
CLEAN_DATASET  = DATA_DIR / "dataset_clean"
CHECKPOINT_DIR = Path("checkpoints")
OUTPUTS_DIR    = Path("outputs")

# ── Modèle
IMG_SIZE      = 640
BASE_CHANNELS = 128

# ── Anchors K-means KITTI
ANCHORS_NORM = [
    [(0.034, 0.077), (0.058, 0.140), (0.058, 0.260)],
    [(0.121, 0.197), (0.082, 0.429), (0.178, 0.320)],
    [(0.143, 0.592), (0.262, 0.476), (0.269, 0.922)],
]

# ── Entraînement
BATCH_SIZE  = 16
NUM_WORKERS = 4

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
CONF_THRESH = 0.5
NMS_THRESH  = 0.35

# ── Nettoyage
MIN_AREA = 0.0005

# ── Classes KITTI
CLASS_NAMES = {
    0: "car", 1: "van", 2: "truck", 3: "pedestrian",
    4: "Person_sitting", 5: "cyclist", 6: "tram", 7: "misc"
}
