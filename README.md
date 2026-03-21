# YOLO Custom — Détection d'objets KITTI

Implémentation from scratch d'un modèle YOLO avec backbone ResNet50 pré-entraîné,
Neck FPN/PAN custom et Detection Head custom, entraîné sur le dataset KITTI.

**mAP50 = 0.60** sur le val set KITTI (8 classes).

---

## 🚀 Installation et lancement rapide

```bash
# 1. Cloner le repo
git clone https://github.com/akramaici/yolo_kitti
cd yolo_kitti

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les credentials Kaggle (une seule fois)
#    → Va sur https://www.kaggle.com/settings → API → Create New Token
#    → Place le fichier kaggle.json dans ~/.kaggle/kaggle.json

# 4. Lancer le pipeline complet (téléchargement + nettoyage + entraînement + évaluation)
python main.py
```

C'est tout. Le pipeline est entièrement automatique.

---

## 📁 Structure du projet

```
yolo_kitti/
├── main.py                   # ← Point d'entrée unique
├── config.py                 # Hyperparamètres centralisés
├── requirements.txt
├── src/
│   ├── download.py           # Téléchargement Kaggle automatique
│   ├── preprocess.py         # Nettoyage dataset
│   ├── dataset.py            # YOLODataset + letterbox
│   ├── model.py              # CustomYOLO + YOLOLossV2
│   ├── train.py              # Entraînement 2 phases
│   ├── evaluate.py           # mAP50 + visualisation
│   └── inference.py          # Inférence image/vidéo
├── data/                     # Dataset (généré automatiquement)
├── checkpoints/              # Modèles sauvegardés
└── outputs/                  # Résultats
```

---

## ⚙️ Options du pipeline

```bash
# Pipeline complet (défaut)
python main.py

# Ignorer le téléchargement si dataset déjà présent
python main.py --skip-download

# Ignorer l'entraînement — évaluation seulement
python main.py --skip-train --checkpoint checkpoints/best_phase2.pth

# Avec exploration du dataset
python main.py --explore

# Inférence sur une image
python main.py --skip-train --image ma_photo.jpg

# Inférence sur une vidéo
python main.py --skip-train --video ma_video.mp4
```

---

## 🔑 Configuration Kaggle

### Option 1 — fichier kaggle.json (recommandé)
```bash
# Linux/Mac
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Placer kaggle.json dans C:\Users\TON_NOM\.kaggle\kaggle.json
```

### Option 2 — Variables d'environnement
```bash
export KAGGLE_USERNAME=ton_username
export KAGGLE_KEY=ta_cle_api
```

---

## 🏗️ Architecture

| Composant | Détail |
|-----------|--------|
| Backbone | ResNet50 pré-entraîné ImageNet |
| Neck | FPN + PAN custom |
| Head | Detection Head 3 échelles |
| Loss | Focal BCE + MSE (YOLOLossV2) |
| Paramètres | 28M |

## 📊 Résultats

| Classe | AP |
|--------|----|
| car | 0.74 |
| van | 0.68 |
| truck | 0.74 |
| pedestrian | 0.47 |
| Person_sitting | 0.42 |
| cyclist | 0.47 |
| tram | 0.77 |
| misc | 0.53 |
| **mAP50** | **0.60** |
