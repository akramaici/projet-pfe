# YOLO Custom — Détection d'objets KITTI

Implémentation from scratch d'un modèle YOLO avec backbone ResNet50 pré-entraîné,
Neck FPN/PAN custom et Detection Head custom, entraîné sur le dataset KITTI.

**mAP@50 = 0.597 | mAP@50:95 = 0.310 | FPS = 102**

---

## 🚀 Lancement rapide

```bash
git clone https://github.com/akramaici/projet-pfe
cd projet-pfe
pip install -r requirements.txt
# Placer kaggle.json dans ~/.kaggle/
python main.py
```

---

## 📁 Structure

```
projet-pfe/
├── main.py                   # Point d'entrée unique
├── config.py                 # Hyperparamètres centralisés
├── requirements.txt
├── src/
│   ├── download.py           # Téléchargement Kaggle automatique
│   ├── preprocess.py         # Nettoyage dataset
│   ├── dataset.py            # YOLODataset + augmentation
│   ├── model.py              # CustomYOLO + YOLOLossV2
│   ├── train.py              # Entraînement 2 phases
│   ├── evaluate.py           # mAP, latence, throughput, confusion
│   └── inference.py          # Inférence image/vidéo
├── data/                     # Dataset (généré automatiquement)
├── checkpoints/              # Modèles sauvegardés
└── outputs/                  # Résultats
```

---

## ⚙️ Options

```bash
python main.py                                    # pipeline complet
python main.py --skip-download                    # dataset déjà présent
python main.py --skip-train --checkpoint path.pth # évaluation seulement
python main.py --skip-train --image photo.jpg     # inférence image
python main.py --skip-train --video video.mp4     # inférence vidéo
python main.py --explore                          # graphiques dataset
```

---

## 🏗️ Architecture

| Composant | Détail |
|-----------|--------|
| Backbone | ResNet50 pré-entraîné ImageNet |
| Neck | FPN + PAN custom |
| Head | Detection Head 3 échelles |
| Loss | Focal BCE + MSE (YOLOLossV2) |
| Anchors | K-means sur KITTI |
| Paramètres | 28M |

---

## 📊 Résultats

| Métrique | Valeur |
|----------|--------|
| mAP@50 | **0.597** |
| mAP@50:95 | 0.310 |
| Recall@100 | 0.407 |
| Latence (1 img) | 10 ms |
| FPS | 102 img/s |
| Throughput max | 412 img/s (batch=64) |

| Classe | AP@50 |
|--------|-------|
| car | 0.442 |
| van | 0.401 |
| truck | 0.490 |
| pedestrian | 0.182 |
| Person_sitting | 0.135 |
| cyclist | 0.233 |
| tram | 0.340 |
| misc | 0.260 |

---

## 🔑 Configuration Kaggle

```bash
# Linux/Mac
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Ou variables d'environnement
export KAGGLE_USERNAME=ton_username
export KAGGLE_KEY=ta_cle_api
```
