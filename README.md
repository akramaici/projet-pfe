# 🚗 Détection d'objets en temps réel - PFE 2026

## 📌 Description
Système de détection d'objets pour la conduite autonome basé sur une architecture YOLO custom entraînée depuis zéro.

## 🎯 Classes détectées (13 classes)
bicycle, bus, car, motorcycle, other person, other vehicle, pedestrian, rider, traffic light, traffic sign, trailer, train, truck

## 📂 Dataset
Dataset filtré depuis MS-COCO (thème conduite autonome) :
👉 [Télécharger dataset_clean.zip](https://drive.google.com/file/d/1tkaSk12SZv7Od0j99zYLbqksnrl680l_/view?usp=sharing)

## 🏗️ Architecture
- Backbone : CSPDarknet (C2f + SPPF)
- Neck : FPN + PAN
- Head : Detection multi-scale (3 échelles)
- Paramètres : 29.4M

## 🚀 Entraînement
- Framework : PyTorch
- Epochs : 20
- Image size : 320×320
- Optimizer : AdamW

## 👨‍🎓 Auteur
Aici Mohamed Akram — Master SITW, Université Oran 1 — 2026
