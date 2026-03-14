# Classification d’images en régime de données limité  
CNN entraîné from scratch vs transfert d’apprentissage

## Présentation

Ce projet étudie la classification d’images dans un contexte de données limitées, en comparant deux stratégies d’apprentissage : un réseau de neurones convolutif (CNN) conçu et entraîné from scratch, et un modèle DenseNet121 pré-entraîné sur ImageNet utilisé en transfert d’apprentissage.  

L’objectif est d’analyser la généralisation, la stabilité d’entraînement et l’impact du pré-entraînement lorsque le nombre d’exemples disponibles est restreint.

## Contexte scientifique

Les réseaux convolutifs profonds nécessitent généralement de grandes quantités de données pour apprendre des représentations visuelles robustes.  

En régime de données limité :  
- Les modèles surparamétrés ont tendance à surapprendre  
- L’écart entre performance d’entraînement et de validation augmente  
- Certaines classes peuvent être sous-représentées  

Le transfert d’apprentissage introduit un prior inductif fort en réutilisant des représentations apprises sur un corpus massif (ImageNet).

Ce projet explore :  
- Le comportement d’un CNN appris from scratch  
- L’impact de la capacité du modèle sur la généralisation  
- Le rôle des représentations pré-entraînées  
- Les différences de stabilité et de structure d’erreur  

## Méthodologie

### CNN entraîné from scratch
- 4 blocs convolutionnels  
- 1–3 millions de paramètres  
- Batch Normalization  
- Dropout  
- Global Average Pooling  
- Early Stopping  
- Expérimentations avec réduction de capacité et régularisation renforcée  

### Transfer Learning (DenseNet121)
- Backbone DenseNet121 pré-entraîné (ImageNet)  
- Poids gelés (feature extractor)  
- Tête de classification légère (Dense + Dropout)  
- Optimisation avec Adam  
- Early Stopping  

## Dataset

- **320 images** réparties en **4 classes** (labels : 0, 2, 4, 9)
- Split : **256 images d'entraînement / 64 images de validation**
- Taille des images : **128×128 pixels**
- Augmentation : flips horizontaux, rotations (±10%), zoom (±10%)

## Évaluation

Les modèles sont évalués via :  
- Accuracy  
- Précision / Rappel / F1-score  
- Matrice de confusion  
- Courbes d’apprentissage (loss et accuracy)  
- Analyse qualitative des erreurs  

## Résultats

Les CNN entraînés from scratch présentent un surapprentissage marqué, un écart important entre train et validation, ainsi qu’une instabilité selon les classes.  

Le modèle DenseNet121 converge rapidement, généralise de manière stable et atteint environ 98 % d’accuracy validation. Il reconnaît correctement l’ensemble des classes.  

Ces résultats confirment qu’en régime de données limité, le transfert d’apprentissage améliore fortement la robustesse et la généralisation.

## Structure du projet

```
flower-image-classification/
├── data/
│   ├── assets/              # Image de test
│   └── raw/                 # Dataset extrait (classes 0, 2, 4, 9)
├── notebooks/
│   └── flower_images_classification.ipynb
├── src/
│   ├── prepare_data.py      # Extraction du dataset (ZIP → data/raw/)
│   ├── models.py            # Définitions des modèles (CNN v1, CNN v2, DenseNet121)
│   └── evaluate.py          # Utilitaires d’évaluation (courbes, matrice de confusion)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/salomefonvielle/flower-image-classification
cd flower-image-classification
python -m venv .venv
source .venv/bin/activate  # Mac / Linux
pip install -r requirements.txt
```

## Données

Placer le fichier `dataset.zip` dans le dossier `src/`, puis exécuter :

```bash
python src/prepare_data.py
```

Les images seront extraites dans `data/raw/`.

> Le fichier `dataset.zip` et le dossier `data/raw/` ne sont pas versionnés.

## Remarques

- Les données brutes ne sont pas versionnées.  
- Le dossier data/raw/ est ignoré par Git.  
- __pycache__/ est automatiquement ignoré.