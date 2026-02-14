# Classification d’images de fleurs avec des CNN

## Présentation du projet
Ce projet porte sur la **classification d’images de fleurs** à l’aide de **réseaux de neurones convolutifs (CNN)**.  
Deux approches sont comparées :
- un **CNN conçu from scratch**,
- un **modèle pré-entraîné DenseNet121** utilisant le **transfert d’apprentissage**.

L’objectif est d’évaluer leurs **performances**, leur capacité de **généralisation** et leur **fiabilité** sur un jeu de données de taille limitée.

---

## Méthodologie
- Conception d’un CNN personnalisé comportant **quatre couches convolutionnelles** (≈ 1 à 3 millions de paramètres)
- Mise en œuvre du **transfert d’apprentissage** avec DenseNet121 (poids ImageNet)
- Entraînement avec **early stopping** pour limiter le surapprentissage
- Évaluation à l’aide des **courbes d’apprentissage**, de **matrices de confusion** et de métriques par classe (précision, rappel, F1-score)

---

## Outils utilisés
- TensorFlow / Keras  
- DenseNet121  
- Scikit-learn  
- Matplotlib  
- NumPy  

---

## Résultats
L’approche par transfert d’apprentissage montre une **meilleure généralisation** et des performances plus stables que le CNN entraîné from scratch, en particulier pour les classes visuellement proches.  
L’analyse des matrices de confusion met en évidence les types d’erreurs commises par chaque modèle.

---

## Données
Le jeu de données utilisé pour ce projet n’est plus disponible. 

---

## Auteur
Salomé Fonvielle
