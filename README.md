# Projet G13 - Fine-tuning TinyBERT pour Emotion Detection

## Vue d'ensemble

Ce projet contient un **Random Search complet** pour comparer différents optimiseurs (AdamW, SGD, Adafactor) et learning rates lors du fine-tuning du modèle **TinyBERT** sur la tâche de détection des émotions.

**Modèle**: TinyBERT_General_4L_312D (14M paramètres)
**Dataset**: Emotion Detection (6 classes: sadness, joy, love, anger, fear, surprise)
**Approche**: Random Search testing 9 configurations

---

## Quickstart

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer le Random Search

```bash
cd projet_transformers_complet
python src/random_search.py
```

**Durée estimée**: 
- CPU moyen: 30-60 minutes
- GPU: 5-10 minutes

### 3. Analyser les résultats

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

##  Structure du projet

```
projet_transformers_complet/
│
├── src/                          # Code source principal
│   ├── __init__.py              # Package init
│   ├── data_loader.py           # Chargement du dataset (120 lignes)
│   ├── model_setup.py           # Setup TinyBERT (110 lignes)
│   ├── train_eval.py            # Loops d'entraînement (240 lignes)
│   ├── random_search.py         # ⭐ POINT D'ENTRÉE (260 lignes)
│   ├── loss_landscape.py        # Analyse du landscape (240 lignes)
│   └── utils.py                 # Utilities (180 lignes)
│
├── notebooks/                   # Jupyter notebooks
│   ├── exploration.ipynb        # Exploration du dataset
│   └── analysis.ipynb           # Analyse des résultats
│
├── results/                     # Résultats (JSON)
│   └── optimization_results_*.json
│
├── plots/                       # Visualisations (PNG)
│   ├── loss_config_*.png
│   ├── acc_config_*.png
│   ├── landscape_*.png
│   └── comparison_all_configs.png
│
├── requirements.txt             # Dépendances
└── README.md                    # Ce fichier
```

---

##  Configuration par défaut

### Dataset
- **Train subset**: 500 exemples (équilibré, ~83 par classe)
- **Validation**: 100 exemples
- **Batch size**: 16

### Entraînement
- **Max steps**: 100
- **Evaluation frequency**: Tous les 25 steps
- **Early stopping patience**: 2 (arrête si pas d'amélioration pendant 2 evaluations)

### Configurations testées (9 au total)

| Config | Optimizer | Learning Rate |
|--------|-----------|---------------|
| 1 | AdamW | 1e-5 |
| 2 | AdamW | 1e-4 |
| 3 | AdamW | 1e-3 |
| 4 | SGD | 1e-5 |
| 5 | SGD | 1e-4 |
| 6 | SGD | 1e-3 |
| 7 | Adafactor | 1e-5 |
| 8 | Adafactor | 1e-4 |
| 9 | Adafactor | 1e-3 |

---

## Ce que vous obtiendrez

Après l'exécution, vous aurez:

### 1. Fichier JSON (résultats complets)
```
results/optimization_results_20260214_153022.json
```

Contient pour chaque configuration:
- Optimizer utilisé
- Learning rate
- Loss final de validation
- Accuracy finale
- F1-macro final
- Historique complet d'entraînement (loss, accuracy, F1)

### 2. Graphiques PNG (18 fichiers)

**Loss curves** (9 fichiers):
- `loss_config_1_AdamW_lr1e-05.png`
- `loss_config_2_AdamW_lr1e-04.png`
- ... (9 au total)

**Accuracy curves** (9 fichiers):
- `acc_config_1_AdamW_lr1e-05.png`
- ... (9 au total)

**Comparaison globale** (1 fichier):
- `comparison_all_configs.png` - Barplot comparant tous les optimiseurs

### 3. Console output
Classement final des configurations par accuracy

---

## Personnalisation

Pour modifier les hyperparamètres, éditez `src/random_search.py`:

```python
# Line ~60-70
TRAIN_SUBSET_SIZE = 500  # Réduire si OutOfMemory
VAL_SIZE = 100
BATCH_SIZE = 16          # Réduire si OutOfMemory
MAX_STEPS = 100          # Réduire si trop lent
EVAL_STEPS = 25
```

### Configurations personnalisées

Remplacez la liste `CONFIGS` pour tester d'autres hyper-paramètres:

```python
CONFIGS = [
    ('AdamW', 5e-5),      # AdamW avec lr=5e-5
    ('SGD', 2e-4),        # SGD avec lr=2e-4
    # ... ajouter vos configs
]
```

---

## Gestion de la mémoire

Si vous avez des problèmes OutOfMemory:

### Option 1: Réduire la taille du subset
```python
TRAIN_SUBSET_SIZE = 200  # Au lieu de 500
VAL_SIZE = 50            # Au lieu de 100
```

### Option 2: Réduire la taille des batches
```python
BATCH_SIZE = 8           # Au lieu de 16
```

### Option 3: Réduire le nombre d'étapes
```python
MAX_STEPS = 50           # Au lieu de 100
EVAL_STEPS = 10
```

---

## Détails techniques

### Modèle TinyBERT
- **Architecture**: 4 couches de transformer, 312 dimensions cachées
- **Paramètres**: ~14 million
- **Pré-entraînement**: BERT généraliste

### Optimiseurs
- **AdamW**: Adaptive Moment Estimation with weight decay
- **SGD**: Stochastic Gradient Descent avec momentum=0.9
- **Adafactor**: Factorized Adam (économe en mémoire)

### Métrique de loss
- CrossEntropyLoss (classification multi-classe)

### Métriques d'évaluation
- **Accuracy**: Nombre de prédictions correctes / Total
- **F1-macro**: Moyenne non pondérée des F1-scores par classe

---

## Pour votre rapport (8-10 pages)

### Structure recommandée

**Introduction** (1-1.5 pages)
- Contexte: Fine-tuning de transformers
- Problème: Quel optimiseur choisir?
- Dataset: Emotion Detection
- Approche: Random Search

**Méthodologie** (2-2.5 pages)
- Modèle: TinyBERT (architecture, paramètres)
- Dataset et préparation des données
- Optimiseurs testés et leurs caractéristiques
- Configuration expérimentale (hyperparamètres)
- Protocole d'évaluation

**Résultats** (3-4 pages)
- Tableau comparatif des 9 configurations
- Graphiques de convergence (loss, accuracy)
- Analyse du loss landscape
- Meilleur optimizer et learning rate
- Discussion des résultats

**Discussion** (1.5-2 pages)
- Pourquoi cet optimizer est meilleur?
- Impact du learning rate
- Relation entre flatness et généralisation
- Limitations de l'approche
- Insights intéressants

**Conclusion** (0.5-1 page)
- Résumé des findings clés
- Recommandations pour le choix d'optimizer
- Améliorations futures possibles

---

## ❓ FAQ

**Q: Ça prend combien de temps?**
A: 30-60 min sur CPU moyen, 5-10 min sur GPU.

**Q: Je dois coder?**
A: Non! Le code est prêt à l'emploi.

**Q: Ça marche du premier coup?**
A: Oui, testé et production-ready. Utilisez les tips de gestion mémoire si nécessaire.

**Q: Où trouver les résultats?**
A: Dossier `results/` (JSON) et `plots/` (PNG).

**Q: Comment analyser les résultats?**
A: Lancez le notebook: `jupyter notebook notebooks/analysis.ipynb`

**Q: Puis-je modifier les learning rates?**
A: Oui! Éditez la liste `CONFIGS` dans `src/random_search.py`.

**Q: Qu'est-ce que le loss landscape?**
A: Visualisation de comment la loss change quand on perturbe le modèle entraîné. Plus plat = plus généraliste.

---

## 📝 Citation et références

### Modèles utilisés
- **TinyBERT**: Jiao et al., "TinyBERT: Distilling BERT for Natural Language Understanding", 2020
- **Transformers**: Hugging Face library

### Datasets
- **Emotion Detection**: Emotion dataset from Hugging Face Datasets

---

## 📧 Support

En cas de problème:
1. Consultez la section "Gestion de la mémoire"
2. Vérifiez que les packages sont bien installés: `pip list`
3. Testez d'abord sur un subset plus petit

---

## 📄 License

Ce projet est fourni pour usage académique dans le cadre du cours G13.

---

**Date de création**: Février 2026
**Version**: 1.0.0
**Statut**: Production-ready ✅