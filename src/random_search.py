"""
random_search.py - Random Search Benchmark pour Fine-tuning TinyBERT

POINT D'ENTRÉE PRINCIPAL
Lancez ceci avec: python src/random_search.py

Ce script:
1. Télécharge le dataset Emotion Detection
2. Lance 9 configurations (3 optimiseurs × 3 learning rates)
3. Entraîne chaque configuration complètement
4. Génère résultats (JSON) et graphiques (PNG)
5. Affiche un classement final
"""

import sys
import os
import logging
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import data_loader, model_setup, train_eval, utils, loss_landscape

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Fonction principale: Lance le Random Search complet
    """
    
    print("\n" + "="*80)
    print(" PROJET G13 - FINE-TUNING TINYBERT POUR EMOTION DETECTION")
    print("="*80)
    print("Random Search: Comparaison d'optimiseurs et learning rates")
    print("="*80 + "\n")
    
    # ==================== CONFIGURATION ====================
    
    # Hyperparamètres du dataset
    TRAIN_SUBSET_SIZE = 10000  # Nombre d'exemples d'entraînement
    VAL_SIZE = 2000           # Nombre d'exemples de validation
    BATCH_SIZE = 32         # Taille des batches
    
    # Hyperparamètres d'entraînement
    MAX_STEPS = 200         # Nombre maximum de steps
    EVAL_STEPS = 25          # Évaluation tous les N steps
    PATIENCE = 5           # Early stopping patience
    
    # Configurations à tester (optimizer, learning_rate)
    CONFIGS = [
        ('AdamW', 1e-5),
        ('AdamW', 1e-4),
        ('AdamW', 1e-3),
        ('SGD', 1e-5),
        ('SGD', 1e-4),
        ('SGD', 1e-3),
        ('Adafactor', 1e-5),
        ('Adafactor', 1e-4),
        ('Adafactor', 1e-3),
    ]
    
    print(f"Configuration:")
    print(f"  Train subset: {TRAIN_SUBSET_SIZE} exemples")
    print(f"  Val set: {VAL_SIZE} exemples")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Eval steps: {EVAL_STEPS}")
    print(f"  Nombre de configurations: {len(CONFIGS)}")
    print()
    
    # ==================== 1. CHARGER LE DATASET ====================
    
    logger.info("ÉTAPE 1: Chargement du dataset Emotion Detection")
    print("-" * 80)
    
    train_loader, val_loader, num_classes, tokenizer = data_loader.load_emotion_dataset(
        subset_size=TRAIN_SUBSET_SIZE,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE
    )
    
    logger.info(f"Dataset chargé: {num_classes} classes")
    print()
    
    # ==================== 2. SETUP DU DEVICE ====================
    
    logger.info("ÉTAPE 2: Configuration du device")
    print("-" * 80)
    
    device = model_setup.get_device()
    logger.info(f"Device: {device}")
    print()
    
    # ==================== 3. RANDOM SEARCH ====================
    
    logger.info("ÉTAPE 3: Random Search - Entraînement de 9 configurations")
    print("-" * 80)
    
    def load_model_fn(device):
        """Fonction pour recharger le modèle à chaque configuration"""
        return model_setup.load_tinybert_model(num_classes=6, device=device)
    
    all_results = train_eval.train_multiple_configs(
        model_fn=load_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        configs=CONFIGS,
        max_steps=MAX_STEPS
    )
    
    print()
    logger.info(f"Entraînement complété: {len(all_results)} configurations")
    
    # ==================== 4. SAUVEGARDER LES RÉSULTATS ====================
    
    logger.info("ÉTAPE 4: Sauvegarde des résultats")
    print("-" * 80)
    
    # Préparer les résultats pour sauvegarde JSON
    results_for_json = {
        'config': {
            'train_subset_size': TRAIN_SUBSET_SIZE,
            'val_size': VAL_SIZE,
            'batch_size': BATCH_SIZE,
            'max_steps': MAX_STEPS,
            'eval_steps': EVAL_STEPS,
            'patience': PATIENCE,
            'num_classes': num_classes
        },
        'results': all_results
    }
    
    # Sauvegarder en JSON
    json_filepath = utils.save_results_json(results_for_json)
    logger.info(f"Résultats JSON: {json_filepath}")
    
    # ==================== 5. GÉNÉRER LES GRAPHIQUES ====================
    
    logger.info("ÉTAPE 5: Génération des graphiques")
    print("-" * 80)
    
    for idx, result in enumerate(all_results, 1):
        # Graphiques de loss
        utils.plot_loss_curves(result, idx)
        
        # Graphiques d'accuracy
        utils.plot_accuracy_curves(result, idx)
    
    # Graphique de comparaison globale
    utils.plot_all_configs_comparison(all_results)
    
    logger.info("Graphiques générés avec succès")
    print()
    
    # ==================== 6. AFFICHER LE CLASSEMENT ====================
    
    logger.info("ÉTAPE 6: Résultats finaux et classement")
    print("-" * 80)
    
    utils.print_ranking(all_results)
    
    # ==================== 7. ANALYSE DU LOSS LANDSCAPE (OPTIONNEL) ====================
    
    logger.info("ÉTAPE 7: Analyse du loss landscape (optionnel)")
    print("-" * 80)
    
    # Charger le meilleur modèle et analyser son landscape
    best_result = sorted(all_results, key=lambda x: x['final_accuracy'], reverse=True)[0]
    best_optimizer = best_result['optimizer']
    best_lr = best_result['learning_rate']
    
    logger.info(f"Analyse du meilleur modèle: {best_optimizer}, lr={best_lr}")
    
    try:
        # Recharger le meilleur modèle
        best_model = load_model_fn(device)
        
        # Entraîner brièvement le meilleur modèle pour avoir ses poids
        best_train_result = train_eval.train_with_early_stopping(
            model=best_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=best_optimizer,
            learning_rate=best_lr,
            device=device,
            max_steps=MAX_STEPS,
            eval_steps=max(1, MAX_STEPS // 4),
            patience=PATIENCE
        )
        
        # Calculer le loss landscape en 1D
        alphas, losses = loss_landscape.get_loss_landscape_1d(
            best_model,
            val_loader,
            device,
            num_perturbations=20,
            radius=0.1
        )
        
        # Tracer le loss landscape
        loss_landscape.plot_loss_landscape(alphas, losses, best_optimizer, best_lr)
        
        logger.info("Loss landscape analysé avec succès")
        
        
         # 2) Sharpness (SAM-like) sur la validation
        sharp = loss_landscape.calculate_sharpness(
            model=best_model,
            val_loader=val_loader,
            device=device,
            radius=0.1,       # garde cohérent avec le landscape
            num_samples=10
        )
        logger.info(f"[Sharpness] best={best_optimizer} lr={best_lr}: {sharp:.6f}")       
        
    except Exception as e:
        logger.warning(f"Analyse du loss landscape impossible: {e}")
    
    print()
    
    # ==================== RÉSUMÉ FINAL ====================
    
    print("="*80)
    print(" RANDOM SEARCH COMPLET!")
    print("="*80)
    print(f"\n Résultats sauvegardés dans:")
    print(f"   • JSON: {json_filepath}")
    print(f"   • Graphiques PNG: outputs/plots/")
    print(f"\n Fichiers générés:")
    print(f"   • {len(all_results)} graphiques de loss")
    print(f"   • {len(all_results)} graphiques d'accuracy")
    print(f"   • 1 graphique de comparaison")
    
    if os.path.exists(os.path.join('outputs', 'plots',
                                   f'landscape_{best_optimizer}_lr{best_lr}.png')):
        print(f"   • 1 loss landscape plot")
    
    print(f"\n Meilleur optimizer: {best_optimizer} (lr={best_lr})")
    print(f"   Accuracy finale: {best_result['final_accuracy']:.4f}")
    print(f"   F1-macro final: {best_result['final_f1_macro']:.4f}")
    
    print(f"\n Prochaines étapes:")
    print(f"   1. Lire les résultats JSON")
    print(f"   2. Analyser les graphiques PNG")
    print(f"   3. Lancer le notebook: jupyter notebook notebooks/analysis.ipynb")
        
    print("\n" + "="*80)
    print(" PROJET TERMINÉ!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        sys.exit(1)