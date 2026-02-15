"""
utils.py - Fonctions utilitaires

Ce module gère:
- Sauvegarde/chargement JSON
- Plotting (loss, accuracy, comparisons)
- Helpers divers

"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_directories(base_path='outputs'):
    """
    Crée les répertoires nécessaires
    
    Args:
        base_path: Chemin de base du projet
    """
    os.makedirs(f'{base_path}/results', exist_ok=True)
    os.makedirs(f'{base_path}/plots', exist_ok=True)
    logger.info(f"Répertoires créés dans {base_path}")


def save_results_json(results, filename=None, base_path='outputs'):
    """
    Sauvegarde les résultats en JSON
    
    Args:
        results: Dictionnaire de résultats
        filename: Nom du fichier (généré automatiquement si None)
        base_path: Chemin de base du projet
    
    Returns:
        str: Chemin du fichier sauvegardé
    """
    create_directories(base_path)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'optimization_results_{timestamp}.json'
    
    filepath = os.path.join(base_path, 'results', filename)
    
    # Convertir les arrays numpy en listes pour JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Résultats sauvegardés: {filepath}")
    return filepath


def load_results_json(filepath):
    """
    Charge les résultats depuis un fichier JSON
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        dict: Résultats chargés
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Résultats chargés: {filepath}")
    return results


def plot_loss_curves(results, config_id, base_path='outputs'):
    """
    Trace les courbes de loss (train + val)
    
    Args:
        results: Dictionnaire de résultats d'une config
        config_id: ID de la configuration
        base_path: Chemin de base du projet
    """
    create_directories(base_path)
    
    history = results['history']
    
    plt.figure(figsize=(10, 6))
    
    # Loss d'entraînement (tous les steps)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3, linewidth=2)
    
    # Loss de validation (tous les eval_steps)
    if history['steps']:
        plt.scatter(history['steps'], history['val_loss'], label='Val Loss', 
                   color='red', s=50, zorder=5)
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f"Loss Curve - {results['optimizer']} (lr={results['learning_rate']})",
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"loss_config_{config_id}_{results['optimizer']}_lr{results['learning_rate']}.png"
    filepath = os.path.join(base_path, 'plots', filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss plot sauvegardé: {filepath}")


def plot_accuracy_curves(results, config_id, base_path='outputs'):
    """
    Trace les courbes d'accuracy et F1-macro
    
    Args:
        results: Dictionnaire de résultats d'une config
        config_id: ID de la configuration
        base_path: Chemin de base du projet
    """
    create_directories(base_path)
    
    history = results['history']
    
    plt.figure(figsize=(10, 6))
    
    if history['steps']:
        plt.plot(history['steps'], history['accuracy'], marker='o', markersize=8,
                label='Accuracy', linewidth=2, color='green')
        plt.plot(history['steps'], history['f1_macro'], marker='s', markersize=8,
                label='F1-macro', linewidth=2, color='blue')
    
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f"Accuracy & F1 - {results['optimizer']} (lr={results['learning_rate']})",
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    filename = f"acc_config_{config_id}_{results['optimizer']}_lr{results['learning_rate']}.png"
    filepath = os.path.join(base_path, 'plots', filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accuracy plot sauvegardé: {filepath}")


def plot_all_configs_comparison(all_results, base_path='outputs'):
    """
    Trace un graphique comparatif de tous les optimiseurs
    
    Args:
        all_results: Liste de résultats pour toutes les configs
        base_path: Chemin de base du projet
    """
    create_directories(base_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique 1: Accuracy finale
    configs = []
    accuracies = []
    
    for result in all_results:
        config_name = f"{result['optimizer']}\nlr={result['learning_rate']}"
        configs.append(config_name)
        accuracies.append(result['final_accuracy'])
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    axes[0].bar(range(len(configs)), accuracies, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Final Accuracy by Configuration', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: F1-macro final
    f1_scores = [result['final_f1_macro'] for result in all_results]
    
    axes[1].bar(range(len(configs)), f1_scores, color=colors, edgecolor='black')
    axes[1].set_ylabel('F1-macro', fontsize=12)
    axes[1].set_title('Final F1-macro by Configuration', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filepath = os.path.join(base_path, 'plots', 'comparison_all_configs.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison plot sauvegardé: {filepath}")


def print_ranking(all_results):
    """
    Affiche un classement des configurations par accuracy
    
    Args:
        all_results: Liste de résultats pour toutes les configs
    """
    # Trier par accuracy
    sorted_results = sorted(all_results, key=lambda x: x['final_accuracy'], reverse=True)
    
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX - CLASSEMENT PAR ACCURACY")
    print("="*70)
    print(f"{'Rank':<6} {'Optimizer':<12} {'LR':<12} {'Accuracy':<12} {'F1-macro':<12}")
    print("-"*70)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<6} {result['optimizer']:<12} {result['learning_rate']:<12.0e} "
              f"{result['final_accuracy']:<12.4f} {result['final_f1_macro']:<12.4f}")
    
    print("="*70)
    
    best = sorted_results[0]
    print(f"\n🏆 MEILLEURE CONFIGURATION:")
    print(f"   Optimizer: {best['optimizer']}")
    print(f"   Learning Rate: {best['learning_rate']}")
    print(f"   Final Accuracy: {best['final_accuracy']:.4f}")
    print(f"   Final F1-macro: {best['final_f1_macro']:.4f}")
    print(f"   Total Steps: {best['total_steps']}")
    print()


def format_learning_rate(lr):
    """
    Formate un learning rate en notation scientifique
    
    Args:
        lr: Learning rate
    
    Returns:
        str: Learning rate formaté
    """
    if lr >= 1e-2:
        return f"{lr:.0e}"
    else:
        return f"{lr:.0e}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module utils importé avec succès")