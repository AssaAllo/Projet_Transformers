"""
loss_landscape.py - Analyse du paysage de loss (loss landscape)

Ce module gère:
- Perturbation 1D du modèle
- Calcul de la métrique de sharpness
- Visualisation du landscape

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def get_loss_landscape_1d(
    model,
    val_loader,
    device,
    num_perturbations=20,
    radius=0.1
):
    """
    Trace le paysage de loss en 1D en perturbant le modèle
    
    Args:
        model: Modèle entraîné
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        num_perturbations: Nombre de perturbations
        radius: Rayon max de perturbation
    
    Returns:
        Tuple: (alphas, losses) pour plotting
    """
    logger.info("Calcul du loss landscape 1D...")
    
    # Sauvegarder les poids d'origine
    original_state = deepcopy(model.state_dict())
    
    # Créer une direction aléatoire pour la perturbation
    perturbation_direction = {}
    for name, param in model.named_parameters():
        perturbation_direction[name] = torch.randn_like(param) / (
            torch.norm(torch.randn_like(param)) + 1e-8
        )
    
    # Normaliser la perturbation
    total_norm = sum(
        torch.sum(d**2).item() for d in perturbation_direction.values()
    ) ** 0.5
    for name in perturbation_direction:
        perturbation_direction[name] /= (total_norm + 1e-8)
    
    # Calculer les losses à différentes perturbations
    alphas = np.linspace(-radius, radius, num_perturbations)
    losses = []
    
    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            # Appliquer la perturbation
            for name, param in model.named_parameters():
                param.data = original_state[name] + alpha * perturbation_direction[name]
            
            # Calculer la loss
            total_loss = 0.0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
            
            avg_loss = total_loss / len(val_loader)
            losses.append(avg_loss)
    
    # Restaurer les poids d'origine
    model.load_state_dict(original_state)
    
    logger.info(f"Loss landscape calculé: min={min(losses):.4f}, max={max(losses):.4f}")
    
    return alphas, np.array(losses)


def calculate_sharpness(
    model,
    val_loader,
    device,
    radius=0.1,
    num_samples=10
):
    """
    Calcule la métrique de sharpness (SAM-like)
    
    Args:
        model: Modèle entraîné
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        radius: Rayon de perturbation
        num_samples: Nombre d'échantillons aléatoires
    
    Returns:
        float: Métrique de sharpness
    """
    logger.info("Calcul de la sharpness...")
    
    original_state = deepcopy(model.state_dict())
    
    max_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            # Perturbation aléatoire
            for name, param in model.named_parameters():
                noise = torch.randn_like(param)
                norm = torch.norm(noise)
                if norm > 0:
                    noise = noise / norm * radius
                param.data = original_state[name] + noise
            
            # Calculer la loss
            total_loss = 0.0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
            
            avg_loss = total_loss / len(val_loader)
            max_loss = max(max_loss, avg_loss)
    
    # Restaurer les poids
    model.load_state_dict(original_state)
    
    logger.info(f"Sharpness calculée: {max_loss:.4f}")
    
    return max_loss


def plot_loss_landscape(alphas, losses, optimizer_name, learning_rate,
                       base_path='projet_transformers_complet'):
    """
    Trace le paysage de loss
    
    Args:
        alphas: Valeurs alpha
        losses: Valeurs de loss
        optimizer_name: Nom de l'optimiseur
        learning_rate: Learning rate
        base_path: Chemin de base du projet
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(alphas, losses, marker='o', linewidth=2, markersize=6, color='blue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Initial weights')
    plt.fill_between(alphas, losses, alpha=0.3)
    
    plt.xlabel('Perturbation (α)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(f"Loss Landscape - {optimizer_name} (lr={learning_rate})",
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    filename = f"landscape_{optimizer_name}_lr{learning_rate}.png"
    filepath = f"{base_path}/plots/{filename}"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss landscape plot sauvegardé: {filepath}")


def compare_sharpness(results_list, base_path='projet_transformers_complet'):
    """
    Compare la sharpness de différents modèles
    
    Args:
        results_list: Liste de (optimizer_name, learning_rate, sharpness)
        base_path: Chemin de base du projet
    """
    optimizers = [r[0] for r in results_list]
    lrs = [r[1] for r in results_list]
    sharpness_values = [r[2] for r in results_list]
    
    configs = [f"{opt}\nlr={lr:.0e}" for opt, lr in zip(optimizers, lrs)]
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
    bars = plt.bar(range(len(configs)), sharpness_values, color=colors, edgecolor='black')
    
    plt.ylabel('Sharpness', fontsize=12)
    plt.title('Sharpness Comparison - Lower is Better (Flatter)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, sharpness_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    filepath = f"{base_path}/plots/sharpness_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sharpness comparison plot sauvegardé: {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module loss_landscape importé avec succès")