"""
loss_landscape.py - Analyse du paysage de loss (loss landscape)

Corrections appliquées:
- Filter-wise normalization (Li et al., 2018) au lieu de normalisation globale
- Radius par défaut augmenté à 1.0
- Sharpness = max_loss - base_loss (SAM)
- Diagnostics de flat landscape
- os.makedirs pour éviter les erreurs de chemin
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import logging
import os

logger = logging.getLogger(__name__)


def _filter_wise_normalization(direction: dict, reference_state: dict) -> dict:
    """
    Filter-wise normalization (Li et al., 2018).
    
    Pour chaque paramètre, normalise la direction par sa propre norme
    puis la met à l'échelle par la norme du paramètre de référence.
    Garantit une perturbation proportionnelle à l'amplitude réelle des poids,
    quelle que soit la taille du modèle.
    """
    normalized = {}
    for name, d in direction.items():
        d_norm = torch.norm(d)
        p_norm = torch.norm(reference_state[name].float())
        if d_norm > 1e-8 and p_norm > 1e-8:
            normalized[name] = d / d_norm * p_norm
        else:
            normalized[name] = d
    return normalized


def _compute_val_loss(model, val_loader, device) -> float:
    """Calcule la loss moyenne sur le val_loader."""
    total_loss = 0.0
    num_batches = 0
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
        num_batches += 1
    return total_loss / max(num_batches, 1)


def get_loss_landscape_1d(
    model,
    val_loader,
    device,
    num_perturbations: int = 40,
    radius: float = 1.0       # ← était 0.1 avant, CORRIGÉ
):
    """
    Trace le paysage de loss en 1D avec filter-wise normalization.

    Args:
        model: Modèle entraîné
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        num_perturbations: Nombre de points sur l'axe alpha (défaut: 40)
        radius: Rayon max de perturbation (défaut: 1.0)
                Augmenter si le landscape reste plat (2.0, 5.0...)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (alphas, losses)
    """
    logger.info(f"Calcul du loss landscape 1D (radius={radius}, n={num_perturbations})...")

    original_state = deepcopy(model.state_dict())

    # Direction aléatoire + filter-wise normalization
    raw_direction = {
        name: torch.randn_like(param)
        for name, param in model.named_parameters()
    }
    direction = _filter_wise_normalization(raw_direction, original_state)

    # Diagnostic
    sample_name = next(iter(direction))
    logger.info(
        f"[Diagnostic] '{sample_name}' — "
        f"perturbation norm: {torch.norm(direction[sample_name]).item():.4f}, "
        f"param norm: {torch.norm(original_state[sample_name].float()).item():.4f}"
    )

    alphas = np.linspace(-radius, radius, num_perturbations)
    losses = []

    model.eval()
    with torch.no_grad():
        base_loss = _compute_val_loss(model, val_loader, device)
        logger.info(f"[Diagnostic] Loss à alpha=0: {base_loss:.6f}")

        for alpha in alphas:
            for name, param in model.named_parameters():
                param.data = original_state[name] + alpha * direction[name]
            losses.append(_compute_val_loss(model, val_loader, device))

    model.load_state_dict(original_state)

    losses = np.array(losses)
    loss_range = losses.max() - losses.min()
    logger.info(
        f"Landscape — min={losses.min():.6f}, max={losses.max():.6f}, range={loss_range:.6f}"
    )

    if loss_range < 1e-4:
        logger.warning(
            "⚠️  Landscape encore plat (range < 1e-4). "
            "Essayez radius=2.0 ou 5.0, ou vérifiez la convergence du modèle."
        )

    return alphas, losses


def calculate_sharpness(
    model,
    val_loader,
    device,
    radius: float = 1.0,
    num_samples: int = 10
) -> float:
    """
    Calcule la sharpness SAM-like avec filter-wise normalization.

    Sharpness = max_loss_perturbed - base_loss
    Une valeur élevée → minimum sharp → moins bonne généralisation.

    Args:
        model: Modèle entraîné
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        radius: Rayon de perturbation (défaut: 1.0)
        num_samples: Nombre de directions aléatoires (défaut: 10)

    Returns:
        float: Sharpness
    """
    logger.info(f"Calcul de la sharpness (radius={radius}, samples={num_samples})...")

    original_state = deepcopy(model.state_dict())

    model.eval()
    with torch.no_grad():
        base_loss = _compute_val_loss(model, val_loader, device)
        logger.info(f"[Sharpness] Loss de base: {base_loss:.6f}")
        max_loss = base_loss

        for i in range(num_samples):
            raw_direction = {
                name: torch.randn_like(param)
                for name, param in model.named_parameters()
            }
            direction = _filter_wise_normalization(raw_direction, original_state)

            for name, param in model.named_parameters():
                param.data = original_state[name] + radius * direction[name]

            avg_loss = _compute_val_loss(model, val_loader, device)
            max_loss = max(max_loss, avg_loss)
            logger.debug(f"  Sample {i+1}/{num_samples}: loss={avg_loss:.6f}")

    model.load_state_dict(original_state)

    sharpness = max_loss - base_loss
    logger.info(f"Sharpness = {sharpness:.6f} (max={max_loss:.6f} - base={base_loss:.6f})")
    return sharpness


def plot_loss_landscape(
    alphas,
    losses,
    optimizer_name: str,
    learning_rate: float,
    base_path: str = 'outputs'
):
    """
    Trace et sauvegarde le paysage de loss 1D.

    Args:
        alphas: Tableau numpy des valeurs alpha
        losses: Tableau numpy des losses
        optimizer_name: Nom de l'optimiseur
        learning_rate: Learning rate utilisé
        base_path: Dossier racine de sauvegarde
    """
    os.makedirs(f"{base_path}/plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, losses, marker='o', linewidth=2, markersize=5,
            color='steelblue', label='Validation Loss', zorder=3)

    min_idx = np.argmin(losses)
    ax.axvline(x=alphas[min_idx], color='green', linestyle=':', linewidth=1.5,
               label=f'Min loss ({losses[min_idx]:.4f})', zorder=2)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               label='Initial weights (α=0)', zorder=2)
    ax.fill_between(alphas, losses, losses.min(), alpha=0.15, color='steelblue')

    loss_range = losses.max() - losses.min()
    status = 'Flat ⚠️' if loss_range < 1e-3 else 'OK ✓'

    ax.set_xlabel('Perturbation (α)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(
        f"Loss Landscape — {optimizer_name} (lr={learning_rate})\n"
        f"Range: {loss_range:.4f} | {status}",
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    filepath = f"{base_path}/plots/landscape_{optimizer_name}_lr{learning_rate}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Loss landscape sauvegardé: {filepath}")
    return filepath


def compare_sharpness(results_list, base_path: str = 'outputs'):
    """
    Barplot comparatif de la sharpness pour plusieurs configs.

    Args:
        results_list: Liste de tuples (optimizer_name, learning_rate, sharpness)
        base_path: Dossier racine de sauvegarde
    """
    os.makedirs(f"{base_path}/plots", exist_ok=True)

    optimizers = [r[0] for r in results_list]
    lrs = [r[1] for r in results_list]
    sharpness_values = [r[2] for r in results_list]
    configs = [f"{opt}\nlr={lr:.0e}" for opt, lr in zip(optimizers, lrs)]
    best_idx = int(np.argmin(sharpness_values))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.5), 6))

    colors = ['#2ecc71' if i == best_idx else '#3498db' for i in range(len(configs))]
    bars = ax.bar(range(len(configs)), sharpness_values,
                  color=colors, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, sharpness_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + max(sharpness_values) * 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    best_bar = bars[best_idx]
    ax.text(
        best_bar.get_x() + best_bar.get_width() / 2.,
        best_bar.get_height() / 2,
        '★ Best', ha='center', va='center', fontsize=9, color='white', fontweight='bold'
    )

    ax.set_ylabel('Sharpness (max_loss - base_loss)', fontsize=12)
    ax.set_title(
        'Sharpness Comparison — Lower = Flatter Minimum = Better Generalization',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(sharpness_values) * 1.2)
    plt.tight_layout()

    filepath = f"{base_path}/plots/sharpness_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Sharpness comparison sauvegardé: {filepath}")
    return filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module loss_landscape importé avec succès")
    print("  ✓ Filter-wise normalization (Li et al., 2018)")
    print("  ✓ Radius par défaut = 1.0 (était 0.1)")
    print("  ✓ Sharpness = max_loss - base_loss")
    print("  ✓ Diagnostics flat landscape")
    print("  ✓ os.makedirs auto")