"""
model_setup.py - Configuration et initialisation du modèle TinyBERT

Ce module gère:
- Chargement du modèle TinyBERT pré-entraîné (14M params)
- Ajustement pour la classification des émotions (6 classes)
- Optimisations CPU (float32, num_threads)
- Détection automatique du device (CPU/GPU)

"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


def get_device():
    """
    Détecte automatiquement le device disponible (GPU ou CPU)
    
    Returns:
        torch.device: Device CUDA si disponible, sinon CPU
    """
    if torch.cuda.is_available():
        logger.info(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        logger.info("Pas de GPU, utilisation du CPU")
        return torch.device('cpu')


def optimize_cpu_settings():
    """
    Optimise les paramètres PyTorch pour une exécution CPU efficace
    """
    # Utiliser float32 au lieu de float64
    torch.set_default_dtype(torch.float32)
    
    # Nombre de threads
    num_threads = torch.get_num_threads()
    logger.info(f"Threads CPU disponibles: {num_threads}")
    
    # Désactiver certaines optimisations si sur CPU
    if not torch.cuda.is_available():
        torch.set_num_threads(min(num_threads, 4))
        torch.set_num_interop_threads(4)
        logger.info("Optimisations CPU appliquées")


def load_tinybert_model(num_classes=6, device=None):
    """
    Charge le modèle TinyBERT pré-entraîné et l'ajuste pour la classification
    
    Args:
        num_classes: Nombre de classes (6 pour Emotion Detection)
        device: Device PyTorch (GPU ou CPU)
    
    Returns:
        model: Modèle TinyBERT fine-tunable
    """
    if device is None:
        device = get_device()
    
    # Optimiser les paramètres CPU
    #optimize_cpu_settings()
    
    logger.info("Chargement du modèle TinyBERT...")
    
    LOCAL_MODEL_PATH = "./models/tinybert-emotion-balanced"
    
    # Charger le modèle pré-entraîné
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Mettre le modèle sur le device
    model = model.to(device)
    model.to(torch.float32)  # Assurer float32
    
    logger.info(f"Modèle chargé sur {device}")
    
    # Afficher les statistiques du modèle
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Paramètres totaux: {total_params:,}")
    logger.info(f"Paramètres entraînables: {trainable_params:,}")
    
    return model


def freeze_encoder(model, freeze_percentage=0.0):
    """
    Gèle une partie de l'encodeur (optionnel)
    
    Args:
        model: Modèle TinyBERT
        freeze_percentage: Pourcentage de couches à geler (0-1)
    """
    if freeze_percentage == 0.0:
        logger.info("Aucune couche gelée")
        return
    
    encoder_layers = model.bert.encoder.layer
    num_layers = len(encoder_layers)
    freeze_count = int(num_layers * freeze_percentage)
    
    for i in range(freeze_count):
        for param in encoder_layers[i].parameters():
            param.requires_grad = False
    
    logger.info(f"{freeze_count}/{num_layers} couches gelées")


def count_parameters(model):
    """
    Compte les paramètres entraînables du modèle
    
    Args:
        model: Modèle PyTorch
    
    Returns:
        int: Nombre de paramètres entraînables
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """
    Retourne les informations détaillées du modèle
    
    Args:
        model: Modèle PyTorch
    
    Returns:
        dict: Informations du modèle
    """
    return {
        'model_name': 'Tinybert-emotion-balanced',
        'num_labels': model.config.num_labels,
        'hidden_size': model.config.hidden_size,
        'num_hidden_layers': model.config.num_hidden_layers,
        'num_attention_heads': model.config.num_attention_heads,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


if __name__ == "__main__":
    # Test de chargement
    logging.basicConfig(level=logging.INFO)
    
    device = get_device()
    model = load_tinybert_model(num_classes=6, device=device)
    
    info = get_model_info(model)
    print("\nInformations du modèle:")
    for key, value in info.items():
        print(f"  {key}: {value}")