"""
train_eval.py - Boucles d'entraînement et évaluation du modèle

Ce module gère:
- Entraînement avec 3 optimiseurs: AdamW, SGD, Adafactor
- Calcul des métriques: Accuracy et F1-macro
- Early stopping automatique
- Sauvegarde des résultats

"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW, SGD
from transformers import Adafactor
import logging

logger = logging.getLogger(__name__)


def get_optimizer(model, optimizer_name, learning_rate):
    """
    Crée l'optimiseur spécifié
    
    Args:
        model: Modèle PyTorch
        optimizer_name: 'AdamW', 'SGD', ou 'Adafactor'
        learning_rate: Taux d'apprentissage
    
    Returns:
        optimizer: Optimiseur PyTorch
    """
    if optimizer_name == 'AdamW':
        return AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adafactor':
        return Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
           # lr=learning_rate
        )
    else:
        raise ValueError(f"Optimiseur inconnu: {optimizer_name}")


def train_step(model, batch, optimizer, device, criterion):
    """
    Entraîne le modèle sur un batch
    
    Args:
        model: Modèle PyTorch
        batch: Batch du DataLoader
        optimizer: Optimiseur
        device: Device (CPU/GPU)
        criterion: Fonction de loss
    
    Returns:
        float: Loss du batch
    """
    model.train()
    
    # Déplacer les données sur le device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return loss.item()


def eval_step(model, val_loader, device):
    """
    Évalue le modèle sur le validation set
    
    Args:
        model: Modèle PyTorch
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
    
    Returns:
        Tuple: (loss_avg, accuracy, f1_macro)
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculer les métriques
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, f1_macro


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer_name,
    learning_rate,
    device,
    max_steps=100,
    eval_steps=25,
    patience=2
):
    """
    Entraîne le modèle avec early stopping
    
    Args:
        model: Modèle PyTorch
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        optimizer_name: Nom de l'optimiseur
        learning_rate: Taux d'apprentissage
        device: Device (CPU/GPU)
        max_steps: Nombre max de steps (par défaut 100)
        eval_steps: Évaluation tous les N steps
        patience: Early stopping patience
    
    Returns:
        dict: Résultats d'entraînement
    """
    logger.info(f"Entraînement: {optimizer_name}, lr={learning_rate}")
    
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialiser les historiques
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'f1_macro': [],
        'steps': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    step = 0
    
    # Entraînement
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Train step
            loss = train_step(model, batch, optimizer, device, criterion)
            history['train_loss'].append(loss)
            
            # Évaluation
            if (step + 1) % eval_steps == 0:
                val_loss, accuracy, f1_macro = eval_step(model, val_loader, device)
                
                history['val_loss'].append(val_loss)
                history['accuracy'].append(accuracy)
                history['f1_macro'].append(f1_macro)
                history['steps'].append(step + 1)
                
                logger.info(
                    f"  Step {step+1}: "
                    f"train_loss={loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"acc={accuracy:.4f}, "
                    f"f1={f1_macro:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at step {step+1}")
                        break
            
            step += 1
    
    # Résultats finaux
    final_val_loss, final_acc, final_f1 = eval_step(model, val_loader, device)
    
    results = {
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'final_val_loss': final_val_loss,
        'final_accuracy': final_acc,
        'final_f1_macro': final_f1,
        'total_steps': step,
        'history': history
    }
    
    return results


def train_multiple_configs(
    model_fn,
    train_loader,
    val_loader,
    device,
    configs,
    max_steps=100
):
    """
    Entraîne le modèle avec plusieurs configurations
    
    Args:
        model_fn: Fonction pour charger le modèle
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        configs: Liste de tuples (optimizer_name, learning_rate)
        max_steps: Nombre max de steps
    
    Returns:
        list: Résultats pour chaque configuration
    """
    all_results = []
    
    for optimizer_name, learning_rate in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration: {optimizer_name}, lr={learning_rate}")
        logger.info(f"{'='*60}")
        
        # Recharger le modèle pour chaque config
        model = model_fn(device=device)
        model.to(device)
        
        # Entraîner
        results = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            device=device,
            max_steps=max_steps,
            eval_steps=max(1, max_steps // 4)
        )
        
        all_results.append(results)
    
    return all_results


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    print("Module train_eval importé avec succès")