"""
data_loader.py - Chargement et préparation du dataset Emotion Detection

Ce module gère:
- Téléchargement du dataset Emotion Detection (6 classes)
- Subsampling équilibré
- Tokenization automatique avec TinyBERT
- DataLoader PyTorch optimisé
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer,  AutoModel
from datasets import load_dataset
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    
    """Dataset personnalisé pour Emotion Detection avec tokenization"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List de textes
            labels: List de labels (0-5)
            tokenizer: Tokenizer TinyBERT
            max_length: Longueur max des séquences
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        """Retourne un exemple tokenisé"""
        
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_emotion_dataset(subset_size=500, val_size=100, batch_size=16):
    
    """
    Charge le dataset Emotion Detection avec subsampling équilibré
    
    Args:
        subset_size: Nombre d'exemples d'entraînement (par défaut 500)
        val_size: Nombre d'exemples de validation (par défaut 100)
        batch_size: Taille des batches (par défaut 16)
    
    Returns:
        Tuple: (train_loader, val_loader, num_classes, tokenizer)
    """
    logger.info("Chargement du dataset Emotion Detection...")
    
    # Charger le dataset depuis Hugging Face
    dataset = load_dataset('emotion')
    
    # Extraire train et validation
    train_data = dataset['train']
    
    # Équilibrer les classes dans le subset
    texts = []
    labels = []
    
    # Nombre d'exemples par classe
    samples_per_class = subset_size // 6
    class_counts = [0] * 6
    
    logger.info(f"Subsampling équilibré: {samples_per_class} par classe")
    
    for example in train_data:
        label = example['label']
        if class_counts[label] < samples_per_class:
            texts.append(example['text'])
            labels.append(label)
            class_counts[label] += 1
        
        if sum(class_counts) >= subset_size:
            break
    
    logger.info(f"Subset créé: {len(texts)} exemples")
    logger.info(f"Distribution: {class_counts}")
    
    


    # Charger ou télécharger le modèle
    LOCAL_MODEL_PATH = "./models/tinybert-imdb"
    
    if Path(LOCAL_MODEL_PATH).exists():
        logger.info("Modèle trouvé en local, chargement...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        logger.info("Tokenizer TinyBERT chargé depuis le stockage local")
        
    else:
        logger.info("Modèle non trouvé localement, téléchargement...")
        Path(LOCAL_MODEL_PATH).mkdir(parents=True, exist_ok=True)
        
        # Télécharger et sauvegarder le tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Harsha901/tinybert-imdb-sentiment-analysis-model')
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        logger.info(f"Tokenizer téléchargé et sauvegardé dans: {LOCAL_MODEL_PATH}")
        
        # Télécharger et sauvegarder le modèle
        model = AutoModel.from_pretrained('Harsha901/tinybert-imdb-sentiment-analysis-model')
        model.save_pretrained(LOCAL_MODEL_PATH)
        logger.info(f"Modèle téléchargé et sauvegardé dans: {LOCAL_MODEL_PATH}")
    
    
    # Créer le dataset
    emotion_dataset = EmotionDataset(texts, labels, tokenizer)
    
    # Split train/val
    train_size = len(emotion_dataset) - val_size
    train_set, val_set = random_split(
        emotion_dataset,
        [train_size, val_size]
    )
    
    logger.info(f"Train set: {len(train_set)} exemples")
    logger.info(f"Val set: {len(val_set)} exemples")
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, 6, tokenizer


def get_label_names():
    """Retourne les noms des 6 classes d'émotions"""
    return {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }


if __name__ == "__main__":
    # Test de chargement
    logging.basicConfig(level=logging.INFO)
    
    train_loader, val_loader, num_classes, tokenizer = load_emotion_dataset(
        subset_size=500,
        val_size=100,
        batch_size=16
    )
    
    print(f"Nombre de classes: {num_classes}")
    print(f"Taille train loader: {len(train_loader)}")
    print(f"Taille val loader: {len(val_loader)}")
    
    # Afficher un exemple
    batch = next(iter(train_loader))
    print(f"\nExemple de batch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")