"""
Core module for generating embeddings with RPT-1-OSS
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List
import numpy as np


class RPT1Embeddings:
    """
    Wrapper for SAP RPT-1-OSS embedding generation
    """
    
    def __init__(self, model_name: str = "sap-ai/rpt-1-oss", device: str = None):
        """
        Initialize RPT-1 embeddings generator
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def encode(
        self, 
        text: Union[str, List[str]], 
        layer: int = -1,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text input
        
        Args:
            text: Single string or list of strings
            layer: Which layer to extract (-1 for last layer)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (768,) or (n, 768)
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]
            return_single = True
        else:
            return_single = False
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract embeddings from specified layer
        if layer == -1:
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            embeddings = outputs.hidden_states[layer][:, 0, :]
        
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
        
        # Return single embedding if input was single string
        if return_single:
            return embeddings[0]
        
        return embeddings
    
    def similarity(
        self, 
        text1: str, 
        text2: str, 
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            metric: 'cosine' or 'euclidean'
            
        Returns:
            Similarity score
        """
        emb1 = self.encode(text1, normalize=True)
        emb2 = self.encode(text2, normalize=True)
        
        if metric == 'cosine':
            return float(np.dot(emb1, emb2))
        elif metric == 'euclidean':
            return float(1 / (1 + np.linalg.norm(emb1 - emb2)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode large batches efficiently
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n, 768)
        """
        from tqdm import tqdm
        
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_emb = self.encode(batch)
            embeddings.append(batch_emb)
        
        return np.vstack(embeddings)


# Convenience function
def create_embeddings_generator(**kwargs):
    """Create an RPT1Embeddings instance with given parameters"""
    return RPT1Embeddings(**kwargs)