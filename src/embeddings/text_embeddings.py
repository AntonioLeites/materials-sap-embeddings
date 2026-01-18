"""
Material Embeddings using Sentence Transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List


class MaterialEmbeddings:
    """
    Generate embeddings for SAP material descriptions
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize material embeddings generator
        
        Args:
            model_name: Sentence Transformer model to use
        """
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded successfully")
    
    def encode(
        self, 
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for material descriptions
        
        Args:
            texts: Single description or list of descriptions
            normalize: L2-normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (768,) or (n, 768)
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two materials
        
        Args:
            text1: First material description
            text2: Second material description
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity (already normalized)
        return float(np.dot(emb1, emb2))
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode large batches efficiently
        
        Args:
            texts: List of material descriptions
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (n, 768)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )


# Convenience function
def create_embedder(**kwargs):
    """Create a MaterialEmbeddings instance"""
    return MaterialEmbeddings(**kwargs)