"""
Categorical feature encoder for SAP material attributes
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class CategoricalEncoder:
    """
    Encode categorical SAP attributes (MaterialGroup, MaterialType, etc.)
    Uses learned embeddings for each category
    """
    
    def __init__(self, vocab_sizes: Optional[Dict[str, int]] = None):
        """
        Initialize categorical encoder
        
        Args:
            vocab_sizes: Dictionary mapping field names to vocabulary sizes
                        If None, uses defaults
        """
        # Default vocabulary sizes for common SAP fields
        self.vocab_sizes = vocab_sizes or {
            'MATKL': 1000,    # Material Groups
            'MTART': 100,     # Material Types
            'MEINS': 50,      # Units of Measure
            'EKGRP': 200,     # Purchasing Groups
            'MATKL_LEVEL1': 50,  # Material Group Level 1
        }
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(size, self._compute_embed_dim(size))
            for field, size in self.vocab_sizes.items()
        })
        
        # Vocabulary mappings (value -> index)
        self.vocabularies = {field: {} for field in self.vocab_sizes.keys()}
        self.next_indices = {field: 0 for field in self.vocab_sizes.keys()}
        
        # Special tokens
        self.UNK_TOKEN = 0
        self.PAD_TOKEN = 1
        
        # Initialize special tokens in vocabularies
        for field in self.vocabularies.keys():
            self.vocabularies[field]['<UNK>'] = self.UNK_TOKEN
            self.vocabularies[field]['<PAD>'] = self.PAD_TOKEN
            self.next_indices[field] = 2
    
    def _compute_embed_dim(self, vocab_size: int) -> int:
        """Compute embedding dimension based on vocabulary size"""
        if vocab_size < 50:
            return 16
        elif vocab_size < 200:
            return 32
        elif vocab_size < 1000:
            return 64
        else:
            return 128
    
    def _get_or_create_index(self, value: str, field: str) -> int:
        """
        Get index for a value, creating it if it doesn't exist
        
        Args:
            value: The categorical value
            field: The field name (e.g., 'MATKL')
            
        Returns:
            Index for the value
        """
        if field not in self.vocabularies:
            raise ValueError(f"Unknown field: {field}")
        
        if value in self.vocabularies[field]:
            return self.vocabularies[field][value]
        
        # Create new index
        if self.next_indices[field] < self.vocab_sizes[field]:
            idx = self.next_indices[field]
            self.vocabularies[field][value] = idx
            self.next_indices[field] += 1
            return idx
        else:
            # Vocabulary full, return UNK
            return self.UNK_TOKEN
    
    def encode(self, material_data: Dict) -> np.ndarray:
        """
        Encode categorical features from material data
        
        Args:
            material_data: Dictionary with categorical fields
                          e.g., {'MATKL': 'FASTENER', 'MTART': 'FERT'}
        
        Returns:
            Concatenated embeddings as numpy array
        """
        embeddings = []
        
        for field, embedding_layer in self.embeddings.items():
            if field in material_data and material_data[field] is not None:
                value = str(material_data[field])
                idx = self._get_or_create_index(value, field)
            else:
                idx = self.PAD_TOKEN
            
            # Get embedding
            with torch.no_grad():
                emb = embedding_layer(torch.tensor([idx]))
                embeddings.append(emb.squeeze().numpy())
        
        if embeddings:
            return np.concatenate(embeddings)
        else:
            # Return zero vector if no categorical features
            total_dim = sum(self._compute_embed_dim(size) 
                          for size in self.vocab_sizes.values())
            return np.zeros(total_dim)
    
    def get_embedding_dim(self) -> int:
        """Get total dimension of categorical embeddings"""
        return sum(self._compute_embed_dim(size) 
                  for size in self.vocab_sizes.values())
    
    def save_vocabularies(self, path: str):
        """Save vocabularies to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocabularies, f, indent=2)
    
    def load_vocabularies(self, path: str):
        """Load vocabularies from file"""
        import json
        with open(path, 'r') as f:
            self.vocabularies = json.load(f)
        
        # Update next_indices
        for field, vocab in self.vocabularies.items():
            if vocab:
                self.next_indices[field] = max(vocab.values()) + 1