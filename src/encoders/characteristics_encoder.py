"""
Technical characteristics encoder (DIAMETER, LENGTH, MATERIAL, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


class CharacteristicsEncoder:
    """
    Encode technical characteristics from SAP classification system
    Examples: DIAMETER=8mm, LENGTH=50mm, MATERIAL=STEEL
    """
    
    def __init__(self, characteristics_config: Optional[Dict] = None):
        """
        Initialize characteristics encoder
        
        Args:
            characteristics_config: Configuration for characteristic types
        """
        # Default characteristics for fasteners/mechanical parts
        self.config = characteristics_config or {
            'DIAMETER': {'type': 'categorical', 'vocab_size': 100, 'embed_dim': 32},
            'LENGTH': {'type': 'categorical', 'vocab_size': 200, 'embed_dim': 32},
            'MATERIAL': {'type': 'categorical', 'vocab_size': 50, 'embed_dim': 32},
            'COATING': {'type': 'categorical', 'vocab_size': 30, 'embed_dim': 16},
            'STANDARD': {'type': 'categorical', 'vocab_size': 100, 'embed_dim': 32},
            'THREAD': {'type': 'categorical', 'vocab_size': 20, 'embed_dim': 16},
            'HEAD_TYPE': {'type': 'categorical', 'vocab_size': 30, 'embed_dim': 16},
            'STRENGTH_CLASS': {'type': 'categorical', 'vocab_size': 20, 'embed_dim': 16},
        }
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict({
            char_name: nn.Embedding(
                config['vocab_size'],
                config['embed_dim']
            )
            for char_name, config in self.config.items()
            if config['type'] == 'categorical'
        })
        
        # Vocabularies
        self.vocabularies = {name: {} for name in self.config.keys()}
        self.next_indices = {name: 2 for name in self.config.keys()}  # 0=UNK, 1=PAD
        
        # Initialize special tokens
        for vocab in self.vocabularies.values():
            vocab['<UNK>'] = 0
            vocab['<PAD>'] = 1
    
    def _normalize_value(self, value: str, char_name: str) -> str:
        """Normalize characteristic value for consistency"""
        value = str(value).strip().upper()
        
        # Specific normalizations
        if char_name == 'DIAMETER':
            # Normalize "8mm", "M8", "8 mm" -> "8MM"
            value = value.replace(' ', '').replace('M', '')
            if not value.endswith('MM'):
                value = value + 'MM'
        
        elif char_name == 'LENGTH':
            # Normalize length values
            value = value.replace(' ', '')
            if not value.endswith('MM'):
                value = value + 'MM'
        
        return value
    
    def _get_or_create_index(self, value: str, char_name: str) -> int:
        """Get or create index for a characteristic value"""
        if char_name not in self.vocabularies:
            return 0  # UNK
        
        value = self._normalize_value(value, char_name)
        
        if value in self.vocabularies[char_name]:
            return self.vocabularies[char_name][value]
        
        # Create new index
        max_idx = self.config[char_name]['vocab_size']
        if self.next_indices[char_name] < max_idx:
            idx = self.next_indices[char_name]
            self.vocabularies[char_name][value] = idx
            self.next_indices[char_name] += 1
            return idx
        else:
            return 0  # UNK if vocabulary full
    
    def encode(self, characteristics: Dict) -> np.ndarray:
        """
        Encode characteristics dictionary to vector
        
        Args:
            characteristics: Dictionary of characteristic values
                            e.g., {'DIAMETER': '8mm', 'LENGTH': '50mm', ...}
        
        Returns:
            Concatenated embeddings as numpy array
        """
        embeddings = []
        
        for char_name, embedding_layer in self.embeddings.items():
            if char_name in characteristics and characteristics[char_name] is not None:
                value = str(characteristics[char_name])
                idx = self._get_or_create_index(value, char_name)
            else:
                idx = 1  # PAD token
            
            # Get embedding
            with torch.no_grad():
                emb = embedding_layer(torch.tensor([idx]))
                embeddings.append(emb.squeeze().numpy())
        
        if embeddings:
            return np.concatenate(embeddings)
        else:
            # Return zero vector
            return np.zeros(self.get_embedding_dim())
    
    def encode_batch(self, characteristics_list: List[Dict]) -> np.ndarray:
        """Encode multiple characteristics dictionaries"""
        return np.array([self.encode(chars) for chars in characteristics_list])
    
    def get_embedding_dim(self) -> int:
        """Get total dimension of characteristics embeddings"""
        return sum(config['embed_dim'] for config in self.config.values())
    
    def get_characteristic_importance(self, characteristics: Dict) -> Dict[str, float]:
        """
        Compute importance scores for each characteristic
        Based on how specific/rare the value is
        """
        importance = {}
        
        for char_name, value in characteristics.items():
            if char_name in self.vocabularies:
                norm_value = self._normalize_value(str(value), char_name)
                vocab_size = len(self.vocabularies[char_name])
                
                # Rarer values are more important
                importance[char_name] = 1.0 / max(vocab_size, 1)
        
        return importance