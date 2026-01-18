"""
Relational context encoder (plants, suppliers, usage patterns)
"""

import numpy as np
from typing import Dict, List, Set
from collections import Counter


class RelationalEncoder:
    """
    Encode relational context for materials
    - Which plants use this material
    - Which suppliers provide it
    - Usage frequency and patterns
    """
    
    def __init__(self, embed_dim: int = 128):
        """
        Initialize relational encoder
        
        Args:
            embed_dim: Output embedding dimension
        """
        self.embed_dim = embed_dim
        
        # Track known entities
        self.known_plants = set()
        self.known_suppliers = set()
        
        # Statistics
        self.plant_frequencies = Counter()
        self.supplier_frequencies = Counter()
    
    def update_knowledge(self, material_data: Dict):
        """
        Update encoder's knowledge with new material data
        
        Args:
            material_data: Material data including plants, suppliers
        """
        if 'plants' in material_data:
            plants = material_data['plants']
            if isinstance(plants, (list, set)):
                self.known_plants.update(plants)
                self.plant_frequencies.update(plants)
        
        if 'suppliers' in material_data:
            suppliers = material_data['suppliers']
            if isinstance(suppliers, (list, set)):
                self.known_suppliers.update(suppliers)
                self.supplier_frequencies.update(suppliers)
    
    def encode(self, material_data: Dict) -> np.ndarray:
        """
        Encode relational context
        
        Args:
            material_data: Dictionary with:
                - plants: List of plant codes
                - suppliers: List of supplier codes
                - usage_frequency: Usage count or frequency
                - po_count: Number of purchase orders
                - last_po_days: Days since last PO
        
        Returns:
            Vector encoding relational context
        """
        features = []
        
        # 1. Plant-based features
        plants = material_data.get('plants', [])
        if isinstance(plants, (list, set)):
            features.extend([
                len(plants),  # Number of plants
                len(plants) / max(len(self.known_plants), 1),  # Plant coverage
                self._compute_diversity(plants, self.plant_frequencies)  # Plant diversity
            ])
        else:
            features.extend([0, 0, 0])
        
        # 2. Supplier-based features
        suppliers = material_data.get('suppliers', [])
        if isinstance(suppliers, (list, set)):
            features.extend([
                len(suppliers),  # Number of suppliers
                len(suppliers) / max(len(self.known_suppliers), 1),  # Supplier coverage
                self._compute_diversity(suppliers, self.supplier_frequencies)  # Supplier diversity
            ])
        else:
            features.extend([0, 0, 0])
        
        # 3. Usage patterns
        features.extend([
            material_data.get('usage_frequency', 0),
            material_data.get('po_count', 0),
            material_data.get('last_po_days', 365) / 365.0,  # Normalize to [0,1]
        ])
        
        # 4. Co-occurrence features (if available)
        features.extend([
            material_data.get('plant_supplier_overlap', 0),  # How many plant-supplier pairs
        ])
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Pad or truncate to embed_dim
        if len(features_array) < self.embed_dim:
            # Pad with zeros
            features_array = np.pad(
                features_array,
                (0, self.embed_dim - len(features_array)),
                mode='constant'
            )
        else:
            # Truncate
            features_array = features_array[:self.embed_dim]
        
        return features_array
    
    def _compute_diversity(
        self,
        entities: List[str],
        frequencies: Counter
    ) -> float:
        """
        Compute diversity score for entities
        Higher score = more diverse/rare entities
        """
        if not entities:
            return 0.0
        
        # Compute inverse frequency scores
        scores = []
        total_count = sum(frequencies.values())
        
        for entity in entities:
            freq = frequencies.get(entity, 0)
            if total_count > 0:
                # Inverse frequency: rare entities have higher scores
                score = 1.0 - (freq / total_count)
            else:
                score = 1.0
            scores.append(score)
        
        # Average diversity
        return np.mean(scores) if scores else 0.0
    
    def encode_batch(self, materials_data: List[Dict]) -> np.ndarray:
        """Encode multiple materials"""
        return np.array([self.encode(mat) for mat in materials_data])
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embed_dim
    
    def get_relational_similarity(
        self,
        material1_data: Dict,
        material2_data: Dict
    ) -> float:
        """
        Compute relational similarity between two materials
        Based on shared plants, suppliers, etc.
        """
        plants1 = set(material1_data.get('plants', []))
        plants2 = set(material2_data.get('plants', []))
        
        suppliers1 = set(material1_data.get('suppliers', []))
        suppliers2 = set(material2_data.get('suppliers', []))
        
        # Jaccard similarity for plants
        plant_sim = self._jaccard_similarity(plants1, plants2)
        
        # Jaccard similarity for suppliers
        supplier_sim = self._jaccard_similarity(suppliers1, suppliers2)
        
        # Combined similarity
        return (plant_sim + supplier_sim) / 2.0
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0