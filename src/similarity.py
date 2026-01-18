"""
Similarity computation and duplicate detection utilities
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import pandas as pd


class DuplicateDetector:
    """
    Detect duplicate materials using embedding similarity
    """
    
    def __init__(self, threshold: float = 0.85):
        """
        Initialize duplicate detector
        
        Args:
            threshold: Similarity threshold for duplicates (0-1)
        """
        self.threshold = threshold
    
    def find_duplicates(
        self, 
        materials: List[str],
        embeddings: np.ndarray = None,
        rpt1_model = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find potential duplicates in material list
        
        Args:
            materials: List of material descriptions
            embeddings: Pre-computed embeddings (optional)
            rpt1_model: RPT1Embeddings instance (if embeddings not provided)
            
        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        # Generate embeddings if not provided
        if embeddings is None:
            if rpt1_model is None:
                raise ValueError("Must provide either embeddings or rpt1_model")
            embeddings = rpt1_model.encode_batch(materials)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Find pairs above threshold
        duplicates = []
        n = len(materials)
        
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    duplicates.append((i, j, sim_matrix[i, j]))
        
        # Sort by similarity (descending)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        return duplicates
    
    def create_duplicate_report(
        self,
        materials: List[str],
        duplicates: List[Tuple[int, int, float]]
    ) -> pd.DataFrame:
        """
        Create a readable duplicate report
        
        Args:
            materials: List of material descriptions
            duplicates: Output from find_duplicates()
            
        Returns:
            pandas DataFrame with duplicate pairs
        """
        report = []
        
        for idx1, idx2, similarity in duplicates:
            report.append({
                'Material_1': materials[idx1],
                'Material_2': materials[idx2],
                'Similarity': f"{similarity:.4f}",
                'Index_1': idx1,
                'Index_2': idx2
            })
        
        return pd.DataFrame(report)


class SimilaritySearch:
    """
    Find similar materials using embedding search
    """
    
    def __init__(self, materials: List[str], embeddings: np.ndarray):
        """
        Initialize similarity search index
        
        Args:
            materials: List of material descriptions
            embeddings: Corresponding embeddings
        """
        self.materials = materials
        self.embeddings = embeddings
    
    def search(
        self, 
        query: str, 
        rpt1_model,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find most similar materials to query
        
        Args:
            query: Query material description
            rpt1_model: RPT1Embeddings instance
            top_k: Number of results to return
            
        Returns:
            List of dicts with material and similarity
        """
        # Encode query
        query_emb = rpt1_model.encode(query, normalize=True)
        
        # Compute similarities
        normalized_embs = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        similarities = np.dot(normalized_embs, query_emb)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'material': self.materials[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            })
        
        return results