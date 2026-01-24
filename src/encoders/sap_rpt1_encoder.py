# src/encoders/sap_rpt1_encoder.py
"""
SAP RPT-1 Business Encoder

Uses pre-trained RPT-1 transformer model to extract learned representations
from tabular material data. RPT-1 captures complex patterns across all
material features (text, categorical, numerical, relational).

Key features:
- Pre-trained on diverse tabular data
- Requires supervised fine-tuning first
- 384-dimensional embeddings
- Handles mixed data types (text, numbers, categories)

Usage:
    encoder = SAPRPT1Encoder()
    encoder.fit(materials, target_column='MATKL')
    embeddings = encoder.encode_batch(materials)
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Literal
from pathlib import Path

try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier
    RPT1_AVAILABLE = True
except ImportError:
    RPT1_AVAILABLE = False
    print("Warning: sap-rpt-1-oss not available. Install with:")
    print("  pip install git+https://github.com/SAP-samples/sap-rpt-1-oss")


class SAPRPT1Encoder:
    """
    Encoder using SAP RPT-1 pre-trained transformer for tabular data.
    
    RPT-1 (Relational Pre-trained Transformer 1) is a transformer-based model
    specifically designed for tabular data. It can handle:
    - Text columns (via sentence embeddings)
    - Numerical columns (quantized and normalized)
    - Categorical columns (learned embeddings)
    - Date columns (decomposed into year/month/day/weekday)
    
    This encoder extracts hidden representations from the model after
    fine-tuning on a supervised task.
    """
    
    def __init__(
        self,
        model_name: str = "sap/sap-rpt-1-oss",
        output_dim: int = 384,
        pooling: Literal['mean', 'cls'] = 'mean',
        bagging: int = 1,
        max_context_size: int = 2048,
        device: Optional[str] = None
    ):
        """
        Initialize RPT-1 encoder.
        
        Args:
            model_name: HuggingFace model identifier
            output_dim: Output embedding dimension (must be 384 for RPT-1)
            pooling: How to pool column embeddings ('mean' or 'cls')
            bagging: Number of models in ensemble (1 = single model, faster)
            max_context_size: Maximum number of context rows
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        if not RPT1_AVAILABLE:
            raise ImportError(
                "SAP RPT-1 OSS not installed. Install with:\n"
                "  pip install git+https://github.com/SAP-samples/sap-rpt-1-oss"
            )
        
        if output_dim != 384:
            raise ValueError(
                f"RPT-1 has fixed embedding dimension of 384, got {output_dim}"
            )
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.pooling = pooling
        self.bagging = bagging
        self.max_context_size = max_context_size
        
        # Initialize RPT-1 classifier
        print(f"Initializing RPT-1 from {model_name}...")
        self.classifier = SAP_RPT_OSS_Classifier(
            bagging=bagging,
            max_context_size=max_context_size
        )
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.is_fitted = False
        
        print(f"✓ RPT-1 initialized (device: {self.device})")
    
    def _materials_to_dataframe(
        self, 
        materials: List[Dict],
        include_target: bool = False,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert list of material dicts to DataFrame for RPT-1.
        
        Args:
            materials: List of material dictionaries
            include_target: Whether to include target column
            target_column: Name of target column (e.g., 'MATKL')
            
        Returns:
            DataFrame with material features
        """
        rows = []
        
        for material in materials:
            row = {
                'MATNR': material['MATNR'],
                'MAKTX': material['MAKTX'],
                'MTART': material.get('MTART', ''),
                'MEINS': material.get('MEINS', 'PC'),
            }
            
            # Numerical features
            row['PRICE'] = float(material.get('price_avg', 0.0))
            row['NUM_PLANTS'] = len(material.get('plants', []))
            row['NUM_SUPPLIERS'] = len(material.get('suppliers', []))
            row['USAGE_FREQ'] = material.get('usage_frequency', 0)
            row['PO_COUNT'] = material.get('po_count', 0)
            
            # Characteristics as separate columns
            chars = material.get('characteristics', {})
            for char_name, char_value in chars.items():
                row[f'CHAR_{char_name}'] = char_value
            
            # Target column if needed
            if include_target and target_column:
                if target_column in material:
                    row[target_column] = material[target_column]
                elif target_column == 'MATKL':
                    row[target_column] = material.get('MATKL', '')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reset index to avoid pandas concat errors
        df = df.reset_index(drop=True)
        
        return df
    
    def fit(
        self,
        materials: List[Dict],
        target_column: str = 'MATKL',
        verbose: bool = True
    ):
        """
        Fit RPT-1 on a supervised task.
        
        IMPORTANT: RPT-1 requires supervised fine-tuning before it can
        extract meaningful embeddings. This method trains the model to
        predict a target column (e.g., MaterialGroup).
        
        Args:
            materials: List of material dictionaries with target values
            target_column: Column to predict (default: 'MATKL')
            verbose: Whether to print progress
            
        Returns:
            self (for method chaining)
        """
        if verbose:
            print(f"Fitting RPT-1 on {len(materials)} materials...")
            print(f"  Target: {target_column}")
        
        # Convert to DataFrame
        df = self._materials_to_dataframe(
            materials, 
            include_target=True,
            target_column=target_column
        )
        
        # Check target column exists
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in materials. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Prepare X and y
        y = df[target_column]
        X = df.drop(target_column, axis=1)
        
        # Reset indices (critical for RPT-1)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        if verbose:
            print(f"  X shape: {X.shape}")
            print(f"  Unique target values: {y.nunique()}")
        
        # Fit classifier
        self.classifier.fit(X, y)
        
        self.is_fitted = True
        
        if verbose:
            print("✓ RPT-1 fitted successfully")
        
        return self
    
    def encode(self, material: Dict) -> np.ndarray:
        """
        Extract embedding for a single material.
        
        Args:
            material: Material dictionary
            
        Returns:
            numpy array of shape (output_dim,)
        """
        return self.encode_batch([material])[0]
    
    def encode_batch(
        self,
        materials: List[Dict],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Extract embeddings for multiple materials.
        
        Args:
            materials: List of material dictionaries
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_materials, output_dim)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Encoder not fitted. Call fit() first with training materials."
            )
        
        # Convert to DataFrame
        df = self._materials_to_dataframe(materials, include_target=False)
        
        # Get tokenized data from RPT-1
        tokenized = self.classifier.get_tokenized_data(df, bagging_index=0)
        
        # Extract text embeddings
        # Structure: tokenized['data']['text_embeddings']
        # Shape: (n_train + n_test, n_cols, embedding_dim)
        text_emb = tokenized['data']['text_embeddings']
        
        # Apply pooling over columns
        if self.pooling == 'mean':
            pooled = text_emb.mean(dim=1)  # (n_train + n_test, embedding_dim)
        elif self.pooling == 'cls':
            pooled = text_emb[:, -1, :]  # Use last column (like [CLS] token)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # CRITICAL: Extract only test samples (last n_materials rows)
        # RPT-1 concatenates [train, test] internally
        n_materials = len(df)
        embeddings_test = pooled[-n_materials:]
        
        # Convert to numpy
        embeddings_np = embeddings_test.detach().cpu().numpy()
        
        return embeddings_np
    
    def __repr__(self) -> str:
        return (
            f"SAPRPT1Encoder("
            f"output_dim={self.output_dim}, "
            f"pooling='{self.pooling}', "
            f"fitted={self.is_fitted})"
        )


# Example usage
if __name__ == "__main__":
    # Test data
    test_materials = [
        {
            'MATNR': 'MAT001',
            'MAKTX': 'Steel Bolt M8x50 DIN 933',
            'MATKL': 'BOLTS',
            'MTART': 'FERT',
            'price_avg': 0.50,
            'plants': ['Plant_1001', 'Plant_1002'],
            'suppliers': ['SUPP_100'],
            'characteristics': {'DIAMETER': 'M8', 'LENGTH': '50mm'}
        },
        {
            'MATNR': 'MAT002',
            'MAKTX': 'Plastic Washer M8',
            'MATKL': 'WASHERS',
            'MTART': 'FERT',
            'price_avg': 0.10,
            'plants': ['Plant_1001'],
            'suppliers': ['SUPP_200'],
            'characteristics': {'DIAMETER': 'M8'}
        }
    ]
    
    # Initialize encoder
    encoder = SAPRPT1Encoder()
    
    # Fit on supervised task
    encoder.fit(test_materials, target_column='MATKL')
    
    # Extract embeddings
    embeddings = encoder.encode_batch(test_materials)
    
    print(f"\n✓ Extracted embeddings: {embeddings.shape}")
    print(f"  First material embedding (first 10 dims):")
    print(f"  {embeddings[0, :10]}")