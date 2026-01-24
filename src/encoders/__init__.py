"""
Encoders for different feature types
"""

from .categorical_encoder import CategoricalEncoder
from .characteristics_encoder import CharacteristicsEncoder
from .relational_encoder import RelationalEncoder

__all__ = [
    'CategoricalEncoder',
    'CharacteristicsEncoder',
    'RelationalEncoder'
]

# SAP RPT-1 encoder (optional dependency)
try:
    from .sap_rpt1_encoder import SAPRPT1Encoder
    __all__.append('SAPRPT1Encoder')
except ImportError:
    # RPT-1 not installed, skip
    pass