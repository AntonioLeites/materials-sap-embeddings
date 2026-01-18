"""
Material Embeddings Package

Provides text-only and multimodal embeddings for SAP materials
"""

from .text_embeddings import MaterialEmbeddings
from .multimodal_embeddings import MultimodalMaterialEmbeddings

__all__ = [
    'MaterialEmbeddings',
    'MultimodalMaterialEmbeddings'
]