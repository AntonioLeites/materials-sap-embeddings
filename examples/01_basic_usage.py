"""
Basic usage example for Material Embeddings
Demonstrates text-only embeddings generation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.text_embeddings import MaterialEmbeddings
from src.sap_connector import create_sample_materials


def main():
    print("=" * 60)
    print("Material Embeddings - Basic Usage Example")
    print("=" * 60)
    
    # 1. Initialize
    print("\n1. Initializing Material Embeddings...")
    embedder = MaterialEmbeddings()
    
    # 2. Single embedding
    print("\n2. Generating embedding for a material...")
    material = "Steel Bolt M8x50 DIN 933"
    embedding = embedder.encode(material)
    
    print(f"   Material: {material}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 dimensions: {embedding[:5]}")
    
    # 3. Compare two materials
    print("\n3. Comparing two materials...")
    material_a = "Steel Bolt M8x50 DIN 933"
    material_b = "Stainless Steel Bolt M8x50 ISO 4017"
    
    similarity = embedder.similarity(material_a, material_b)
    
    print(f"   Material A: {material_a}")
    print(f"   Material B: {material_b}")
    print(f"   Similarity: {similarity:.4f}")
    
    # 4. Batch encoding
    print("\n4. Encoding multiple materials...")
    materials_data = create_sample_materials(n_materials=5)
    materials = [m['MAKTX'] for m in materials_data]
    
    embeddings = embedder.encode_batch(materials, show_progress=False)
    
    print(f"   Encoded {len(materials)} materials")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    print("\n✓ Example completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()