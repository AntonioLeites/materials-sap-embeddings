"""
Multimodal embeddings usage example
Demonstrates the power of combining multiple features
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.multimodal_embeddings import MultimodalMaterialEmbeddings
from src.sap_connector import create_sample_materials, print_material_summary


def main():
    print("=" * 70)
    print("Multimodal Material Embeddings - Complete Example")
    print("=" * 70)
    
    # 1. Generate sample data
    print("\n1. Generating sample materials with full context...")
    materials = create_sample_materials(n_materials=10)
    
    print(f"   Created {len(materials)} materials")
    print("\n   Example material:")
    print_material_summary(materials[0])
    
    # 2. Initialize multimodal embedder
    print("\n2. Initializing Multimodal Embeddings...")
    embedder = MultimodalMaterialEmbeddings()
    
    # 3. Update relational knowledge
    print("\n3. Building relational knowledge base...")
    embedder.update_relational_knowledge(materials)
    
    # 4. Generate multimodal embedding
    print("\n4. Generating multimodal embedding...")
    embedding = embedder.encode_multimodal(materials[0])
    
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 dimensions: {embedding[:5]}")
    
    # 5. Compare two similar materials
    print("\n5. Comparing two materials...")
    print("\n   Material A:")
    print(f"   {materials[0]['MAKTX']}")
    print(f"   Plants: {materials[0]['plants'][:2]}")
    print(f"   Suppliers: {materials[0]['suppliers'][:2]}")
    
    print("\n   Material B:")
    print(f"   {materials[1]['MAKTX']}")
    print(f"   Plants: {materials[1]['plants'][:2]}")
    print(f"   Suppliers: {materials[1]['suppliers'][:2]}")
    
    # Overall similarity
    similarity = embedder.similarity(materials[0], materials[1])
    print(f"\n   Overall Similarity: {similarity:.4f}")
    
    # 6. Explain similarity breakdown
    print("\n6. Similarity breakdown by component:")
    explanation = embedder.explain_similarity(materials[0], materials[1])
    
    for component, score in explanation.items():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {component:20s} {score:.4f} {bar}")
    
    # 7. Compare with different modalities
    print("\n7. Testing different feature combinations...")
    
    # Text only
    emb_text = embedder.encode_multimodal(
        materials[0],
        include_categorical=False,
        include_characteristics=False,
        include_relational=False
    )
    emb_text2 = embedder.encode_multimodal(
        materials[1],
        include_categorical=False,
        include_characteristics=False,
        include_relational=False
    )
    sim_text = float(np.dot(emb_text, emb_text2))
    print(f"   Text only:              {sim_text:.4f}")
    
    # Text + Categorical
    emb_cat = embedder.encode_multimodal(
        materials[0],
        include_characteristics=False,
        include_relational=False
    )
    emb_cat2 = embedder.encode_multimodal(
        materials[1],
        include_characteristics=False,
        include_relational=False
    )
    sim_cat = float(np.dot(emb_cat, emb_cat2))
    print(f"   Text + Categorical:     {sim_cat:.4f}")
    
    # All features
    print(f"   All features (multimodal): {similarity:.4f}")
    
    print("\n" + "=" * 70)
    print("Key Insight:")
    print("Multimodal embeddings capture MORE than just text similarity.")
    print("They understand:")
    print("  - Semantic meaning (from text)")
    print("  - Business context (from categories)")
    print("  - Technical specs (from characteristics)")
    print("  - Usage patterns (from relations)")
    print("=" * 70)
    
    print("\n✓ Example completed successfully")


if __name__ == "__main__":
    import numpy as np
    main()