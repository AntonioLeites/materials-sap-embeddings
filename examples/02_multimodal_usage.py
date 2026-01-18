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
import numpy as np


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
    
    # 5. Compare two materials
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
    
    # 7. Understanding component contributions
    print("\n7. Understanding component contributions...")
    
    # Show which components contributed most
    print("\n   Component importance:")
    components_only = {k: v for k, v in explanation.items() if k != 'overall'}
    for component, score in sorted(components_only.items(), key=lambda x: x[1], reverse=True):
        contribution = (score / sum(components_only.values())) * 100
        print(f"   {component:20s} {score:.4f} ({contribution:5.1f}% of total)")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Multimodal embeddings combine multiple information sources:")
    print("   - Text: Semantic meaning of description")
    print("   - Categorical: Business classification (MaterialGroup, Type)")
    print("   - Characteristics: Technical specifications (size, material)")
    print("   - Relational: Usage context (plants, suppliers)")
    print()
    print("2. Different components contribute differently to similarity:")
    most_important = max(components_only.items(), key=lambda x: x[1])
    least_important = min(components_only.items(), key=lambda x: x[1])
    print(f"   - Most important: {most_important[0]} ({most_important[1]:.3f})")
    print(f"   - Least important: {least_important[0]} ({least_important[1]:.3f})")
    print()
    print("3. This enables Tensor Logic reasoning:")
    print("   Materials can be similar even with different descriptions")
    print("   if they share plants, suppliers, or characteristics.")
    print("=" * 70)
    
    print("\n✓ Example completed successfully")


if __name__ == "__main__":
    main()