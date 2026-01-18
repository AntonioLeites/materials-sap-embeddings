"""
Basic usage example for RPT-1 embeddings
"""

from src.embeddings import RPT1Embeddings


def main():
    print("=" * 60)
    print("RPT-1 Embeddings - Basic Usage Example")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing RPT-1...")
    rpt1 = RPT1Embeddings()
    
    # Single embedding
    print("\n2. Generating embedding for a material...")
    material = "Steel Bolt M8x50 DIN 933"
    embedding = rpt1.encode(material)
    
    print(f"   Material: {material}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 dimensions: {embedding[:5]}")
    
    # Compare two materials
    print("\n3. Comparing two materials...")
    material_a = "Steel Bolt M8x50 DIN 933"
    material_b = "Stainless Steel Bolt M8x50 ISO 4017"
    
    similarity = rpt1.similarity(material_a, material_b)
    
    print(f"   Material A: {material_a}")
    print(f"   Material B: {material_b}")
    print(f"   Similarity: {similarity:.4f}")
    
    # Batch encoding
    print("\n4. Encoding multiple materials...")
    materials = [
        "Steel Bolt M8x50",
        "Plastic Washer M8",
        "Stainless Bolt M10x60"
    ]
    
    embeddings = rpt1.encode_batch(materials, show_progress=False)
    
    print(f"   Encoded {len(materials)} materials")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    print("\n✓ Example completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()