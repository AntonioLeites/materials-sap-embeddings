"""
Multimodal duplicate detection with visualizations
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Multimodal vs Text-Only Duplicate Detection with Visualizations")
print("=" * 70)

try:
    # Imports
    print("\nInitializing...")
    from src.embeddings.multimodal_embeddings import MultimodalMaterialEmbeddings
    from src.embeddings.text_embeddings import MaterialEmbeddings
    from src.sap_connector import create_sample_materials, create_duplicate_pairs
    from src.similarity import DuplicateDetector
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 1. Generate data
    print("\n1. Generating materials with duplicates...")
    base_materials = create_sample_materials(n_materials=30)
    all_materials = create_duplicate_pairs(base_materials, n_duplicates=10)
    print(f"   Total: {len(all_materials)} ({len(base_materials)} base + {len(all_materials)-len(base_materials)} duplicates)")
    
    # 2. Text embeddings
    print("\n2. Text-only embeddings...")
    text_embedder = MaterialEmbeddings()
    text_descriptions = [m['MAKTX'] for m in all_materials]
    text_embeddings = text_embedder.encode_batch(text_descriptions, show_progress=False)
    print(f"   ✓ Generated {text_embeddings.shape}")
    
    # 3. Multimodal embeddings
    print("\n3. Multimodal embeddings...")
    multimodal_embedder = MultimodalMaterialEmbeddings()
    multimodal_embedder.update_relational_knowledge(all_materials)
    multimodal_embeddings = multimodal_embedder.encode_batch(all_materials, show_progress=False)
    print(f"   ✓ Generated {multimodal_embeddings.shape}")
    
    # 4. Detect duplicates
    print("\n4. Detecting duplicates (threshold=0.85)...")
    detector = DuplicateDetector(threshold=0.85)
    
    text_dups = detector.find_duplicates(text_descriptions, embeddings=text_embeddings)
    multi_dups = detector.find_duplicates(text_descriptions, embeddings=multimodal_embeddings)
    
    print(f"\n   Text-only:  {len(text_dups)} pairs")
    print(f"   Multimodal: {len(multi_dups)} pairs")
    
    improvement = len(multi_dups) - len(text_dups)
    if improvement > 0:
        pct = (improvement / max(len(text_dups), 1)) * 100
        print(f"   ✓ Improvement: +{improvement} pairs (+{pct:.0f}%)")
    
    # 5. Show examples
    if multi_dups:
        print("\n5. Top 3 duplicate pairs:")
        for i, (idx1, idx2, sim) in enumerate(multi_dups[:3], 1):
            mat1, mat2 = all_materials[idx1], all_materials[idx2]
            print(f"\n   Pair {i} (sim={sim:.3f}):")
            print(f"   - {mat1['MAKTX']}")
            print(f"   - {mat2['MAKTX']}")
    
    # 6. Visualizations
    print("\n6. Creating visualizations...")
    
    labels = ['Original'] * len(base_materials) + ['Duplicate'] * (len(all_materials) - len(base_materials))
    
    # VIZ 1: 2D Embeddings Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    pca = PCA(n_components=2, random_state=42)
    text_2d = pca.fit_transform(text_embeddings)
    multi_2d = pca.fit_transform(multimodal_embeddings)
    
    for label, color in [('Original', 'blue'), ('Duplicate', 'red')]:
        mask = np.array(labels) == label
        ax1.scatter(text_2d[mask, 0], text_2d[mask, 1], 
                   c=color, label=label, alpha=0.6, s=100, edgecolors='black')
        ax2.scatter(multi_2d[mask, 0], multi_2d[mask, 1], 
                   c=color, label=label, alpha=0.6, s=100, edgecolors='black')
    
    ax1.set_title('Text-Only Embeddings', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Multimodal Embeddings', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embeddings_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ embeddings_comparison.png")
    plt.close()
    
    # VIZ 2: Similarity Matrices
    n = min(15, len(all_materials))
    text_sim = cosine_similarity(text_embeddings[:n])
    multi_sim = cosine_similarity(multimodal_embeddings[:n])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = ax1.imshow(text_sim, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title('Text-Only Similarity', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(multi_sim, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Multimodal Similarity', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('similarity_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✓ similarity_matrices.png")
    plt.close()
    
    # VIZ 3: Bar Chart Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(['Text-Only', 'Multimodal'], 
                  [len(text_dups), len(multi_dups)],
                  color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    
    if improvement > 0:
        ax.annotate(f'+{improvement}\n(+{pct:.0f}%)',
                   xy=(1, len(multi_dups)), xytext=(1.3, len(multi_dups)*0.8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12, color='green', fontweight='bold')
    
    ax.set_ylabel('Duplicate Pairs Found', fontsize=12)
    ax.set_title('Duplicate Detection Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('duplicate_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ duplicate_comparison.png")
    plt.close()
    
    # VIZ 4: Component Contributions
    if multi_dups:
        components = ['text', 'categorical', 'characteristics', 'relational']
        scores = {comp: [] for comp in components}
        
        for idx1, idx2, _ in multi_dups[:10]:
            exp = multimodal_embedder.explain_similarity(all_materials[idx1], all_materials[idx2])
            for comp in components:
                scores[comp].append(exp.get(comp, 0))
        
        avg_scores = [np.mean(scores[comp]) for comp in components]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(components, avg_scores, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, score in zip(bars, avg_scores):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                   f'{score:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Average Similarity', fontsize=12)
        ax.set_title('Component Contribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('component_contribution.png', dpi=300, bbox_inches='tight')
        print("   ✓ component_contribution.png")
        plt.close()
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nMultimodal found {improvement} more duplicates (+{pct:.0f}%)")
    print("\nGenerated visualizations:")
    print("  • embeddings_comparison.png")
    print("  • similarity_matrices.png")
    print("  • duplicate_comparison.png")
    print("  • component_contribution.png")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()