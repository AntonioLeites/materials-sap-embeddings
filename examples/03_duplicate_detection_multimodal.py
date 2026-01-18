"""
Multimodal duplicate detection with visualizations
Shows how multimodal embeddings improve duplicate detection
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.multimodal_embeddings import MultimodalMaterialEmbeddings
from src.embeddings.text_embeddings import MaterialEmbeddings
from src.sap_connector import create_sample_materials, create_duplicate_pairs
from src.similarity import DuplicateDetector
from src.visualization import (
    plot_embeddings_2d,
    plot_similarity_matrix,
    plot_duplicate_comparison
)
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("Multimodal vs Text-Only Duplicate Detection")
    print("=" * 70)
    
    # 1. Generate materials with duplicates
    print("\n1. Generating materials with intentional duplicates...")
    base_materials = create_sample_materials(n_materials=30)
    all_materials = create_duplicate_pairs(base_materials, n_duplicates=10)
    
    print(f"   Total materials: {len(all_materials)}")
    print(f"   Base materials: {len(base_materials)}")
    print(f"   Duplicates added: {len(all_materials) - len(base_materials)}")
    
    # 2. Text-only embeddings
    print("\n2. Generating text-only embeddings...")
    text_embedder = MaterialEmbeddings()
    text_descriptions = [m['MAKTX'] for m in all_materials]
    text_embeddings = text_embedder.encode_batch(text_descriptions, show_progress=False)
    
    # 3. Multimodal embeddings
    print("\n3. Generating multimodal embeddings...")
    multimodal_embedder = MultimodalMaterialEmbeddings()
    multimodal_embedder.update_relational_knowledge(all_materials)
    multimodal_embeddings = multimodal_embedder.encode_batch(all_materials, show_progress=True)
    
    # 4. Detect duplicates with both methods
    print("\n4. Detecting duplicates...")
    
    threshold = 0.85
    detector = DuplicateDetector(threshold=threshold)
    
    text_duplicates = detector.find_duplicates(
        text_descriptions,
        embeddings=text_embeddings
    )
    
    multimodal_duplicates = detector.find_duplicates(
        text_descriptions,
        embeddings=multimodal_embeddings
    )
    
    # 5. Compare results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\nThreshold: {threshold:.2f}")
    print(f"\nText-only approach:")
    print(f"  Duplicates found: {len(text_duplicates)}")
    
    print(f"\nMultimodal approach:")
    print(f"  Duplicates found: {len(multimodal_duplicates)}")
    
    # Show improvement
    improvement = len(multimodal_duplicates) - len(text_duplicates)
    if improvement > 0:
        print(f"\n✓ Multimodal found {improvement} MORE duplicates (+{improvement/max(len(text_duplicates),1)*100:.1f}%)")
    elif improvement < 0:
        print(f"\n⚠️ Multimodal found {abs(improvement)} FEWER duplicates ({improvement/len(text_duplicates)*100:.1f}%)")
    else:
        print(f"\n= Both methods found the same number")
    
    # 6. Show examples
    if multimodal_duplicates:
        print("\n" + "=" * 70)
        print("Example duplicate pairs found by multimodal:")
        print("=" * 70)
        
        for i, (idx1, idx2, sim) in enumerate(multimodal_duplicates[:3], 1):
            mat1 = all_materials[idx1]
            mat2 = all_materials[idx2]
            
            print(f"\nPair {i} (similarity: {sim:.4f}):")
            print(f"  Material 1: {mat1['MAKTX']}")
            print(f"    Group: {mat1['MATKL']}, Plants: {len(mat1['plants'])}, Suppliers: {len(mat1['suppliers'])}")
            print(f"  Material 2: {mat2['MAKTX']}")
            print(f"    Group: {mat2['MATKL']}, Plants: {len(mat2['plants'])}, Suppliers: {len(mat2['suppliers'])}")
            
            # Show why multimodal found it
            explanation = multimodal_embedder.explain_similarity(mat1, mat2)
            print(f"  Breakdown: Text={explanation['text']:.2f}, Cat={explanation['categorical']:.2f}, "
                  f"Char={explanation.get('characteristics', 0):.2f}, Rel={explanation['relational']:.2f}")
    
    # 7. Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Create labels for duplicates
    labels = ['Original'] * len(base_materials) + ['Duplicate'] * (len(all_materials) - len(base_materials))
    
    # Visualization 1: 2D projection comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Text-only projection
    from sklearn.decomposition import PCA
    pca_text = PCA(n_components=2, random_state=42)
    text_2d = pca_text.fit_transform(text_embeddings)
    
    for i, label in enumerate(set(labels)):
        mask = np.array(labels) == label
        color = 'blue' if label == 'Original' else 'red'
        ax1.scatter(text_2d[mask, 0], text_2d[mask, 1], 
                   c=color, label=label, alpha=0.6, s=100)
    
    ax1.set_title('Text-Only Embeddings (PCA)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Multimodal projection
    pca_multi = PCA(n_components=2, random_state=42)
    multi_2d = pca_multi.fit_transform(multimodal_embeddings)
    
    for i, label in enumerate(set(labels)):
        mask = np.array(labels) == label
        color = 'blue' if label == 'Original' else 'red'
        ax2.scatter(multi_2d[mask, 0], multi_2d[mask, 1], 
                   c=color, label=label, alpha=0.6, s=100)
    
    ax2.set_title('Multimodal Embeddings (PCA)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embeddings_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: embeddings_comparison.png")
    
    # Visualization 2: Similarity matrix comparison
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Select subset for visualization (first 15 materials)
    subset_size = 15
    text_sim_matrix = cosine_similarity(text_embeddings[:subset_size])
    multi_sim_matrix = cosine_similarity(multimodal_embeddings[:subset_size])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Text similarity matrix
    im1 = ax1.imshow(text_sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title('Text-Only Similarity Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Material Index')
    ax1.set_ylabel('Material Index')
    plt.colorbar(im1, ax=ax1, label='Similarity')
    
    # Multimodal similarity matrix
    im2 = ax2.imshow(multi_sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Multimodal Similarity Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Material Index')
    ax2.set_ylabel('Material Index')
    plt.colorbar(im2, ax=ax2, label='Similarity')
    
    plt.tight_layout()
    plt.savefig('similarity_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: similarity_matrices.png")
    
    # Visualization 3: Duplicate detection comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Text-Only', 'Multimodal']
    counts = [len(text_duplicates), len(multimodal_duplicates)]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Number of Duplicate Pairs Found', fontsize=12)
    ax.set_title(f'Duplicate Detection Comparison (threshold={threshold})', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotation
    if improvement > 0:
        ax.annotate(f'+{improvement} duplicates\n(+{improvement/max(counts[0],1)*100:.0f}%)',
                   xy=(1, counts[1]), xytext=(1.3, counts[1]*0.8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('duplicate_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: duplicate_comparison.png")
    
    # Visualization 4: Component contribution
    if multimodal_duplicates:
        fig, ax = plt