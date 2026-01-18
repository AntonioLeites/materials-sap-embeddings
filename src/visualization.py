"""
Visualization utilities for embeddings and similarity analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import pandas as pd


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "pca",
    title: str = "Embedding Visualization",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
):
    """
    Reduce embeddings to 2D and plot
    
    Args:
        embeddings: NxD embedding matrix
        labels: Optional labels for points
        method: 'tsne' or 'pca'
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=figsize)
    
    if labels:
        # Color by label if provided
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.legend(fontsize=10)
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(f"{title} ({method.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Similarity Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "YlOrRd",
    save_path: Optional[str] = None
):
    """
    Plot a similarity matrix as a heatmap
    
    Args:
        similarity_matrix: NxN similarity matrix
        labels: Optional labels for materials
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        similarity_matrix,
        annot=True if len(similarity_matrix) <= 10 else False,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels if labels and len(labels) <= 20 else False,
        yticklabels=labels if labels and len(labels) <= 20 else False,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Similarity'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_duplicate_comparison(
    text_count: int,
    multimodal_count: int,
    threshold: float = 0.85,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot comparison of duplicate detection methods
    
    Args:
        text_count: Number of duplicates found by text-only
        multimodal_count: Number found by multimodal
        threshold: Similarity threshold used
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = ['Text-Only', 'Multimodal']
    counts = [text_count, multimodal_count]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, counts, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    
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
    improvement = multimodal_count - text_count
    if improvement > 0:
        ax.annotate(f'+{improvement} duplicates\n(+{improvement/max(text_count,1)*100:.0f}%)',
                   xy=(1, multimodal_count), 
                   xytext=(1.3, multimodal_count*0.8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_component_breakdown(
    component_scores: dict,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot breakdown of similarity by component
    
    Args:
        component_scores: Dict with component names and scores
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove 'overall' from components
    components = {k: v for k, v in component_scores.items() if k != 'overall'}
    
    colors = {
        'text': '#3498db',
        'categorical': '#e74c3c',
        'characteristics': '#2ecc71',
        'relational': '#f39c12'
    }
    
    names = list(components.keys())
    values = list(components.values())
    bar_colors = [colors.get(name, '#95a5a6') for name in names]
    
    bars = ax.barh(names, values, color=bar_colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
               f'{value:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_title('Component Contribution to Similarity', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()