"""
Visualization utilities for embeddings and similarity matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: list = None,
    title: str = "Similarity Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "YlOrRd"
):
    """
    Plot a similarity matrix as a heatmap
    
    Args:
        similarity_matrix: NxN similarity matrix
        labels: Optional labels for materials
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        similarity_matrix,
        annot=True if len(similarity_matrix) <= 10 else False,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels if labels else False,
        yticklabels=labels if labels else False,
        vmin=0,
        vmax=1
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: list = None,
    method: str = "tsne",
    title: str = "Embedding Visualization",
    figsize: tuple = (12, 8)
):
    """
    Reduce embeddings to 2D and plot
    
    Args:
        embeddings: NxD embedding matrix
        labels: Optional labels for points
        method: 'tsne' or 'pca'
        title: Plot title
        figsize: Figure size
    """
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
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
                s=100
            )
        
        plt.legend()
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=100
        )
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_duplicate_groups(
    duplicates: list,
    materials: list,
    figsize: tuple = (14, 6)
):
    """
    Visualize duplicate groups
    
    Args:
        duplicates: List of (idx1, idx2, similarity) tuples
        materials: List of material descriptions
        figsize: Figure size
    """
    # Build groups
    from collections import defaultdict
    
    groups = defaultdict(set)
    for idx1, idx2, sim in duplicates:
        groups[idx1].add(idx2)
        groups[idx2].add(idx1)
    
    # Find connected components
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in groups[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in groups.keys():
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Group sizes
    sizes = [len(comp) for comp in components]
    ax1.bar(range(len(sizes)), sorted(sizes, reverse=True))
    ax1.set_xlabel("Group")
    ax1.set_ylabel("Number of Materials")
    ax1.set_title("Duplicate Group Sizes")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Similarity distribution
    similarities = [sim for _, _, sim in duplicates]
    ax2.hist(similarities, bins=20, edgecolor='black')
    ax2.set_xlabel("Similarity Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Similarity Scores")
    ax2.axvline(np.mean(similarities), color='r', linestyle='--', 
                label=f'Mean: {np.mean(similarities):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Found {len(components)} duplicate groups")
    print(f"Total materials involved: {sum(sizes)}")