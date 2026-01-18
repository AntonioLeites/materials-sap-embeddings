# SAP Materials Embeddings

A practical guide to generating and working with embeddings for SAP material master data using Sentence Transformers.

## 🎯 What This Repository Provides

- **Text embeddings** for SAP material descriptions
- **Semantic similarity** search and duplicate detection
- **Working code examples** with real SAP use cases
- **Jupyter notebooks** with step-by-step explanations
- **Reusable utilities** for material similarity analysis

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/AntonioLeites/materials-sap-embeddings.git
cd materials-sap-embeddings

# Create environment
python3.11 -m venv ~/envs/materials-embeddings
source ~/envs/materials-embeddings/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.embeddings import MaterialEmbeddings

# Initialize
embedder = MaterialEmbeddings()

# Generate embedding
embedding = embedder.encode("Steel Bolt M8x50 DIN 933")
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Compare materials
similarity = embedder.similarity(
    "Steel Bolt M8x50 DIN 933",
    "Stainless Steel Bolt M8x50 ISO 4017"
)
print(f"Similarity: {similarity:.4f}")  # 0.8923
```

## 📚 Use Cases

### 1. Duplicate Detection
Find materials with similar descriptions that might be duplicates.

### 2. Material Search
Search for materials by semantic similarity, not just exact matches.

### 3. Classification Support
Generate embeddings as features for ML models.

### 4. Supplier Matching
Find materials typically supplied by similar vendors.

## 🏗️ Architecture

This project uses **Sentence Transformers** for generating embeddings:
- Model: `sentence-transformers/all-mpnet-base-v2`
- Embedding dimension: 768
- Optimized for semantic similarity

Unlike SAP RPT-1-OSS (which is for tabular classification/regression),
this project focuses on **text embeddings** for material descriptions.

## 📖 Documentation

- [Examples](examples/) - Runnable Python scripts
- [Notebooks](notebooks/) - Interactive tutorials
- [API Reference](docs/API.md)


## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🎯 Multimodal Embeddings Results

### Example: Comparing Different Materials

**Material A:** Steel Rivet 5x15  
**Material B:** Plastic Pin 4x30

**Similarity Breakdown:**

| Component | Score | Contribution |
|-----------|-------|--------------|
| Text (description) | 44.3% | Semantic similarity |
| Categorical (groups) | 46.3% | Business classification |
| Characteristics (specs) | 40.1% | Technical properties |
| Relational (usage) | 16.7% | Shared context |
| **Overall (fused)** | **58.9%** | Combined understanding |

### Key Insight

Multimodal embeddings achieve **33% higher similarity** (44% → 59%) compared to text-only by incorporating:
- ✅ Business context (MaterialGroup)
- ✅ Technical specifications (DIAMETER, MATERIAL)
- ✅ Usage patterns (plants, suppliers)

This demonstrates **Tensor Logic**: similarity emerges from learned fusion of multiple features, not explicit rules.