# SAP RPT-1 Embeddings Tutorial

A practical guide to generating and working with 768-dimensional embeddings from SAP's Relational Pre-Trained Transformer (RPT-1-OSS).

## 🎯 What This Repository Provides

- **Working code examples** for generating embeddings with RPT-1-OSS
- **Practical use cases**: duplicate detection, similarity search, classification
- **Jupyter notebooks** with step-by-step explanations
- **Reusable utilities** for SAP material and supplier embeddings
- **Sample datasets** for experimentation

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/sap-rpt1-embeddings-tutorial.git
cd sap-rpt1-embeddings-tutorial
pip install -r requirements.txt
```

### Basic Usage
```python
from src.embeddings import RPT1Embeddings

# Initialize
rpt1 = RPT1Embeddings()

# Generate embedding
embedding = rpt1.encode("Steel Bolt M8x50 DIN 933")
print(f"Embedding shape: {embedding.shape}")  # (768,)

# Compare materials
similarity = rpt1.similarity(
    "Steel Bolt M8x50 DIN 933",
    "Stainless Steel Bolt M8x50 ISO 4017"
)
print(f"Similarity: {similarity:.4f}")  # 0.8923
```

## 📚 Notebooks

1. [**Basic Embeddings**](notebooks/01_basic_embeddings.ipynb) - Generate your first embeddings
2. [**Similarity Comparison**](notebooks/02_similarity_comparison.ipynb) - Compare materials semantically
3. [**Duplicate Detection**](notebooks/03_duplicate_detection.ipynb) - Find duplicate materials at scale
4. [**Intermediate Layers**](notebooks/04_intermediate_layers.ipynb) - Access hidden representations
5. [**Fine-Tuning**](notebooks/05_fine_tuning.ipynb) - Adapt RPT-1 to your data

## 🔧 Use Cases

### Duplicate Detection
```python
from src.similarity import DuplicateDetector

detector = DuplicateDetector(threshold=0.85)
duplicates = detector.find_duplicates(material_catalog)
```

### Supplier Matching
```python
from examples.supplier_matching import recommend_suppliers

suppliers = recommend_suppliers(
    material="Steel Bolt M8x50",
    top_k=5
)
```

### Material Classification
```python
from examples.material_classification import classify_material

material_group = classify_material(
    "Hexagonal Bolt Stainless Steel M8"
)
```

## 📊 Sample Data

The repository includes sample SAP-like data:
- `data/sample_materials.csv` - 1000 material descriptions
- `data/sample_suppliers.csv` - 100 supplier profiles

## 🏗️ Architecture
```
User Input (Material Description)
        ↓
   Tokenization
        ↓
  RPT-1-OSS Model
        ↓
  768-D Embedding
        ↓
  Downstream Tasks
  (Similarity, Classification, etc.)
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 📖 Documentation

- [API Reference](docs/API.md)
- [Examples](docs/EXAMPLES.md)
- [Architecture](docs/ARCHITECTURE.md)

## 🧪 Testing
```bash
pytest tests/
```

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Related Articles

- [How to Generate Embeddings with SAP RPT-1-OSS](link-to-linkedin-post)
- [From Business Rules to Neural Embeddings in SAP S/4HANA](link-to-linkedin-article)

## 💡 Questions?

Open an issue or reach out on LinkedIn.

## ⭐ Star This Repo

If you find this useful, please star the repository!