# Interactive Notebooks

📚 Step-by-step for SAP material embeddings.

## 📚 Notebooks

### 01 - Introduction
**File:** `01_introduction.ipynb`

- Overview of the project
- Key concepts (FOL, Fuzzy Logic, Tensor Logic)
- Quick start guide
- What you'll learn

### 02 - Text Embeddings  
**File:** `02_text_embeddings.ipynb`

- Generate 768-d semantic embeddings
- Compute cosine similarity
- Batch processing
- Visualize in 2D with PCA
- Similarity matrices

**Result:** Text-only baseline

### 03 - Multimodal Embeddings
**File:** `03_multimodal_embeddings.ipynb`

- Combine text + categorical + characteristics + relational
- See dimension breakdown (768 + 288 + 192 + 128)
- Fusion layer (1376-d → 768-d)
- Component contribution analysis
- Compare with text-only

**Result:** 33% improvement over text-only

### 04 - Duplicate Detection
**File:** `04_duplicate_detection.ipynb`

- Full duplicate detection pipeline
- Text-only vs Multimodal comparison
- Visualizations (PCA, heatmaps, bar charts)
- Top duplicate pairs analysis

**Result:** 1481% improvement 🚀

## 🚀 Running the Notebooks
```bash
# Activate environment
source ~/envs/materials-embeddings/bin/activate

# Navigate to notebooks
cd notebooks

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## 📋 Prerequisites

Make sure you have:
- ✅ Activated `materials-embeddings` virtual environment
- ✅ Installed all packages from `requirements.txt`
- ✅ At least 4GB RAM (8GB recommended for larger datasets)

## 🎯 Learning Path

**Recommended order:**

1. **01_introduction** → Understand the problem and approach
2. **02_text_embeddings** → Learn the baseline (text-only)
3. **03_multimodal_embeddings** → See the power of multiple features
4. **04_duplicate_detection** → The spectacular result

Each notebook builds on the previous one.

## 💡 Tips

- Run cells in order (top to bottom)
- Restart kernel if you encounter errors
- Each notebook takes ~5-10 minutes to complete
- Visualizations may take a few seconds to render

## 🐛 Troubleshooting

**Import errors:**
```bash
# Make sure you're in the right environment
which python
# Should show: .../envs/materials-embeddings/bin/python
```

**Memory errors:**
```python
# Reduce sample size in create_sample_materials()
materials = create_sample_materials(n_materials=10)  # Instead of 30
```

## 📖 Next Steps

After completing the notebooks:
- Explore the Python scripts in `examples/`
- Read the main `README.md` for more context
- Try with your own SAP data
- Contribute improvements via GitHub

---