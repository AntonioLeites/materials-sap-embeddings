#!/bin/bash
# Activation script for materials-sap-embeddings

echo "🚀 Activating Materials Embeddings environment..."
source ~/envs/materials-embeddings/bin/activate

echo "✓ Environment activated!"
echo "  Python: $(python --version)"
echo "  Location: $(which python)"
echo ""

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "📦 Key packages:"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  NumPy: not installed"
python -c "import sentence_transformers; print(f'  Sentence Transformers: {sentence_transformers.__version__}')" 2>/dev/null || echo "  Sentence Transformers: not installed"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  PyTorch: not installed"

echo ""
echo "Ready to work! 🎉"
echo ""
echo "Run: python -m examples.basic_usage"
