#!/bin/bash

echo "🚀 Setting up FlashRAG..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/documents cache/chroma_db logs

# Create .env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env and add your GROQ_API_KEY"
fi

# Create sample document
cat > data/documents/sample.txt << 'EOF'
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed.

Neural Networks
Neural networks are computing systems inspired by biological neural networks.
EOF

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GROQ_API_KEY from https://console.groq.com"
echo "2. Run: python main.py index"
echo "3. Run: python main.py serve"