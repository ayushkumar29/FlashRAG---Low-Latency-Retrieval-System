#!/bin/bash

# FlashRAG Setup Script
# Automates the complete setup process

set -e  # Exit on error

echo "🚀 FlashRAG Setup Script"
echo "======================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}❌ Python 3.9+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv already exists, skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/documents
mkdir -p cache/chroma_db
mkdir -p logs
mkdir -p src
mkdir -p tests
mkdir -p benchmarks
mkdir -p scripts
echo -e "${GREEN}✓ Directories created${NC}"

# Setup .env file
echo ""
echo "⚙️  Setting up environment file..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env already exists, skipping...${NC}"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY${NC}"
        echo -e "${YELLOW}   Get free key from: https://console.groq.com${NC}"
    else
        cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
WEB_HOST=0.0.0.0
WEB_PORT=8000
MAX_WORKERS=4
RATE_LIMIT_PER_MINUTE=60
EOF
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY${NC}"
    fi
fi

# Create sample document
echo ""
echo "📄 Creating sample document..."
cat > data/documents/ml_basics.txt << 'EOF'
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on developing computer programs that can access data and use it 
to learn for themselves.

Types of Machine Learning

1. Supervised Learning
Learning from labeled data where the correct answer is provided. The algorithm 
learns to map inputs to outputs based on example input-output pairs. Common 
applications include classification (spam detection, image recognition) and 
regression (price prediction, weather forecasting).

2. Unsupervised Learning
Finding patterns in unlabeled data where no correct answer is provided. The 
algorithm discovers hidden structures in the data. Used for clustering 
(customer segmentation), dimensionality reduction (data visualization), and 
anomaly detection (fraud detection).

3. Reinforcement Learning
Learning through trial and error by interacting with an environment. The 
algorithm learns to make decisions by receiving rewards or penalties. Used 
in robotics, game playing (AlphaGo, chess), autonomous vehicles, and 
recommendation systems.

Neural Networks

Neural networks are computing systems inspired by biological neural networks 
that constitute animal brains. They consist of interconnected nodes (neurons) 
organized in layers:

- Input Layer: Receives the raw input data (images, text, numbers)
- Hidden Layers: Process and transform the data through weighted connections
- Output Layer: Produces the final prediction or classification

Each neuron receives inputs, applies weights, adds a bias term, and passes 
the result through an activation function. The network learns by adjusting 
weights through backpropagation.

Deep Learning

Deep learning uses neural networks with multiple hidden layers (deep networks) 
to learn complex patterns in data. This has revolutionized many fields:

- Computer Vision: Image classification, object detection, facial recognition
- Natural Language Processing: Translation, sentiment analysis, text generation
- Speech Recognition: Voice assistants, transcription services
- Generative AI: Creating images, text, music, and video

Popular architectures include:
- Convolutional Neural Networks (CNNs) for images
- Recurrent Neural Networks (RNNs) for sequences
- Transformers for language understanding
- Generative Adversarial Networks (GANs) for content creation

Applications of Machine Learning

Machine learning powers many modern applications:

1. Healthcare: Disease diagnosis, drug discovery, personalized treatment
2. Finance: Fraud detection, algorithmic trading, credit scoring
3. E-commerce: Product recommendations, demand forecasting, price optimization
4. Transportation: Autonomous vehicles, route optimization, traffic prediction
5. Entertainment: Content recommendation (Netflix, Spotify), game AI
6. Manufacturing: Quality control, predictive maintenance, supply chain optimization
7. Agriculture: Crop yield prediction, pest detection, precision farming

Challenges and Considerations

- Data Quality: ML models require large amounts of clean, representative data
- Bias and Fairness: Models can perpetuate biases present in training data
- Interpretability: Complex models (deep learning) can be "black boxes"
- Computational Resources: Training large models requires significant compute
- Privacy: Handling sensitive data requires careful consideration
- Generalization: Models must perform well on new, unseen data

The field continues to evolve with new techniques like transfer learning, 
few-shot learning, and self-supervised learning, making AI more accessible 
and powerful.
EOF

echo -e "${GREEN}✓ Sample document created${NC}"

# Create sample queries file
echo ""
echo "📝 Creating sample queries file..."
cat > queries.txt << 'EOF'
What is machine learning?
Explain supervised learning
What are neural networks?
How does deep learning work?
What are the applications of ML?
EOF
echo -e "${GREEN}✓ Sample queries file created${NC}"

# Test imports
echo ""
echo "🧪 Testing imports..."
python3 << 'PYEOF'
try:
    import chromadb
    import sentence_transformers
    import fastapi
    import groq
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
PYEOF

# Summary
echo ""
echo "======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Add your GROQ API key to .env:"
echo "   nano .env  (or use your favorite editor)"
echo "   Get free key from: https://console.groq.com"
echo ""
echo "2. Index your documents:"
echo "   python main.py index"
echo ""
echo "3. Test a query:"
echo "   python main.py query \"What is machine learning?\""
echo ""
echo "4. Start the web server:"
echo "   python main.py serve"
echo "   Then open: http://localhost:8000"
echo ""
echo "5. Run load test:"
echo "   python benchmarks/load_test.py"
echo ""
echo "📚 For more info, see README.md"
echo ""