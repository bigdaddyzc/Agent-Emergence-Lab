#!/bin/bash
set -e

echo "========================================"
echo "  Agent Emergence Lab - Setup"
echo "========================================"
echo ""

# Step 1: Install Ollama
echo "[1/5] Installing Ollama..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed."
else
    echo "  Ollama already installed, skipping."
fi

# Step 2: Create Python virtual environment
echo "[2/5] Setting up Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists, skipping."
fi

source venv/bin/activate

# Step 3: Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."

# Step 4: Pull models
echo "[4/5] Pulling Ollama models (this may take a while)..."
ollama pull qwen2.5:3b 2>&1 | tail -1
ollama pull llama3.2:3b 2>&1 | tail -1
echo "  Models downloaded."

# Step 5: Verify
echo "[5/5] Verifying installation..."
python3 -c "import requests, yaml, numpy, flask; print('  All Python dependencies OK')"
ollama list 2>&1 | tail -n +2
echo ""

echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "To run:"
echo "  1. Start Ollama (if not running):  ollama serve &"
echo "  2. Activate venv:                  source venv/bin/activate"
echo "  3. Launch the lab:                 python3 src/main.py --topic 'consciousness' --turns 10"
echo "  4. Open browser:                   http://localhost:5000"
echo ""
