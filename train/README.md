# Hanamikoji AI Training
This directory contains the code to train a high-performance AI for Hanamikoji using Reinforcement Learning (PPO).

## Prerequisites
- Python 3.8+
- PyTorch (with MPS support for M1/M2/M3 Macs)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
To start training the AI:
```bash
python train_hanamikoji.py
```

### Configuration
You can adjust the training parameters in `train_hanamikoji.py` under the `CONFIG` dictionary:
- `max_episodes`: Total number of games to play (default: 50,000).
- `hidden_dim`: Size of the neural network hidden layers.
- `device`: Automatically detects 'mps' for Apple Silicon.

### Output
- The script will save the trained model to `hanamikoji_ai.pth` every 1000 episodes.
- You can stop training at any time (Ctrl+C).

## Using the Model
Once trained, you can load `hanamikoji_ai.pth` in your game backend (requires converting the PyTorch model to ONNX or TensorFlow.js format for web use, or running a Python inference server).
