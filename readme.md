# Turkish BERT

A PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) pre-trained on Turkish text data. This implementation includes custom Turkish tokenization with morphological analysis and supports training on various devices (CUDA, MPS, CPU).

## Quick Start

```bash
# Clone and setup
git clone https://github.com/malibayram/turkish-bert.git
cd turkish-bert
pip install -r requirements.txt

# Prepare data directory
mkdir -p data
# Add your text data to data/combined_reviews.csv

# Run minimal training (for testing)
python train.py \
    --d_model 64 \
    --n_layers 2 \
    --heads 4 \
    --batch_size 8 \
    --seq_len 32 \
    --num_epochs 2

# Run full training
python train.py \
    --d_model 768 \
    --n_layers 12 \
    --heads 12 \
    --batch_size 32 \
    --seq_len 512 \
    --num_epochs 40
```

## Features

- **Custom Turkish Tokenizer**
  - Morphological analysis
  - Root word detection
  - Suffix handling
  - Special token support ([CLS], [SEP], [MASK])
  - Vocabulary size of 32,768 tokens

- **Model Architecture**
  - Based on BERT-base architecture
  - 12 transformer layers (configurable)
  - 12 attention heads (configurable)
  - 768 hidden size (configurable)
  - ~110M parameters
  - Maximum sequence length of 512 tokens

- **Training Features**
  - Masked Language Model (MLM) pre-training
  - Next Sentence Prediction (NSP)
  - Learning rate scheduling with warmup
  - Multi-device support (CUDA, MPS, CPU)
  - Automatic checkpointing
  - Progress tracking and logging

## Requirements

```bash
torch>=1.12.0
numpy>=1.21.0
tqdm>=4.62.0
pandas>=1.3.0
```

## Memory Requirements

Recommended configurations for different setups:

### Entry Level (16GB RAM, No GPU)
```bash
python train.py \
    --d_model 256 \
    --n_layers 4 \
    --heads 4 \
    --batch_size 8 \
    --seq_len 128 \
    --device cpu
```

### Mid Range (Apple Silicon, 16GB RAM)
```bash
python train.py \
    --d_model 512 \
    --n_layers 6 \
    --heads 8 \
    --batch_size 32 \
    --seq_len 256 \
    --device mps
```

### High End (NVIDIA GPU, 24GB+ VRAM)
```bash
python train.py \
    --d_model 768 \
    --n_layers 12 \
    --heads 12 \
    --batch_size 64 \
    --seq_len 512 \
    --device cuda
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/malibayram/turkish-bert.git
cd turkish-bert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
```bash
mkdir -p data
# Place your combined_reviews.csv in the data directory
```

## Data Format

The training script expects a CSV file with the following format:
- One text sample per line
- UTF-8 encoding
- No header row
- Single column containing the text

Example `combined_reviews.csv`:
```text
Bu film gerçekten çok güzeldi.
Yemekler lezzetli değildi.
Hizmet kalitesi mükemmeldi.
...
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train.py
├── .gitignore
├── bert/
│   ├── __init__.py
│   ├── bert_model.py              # Core BERT model
│   ├── dataset.py                 # Dataset and data loading
│   └── positional_embedding.py    # Positional encodings
└── turkish_tokenizer/
    ├── __init__.py
    ├── turkish_tokenizer.py       # Python implementation
    ├── src/
    │   └── main.rs                # Rust implementation
    ├── Cargo.toml                 # Rust package configuration
    ├── kokler_v05.json
    ├── ekler_v05.json
    └── bpe_v05.json
```

## Turkish Tokenizer

The project includes two implementations of the Turkish tokenizer:

### Python Implementation
Located in `turkish_tokenizer/turkish_tokenizer.py`, this is the default implementation used by the training script. It provides:
- Morphological analysis
- Root word detection
- Suffix handling
- Special token support ([CLS], [SEP], [MASK], [PAD])
- Case handling

### Rust Implementation (Optional)
Located in `turkish_tokenizer/src/main.rs`, this is a high-performance alternative that provides:
- Parallel processing using Rayon
- Memory-efficient caching
- Same tokenization features as Python version
- Significantly faster processing for large texts

To use the Rust version:
```bash
# Build the Rust tokenizer
cd turkish_tokenizer
cargo build --release

# Test the tokenizer
./target/release/turkish_tokenizer "Bu bir test cümlesidir."
```

Both implementations use the same dictionary files:
- `kokler_v05.json`: Turkish root words dictionary
- `ekler_v05.json`: Turkish suffixes dictionary
- `bpe_v05.json`: BPE tokens dictionary

## Training

### Command Line Arguments

```bash
python train.py --help
```

#### Data Parameters
- `--corpus_path`: Path to training corpus (default: 'combined_reviews.csv')
- `--train_test_split`: Train/test split ratio (default: 0.9)
- `--seq_len`: Maximum sequence length (default: 512)

#### Model Parameters
- `--vocab_size`: Vocabulary size (default: 32768)
- `--d_model`: Model dimension (default: 768)
- `--n_layers`: Number of transformer layers (default: 12)
- `--heads`: Number of attention heads (default: 12)
- `--dropout`: Dropout rate (default: 0.1)

#### Training Parameters
- `--batch_size`: Training batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 1)
- `--lr`: Learning rate (default: 1e-4)
- `--warmup_steps`: Number of warmup steps (default: 10000)
- `--num_workers`: Number of data loading workers (default: auto)

#### Device Parameters
- `--device`: Device to use ['cuda', 'mps', 'cpu'] (default: auto-detect)
- `--save_dir`: Directory for checkpoints (default: 'checkpoints')
- `--save_freq`: Save frequency in epochs (default: 1)

## Performance Tips

### Training Speed
- Use the Rust tokenizer for faster preprocessing
- Increase batch_size when possible
- Use appropriate num_workers for your CPU
- Enable pin_memory for GPU training

### Memory Usage
- Start with small model and gradually increase
- Monitor GPU memory usage
- Use gradient checkpointing for large models
- Adjust sequence length based on your data needs

### Training Stability
- Start with lower learning rate (1e-5)
- Increase warmup_steps for larger models
- Use gradient clipping
- Monitor loss values for convergence

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch_size
   - Reduce seq_len
   - Reduce model size (d_model, n_layers, heads)
   ```bash
   python train.py --batch_size 8 --seq_len 128 --d_model 256
   ```

2. **Slow Training**
   - Increase batch_size if memory allows
   - Adjust num_workers
   - Use Rust tokenizer for faster preprocessing
   ```bash
   python train.py --batch_size 64 --num_workers 4
   ```

3. **MPS (Apple Silicon) Issues**
   - Use num_workers=0
   - Keep batch_size ≤ 32
   - Keep seq_len ≤ 512
   ```bash
   python train.py --device mps --num_workers 0 --batch_size 32
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{turkish_bert,
  author = {M. Ali Bayram},
  title = {Turkish BERT: Pre-trained BERT Models for Turkish Language},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/malibayram/turkish-bert}
}
```

## Acknowledgments

- BERT paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Implementation based on [Complete Guide to Building BERT Model from Scratch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891) by CheeKean
- Turkish morphological analysis resources
- PyTorch team for the deep learning framework

## Contact

M. Ali Bayram - [@malibayram](https://github.com/malibayram)

Project Link: [https://github.com/malibayram/turkish-bert](https://github.com/malibayram/turkish-bert)
