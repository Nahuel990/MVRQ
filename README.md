# Early Stopping in Residual Quantization: An Empirical Negative Result

**Published**: TechRxiv, October 24, 2025  
**DOI**: 10.36227/techrxiv.176127327.72667338/v1  
**Author**: Nahuel Alejandro Nucera

This repository contains the experimental code for investigating Multi-Vector Residual Quantization (MVRQ) with adaptive early stopping mechanisms. The research explores whether predictive early stopping can improve the compression-accuracy trade-off in residual quantization systems.

## Abstract

Residual Quantization (RQ) is a common method for compressing high-dimensional embeddings used in vector search systems. We investigate a variant, MVRQ (Margin-Variable Residual Quantization), which applies early stopping to reduce storage by dynamically selecting the number of quantization stages per vector. While the approach intuitively aims to allocate more bits to "hard" vectors and fewer to "easy" ones, our experiments demonstrate that early stopping in RQ leads to significant recall degradation with minimal actual storage benefit. This paper reports an honest negative result, showing that naive adaptive RQ schemes fail to outperform standard fixed-stage RQ even under favorable settings.

## Overview

Residual Quantization (RQ) is a powerful technique for compressing high-dimensional vectors while maintaining retrieval accuracy. This project investigates whether adaptive early stopping based on vector complexity predictors can achieve better compression ratios without significant accuracy loss.

**Key Research Question**: Can we predict when to stop quantization early based on vector characteristics, achieving better compression while maintaining retrieval performance?

## Methodology

### Multi-Vector Residual Quantization (MVRQ)

MVRQ extends traditional RQ by using adaptive stopping criteria:
- **Traditional RQ**: Uses a fixed number of quantization stages (M=8)
- **MVRQ**: Stops early when residual norm ≤ α × g(x), where g(x) is a complexity predictor

### Complexity Predictors

The implementation explores several predictors g(x):

1. **Local Density** (10-NN): Average distance to 10 nearest neighbors
2. **Cluster Variance**: Intra-cluster variance via k-means clustering  
3. **Vector Norm**: Magnitude of the original vector
4. **Combined Predictor**: Weighted combination of the above signals

### Evaluation Framework

- **Dataset**: 100K synthetic vectors (d=256) + 2K queries
- **Real Embeddings**: SBERT (`all-MiniLM-L6-v2`) embeddings for realistic evaluation
- **Metrics**: Recall@10, compression ratio (bytes/vector)
- **Baseline**: Standard RQ-8 quantization

## Files Description

### Core Experiments

- **`mvrq_sweep_sbert.py`**: Main experiment with SBERT embeddings
  - Uses real sentence embeddings instead of synthetic data
  - Implements combined predictor (variance + density + norm)
  - Comprehensive evaluation with multiple α values

- **`mvrq_sweep_v2.py`**: Enhanced synthetic data experiment
  - Configurable predictor weights (W parameter)
  - Variance-based predictor with k-means clustering
  - Detailed analysis of compression-accuracy trade-offs

- **`mvrq_sweep.py`**: Basic MVRQ implementation
  - Simple local density predictor (10-NN)
  - Foundation for more complex experiments

- **`mvrq_test.py`**: Minimal test implementation
  - Quick validation of MVRQ concept
  - Single α value evaluation

### Key Features

- **Adaptive Quantization**: Early stopping based on residual norm thresholds
- **Multiple Predictors**: Various complexity estimation methods
- **Comprehensive Evaluation**: Compression vs. accuracy analysis
- **Visualization**: Plots showing trade-offs and predictor effectiveness

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mvrq

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy faiss-cpu matplotlib sentence-transformers tqdm scikit-learn
```

## Usage

### Basic MVRQ Experiment
```bash
python mvrq_sweep.py
```

### SBERT Embeddings Experiment
```bash
python mvrq_sweep_sbert.py
```

### Enhanced Synthetic Data Experiment
```bash
python mvrq_sweep_v2.py
```

## Configuration

Key parameters can be modified in each script:

```python
N = 100_000      # Number of documents
Qn = 2_000       # Number of queries  
M_MAX = 8        # Maximum RQ stages
NBITS = 8        # Bits per stage
ALPHAS = [0.4, 0.6, 0.8, 1.0]  # Early stopping thresholds
```

## Results

The experiments evaluate:
- **Compression Ratio**: Average bytes per vector
- **Retrieval Accuracy**: Recall@10 performance
- **Predictor Effectiveness**: Correlation between predictor and residual error
- **Stage Distribution**: How many stages are used per vector

### Expected Findings

The research investigates whether MVRQ can achieve:
- Better compression ratios than fixed RQ
- Maintained or improved retrieval accuracy
- Effective complexity prediction

## Technical Details

### Residual Quantization
- Uses FAISS ResidualQuantizer implementation
- 8 stages maximum, 8 bits per stage
- Inner product similarity for retrieval

### Early Stopping Criteria
- Threshold: `||residual|| ≤ α × g(x)`
- α controls aggressiveness of early stopping
- g(x) estimates vector complexity

### Evaluation Metrics
- **Recall@10**: Fraction of ground-truth neighbors found in top-10
- **Compression**: Average bytes per vector (including overhead)
- **Correlation**: How well predictor correlates with reconstruction error

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nucera2025,
  title={Early Stopping in Residual Quantization: An Empirical Negative Result},
  author={Nahuel Alejandro Nucera},
  journal={TechRxiv},
  year={2025},
  doi={10.36227/techrxiv.176127327.72667338/v1},
  url={https://www.techrxiv.org/articles/preprint/Early_Stopping_in_Residual_Quantization_An_Empirical_Negative_Result/176127327}
}
```

**TechRxiv Link**: https://www.techrxiv.org/articles/preprint/Early_Stopping_in_Residual_Quantization_An_Empirical_Negative_Result/176127327

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Publication Details

- **Preprint**: TechRxiv
- **Publication Date**: October 24, 2025
- **DOI**: 10.36227/techrxiv.176127327.72667338/v1
- **License**: Public Domain (CC0 1.0)
- **Funder**: Independent Sector (Identifier: 100003880)
