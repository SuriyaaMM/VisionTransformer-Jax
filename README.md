# JAX Vision Transformer

A complete implementation of the Vision Transformer (ViT) architecture from scratch using JAX and optax.

## Overview

This repository contains a full implementation of the Vision Transformer as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. The implementation is written in JAX and trained on the MNIST dataset.

![Vision Transformer Architecture](https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_figure.png)

## Features

- Complete Vision Transformer implementation including:
  - Patch Embedding
  - Class Token
  - Positional Embeddings
  - Transformer Blocks with Multi-Head Self-Attention
  - Layer Normalization
  - Classification Head
- Training pipeline for MNIST classification
- Visualization of attention weights
- Pure JAX implementation for efficient computation

## Requirements

This project requires the following dependencies:

- Python 3.9+
- JAX and JAXlib with GPU support
- PyTorch and torchvision (for dataset loading)
- optax (for optimization)
- matplotlib and seaborn (for visualization)

You can set up the environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate vit-jax
```

Or install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Files Structure

- `VisionTransformer.py`: Main Vision Transformer model
- `PatchEmbedding.py`: Implementation of patch-based image embedding
- `Transformer.py`: Implementation of Transformer blocks
- `MultiheadAttention.py`: Implementation of Multi-head Self-Attention
- `Feedforward.py`: Implementation of Feedforward networks
- `LayerNorm.py`: Implementation of Layer Normalization
- `TrainVisionTransformer.py`: Training script for MNIST dataset

## Usage

To train the Vision Transformer on MNIST:

```bash
python TrainVisionTransformer.py
```

## Model Architecture

The Vision Transformer architecture implemented here consists of:

1. **Patch Embedding**: Divides the input image into patches and projects them to a fixed embedding dimension.
2. **Class Token**: A learnable token prepended to the sequence of embedded patches.
3. **Positional Embedding**: Position information added to the patch embeddings.
4. **Transformer Encoder**: Multiple layers of transformer blocks, each containing:
   - Multi-head Self-Attention
   - Feedforward Neural Network
   - Layer Normalization
   - Residual Connections
5. **Classification Head**: A simple linear layer that maps the class token output to logits.

## Attention Visualization

During training, the model visualizes attention weights for each layer and attention head, saving them to the `./AttentionWeightsHeatmap/` directory. These visualizations show what parts of the image the model focuses on when making predictions.

## Performance

When trained on MNIST, the model typically achieves around 80-85% accuracy after 25 epochs.
