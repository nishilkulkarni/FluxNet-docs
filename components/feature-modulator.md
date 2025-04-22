---
layout: default
title: FeatureModulator
parent: Components
nav_order: 1
---

# FeatureModulator

```python
fluxnet.FeatureModulator(edge_dim, node_dim, hidden_dim=64, dropout=0.0)
```

A neural network module that transforms edge features to modulate node features, acting as the continuous convolution kernel.

## Parameters:

- **edge_dim** (`int`) – Dimensionality of edge features
- **node_dim** (`int`) – Dimensionality of node features
- **hidden_dim** (`int`, *optional*) – Hidden layer dimension. Default: `64`
- **dropout** (`float`, *optional*) – Dropout probability. Default: `0.0`

## Inputs:

- **edge_features** (`Tensor`) – Edge feature matrix of shape `[num_edges, edge_dim]`

## Returns:

- **modulation_weights** (`Tensor`) – Weights for node features of shape `[num_edges, node_dim]`

## Architecture:

The FeatureModulator consists of a simple MLP with:
- Input layer: Edge features (`edge_dim`)
- Hidden layer with GELU activation (`hidden_dim`)
- Dropout for regularization
- Output layer: Modulation weights (`node_dim`)

## Processing Flow:

1. Take edge features as input
2. Transform through the first linear layer
3. Apply GELU activation
4. Apply dropout for regularization
5. Transform through the second linear layer to match node dimensions
6. Output modulation weights to be applied to node features

## Example:

```python
import torch
from fluxnet import FeatureModulator

# Create a FeatureModulator
modulator = FeatureModulator(
    edge_dim=24,
    node_dim=40,
    hidden_dim=128,
    dropout=0.1
)

# Sample edge features
num_edges = 500
edge_features = torch.randn(num_edges, 24)

# Get modulation weights
modulation_weights = modulator(edge_features)
print(modulation_weights.shape)  # [500, 40]

# These weights can now be used to modulate node features via element-wise multiplication
source_node_features = torch.randn(num_edges, 40)  # Features of source nodes for each edge
modulated_features = source_node_features * modulation_weights
```

## Notes:

- The output weights are typically applied via element-wise multiplication to node features
- Increasing `hidden_dim` can provide more expressive transformations but increases computational cost
- The GELU activation provides smooth non-linearities suitable for gradient-based learning
