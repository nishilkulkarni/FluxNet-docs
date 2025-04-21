---
layout: default
title: FeatureModulator
parent: Components
nav_order: 1
---

# FeatureModulator

The `FeatureModulator` class is a neural network module that transforms edge features to modulate node features.

## Purpose
- Acts as the ψ function mentioned in the paper
- Transforms edge features into weights that will be applied to node features

## Parameters
- `edge_dim`: Dimensionality of edge features
- `node_dim`: Dimensionality of node features
- `hidden_dim`: Hidden layer dimension (default: 64)
- `dropout`: Dropout probability (default: 0.0)

## Architecture
- MLP with a single hidden layer
- GELU activation function
- Dropout regularization
- Input: Edge features → Output: Modulation weights for node features

## Implementation

```python
class FeatureModulator(nn.Module):
    """
    Neural network that modulates node features based on edge features.
    Corresponds to the ψ function in the paper.
    """
    def __init__(self, edge_dim, node_dim, hidden_dim=64, dropout=0.0):
        super(FeatureModulator, self).__init__()
        self.mlp = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, edge_features):
        return self.mlp(edge_features)