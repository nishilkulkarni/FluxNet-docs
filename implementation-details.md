---
layout: default
title: Implementation Details
nav_order: 4
---

# Implementation Details

This page covers specific implementation details of the FluxNet components.

## Adaptive Degree Scaling

The adaptive degree scaling is implemented in the `CKGConv` class with two learnable parameters:
- `theta1`: Scaling factor for the aggregated messages
- `theta2`: Scaling factor for the degree-adjusted messages

The scaling is applied as:
```python
out = out * self.theta1 + deg_sqrt * (out * self.theta2)
```
where `deg_sqrt` is the square root of the node degrees.

This mechanism helps the model adapt to graphs with varying node degrees by applying different weightings to node features based on their connectivity patterns.

## Normalization Options

The `FluxNet` class supports multiple normalization types:

| Type | Implementation | Description |
|------|---------------|-------------|
| `batch` | BatchNorm1d | Normalizes across batch dimension |
| `layer` | LayerNorm | Normalizes across feature dimension |
| `instance` | InstanceNorm1d | Normalizes each instance independently |
| `none` | Identity | No normalization is applied |

### When to use each type:

- **BatchNorm**: Good for large batch sizes and when data distribution is consistent
- **LayerNorm**: Better for varying input distributions or when batch size is small
- **InstanceNorm**: Helpful for graph data where each graph can have very different distributions
- **None**: When you want to avoid any normalization, e.g., for debugging

## GAT Attention

The GATv2 attention mechanism is implemented using PyTorch Geometric's `GATv2Conv` class with:
- Multi-head attention (default: 4 heads)
- Edge feature integration
- Non-concatenated output (heads are averaged)

### Differences from GAT:

GATv2Conv is an improvement over the original GAT attention mechanism with:
1. Dynamic attention computation (addresses the static attention problem)
2. Better expressive power
3. Generally improved performance on graph tasks

## Feed-Forward Network

The feed-forward network follows a typical design:
- Expansion layer: `out_channels` → `ffn_hidden_dim`
- GELU activation: Non-linear transformation
- Dropout: Regularization
- Contraction layer: `ffn_hidden_dim` → `out_channels`

By default, `ffn_hidden_dim` is set to 4 times the output dimension, which is a common practice in transformer architectures.