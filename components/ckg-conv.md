---
layout: default
title: CKGConv
parent: Components
nav_order: 2
---

# CKGConv

```python
fluxnet.CKGConv(node_in_dim, edge_in_dim, pe_dim, out_channels,
                modulator_hidden_dim=64, dropout=0.0, add_self_loops=True,
                aggr='mean')
```

A specialized graph convolution layer based on PyTorch Geometric's `MessagePassing` framework that performs graph convolution operations using concatenated node/edge features with positional encodings.

## Parameters:

- **node_in_dim** (`int`) – Input dimension of node features
- **edge_in_dim** (`int`) – Input dimension of edge features
- **pe_dim** (`int`) – Dimension of positional encodings
- **out_channels** (`int`) – Output dimension of the convolution
- **modulator_hidden_dim** (`int`, *optional*) – Hidden dimension for the feature modulator. Default: `64`
- **dropout** (`float`, *optional*) – Dropout probability. Default: `0.0`
- **add_self_loops** (`bool`, *optional*) – Whether to add self-loops to the graph. Default: `True`
- **aggr** (`str`, *optional*) – Aggregation method (`'mean'`, `'sum'`, `'max'`, etc.). Default: `'mean'`

## Inputs:

- **x** (`Tensor`) – Node feature matrix of shape `[num_nodes, node_in_dim]`
- **x_pe** (`Tensor`) – Node positional encoding matrix of shape `[num_nodes, pe_dim]`
- **edge_index** (`LongTensor`) – Graph connectivity matrix of shape `[2, num_edges]`
- **edge_attr** (`Tensor`) – Edge feature matrix of shape `[num_edges, edge_in_dim]`
- **edge_pe** (`Tensor`) – Edge positional encoding matrix of shape `[num_edges, pe_dim]`
- **batch** (`LongTensor`, *optional*) – Batch vector of shape `[num_nodes]` indicating node assignment to batch instances. Default: `None`

## Returns:

- **out** (`Tensor`) – Updated node feature matrix of shape `[num_nodes, out_channels]`

## Key Components:

1. **Feature Modulator**: Transforms edge features to modulate node features
2. **Linear Transformation**: Transforms aggregated node features
3. **Adaptive Degree Scaling**: Scales node features based on node degrees using learnable parameters
   - `theta1`: Scaling factor for the aggregated messages
   - `theta2`: Scaling factor for the degree-adjusted messages

## Processing Flow:

1. Concatenates raw node features with positional encodings
2. Concatenates edge features with positional encodings
3. Propagates messages through the graph
   - Modulates source node features with edge features
   - Aggregates messages at each target node
4. Applies adaptive degree scaling with learnable parameters
5. Applies linear transformation to produce output features

## Example:

```python
import torch
from fluxnet import CKGConv

# Create a CKGConv layer
conv = CKGConv(
    node_in_dim=32, 
    edge_in_dim=16, 
    pe_dim=8,
    out_channels=64,
    modulator_hidden_dim=128,
    aggr='mean'
)

# Input features
num_nodes = 100
num_edges = 500
x = torch.randn(num_nodes, 32)
x_pe = torch.randn(num_nodes, 8)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.randn(num_edges, 16)
edge_pe = torch.randn(num_edges, 8)

# Forward pass
output = conv(x, x_pe, edge_index, edge_attr, edge_pe)
print(output.shape)  # [100, 64]
```

## Notes:

- The adaptive degree scaling helps the model adapt to graphs with varying node degrees
- The modulator uses an MLP to transform edge features, which are then used to modulate node features
- For large graphs, the aggregation method (`aggr`) can significantly impact both performance and results
- When `add_self_loops=True`, the layer adds self-connections to each node, which helps with information propagation
