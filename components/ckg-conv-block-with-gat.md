---
layout: default
title: FluxNet
parent: Components
nav_order: 3
---

# FluxNet

```python
fluxnet.FluxNet(node_in_dim, edge_in_dim, pe_dim, out_channels, 
                ffn_hidden_dim=None, modulator_hidden_dim=64,
                dropout=0.0, norm_type='batch', add_self_loops=True, 
                aggr='mean', num_heads=4, use_attention=True)
```

Combines `CKGConv` with a GATv2 attention mechanism to create a comprehensive continuous kernel graph convolution block.

## Parameters:

- **node_in_dim** (`int`) – Input dimension of node features
- **edge_in_dim** (`int`) – Input dimension of edge features
- **pe_dim** (`int`) – Dimension of positional encodings
- **out_channels** (`int`) – Output dimension of the convolution
- **ffn_hidden_dim** (`int`, *optional*) – Hidden dimension for the feed-forward network. Default: `4 * out_channels`
- **modulator_hidden_dim** (`int`, *optional*) – Hidden dimension for the feature modulator. Default: `64`
- **dropout** (`float`, *optional*) – Dropout probability. Default: `0.0`
- **norm_type** (`str`, *optional*) – Normalization type, one of [`'batch'`, `'layer'`, `'instance'`, `'none'`]. Default: `'batch'`
- **add_self_loops** (`bool`, *optional*) – Whether to add self-loops to the graph. Default: `True`
- **aggr** (`str`, *optional*) – Aggregation method. Default: `'mean'`
- **num_heads** (`int`, *optional*) – Number of attention heads for GATv2. Default: `4`
- **use_attention** (`bool`, *optional*) – Whether to use the GATv2 attention mechanism. Default: `True`

## Inputs:

- **x** (`Tensor`) – Node feature matrix of shape `[num_nodes, node_in_dim]`
- **x_pe** (`Tensor`) – Node positional encoding matrix of shape `[num_nodes, pe_dim]`
- **edge_index** (`LongTensor`) – Graph connectivity matrix of shape `[2, num_edges]`
- **edge_attr** (`Tensor`) – Edge feature matrix of shape `[num_edges, edge_in_dim]`
- **edge_pe** (`Tensor`) – Edge positional encoding matrix of shape `[num_edges, pe_dim]`
- **batch** (`LongTensor`, *optional*) – Batch vector of shape `[num_nodes]` indicating node assignment to batch instances. Default: `None`

## Returns:

- **out** (`Tensor`) – Updated node feature matrix of shape `[num_nodes, out_channels]`

## Architecture:

`FluxNet` combines several components:

1. **CKGConv Layer**: Base graph convolution operation
2. **Normalization**: Configurable normalization applied after each major component
3. **GATv2 Attention**: Multi-head graph attention mechanism (optional)
4. **Feed-Forward Network**: Two-layer MLP with GELU activation
5. **Residual Connections**: Added after each major component
6. **Dropout**: Applied to outputs of attention and feed-forward network

## Processing Flow:

1. Apply `CKGConv` to input features
2. Apply normalization
3. Add residual connection if dimensions match
4. If `use_attention=True`:
   - Apply GATv2 attention mechanism
   - Add residual connection with dropout
   - Apply normalization
5. Apply feed-forward network
6. Add residual connection with dropout
7. Apply final normalization

## Example:

```python
import torch
from fluxnet import FluxNet

# Create a FluxNet layer
model = FluxNet(
    node_in_dim=32, 
    edge_in_dim=16, 
    pe_dim=8,
    out_channels=64,
    dropout=0.1,
    norm_type='layer'
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
output = model(x, x_pe, edge_index, edge_attr, edge_pe)
print(output.shape)  # [100, 64]
```

## Notes:

- When `ffn_hidden_dim` is not provided, it defaults to 4 times the `out_channels`
- The choice of normalization type significantly impacts performance on different graph datasets
- If `use_attention=False`, the model skips the GATv2 attention mechanism
- For best performance with large graphs, consider using `norm_type='instance'`
