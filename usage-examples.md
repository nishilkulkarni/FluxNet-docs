---
layout: default
title: Usage Examples
nav_order: 3
---

# Usage Examples

This page provides examples of how to use the FluxNet in practice.

## Basic Example

```python
import torch
from torch_geometric.data import Data

# Define dimensions
node_in_dim = 32
edge_in_dim = 16
pe_dim = 8
out_channels = 64

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, node_in_dim)  # 3 nodes with 32 features each
edge_attr = torch.randn(4, edge_in_dim)  # 4 edges with 16 features each
x_pe = torch.randn(3, pe_dim)  # Positional encoding for nodes
edge_pe = torch.randn(4, pe_dim)  # Positional encoding for edges

# Initialize model
model = FluxNet(
    node_in_dim=node_in_dim,
    edge_in_dim=edge_in_dim,
    pe_dim=pe_dim,
    out_channels=out_channels,
    dropout=0.1,
    norm_type='layer'
)

# Forward pass
output = model(x, x_pe, edge_index, edge_attr, edge_pe)
print(output.shape)  # Should be [3, 64]
```

## Creating a Complete GNN Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class FluxNetModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, pe_dim, hidden_dim, output_dim, num_layers=3):
        super(FluxNetModel, self).__init__()
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                FluxNet(
                    node_in_dim=hidden_dim,
                    edge_in_dim=hidden_dim,
                    pe_dim=pe_dim,
                    out_channels=hidden_dim,
                    dropout=0.1
                )
            )
            
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, x_pe, edge_index, edge_attr, edge_pe, batch):
        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = conv(x, x_pe, edge_index, edge_attr, edge_pe, batch)
            
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

# Example usage
model = FluxNetModel(
    node_in_dim=32,
    edge_in_dim=16,
    pe_dim=8,
    hidden_dim=64,
    output_dim=10
)
```

## Training Loop Example

```python
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

# Assuming you have a dataset of PyG Data objects
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = FluxNetModel(...)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epoch):
    model.train()
    total_loss = 0
    
    for data in loader:
        optimizer.zero_grad()
        
        # Get data attributes
        x, edge_index = data.x, data.edge_index
        edge_attr, batch = data.edge_attr, data.batch
        x_pe, edge_pe = data.x_pe, data.edge_pe  # Assuming these are included in your dataset
        y = data.y
        
        # Forward pass
        out = model(x, x_pe, edge_index, edge_attr, edge_pe, batch)
        
        # Calculate loss
        loss = F.cross_entropy(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Run training for multiple epochs
for epoch in range(1, 101):
    loss = train(epoch)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')
```