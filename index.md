---
layout: default
title: Home
nav_order: 1
permalink: /
---

# FluxNet Neural Network Components
{: .fs-9 }

Documentation for FluxNet.
{: .fs-6 .fw-300 }

[View on GitHub](https://github.com/nishilkulkarni/FluxNet){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }

---

## Overview

This documentation covers a set of neural network components for graph-based learning that integrate:
- Node and edge features
- Positional encodings
- Adaptive degree scaling
- GATv2 attention mechanisms
- Residual connections
- Various normalization options

The architecture follows a message-passing paradigm where node features are modulated based on edge features, and an adaptive degree scaling is applied to handle graphs with varying node degrees.

[Get Started Now](./components/feature-modulator.html){: .btn .btn-blue .fs-5 .mb-4 .mb-md-0 .mr-2 }