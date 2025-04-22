---
layout: default
title: Installation
nav_order: 3
---

# Installation

This page covers the installation process and requirements for FluxNet.

## Virtual Environment Setup

It's recommended to use a virtual environment for FluxNet installation:

```bash
# Create a virtual environment
python -m venv fluxnet-env

# Activate the virtual environment
# On Windows
fluxnet-env\Scripts\activate
# On macOS/Linux
source fluxnet-env/bin/activate
```

## Installing Dependencies

Install dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Installing FluxNet

### From PyPI

```bash
pip install fluxnet
```

### From Source

```bash
git clone https://github.com/username/fluxnet.git
cd fluxnet
pip install -e .
```

## GPU Support

For optimal performance with GPU acceleration:

```bash
# Ensure you have compatible CUDA drivers installed
pip install fluxnet[cuda]
```

## Verifying Installation

```python
import fluxnet
print(fluxnet.__version__)

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Conda Installation

Alternative installation using Conda:

```bash
conda create -n fluxnet-env python=3.8
conda activate fluxnet-env
pip install -r requirements.txt
pip install -e .
```
