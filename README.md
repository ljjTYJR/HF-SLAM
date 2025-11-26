# RGB-D Gaussian Splatting SLAM

## Installation

### Prerequisites
- Python 3.10
- CUDA 11.8 or compatible version
- [uv](https://github.com/astral-sh/uv) package manager

### Environment Setup with uv

#### Quick Install (Recommended)
Create a virtual environment and install all dependencies:
```bash
# Create virtual environment with Python 3.10
uv venv --python 3.10
```

# Activate the virtual environment
source .venv/bin/activate
# Install all dependencies
uv pip install -r requirements.txt

### Install Gaussian Rasterization Package
After installing the base dependencies, install the custom Gaussian rasterization package:
```bash
uv pip install submodules/diff_rasterization_w_d --no-build-isolation
uv pip install git+ssh://git@github.com/cvg/LightGlue.git@b1cd942
```