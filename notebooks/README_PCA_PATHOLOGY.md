# PCA Pathology Visualization Module

A modular Python toolkit for visualizing DINOv3 features on pathology images using PCA decomposition and interactive ROI selection.

## Overview

This module provides tools for:
- Loading whole-slide images (WSI) and extracting patches
- Computing DINOv3 features on pathology patches
- Fitting PCA for feature visualization
- Creating rainbow-colored PCA visualizations
- Interactive ROI selection for focused analysis
- Creating overlay visualizations (PCA + original images)

## Files

- **`pca_pathology_utils.py`**: Main module with all functions and classes
- **`example_pca_usage.py`**: End-to-end pipeline example
- **`example_modular_usage.py`**: Individual component examples
- **`pca_pathology.ipynb`**: Original Jupyter notebook

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install numpy matplotlib scikit-learn
pip install Pillow h5py

# For WSI handling (Trident)
# See: https://github.com/your-trident-repo
```

### Setup

```bash
# Add to Python path
export PYTHONPATH="/home/nadim/Source/dinov3/notebooks:$PYTHONPATH"

# Or in Python
import sys
sys.path.insert(0, '/home/nadim/Source/dinov3/notebooks')
```

## Quick Start

### Full Pipeline

```python
from pca_pathology_utils import (
    load_h5_from_trident,
    PathologyPCAPipeline
)
import torch

# Load coordinates
coords, _, _ = load_h5_from_trident("path/to/file.h5")

# Load model
model = torch.hub.load(
    "facebookresearch/dinov3",
    "dinov3_vith16plus",
    pretrained=True
)
model.cuda()

# Create pipeline
pipeline = PathologyPCAPipeline(
    wsi_path="path/to/slide.tiff",
    coords=coords,
    model=model,
    model_name="dinov3_vith16plus",
    device='cuda'
)

# Run pipeline
pipeline.load_patches(patch_size=256, target_mag=20)
pipeline.extract_features(batch_size=32)
pipeline.fit_pca(n_components=3)

# Interactive ROI selection
pipeline.load_overview(overview_mag=10)
roi_patches = pipeline.select_rois_interactive(max_rois=10)
pipeline.visualize_roi_patches(roi_patches)
```

## Module Components

### 1. Data Loading

```python
from pca_pathology_utils import (
    load_h5_from_trident,
    load_wsi_and_extract_patches,
    load_wsi_overview
)

# Load patch coordinates from H5
coords, features, metadata = load_h5_from_trident("file.h5")

# Extract patches from WSI
patches, wsi_info = load_wsi_and_extract_patches(
    "slide.tiff",
    coords,
    patch_size=256,
    target_mag=20
)

# Load overview for ROI selection
overview_img, scale, info = load_wsi_overview("slide.tiff", overview_mag=10)
```

### 2. Feature Extraction

```python
from pca_pathology_utils import DINOv3FeatureExtractor

extractor = DINOv3FeatureExtractor(
    model=model,
    model_name="dinov3_vith16plus",
    device='cuda'
)

features = extractor.extract_features(
    patches,
    batch_size=32,
    verbose=True
)
# Returns: torch.Tensor of shape (N, feature_dim, 16, 16)
```

### 3. PCA Processing

```python
from pca_pathology_utils import PathologyPCA

# Create and fit PCA
pca_processor = PathologyPCA(n_components=3, whiten=True)
pca_processor.fit(features, verbose=True)

# Transform features
pca_rgb = pca_processor.transform(
    features,
    apply_sigmoid=True,
    sigmoid_scale=2.0
)
# Returns: torch.Tensor of shape (N, 3, 16, 16)
```

### 4. Interactive ROI Selection

```python
from pca_pathology_utils import ROISelector, find_patches_in_roi

# Create ROI selector
roi_selector = ROISelector(overview_img, max_rois=10)
plt.show()  # Interactive plot

# Get selected ROIs
selected_rois = roi_selector.get_rois()

# Find patches in each ROI
for roi_coords in selected_rois:
    patch_indices, roi_native = find_patches_in_roi(
        roi_coords,
        coords,
        overview_scale,
        patch_size=256
    )
    print(f"Found {len(patch_indices)} patches in ROI")
```

### 5. Visualization

```python
from pca_pathology_utils import (
    create_overlay_patch,
    visualize_patches_with_pca,
    visualize_roi_summary
)

# Create single overlay
overlay = create_overlay_patch(
    original_patch=patches[0],
    pca_result=pca_rgb[0].permute(1, 2, 0).numpy(),
    alpha=0.3  # 30% original, 70% PCA
)

# Visualize multiple patches
visualize_patches_with_pca(
    patches,
    pca_rgb,
    indices=[0, 1, 2, 3, 4],
    max_display=5,
    title="My Patches",
    alpha=0.3
)

# Visualize ROI summary
visualize_roi_summary(overview_img, roi_patches, coords, overview_scale)
```

## Class Reference

### PathologyPCAPipeline

End-to-end pipeline for pathology PCA visualization.

**Methods:**
- `load_patches(patch_size, target_mag)`: Load all patches from WSI
- `extract_features(batch_size)`: Extract DINOv3 features
- `fit_pca(n_components)`: Fit PCA on features
- `load_overview(overview_mag)`: Load WSI overview
- `select_rois_interactive(max_rois)`: Interactive ROI selection
- `visualize_roi_patches(roi_patches, max_display)`: Visualize ROI patches

### DINOv3FeatureExtractor

Extract DINOv3 features from patches.

**Methods:**
- `normalize_patches(patches)`: Normalize to ImageNet mean/std
- `extract_features(patches, batch_size, verbose)`: Extract features

### PathologyPCA

Fit and apply PCA to pathology features.

**Methods:**
- `fit(features, verbose)`: Fit PCA on features
- `transform(features, apply_sigmoid, sigmoid_scale)`: Transform features

### ROISelector

Interactive ROI selector for WSI overview.

**Methods:**
- `get_rois()`: Return list of selected ROIs as (x1, y1, x2, y2) tuples

## Function Reference

### Data Loading
- `load_h5_from_trident(h5_path)`: Load coords and features from H5
- `load_wsi_and_extract_patches(wsi_path, coords, ...)`: Extract patches
- `load_wsi_overview(wsi_path, overview_mag)`: Load overview image

### ROI Processing
- `find_patches_in_roi(roi_coords, patch_coords, scale, patch_size)`: Find patches in ROI

### Visualization
- `create_overlay_patch(patch, pca_result, alpha)`: Create overlay
- `visualize_patches_with_pca(patches, pca_rgb, ...)`: Visualize patches
- `visualize_roi_summary(overview_img, roi_patches, ...)`: Visualize ROI summary

## Examples

See the example files for detailed usage:
- **`example_pca_usage.py`**: Full pipeline with interactive prompts
- **`example_modular_usage.py`**: Individual component examples

### Example 1: Basic Workflow

```python
# 1. Load data
coords, _, _ = load_h5_from_trident("file.h5")
patches, _ = load_wsi_and_extract_patches("slide.tiff", coords)

# 2. Extract features
extractor = DINOv3FeatureExtractor(model, "dinov3_vith16plus", "cuda")
features = extractor.extract_features(patches)

# 3. Fit PCA
pca_processor = PathologyPCA(n_components=3)
pca_processor.fit(features)
pca_rgb = pca_processor.transform(features)

# 4. Visualize
visualize_patches_with_pca(patches, pca_rgb, indices=range(10))
```

### Example 2: Custom Overlay Alpha

```python
# Test different alpha values
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for alpha in alphas:
    overlay = create_overlay_patch(patches[0], pca_result, alpha=alpha)
    plt.figure()
    plt.imshow(overlay)
    plt.title(f"Alpha = {alpha}")
    plt.axis('off')
    plt.show()
```

## Key Features

### PCA Visualization
- **3-component PCA**: RGB visualization of feature space
- **Sigmoid scaling**: Vibrant color mapping (scale × 2.0)
- **Whitening**: Optional for better component separation

### Overlay Creation
- **Alpha blending**: Adjustable balance between original and PCA
- **PIL-based resizing**: Compatible without OpenCV
- **Upsampling**: 16×16 PCA features → 256×256 patch resolution

### ROI Selection
- **Interactive clicking**: Click 2 points per ROI rectangle
- **Visual feedback**: Yellow rectangles with labels
- **Coordinate conversion**: 10x overview → native resolution
- **Patch filtering**: Find patches within ROI bounds

## Technical Details

### Feature Dimensions
- **Input patches**: 256×256 pixels at 20x magnification
- **DINOv3 patch size**: 16×16 pixels
- **Patch features per image**: 16×16 = 256 feature vectors
- **Feature dimension**: 1280 (for ViT-H16+)

### Coordinate Systems
- **Native resolution**: Original WSI coordinates (e.g., 40x)
- **Overview resolution**: Downsampled for visualization (e.g., 10x)
- **Scale factor**: `overview_mag / native_mag`

### Performance
- **Batch processing**: 32 patches per batch (adjustable)
- **GPU acceleration**: CUDA support for feature extraction
- **Memory efficient**: Processes in batches to avoid OOM

## Troubleshooting

### Import Errors
```python
# Add module to path
import sys
sys.path.insert(0, '/path/to/notebooks')
```

### OpenCV cv2.resize Error
Module uses `PIL.Image.resize()` instead of `cv2.resize()` to avoid compatibility issues.

### CUDA Out of Memory
Reduce batch size:
```python
pipeline.extract_features(batch_size=16)  # Default: 32
```

### Trident Import Error
Ensure Trident is installed and in Python path:
```python
sys.path.insert(0, '/home/nadim/Source/trident')
```

## Citation

If you use this module, please cite:
- DINOv3: [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Inspired by prov-gigapath overlay techniques

## License

Follow the license of the DINOv3 repository and Trident.

## Contributing

Contributions welcome! Feel free to:
- Add new visualization methods
- Improve ROI selection UI
- Add support for other foundation models
- Optimize performance

## Contact

For questions or issues, please open an issue on the repository.
