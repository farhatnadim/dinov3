"""
Example: Using individual components from pca_pathology_utils

This script shows how to use specific functions and classes
without the full pipeline.
"""

import torch
import numpy as np
from pca_pathology_utils import (
    load_h5_from_trident,
    load_wsi_and_extract_patches,
    load_wsi_overview,
    DINOv3FeatureExtractor,
    PathologyPCA,
    ROISelector,
    find_patches_in_roi,
    create_overlay_patch,
    visualize_patches_with_pca
)


# =============================================================================
# Example 1: Load data from H5 and WSI
# =============================================================================

def example_load_data():
    """Load patch coordinates and patches from WSI."""

    H5_PATH = "/path/to/your/file.h5"
    WSI_PATH = "/path/to/your/slide.tiff"

    # Load coordinates
    coords, features, metadata = load_h5_from_trident(H5_PATH)
    print(f"Loaded {len(coords)} coordinates")

    # Extract first 10 patches
    patches, wsi_info = load_wsi_and_extract_patches(
        WSI_PATH,
        coords[:10],
        patch_size=256,
        target_mag=20
    )
    print(f"Extracted {len(patches)} patches")

    return coords, patches


# =============================================================================
# Example 2: Extract features using DINOv3
# =============================================================================

def example_extract_features(model, model_name, patches):
    """Extract DINOv3 features from patches."""

    # Create feature extractor
    extractor = DINOv3FeatureExtractor(
        model=model,
        model_name=model_name,
        device='cuda'
    )

    # Extract features
    features = extractor.extract_features(
        patches,
        batch_size=32,
        verbose=True
    )

    print(f"Features shape: {features.shape}")

    return features


# =============================================================================
# Example 3: Fit and apply PCA
# =============================================================================

def example_pca_analysis(features):
    """Fit PCA and transform features."""

    # Create PCA processor
    pca_processor = PathologyPCA(n_components=3, whiten=True)

    # Fit PCA
    pca_processor.fit(features, verbose=True)

    # Transform features
    pca_rgb = pca_processor.transform(
        features,
        apply_sigmoid=True,
        sigmoid_scale=2.0
    )

    print(f"PCA RGB shape: {pca_rgb.shape}")

    return pca_rgb


# =============================================================================
# Example 4: Create overlays
# =============================================================================

def example_create_overlays(patches, pca_rgb):
    """Create overlay visualizations."""

    # Get first patch and its PCA result
    patch = patches[0]
    pca_result = pca_rgb[0].permute(1, 2, 0).numpy()  # (16, 16, 3)

    # Create overlays with different alpha values
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    overlays = []

    for alpha in alphas:
        overlay = create_overlay_patch(patch, pca_result, alpha=alpha)
        overlays.append(overlay)
        print(f"Created overlay with alpha={alpha}")

    return overlays


# =============================================================================
# Example 5: Interactive ROI selection
# =============================================================================

def example_roi_selection(wsi_path, coords):
    """Interactive ROI selection workflow."""

    # Load overview
    overview_img, overview_scale, wsi_info = load_wsi_overview(
        wsi_path,
        overview_mag=10
    )
    print(f"Overview size: {overview_img.size}")
    print(f"Scale factor: {overview_scale}")

    # Create ROI selector
    roi_selector = ROISelector(overview_img, max_rois=5)

    # Note: This will open an interactive plot
    # Click to select ROIs, then close the window
    import matplotlib.pyplot as plt
    plt.show()

    # Get selected ROIs
    selected_rois = roi_selector.get_rois()
    print(f"Selected {len(selected_rois)} ROIs")

    # Find patches in each ROI
    roi_patches = {}
    for roi_idx, roi_coords in enumerate(selected_rois):
        patch_indices, roi_native = find_patches_in_roi(
            roi_coords,
            coords,
            overview_scale,
            patch_size=256
        )

        roi_patches[roi_idx] = {
            'indices': patch_indices,
            'roi_10x': roi_coords,
            'roi_native': roi_native,
            'num_patches': len(patch_indices)
        }

        print(f"ROI {roi_idx}: {len(patch_indices)} patches")

    return roi_patches


# =============================================================================
# Example 6: Visualize specific patches
# =============================================================================

def example_visualize_patches(patches, pca_rgb):
    """Visualize specific patches with PCA."""

    # Visualize first 5 patches
    indices = [0, 1, 2, 3, 4]

    visualize_patches_with_pca(
        patches,
        pca_rgb,
        indices=indices,
        max_display=5,
        title="First 5 Patches",
        alpha=0.3
    )


# =============================================================================
# Example 7: Custom PCA processing
# =============================================================================

def example_custom_pca(features):
    """Custom PCA with different parameters."""

    # Fit PCA without sigmoid
    pca_processor = PathologyPCA(n_components=3, whiten=False)
    pca_processor.fit(features, verbose=False)

    # Transform with custom settings
    pca_rgb = pca_processor.transform(
        features,
        apply_sigmoid=False,  # No sigmoid
        sigmoid_scale=1.0
    )

    # Manually normalize to [0, 1]
    pca_np = pca_rgb.numpy()
    pca_min = pca_np.min()
    pca_max = pca_np.max()
    pca_normalized = (pca_np - pca_min) / (pca_max - pca_min)

    pca_rgb_normalized = torch.from_numpy(pca_normalized)

    print(f"Custom PCA range: [{pca_min:.3f}, {pca_max:.3f}]")
    print(f"Normalized range: [0, 1]")

    return pca_rgb_normalized


# =============================================================================
# Example 8: Batch processing multiple slides
# =============================================================================

def example_batch_processing(slide_paths, h5_paths, model, model_name):
    """Process multiple slides in batch."""

    results = {}

    for slide_path, h5_path in zip(slide_paths, h5_paths):
        print(f"\nProcessing {slide_path}...")

        # Load data
        coords, _, _ = load_h5_from_trident(h5_path)
        patches, _ = load_wsi_and_extract_patches(slide_path, coords[:100])

        # Extract features
        extractor = DINOv3FeatureExtractor(model, model_name, 'cuda')
        features = extractor.extract_features(patches, batch_size=32, verbose=False)

        # Fit PCA
        pca_processor = PathologyPCA(n_components=3)
        pca_processor.fit(features, verbose=False)
        pca_rgb = pca_processor.transform(features)

        # Store results
        results[slide_path] = {
            'patches': patches,
            'features': features,
            'pca_rgb': pca_rgb,
            'pca_processor': pca_processor
        }

        print(f"✓ Processed {len(patches)} patches")

    return results


# =============================================================================
# Main function to run examples
# =============================================================================

def main():
    """Run selected examples."""

    print("PCA Pathology Utils - Modular Examples")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  1. Load data from H5 and WSI")
    print("  2. Extract DINOv3 features")
    print("  3. Fit and apply PCA")
    print("  4. Create overlays")
    print("  5. Interactive ROI selection")
    print("  6. Visualize specific patches")
    print("  7. Custom PCA processing")
    print("  8. Batch process multiple slides")
    print("\nNote: Update file paths in each example before running!")
    print("=" * 70)


if __name__ == "__main__":
    main()
