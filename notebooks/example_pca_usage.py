"""
Example usage of the PCA Pathology Visualization Pipeline

This script demonstrates how to use the pca_pathology_utils module
for end-to-end pathology PCA analysis.
"""

import sys
import torch
from pathlib import Path

# Import the utilities module
from pca_pathology_utils import (
    load_h5_from_trident,
    PathologyPCAPipeline,
    visualize_patches_with_pca
)


def main():
    """Main execution function."""

    # ===========================================================================
    # Configuration
    # ===========================================================================

    # Paths
    H5_PATH = "/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/0005f7aaab2800f6170c399693a96917.h5"
    WSI_PATH = "/media/nadim/Data/prostate-cancer-grade-assessment/train_images/0005f7aaab2800f6170c399693a96917.tiff"
    CHECKPOINT_PATH = "/home/nadim/Downloads/dinov3h16/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

    # Model configuration
    MODEL_NAME = "dinov3_vith16plus"
    DINOV3_LOCATION = "facebookresearch/dinov3"

    # Processing parameters
    PATCH_SIZE = 256
    TARGET_MAG = 20
    OVERVIEW_MAG = 10
    BATCH_SIZE = 32
    N_COMPONENTS = 3
    MAX_ROIS = 10

    # ===========================================================================
    # Step 1: Load patch coordinates from H5
    # ===========================================================================

    print("=" * 70)
    print("STEP 1: Loading patch coordinates")
    print("=" * 70)

    coords, features, metadata = load_h5_from_trident(H5_PATH)
    print(f"Loaded {len(coords)} patch coordinates")
    print(f"Feature dimension: {metadata['feature_dim']}")

    # ===========================================================================
    # Step 2: Load DINOv3 model
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Loading DINOv3 model")
    print("=" * 70)

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        pretrained=True,
        weights=CHECKPOINT_PATH
    )
    model.cuda()
    model.eval()

    print(f"✓ Loaded {MODEL_NAME}")

    # ===========================================================================
    # Step 3: Initialize pipeline
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 3: Initializing pipeline")
    print("=" * 70)

    pipeline = PathologyPCAPipeline(
        wsi_path=WSI_PATH,
        coords=coords,
        model=model,
        model_name=MODEL_NAME,
        device='cuda'
    )

    print("✓ Pipeline initialized")

    # ===========================================================================
    # Step 4: Load patches and extract features
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 4: Loading patches and extracting features")
    print("=" * 70)

    pipeline.load_patches(patch_size=PATCH_SIZE, target_mag=TARGET_MAG)
    pipeline.extract_features(batch_size=BATCH_SIZE)

    # ===========================================================================
    # Step 5: Fit PCA
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 5: Fitting PCA")
    print("=" * 70)

    pipeline.fit_pca(n_components=N_COMPONENTS)

    # ===========================================================================
    # Step 6: Visualize some random patches
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 6: Visualizing random patches")
    print("=" * 70)

    import numpy as np
    random_indices = np.random.choice(len(coords), size=10, replace=False).tolist()

    visualize_patches_with_pca(
        pipeline.patches,
        pipeline.pca_rgb,
        indices=random_indices,
        max_display=10,
        title="Random Patches with PCA Features"
    )

    # ===========================================================================
    # Step 7: Interactive ROI selection (optional)
    # ===========================================================================

    print("\n" + "=" * 70)
    print("STEP 7: Interactive ROI selection (optional)")
    print("=" * 70)
    print("Would you like to select ROIs interactively? (y/n)")

    response = input().strip().lower()

    if response == 'y':
        # Load overview
        pipeline.load_overview(overview_mag=OVERVIEW_MAG)

        # Select ROIs
        roi_patches = pipeline.select_rois_interactive(max_rois=MAX_ROIS)

        # Visualize ROI patches
        pipeline.visualize_roi_patches(roi_patches, max_display=20)
    else:
        print("Skipping ROI selection")

    print("\n" + "=" * 70)
    print("✓ Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
