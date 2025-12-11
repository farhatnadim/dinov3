"""
Quick test script to verify pca_pathology_utils module imports correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all components can be imported."""

    print("Testing pca_pathology_utils imports...")
    print("=" * 60)

    try:
        # Test main imports
        from pca_pathology_utils import (
            load_h5_from_trident,
            load_wsi_and_extract_patches,
            load_wsi_overview,
            DINOv3FeatureExtractor,
            PathologyPCA,
            ROISelector,
            find_patches_in_roi,
            create_overlay_patch,
            visualize_patches_with_pca,
            visualize_roi_summary,
            PathologyPCAPipeline
        )
        print("✓ All imports successful!")

        # Test class instantiation
        print("\nTesting class instantiation...")

        # Test PathologyPCA
        pca = PathologyPCA(n_components=3)
        print(f"  ✓ PathologyPCA: n_components={pca.n_components}")

        # Test that functions exist
        print("\nTesting function signatures...")
        assert callable(load_h5_from_trident), "load_h5_from_trident not callable"
        assert callable(load_wsi_and_extract_patches), "load_wsi_and_extract_patches not callable"
        assert callable(create_overlay_patch), "create_overlay_patch not callable"
        print("  ✓ All functions callable")

        # Test class methods
        print("\nTesting class methods...")
        assert hasattr(PathologyPCA, 'fit'), "PathologyPCA missing fit method"
        assert hasattr(PathologyPCA, 'transform'), "PathologyPCA missing transform method"
        assert hasattr(DINOv3FeatureExtractor, 'extract_features'), "DINOv3FeatureExtractor missing extract_features"
        assert hasattr(PathologyPCAPipeline, 'load_patches'), "PathologyPCAPipeline missing load_patches"
        print("  ✓ All class methods present")

        print("\n" + "=" * 60)
        print("✓ All tests passed! Module is ready to use.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Import test failed!")
        print(f"Error: {e}")
        print("\nMake sure:")
        print("  1. pca_pathology_utils.py is in the same directory")
        print("  2. All dependencies are installed (torch, numpy, PIL, etc.)")
        print("  3. Trident is in your Python path if using WSI functions")
        return False


def show_module_info():
    """Display module information."""

    print("\nModule Information:")
    print("=" * 60)

    try:
        import pca_pathology_utils

        # Show docstring
        if pca_pathology_utils.__doc__:
            print("Description:")
            print(pca_pathology_utils.__doc__.strip())

        # List all public functions and classes
        print("\nAvailable functions and classes:")
        members = [m for m in dir(pca_pathology_utils) if not m.startswith('_')]

        classes = []
        functions = []

        for member in members:
            obj = getattr(pca_pathology_utils, member)
            if isinstance(obj, type):
                classes.append(member)
            elif callable(obj):
                functions.append(member)

        print(f"\n  Classes ({len(classes)}):")
        for cls in sorted(classes):
            print(f"    - {cls}")

        print(f"\n  Functions ({len(functions)}):")
        for func in sorted(functions):
            print(f"    - {func}")

        print("=" * 60)

    except ImportError as e:
        print(f"Could not import module: {e}")


if __name__ == "__main__":
    success = test_imports()

    if success:
        show_module_info()

        print("\nNext steps:")
        print("  1. See example_pca_usage.py for full pipeline example")
        print("  2. See example_modular_usage.py for component examples")
        print("  3. See README_PCA_PATHOLOGY.md for documentation")
    else:
        sys.exit(1)
