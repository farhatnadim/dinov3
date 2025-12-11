"""
PCA Pathology Visualization Utilities

This module provides tools for visualizing DINOv3 features on pathology images using PCA.
Includes functionality for:
- Loading WSI images and extracting patches
- Computing DINOv3 features
- Fitting and visualizing PCA
- Interactive ROI selection
- Creating overlay visualizations

Author: Generated for DINOv3 pathology analysis
"""

import os
import sys
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from PIL import Image
import h5py

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Normalize function (replaces torchvision.transforms.functional)
def normalize_tensor(tensor, mean, std):
    """Normalize a tensor with mean and std."""
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean) / std

def to_tensor(img):
    """Convert PIL Image to tensor."""
    if isinstance(img, Image.Image):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_h5_from_trident(h5_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load patch coordinates and features from a Trident-generated H5 file.

    Parameters
    ----------
    h5_path : str
        Path to the H5 file containing coords and features

    Returns
    -------
    coords : np.ndarray
        Array of patch coordinates (N, 2) where N is number of patches
    features : np.ndarray
        Array of feature vectors (N, D) where D is feature dimension
    metadata : dict
        Dictionary containing H5 file metadata and attributes
    """
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        features = f['features'][:]

        metadata = {}
        for key in ['coords', 'features']:
            if key in f and hasattr(f[key], 'attrs'):
                metadata[key] = dict(f[key].attrs)

        metadata['h5_keys'] = list(f.keys())
        metadata['num_patches'] = len(coords)
        metadata['feature_dim'] = features.shape[1] if len(features.shape) > 1 else 1

    return coords, features, metadata


def load_wsi_and_extract_patches(
    wsi_path: str,
    coords: np.ndarray,
    patch_size: int = 256,
    target_mag: int = 20
) -> Tuple[list, Dict]:
    """
    Load a whole-slide image and extract patches at specified coordinates.

    Parameters
    ----------
    wsi_path : str
        Path to the WSI file (.tiff, .svs, etc.)
    coords : np.ndarray
        Array of patch coordinates (N, 2) at level 0
    patch_size : int
        Size of patches to extract (default: 256)
    target_mag : int
        Target magnification for extraction (default: 20x)

    Returns
    -------
    patches : list
        List of PIL Image patches
    wsi_info : dict
        Dictionary containing WSI metadata
    """
    from trident import load_wsi

    wsi = load_wsi(wsi_path, lazy_init=False)

    wsi_info = {
        'filename': os.path.basename(wsi_path),
        'width': wsi.width,
        'height': wsi.height,
        'magnification': wsi.mag,
        'num_coords': len(coords),
        'patch_size': patch_size,
        'target_mag': target_mag
    }

    patcher = wsi.create_patcher(
        patch_size=patch_size,
        src_mag=wsi.mag,
        dst_mag=target_mag,
        custom_coords=coords,
        coords_only=False,
        pil=True
    )

    patches = []
    for i, (patch, x, y) in enumerate(patcher):
        patches.append(patch)

    print(f"Extracted {len(patches)} patches from {wsi_info['filename']}")

    return patches, wsi_info


def load_wsi_overview(wsi_path: str, overview_mag: int = 10) -> Tuple[Image.Image, float, Dict]:
    """
    Load WSI overview at specified magnification.

    Parameters
    ----------
    wsi_path : str
        Path to the WSI file
    overview_mag : int
        Magnification for overview (default: 10x)

    Returns
    -------
    overview_img : PIL.Image
        Overview image
    overview_scale : float
        Scale factor from native to overview magnification
    wsi_info : dict
        WSI metadata
    """
    from trident import load_wsi

    wsi = load_wsi(wsi_path, lazy_init=False)
    overview_scale = overview_mag / wsi.mag

    overview_img = wsi.read_region((0, 0), wsi.mag, overview_mag, (wsi.width, wsi.height))

    wsi_info = {
        'filename': os.path.basename(wsi_path),
        'width': wsi.width,
        'height': wsi.height,
        'magnification': wsi.mag,
        'overview_mag': overview_mag,
        'overview_scale': overview_scale
    }

    return overview_img, overview_scale, wsi_info


# ==============================================================================
# Feature Extraction
# ==============================================================================

class DINOv3FeatureExtractor:
    """Extract DINOv3 features from patches."""

    def __init__(self, model, model_name: str, device: str = 'cuda'):
        """
        Initialize feature extractor.

        Parameters
        ----------
        model : torch.nn.Module
            DINOv3 model
        model_name : str
            Name of the model (e.g., 'dinov3_vith16plus')
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.model_name = model_name
        self.device = device

        # Model layer mapping
        self.model_to_layers = {
            'dinov3_vits16': 12,
            'dinov3_vits16plus': 12,
            'dinov3_vitb16': 12,
            'dinov3_vitl16': 24,
            'dinov3_vith16plus': 32,
            'dinov3_vit7b16': 40,
        }

        self.n_layers = self.model_to_layers.get(model_name, 32)

        # ImageNet normalization
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)

    def normalize_patches(self, patches: List[Image.Image]) -> torch.Tensor:
        """
        Normalize patches to ImageNet mean/std.

        Parameters
        ----------
        patches : list of PIL.Image
            List of patches to normalize

        Returns
        -------
        patches_batch : torch.Tensor
            Normalized batch of patches (N, 3, H, W)
        """
        normalized_patches = []
        for patch in patches:
            patch_tensor = to_tensor(patch)
            patch_normalized = normalize_tensor(
                patch_tensor,
                mean=self.imagenet_mean,
                std=self.imagenet_std
            )
            normalized_patches.append(patch_normalized)

        return torch.stack(normalized_patches)

    def extract_features(
        self,
        patches: List[Image.Image],
        batch_size: int = 32,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Extract DINOv3 features from patches.

        Parameters
        ----------
        patches : list of PIL.Image
            Patches to extract features from
        batch_size : int
            Batch size for processing (default: 32)
        verbose : bool
            Print progress (default: True)

        Returns
        -------
        features : torch.Tensor
            Extracted features (N, feature_dim, h, w)
        """
        if verbose:
            print(f"Computing DINOv3 features for {len(patches)} patches...")
            print(f"Using {self.n_layers} layers from {self.model_name}")

        # Normalize patches
        patches_batch = self.normalize_patches(patches)

        all_features = []
        num_batches = (len(patches) + batch_size - 1) // batch_size

        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(patches))

                    batch = patches_batch[start_idx:end_idx].to(self.device)

                    feats = self.model.get_intermediate_layers(
                        batch,
                        n=range(self.n_layers),
                        reshape=True,
                        norm=True
                    )

                    batch_feats = feats[-1].detach().cpu()
                    all_features.append(batch_feats)

                    if verbose and ((i + 1) % 10 == 0 or (i + 1) == num_batches):
                        print(f"  Processed {end_idx}/{len(patches)} patches")

        features = torch.cat(all_features, dim=0)

        if verbose:
            print(f"Feature extraction complete! Shape: {features.shape}")

        return features


# ==============================================================================
# PCA Processing
# ==============================================================================

class PathologyPCA:
    """Fit and apply PCA to pathology features."""

    def __init__(self, n_components: int = 3, whiten: bool = True):
        """
        Initialize PCA processor.

        Parameters
        ----------
        n_components : int
            Number of PCA components (default: 3 for RGB)
        whiten : bool
            Whether to whiten features (default: True)
        """
        self.pca = PCA(n_components=n_components, whiten=whiten)
        self.n_components = n_components
        self.is_fitted = False

    def fit(self, features: torch.Tensor, verbose: bool = True) -> 'PathologyPCA':
        """
        Fit PCA on features.

        Parameters
        ----------
        features : torch.Tensor
            Features with shape (num_patches, feature_dim, h, w)
        verbose : bool
            Print statistics (default: True)

        Returns
        -------
        self : PathologyPCA
            Returns self for chaining
        """
        # Reshape from (N, D, H, W) to (N*H*W, D)
        num_patches, feature_dim, h, w = features.shape
        features_reshaped = features.permute(0, 2, 3, 1)
        features_flat = features_reshaped.reshape(-1, feature_dim)

        if verbose:
            print(f"Fitting PCA on {features_flat.shape[0]} feature vectors...")

        self.pca.fit(features_flat.numpy())
        self.is_fitted = True

        if verbose:
            print(f"✓ PCA fitted successfully!")
            print(f"  Explained variance: {self.pca.explained_variance_ratio_}")
            print(f"  Total variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        return self

    def transform(
        self,
        features: torch.Tensor,
        apply_sigmoid: bool = True,
        sigmoid_scale: float = 2.0
    ) -> torch.Tensor:
        """
        Transform features using fitted PCA.

        Parameters
        ----------
        features : torch.Tensor
            Features with shape (num_patches, feature_dim, h, w)
        apply_sigmoid : bool
            Apply sigmoid for vibrant colors (default: True)
        sigmoid_scale : float
            Scaling factor before sigmoid (default: 2.0)

        Returns
        -------
        pca_rgb : torch.Tensor
            PCA-transformed features (num_patches, n_components, h, w)
        """
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")

        # Reshape features
        num_patches, feature_dim, h, w = features.shape
        features_reshaped = features.permute(0, 2, 3, 1)
        features_flat = features_reshaped.reshape(-1, feature_dim)

        # Apply PCA
        pca_features = self.pca.transform(features_flat.numpy())
        pca_features_reshaped = pca_features.reshape(num_patches, h, w, self.n_components)
        pca_features_tensor = torch.from_numpy(pca_features_reshaped)

        # Apply sigmoid for vibrant colors
        if apply_sigmoid:
            pca_rgb = torch.nn.functional.sigmoid(pca_features_tensor.mul(sigmoid_scale))
        else:
            pca_rgb = pca_features_tensor

        # Permute to (N, C, H, W)
        pca_rgb = pca_rgb.permute(0, 3, 1, 2)

        return pca_rgb


# ==============================================================================
# ROI Selection
# ==============================================================================

class ROISelector:
    """Interactive ROI selector for WSI overview."""

    def __init__(self, img: Image.Image, max_rois: int = 10):
        """
        Initialize ROI selector.

        Parameters
        ----------
        img : PIL.Image
            Overview image to select ROIs from
        max_rois : int
            Maximum number of ROIs to select (default: 10)
        """
        self.img = img
        self.max_rois = max_rois
        self.rois = []
        self.current_point = None
        self.rectangles = []

        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 16))
        self.ax.imshow(self.img)
        self.ax.set_title(
            f"Click to select ROIs (0/{max_rois})\nClick 2 points per ROI rectangle",
            fontsize=14
        )
        self.ax.axis('off')

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        if len(self.rois) >= self.max_rois:
            print(f"Maximum {self.max_rois} ROIs reached!")
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.current_point is None:
            self.current_point = (x, y)
            self.ax.plot(x, y, 'go', markersize=10)
            self.fig.canvas.draw()
            print(f"First corner: ({x}, {y}). Click second corner...")

        else:
            x1, y1 = self.current_point
            x2, y2 = x, y

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            self.rois.append((x1, y1, x2, y2))

            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='yellow', facecolor='none')
            self.ax.add_patch(rect)
            self.rectangles.append(rect)

            self.ax.text(x1, y1 - 10, f'ROI {len(self.rois)}',
                        color='yellow', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            self.ax.set_title(
                f"Click to select ROIs ({len(self.rois)}/{self.max_rois})\n"
                f"Click 2 points per ROI rectangle",
                fontsize=14
            )

            self.fig.canvas.draw()
            print(f"ROI {len(self.rois)}: ({x1}, {y1}) to ({x2}, {y2})")

            self.current_point = None

            if len(self.rois) >= self.max_rois:
                print(f"\n✓ All {self.max_rois} ROIs selected!")

    def get_rois(self) -> List[Tuple[int, int, int, int]]:
        """Return list of selected ROIs as (x1, y1, x2, y2) tuples."""
        return self.rois


def find_patches_in_roi(
    roi_coords_10x: Tuple[int, int, int, int],
    patch_coords_native: np.ndarray,
    overview_scale: float,
    patch_size: int = 256
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Find patches that fall within an ROI.

    Parameters
    ----------
    roi_coords_10x : tuple
        ROI coordinates at overview resolution (x1, y1, x2, y2)
    patch_coords_native : np.ndarray
        Patch coordinates at native resolution (N, 2)
    overview_scale : float
        Scale factor from native to overview magnification
    patch_size : int
        Size of patches in pixels (default: 256)

    Returns
    -------
    indices : np.ndarray
        Indices of patches within the ROI
    roi_native : tuple
        ROI coordinates at native resolution (x1, y1, x2, y2)
    """
    x1_10x, y1_10x, x2_10x, y2_10x = roi_coords_10x

    scale_factor = 1.0 / overview_scale
    x1_native = int(x1_10x * scale_factor)
    y1_native = int(y1_10x * scale_factor)
    x2_native = int(x2_10x * scale_factor)
    y2_native = int(y2_10x * scale_factor)

    patch_centers = patch_coords_native + patch_size // 2

    within_x = (patch_centers[:, 0] >= x1_native) & (patch_centers[:, 0] <= x2_native)
    within_y = (patch_centers[:, 1] >= y1_native) & (patch_centers[:, 1] <= y2_native)
    within_roi = within_x & within_y

    indices = np.where(within_roi)[0]

    return indices, (x1_native, y1_native, x2_native, y2_native)


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def create_overlay_patch(
    original_patch: Image.Image,
    pca_result: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Create overlay of PCA result on original patch.

    Parameters
    ----------
    original_patch : PIL.Image or np.ndarray
        Original patch (256x256)
    pca_result : np.ndarray
        PCA result (16x16x3) in [0, 1] range
    alpha : float
        Blending factor (0=full PCA, 1=full original)

    Returns
    -------
    overlay : np.ndarray
        Overlaid image (256x256x3) as uint8
    """
    if hasattr(original_patch, 'size'):
        original_np = np.array(original_patch)
    else:
        original_np = original_patch

    pca_pil = Image.fromarray((pca_result * 255).astype(np.uint8))
    pca_resized_pil = pca_pil.resize(
        (original_np.shape[1], original_np.shape[0]),
        Image.BILINEAR
    )
    pca_resized = np.array(pca_resized_pil).astype(np.float32) / 255.0

    overlay = (alpha * original_np + (1 - alpha) * pca_resized * 255).astype(np.uint8)

    return overlay


def visualize_patches_with_pca(
    patches: List[Image.Image],
    pca_rgb: torch.Tensor,
    indices: Optional[List[int]] = None,
    max_display: int = 20,
    title: str = "Patches with PCA",
    alpha: float = 0.3
):
    """
    Visualize patches with their PCA features.

    Parameters
    ----------
    patches : list of PIL.Image
        Original patches
    pca_rgb : torch.Tensor
        PCA-transformed features (N, 3, H, W)
    indices : list of int, optional
        Indices of patches to display (default: all)
    max_display : int
        Maximum patches to display (default: 20)
    title : str
        Figure title
    alpha : float
        Overlay blending factor (default: 0.3)
    """
    if indices is None:
        indices = list(range(len(patches)))

    indices = indices[:max_display]

    cols = min(5, len(indices))
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 3, rows * 9))

    if rows == 1:
        axes = axes.reshape(3, -1)

    fig.suptitle(f"{title} ({len(indices)} patches)", fontsize=16)

    for idx, patch_idx in enumerate(indices):
        col = idx % cols
        row_base = (idx // cols) * 3

        patch = patches[patch_idx]
        pca_result = pca_rgb[patch_idx].permute(1, 2, 0).numpy()
        overlay = create_overlay_patch(patch, pca_result, alpha=alpha)

        axes[row_base, col].imshow(patch)
        axes[row_base, col].axis('off')
        axes[row_base, col].set_title(f"Patch {patch_idx}", fontsize=8)

        axes[row_base + 1, col].imshow(pca_result)
        axes[row_base + 1, col].axis('off')

        axes[row_base + 2, col].imshow(overlay)
        axes[row_base + 2, col].axis('off')

    # Turn off unused subplots
    total_subplots = rows * cols
    for idx in range(len(indices), total_subplots):
        col = idx % cols
        row_base = (idx // cols) * 3
        axes[row_base, col].axis('off')
        axes[row_base + 1, col].axis('off')
        axes[row_base + 2, col].axis('off')

    if rows > 0:
        axes[0, 0].set_ylabel('Original', fontsize=10, rotation=0, ha='right', va='center')
        axes[1, 0].set_ylabel('PCA', fontsize=10, rotation=0, ha='right', va='center')
        axes[2, 0].set_ylabel('Overlay', fontsize=10, rotation=0, ha='right', va='center')

    plt.tight_layout()
    plt.show()


def visualize_roi_summary(
    overview_img: Image.Image,
    roi_patches: Dict,
    coords: np.ndarray,
    overview_scale: float
):
    """
    Visualize summary of all selected ROIs.

    Parameters
    ----------
    overview_img : PIL.Image
        Overview image
    roi_patches : dict
        Dictionary mapping ROI indices to patch info
    coords : np.ndarray
        All patch coordinates at native resolution
    overview_scale : float
        Scale factor from native to overview
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(overview_img)
    ax.set_title(f"Selected ROIs Summary - {len(roi_patches)} regions", fontsize=16)

    for roi_idx, roi_info in roi_patches.items():
        x1, y1, x2, y2 = roi_info['roi_10x']
        num_patches = roi_info['num_patches']

        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height,
                        linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)

        label_text = f"ROI {roi_idx + 1}\n{num_patches} patches"
        ax.text(x1, y1 - 20, label_text,
               color='yellow', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        patch_indices = roi_info['indices']
        if len(patch_indices) > 0:
            roi_patch_coords = coords[patch_indices] * overview_scale
            ax.scatter(roi_patch_coords[:, 0], roi_patch_coords[:, 1],
                      c='lime', s=10, alpha=0.6, marker='s')

    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Pipeline Class
# ==============================================================================

class PathologyPCAPipeline:
    """End-to-end pipeline for pathology PCA visualization."""

    def __init__(
        self,
        wsi_path: str,
        coords: np.ndarray,
        model,
        model_name: str,
        device: str = 'cuda'
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        wsi_path : str
            Path to WSI file
        coords : np.ndarray
            Patch coordinates
        model : torch.nn.Module
            DINOv3 model
        model_name : str
            Model name
        device : str
            Device ('cuda' or 'cpu')
        """
        self.wsi_path = wsi_path
        self.coords = coords
        self.model = model
        self.model_name = model_name
        self.device = device

        self.patches = None
        self.features = None
        self.pca_processor = None
        self.pca_rgb = None
        self.overview_img = None
        self.overview_scale = None

    def load_patches(self, patch_size: int = 256, target_mag: int = 20):
        """Load all patches from WSI."""
        print(f"Loading {len(self.coords)} patches from WSI...")
        self.patches, self.wsi_info = load_wsi_and_extract_patches(
            self.wsi_path,
            self.coords,
            patch_size=patch_size,
            target_mag=target_mag
        )
        return self

    def extract_features(self, batch_size: int = 32):
        """Extract DINOv3 features from patches."""
        extractor = DINOv3FeatureExtractor(self.model, self.model_name, self.device)
        self.features = extractor.extract_features(self.patches, batch_size=batch_size)
        return self

    def fit_pca(self, n_components: int = 3):
        """Fit PCA on features."""
        self.pca_processor = PathologyPCA(n_components=n_components)
        self.pca_processor.fit(self.features)
        self.pca_rgb = self.pca_processor.transform(self.features)
        return self

    def load_overview(self, overview_mag: int = 10):
        """Load WSI overview."""
        self.overview_img, self.overview_scale, _ = load_wsi_overview(
            self.wsi_path,
            overview_mag=overview_mag
        )
        return self

    def select_rois_interactive(self, max_rois: int = 10) -> Dict:
        """
        Interactive ROI selection.

        Returns
        -------
        roi_patches : dict
            Dictionary mapping ROI indices to patch info
        """
        if self.overview_img is None:
            self.load_overview()

        print("Interactive ROI Selection")
        print("=" * 60)
        print("Click 2 points to define each rectangular ROI")
        print(f"Select up to {max_rois} ROIs")
        print("Close the plot when done")
        print("=" * 60)

        roi_selector = ROISelector(self.overview_img, max_rois=max_rois)
        plt.show()

        selected_rois = roi_selector.get_rois()

        # Find patches in each ROI
        roi_patches = {}
        for roi_idx, roi_coords in enumerate(selected_rois):
            patch_indices, roi_native = find_patches_in_roi(
                roi_coords,
                self.coords,
                self.overview_scale
            )

            roi_patches[roi_idx] = {
                'indices': patch_indices,
                'roi_10x': roi_coords,
                'roi_native': roi_native,
                'num_patches': len(patch_indices)
            }

            print(f"ROI {roi_idx + 1}: {len(patch_indices)} patches")

        return roi_patches

    def visualize_roi_patches(self, roi_patches: Dict, max_display: int = 20):
        """Visualize patches from selected ROIs."""
        for roi_idx, roi_info in roi_patches.items():
            patch_indices = roi_info['indices']

            if len(patch_indices) == 0:
                print(f"ROI {roi_idx + 1}: No patches, skipping...")
                continue

            visualize_patches_with_pca(
                self.patches,
                self.pca_rgb,
                indices=patch_indices.tolist(),
                max_display=max_display,
                title=f"ROI {roi_idx + 1}"
            )

        # Show summary
        visualize_roi_summary(
            self.overview_img,
            roi_patches,
            self.coords,
            self.overview_scale
        )
