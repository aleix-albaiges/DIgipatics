import os
import random
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch.utils.data import DataLoader
from tile_dataset import TileDataset

CLASS_COLORS = {
    0: [0,   0,   0  ],  # Background
    1: [0,   200, 0  ],  # Normal
    2: [255, 255, 0  ],  # Polyp
    3: [255, 165, 0  ],  # Serrated adenoma
    4: [255, 100, 100],  # Low-grade IN
    5: [255, 0,   0  ],  # High-grade IN
    6: [139, 0,   0  ],  # Adenocarcinoma
}
CLASS_NAMES = {
    0: "Background", 1: "Normal", 2: "Polyp",
    3: "Serrated adenoma", 4: "Low-grade IN",
    5: "High-grade IN", 6: "Adenocarcinoma",
}


def create_gaussian_kernel(size: int, sigma: float = None) -> torch.Tensor:
    """
    Create a 2D Gaussian weight map of shape (size, size) with peak value 1.0 at the center.

    The kernel is built as the outer product of two 1D Gaussians, then normalised so
    that the maximum weight is exactly 1.0.  Using sigma = size / 4 means the weight
    drops to ~e^{-2} ≈ 0.135 at the patch edge, giving a smooth taper that is ideal
    for 50% overlap (stride = size // 2).

    Args:
        size: Side length of the square kernel (should match tile_size, e.g. 224).
        sigma: Standard deviation of the Gaussian. Defaults to size / 4.

    Returns:
        torch.Tensor of shape (size, size), dtype=float32, with values in (0, 1].
    """
    if sigma is None:
        sigma = size / 4.0

    # 1D Gaussian centred at zero
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))

    # 2D kernel via outer product; normalise peak to 1.0
    kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    kernel = kernel / kernel.max()
    return kernel  # (size, size)


def tensor_to_rgb_uint8(patch: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized tensor [3,H,W] to uint8 RGB [H,W,3]."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = patch.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def save_debug_tiles(patches, pred_masks, coords_list, debug_dir, n_samples=20):
    """Save random tiles with their predicted segmentation mask side by side."""
    os.makedirs(debug_dir, exist_ok=True)
    indices = random.sample(range(len(patches)), min(n_samples, len(patches)))
    for idx in indices:
        patch = patches[idx]  # [3, H, W] float tensor, ImageNet-normalized
        pred = pred_masks[idx]  # [H, W] uint8

        # Denormalize for display
        img = tensor_to_rgb_uint8(patch).astype(np.float32) / 255.0

        # Build colored mask
        colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            colored[pred == cls] = color

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input tile")
        axes[0].axis("off")
        axes[1].imshow(colored)
        axes[1].set_title("Predicted mask")
        axes[1].axis("off")

        present = np.unique(pred)
        patches_legend = [
            mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255, label=CLASS_NAMES[c])
            for c in present
        ]
        fig.legend(handles=patches_legend, loc="lower center", ncol=len(present), fontsize=8)

        x, y = coords_list[idx]
        fname = os.path.join(debug_dir, f"tile_{idx}_x{x}_y{y}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=100)
        plt.close(fig)


def inference(args, tiles_mask, wsi, model, transform, device, tile_size_lvl0, offset_x, offset_y,
              tile_level, downsample, output_dir, num_classes, debug=False):
    """
    Performs patch-wise inference on a WSI guided by a tiles mask using
    Gaussian-weighted overlapping patch accumulation.

    Overlapping patches are run through the model. In Gaussian mode (args.gaussian),
    per-tile softmax probabilities are weighted by a 2D Gaussian kernel (peak = 1.0
    at centre) and either summed+normalised (args.gaussian_aggregation='sum') or
    max-pooled (args.gaussian_aggregation='max') across overlapping tiles before the
    final argmax. In center-crop mode the argmax of the central stride×stride region
    is pasted directly into the output canvas.

    Args:
        args: Argument namespace with tile_size, overlap, batch_size.
        tiles_mask: Binary mask where each pixel represents a tile.
        wsi: WSI wrapper with read_region_at_level method.
        model: Trained UNet_FiLM model.
        transform: Transform to apply to extracted patches.
        device: 'cuda' or 'cpu'.
        tile_size_lvl0: Tile footprint at level 0.
        offset_x, offset_y: Bounding box origin in level-0 coords.
        tile_level: Pyramid level (may be fractional, e.g. 2.5).
        downsample: Actual downsample factor (2 ** tile_level).
        output_dir: Directory where output PNG will be saved.
        num_classes: Number of segmentation classes (including background).

    Returns:
        prediction_image: Full prediction mask at tile-level resolution (np.ndarray uint8).
    """
    os.makedirs(output_dir, exist_ok=True)
    wsi_name = wsi.wsi_name()
    print("WSI name:", wsi_name)

    model.eval()

    dataset = TileDataset(
        tiles_mask=tiles_mask,
        wsi=wsi,
        tile_size=args.tile_size,
        tile_size_lvl0=tile_size_lvl0,
        offset_x=offset_x,
        offset_y=offset_y,
        overlap=args.overlap,
        tile_level=tile_level,
        downsample=downsample,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    tiles_mask_h, tiles_mask_w = tiles_mask.shape
    stride = args.tile_size - args.overlap      # pixels between tile origins (e.g. 112)
    margin = args.overlap // 2                  # half-overlap used when reading patches (e.g. 56)
    tile_h = tiles_mask_h * stride
    tile_w = tiles_mask_w * stride

    margin_map = None

    if args.gaussian:
        # ------------------------------------------------------------------ #
        # Accumulators for Gaussian-weighted blending (kept on CPU)           #
        # Both sum and max modes accumulate gaussian-weighted softmax probs.  #
        # pred_accumulator : weighted sum (sum) or running max (max) per pixel#
        # weight_accumulator: sum of Gaussian weights per pixel (sum only)    #
        # ------------------------------------------------------------------ #
        pred_accumulator = torch.zeros((num_classes, tile_h, tile_w), dtype=torch.float16)
        if args.gaussian_aggregation == 'sum':
            weight_accumulator = torch.zeros((1, tile_h, tile_w), dtype=torch.float16)

        # Build the Gaussian kernel once; shape (tile_size, tile_size)
        gaussian_kernel = create_gaussian_kernel(args.tile_size).half()  # values in (0, 1]
    else:
        # ------------------------------------------------------------------ #
        # Center-crop mode: argmax of the central stride×stride region is     #
        # pasted directly into a uint8 canvas — no blending, no float canvas. #
        # ------------------------------------------------------------------ #
        prediction_image = torch.zeros((tile_h, tile_w), dtype=torch.uint8)

    if debug:
        all_patches = []
        all_pred_masks = []
        all_coords = []
        debug_dir = os.path.join(output_dir, "debug_tiles")
        patches_dir = os.path.join(debug_dir, "patches")
        os.makedirs(patches_dir, exist_ok=True)
        tile_counter = 0

    forward_time = 0.0
    postproc_time = 0.0

    for batch in dataloader:
        patch_img, coords = batch["patch"], batch["coords"]
        t0 = time.perf_counter()
        with torch.no_grad():
            patch_tensor = patch_img.to(device)
            seg_map, class_logits = model(patch_tensor)
            # Detach before moving to CPU to free the computational graph and
            # prevent GPU/RAM memory from accumulating across the entire WSI.
            logits_cpu = seg_map.detach().cpu().half()   # (B, num_classes, tile_size, tile_size)
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        for i in range(coords[0].shape[0]):
            x_super, y_super = int(coords[0][i].item()), int(coords[1][i].item())

            # Tile-level top-left of the *grid cell* (without margin)
            x_tile = int(round((x_super - offset_x) / downsample))
            y_tile = int(round((y_super - offset_y) / downsample))

            patch_logits = logits_cpu[i]  # (num_classes, tile_size, tile_size)

            if args.gaussian:
                # The full tile_size patch starts margin pixels before the grid cell
                x_start = x_tile - margin
                y_start = y_tile - margin

                # ---------------------------------------------------------- #
                # Clip canvas coordinates to valid range.                     #
                # The patch/kernel slices are derived from the same offsets   #
                # so they are always synchronised with the canvas slices.     #
                # ---------------------------------------------------------- #
                cy0 = max(0, y_start)
                cx0 = max(0, x_start)
                cy1 = min(tile_h, y_start + args.tile_size)
                cx1 = min(tile_w, x_start + args.tile_size)

                # Corresponding region within the tile_size×tile_size prediction / kernel
                py0 = cy0 - y_start
                px0 = cx0 - x_start
                py1 = py0 + (cy1 - cy0)
                px1 = px0 + (cx1 - cx0)

                kernel_slice = gaussian_kernel[py0:py1, px0:px1]  # (h_clip, w_clip)

                # Softmax in float32 for stability, cast back to float16 before slicing
                patch_probs = torch.softmax(patch_logits.float(), dim=0).half()

                if args.gaussian_aggregation == 'sum':
                    pred_accumulator[:, cy0:cy1, cx0:cx1] += patch_probs[:, py0:py1, px0:px1] * kernel_slice
                    weight_accumulator[:, cy0:cy1, cx0:cx1] += kernel_slice
                else:
                    weighted = patch_probs[:, py0:py1, px0:px1] * kernel_slice
                    region = pred_accumulator[:, cy0:cy1, cx0:cx1]
                    pred_accumulator[:, cy0:cy1, cx0:cx1] = torch.maximum(region, weighted)
            else:
                # ---------------------------------------------------------- #
                # Center-crop: argmax over the central stride×stride region   #
                # and paste it at the grid-cell position in the canvas.       #
                # ---------------------------------------------------------- #
                patch_pred = torch.argmax(patch_logits, dim=0).to(torch.uint8)  # (tile_size, tile_size)
                center = patch_pred[margin:margin + stride, margin:margin + stride]  # (stride, stride)

                # Defensive clipping (grid cells are in-bounds by construction)
                cy0 = max(0, y_tile)
                cx0 = max(0, x_tile)
                cy1 = min(tile_h, y_tile + stride)
                cx1 = min(tile_w, x_tile + stride)
                py0 = cy0 - y_tile
                px0 = cx0 - x_tile
                py1 = py0 + (cy1 - cy0)
                px1 = px0 + (cx1 - cx0)

                prediction_image[cy0:cy1, cx0:cx1] = center[py0:py1, px0:px1]

            if debug:
                pred_mask_full = torch.argmax(patch_logits, dim=0).numpy().astype(np.uint8)
                patch_name = f"tile_{tile_counter:06d}_x{x_super}_y{y_super}.png"
                patch_rgb = tensor_to_rgb_uint8(patch_img[i])
                cv2.imwrite(os.path.join(patches_dir, patch_name), cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                tile_counter += 1
                all_patches.append(patch_img[i])
                all_pred_masks.append(pred_mask_full)
                all_coords.append((x_super, y_super))
        postproc_time += time.perf_counter() - t1

    t2 = time.perf_counter()
    if args.gaussian:
        if args.gaussian_aggregation == 'sum':
            # Normalise weighted-sum softmax probs by accumulated kernel weights,
            # then argmax to get the final class label per pixel.
            pred_accumulator = pred_accumulator.float() / weight_accumulator.float().clamp(min=1e-8)
        # max mode: no normalisation needed — argmax directly on the running max probs.
        if getattr(args, 'visualize_transitions', False):
            from transition_zones import compute_margin_map
            margin_map = compute_margin_map(pred_accumulator.float().numpy())
        prediction_image = torch.argmax(pred_accumulator, dim=0).numpy().astype(np.uint8)
    else:
        prediction_image = prediction_image.numpy()

    if debug:
        print(f"Saved {tile_counter} tile patches to {patches_dir}")
        print(f"Saving debug tiles to {debug_dir}...")
        save_debug_tiles(all_patches, all_pred_masks, all_coords, debug_dir, n_samples=20)

    cv2.imwrite(f"{output_dir}/{wsi_name}.png", prediction_image)
    postproc_time += time.perf_counter() - t2
    return prediction_image, margin_map, {"forward": forward_time, "postproc": postproc_time}
