import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import shutil
import pandas as pd
import copy

import torch
import albumentations as A

import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed

import scipy.ndimage as ndi

import time

import sys
sys.path.append('../../dpcore')
from dpcore.config import TILE_COORDS, TILE, CELL_BOXES, CELL_MASKS, CELL_INST_MASK, CELL_CAT_MASK, CELL_LABELS, CELL_CENTROIDS
from dpcore.data.wsi.oslide import Slide
from dpcore.data.dataset.tile import SlideTileDataset
from dpcore.data.loader.tile import TileDataLoader
import dpcore.transforms as T
import dpcore.transforms.functional as F
from dpcore.models.segmentation.dualunet import DualUNet
from dpcore.models.segmentation.parunet import ParUNet
from dpcore.models.classification import ImageClassifier


SLIDE     = "slide"
THUMBNAIL = "thumbnail"
INDICES   = "indices"
FG_MSK = "foreground_mask"

GAUSS_MSK = "gaussian_mask"
SEGM_MSK  = "segmentation_mask"

MODELS ={
    'dualunet': DualUNet,
    'parunet': ParUNet
}

def get_args():
    parser = argparse.ArgumentParser(description='WSI Inference')

    parser.add_argument('--num-cls',
                        type=int, default=2,
                        help='Number of classes for the model')

    # Whole Slide Image (WSI)
    parser.add_argument('--wsi-path',
                        type=str, required=True,
                        help='Path to the whole slide image file')
    
    # Tiling parameters
    parser.add_argument('--tile-size',
                        type=int, default=1024,
                        help='Size of the tiles to extract from the WSI')
    parser.add_argument('--tile-overlap',
                        type=int, default=100,
                        help='Overlap between tiles in pixels')
    parser.add_argument('--tile-mpp',
                        type=float, default=0.25,
                        help='Microns per pixel for the tiles')
    
    # Output Directory
    parser.add_argument('--output-dir',
                        type=str, required=True,
                        help='Directory to save the output results')

    # Cell Segmentation Model
    parser.add_argument('--cellseg-model-name',
                        type=str, default='dualunet',
                        choices=['dualunet', 'parunet'],
                        help='Model name')
    parser.add_argument('--cellseg-model-encoder',
                        type=str, default='resnet50',
                        help='Encoder name')
    parser.add_argument('--cellseg-model-checkpoint',
                        type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--cellseg-model-tile-size',
                        type=int, required=True,
                        help='Size of the tiles for the cell segmentation model')
    
    # Heatmap configuration
    parser.add_argument('--heatmap-blocks-tile',
                        type=int, default=14,
                        help='Number of bocks for each tile to obtain information about unique tile area')
    parser.add_argument('--heatmap-conv-size',
                        type=int, default=3,
                        help='Size of convolution filter to smooth the heatmap')
    parser.add_argument('--heatmap-neighbourhood',
                        type=int, default=3,
                        help='Neighbourhood')
    parser.add_argument('--heatmap-min-num-cells',
                        type=int, default=2,
                        help='Minimum number of cells in the neighbourhood')

    # Loading
    parser.add_argument('--batch-size',
                        type=int, default=4, 
                        help='Batch size')
    parser.add_argument('--num-workers',
                        type=int, default=8, 
                        help='Number of workers')
    
    return parser.parse_args()

def main(args):
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    tissue_dir = osp.join(args.output_dir, 'Tissue')
    os.makedirs(tissue_dir, exist_ok=True)
    segmentation_dir = osp.join(args.output_dir, 'Segmentation')
    os.makedirs(segmentation_dir, exist_ok=True)
    heatmap_dir = osp.join(args.output_dir, 'Heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmapq_dir = osp.join(args.output_dir, 'HeatmapQ')
    os.makedirs(heatmapq_dir, exist_ok=True)
    tmp_dir = osp.join(args.output_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load WSI
    slide = Slide(args.wsi_path)
    print(f"Loaded slide {slide.slide_name}")

    ########################################################################################
    ################################# TISSUE SEGMENTATION ##################################
    ########################################################################################
    
    t0 = time.time()
    print(f"Starting tissue segmentation for {slide.slide_name}...")
    
    # Tissue segmentation pipeline
    tissue_seg = T.Pipeline([
        # Get thumbnail from the slide
        T.Lambda(lambda slide : slide.get_thumbnail(), 
                 required=[SLIDE], returned=[THUMBNAIL]),
        # Separete foreground from background
        T.GraySaturationForegroundMask(
            threshold_g=0.95,
            kernel_size_g=8,
            median_size_g=12,
            threshold_s=0.015,
            kernel_size_s=6,
            sigma_s=4.0,
            image=THUMBNAIL, mask=FG_MSK),
        # Remove dark marks
        # T.RemoveMarks(
        #     threshold=0.3,
        #     kernel_size=8,
        #     mean_size=12,
        #     image=THUMBNAIL, mask=FG_MSK),
        # Remove small objects from the foreground mask
        T.RemoveSmallObjects(
            min_size=50,
            mask=FG_MSK),
        # Extract tile coordinates and indices
        T.SlideTileCoordinates(
            size=args.tile_size, overlap=args.tile_overlap, mpp=args.tile_mpp,
            slide=SLIDE, mask=FG_MSK,
            coords=TILE_COORDS, indices=INDICES,),
    ])

    # Run the pipeline on the slide
    data = tissue_seg({SLIDE : slide})

    # Save thumbnail
    thumbnail = data[THUMBNAIL]
    Image.fromarray(thumbnail).save(osp.join(tissue_dir, 'thumbnail.png'))

    # Save foreground mask
    fg_mask = data[FG_MSK]
    Image.fromarray((fg_mask * 255).astype(np.uint8)).save(
        osp.join(tissue_dir, 'tissue_tiles_mask.png'))
        
    # Save auxiliary metadata
    x0, y0, w, h = slide.bounds
    with open(osp.join(tissue_dir, 'WSI_bounds.txt'), 'w') as f:
        f.write(f"{x0} {y0}")

    t1 = time.time()
    print(f"Tissue segmentation completed in {t1 - t0:.2f} seconds.")

    ########################################################################################
    ################################## CELL SEGMENTATION ###################################
    ########################################################################################

    t2 = time.time()
    print(f"Starting cell segmentation on {data[TILE_COORDS].shape[0]} tiles for {slide.slide_name}...")

    # Preprocessing and post-processing cell segmentation pipelines
    cellseg_preproc  = get_tile_preproc_pipe(args.cellseg_model_tile_size)
    cellseg_postproc = T.Pipeline([
        # Convert output keys to numpy
        T.Pipeline([T.ToNumpy(input=k) \
            for k in [SEGM_MSK, GAUSS_MSK]]),
        # Transpose the segmentation mask to H,W,C
        T.Pipeline([T.Transpose(input=k, axes=(1,2,0)) \
            for k in [SEGM_MSK, GAUSS_MSK]]),
        # Resize output masks to original tiling size
        T.Pipeline([T.Resize(input=k, dsize=(args.tile_size, args.tile_size)) \
            for k in [SEGM_MSK, GAUSS_MSK]]),
        # Post-process DualUnet output
        T.PostprocessBiUNet(
            merge_mode='A',
            remove_small_objects=True, extend_boundaries=True,
            segmentation_mask=SEGM_MSK, gaussian_mask=GAUSS_MSK,
            labels=CELL_LABELS, centroids=CELL_CENTROIDS, instance_map=CELL_INST_MASK),
        # Remove edge cells
        T.RemoveEdgeCells(
            tile_size=args.tile_size, min_distance=(args.tile_overlap // 2),
            centroids=CELL_CENTROIDS, labels=CELL_LABELS, instance_mask=CELL_INST_MASK),
        # Add instance category mask
        T.InstanceCategoryMask(
            mask=CELL_INST_MASK, labels=CELL_LABELS,
            segmentation_mask=CELL_CAT_MASK),
    ])
    
    # Data loader
    cellseg_dataset, cellseg_loader = get_tile_loader(
        args, data[TILE_COORDS], cellseg_preproc,
        batch_size=args.batch_size, num_workers=args.num_workers)
        
    # Create the cell segmentation model
    cellseg_model = MODELS[args.cellseg_model_name](
        encoder_name=args.cellseg_model_encoder, 
        encoder_depth=5, 
        classes_s=args.num_cls+1,
        classes_c=1)
    cellseg_model.load_state_dict(
        torch.load(args.cellseg_model_checkpoint, map_location='cpu'))
    cellseg_model.to(device)
    cellseg_model.eval()

    # Calculate tile and overlap sizes at level 0 for metadata
    size         = args.tile_size - args.tile_overlap
    size_lvl0    = slide.adjust_size(current_size=size, current_mpp=args.tile_mpp, target_level=0)
    overlap_lvl0 = slide.adjust_size(current_size=args.tile_overlap, current_mpp=args.tile_mpp, target_level=0)

    # Define cell segmentation post-processing
    def process_and_save(args, save_path, postproc_pipe, 
                         pred_segm, pred_gauss):
        
        result = postproc_pipe({SEGM_MSK: pred_segm, GAUSS_MSK: pred_gauss,})
        labels = result[CELL_LABELS]
        cells = np.column_stack((np.arange(1,len(labels)+1), labels))
        counts_classes = np.bincount(labels, minlength=args.num_cls+1)[1:]
        centroids = result[CELL_CENTROIDS]
        
        # Heatmap
        tile_pn_map = create_tile_block_map(cells, centroids, blocks_tile=args.heatmap_blocks_tile)

        np.savez_compressed(
            save_path,
            mask = result[CELL_INST_MASK],
            mask_class = result[CELL_CAT_MASK][..., 0],
            cells = cells, cells_class = counts_classes,
            pn_map = tile_pn_map,
        )
    
    # Run inference
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        with torch.no_grad():
            for j, batch in enumerate(cellseg_loader):
                
                # Get images and coordinates
                images = batch[TILE].to(device)
                coords = batch[TILE_COORDS]
                
                # Forward pass
                pred_segm, pred_gauss = cellseg_model(images)
                pred_segm, pred_gauss = pred_segm.cpu(), pred_gauss.cpu()
                
                # Post-process the predictions (in parallel)
                for jj in range(len(pred_segm)):
                    
                    # Get tile coordinates
                    tx, ty = coords[jj][0].item(), coords[jj][1].item()
                    save_path = osp.join(
                        tmp_dir,
                        f"{slide.slide_name} (x={tx + overlap_lvl0//2}, y={ty + overlap_lvl0//2}, w={size_lvl0}, h={size_lvl0}).npz"
                    )

                    # Submit the future
                    futures.append(
                        executor.submit(
                            process_and_save,
                            args,
                            save_path,
                            cellseg_postproc,
                            pred_segm[jj],
                            pred_gauss[jj]
                        )
                    )
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with exception: {e}")
    
    t3 = time.time()
    print(f"Cell segmentation completed in {t3 - t2:.2f} seconds.")

    ########################################################################################
    ##################################     MERGING       ###################################
    ########################################################################################
    
    t4 = time.time()
    print(f"Starting merging results for {slide.slide_name}...")

    # Define function to save the final result
    def save_final_result(save_path, mask, mask_class, cells, cells_class):
        np.savez_compressed(
            save_path,
            mask=mask,
            mask_class=mask_class,
            cells=cells,
            cells_class=cells_class
        )
    
    # Run merging
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        coordinates = data[TILE_COORDS]
        indices = data[INDICES]
        init_label = 0
        slide_pn_map = np.zeros((data[FG_MSK].shape[0], data[FG_MSK].shape[1],
                                 args.heatmap_blocks_tile, args.heatmap_blocks_tile, args.num_cls), dtype=np.uint8)
        for coord, ind in zip(coordinates, indices):
            
            # Original information
            i, j = ind
            tx, ty = coord[0].item(), coord[1].item()
            name = f"{slide.slide_name} (x={tx + overlap_lvl0//2}, y={ty + overlap_lvl0//2}, w={size_lvl0}, h={size_lvl0}).npz"
            
            tile_pred = np.load(os.path.join(tmp_dir, name))
            instances_mask = tile_pred['mask']
            cells          = tile_pred['cells']
            pn_map         = tile_pred['pn_map']

            # Update identifiers
            instances_mask[instances_mask > 0] += init_label
            cells[:,0] += init_label
            init_label += len(cells)

            # Heatmap
            slide_pn_map[j, i] = pn_map

            # Submit the future
            futures.append(
                executor.submit(save_final_result, osp.join(segmentation_dir, name),
                                instances_mask, tile_pred['mask_class'],
                                cells, tile_pred['cells_class'])
            )

        # Process results as they complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with exception: {e}")
    
    # Heatmap processing
    slide_pn_map = np.concatenate(np.concatenate(slide_pn_map, axis=1), axis=1)
    slide_percentage_map, slide_percentage_qmap = create_percentage_map(slide_pn_map)
    neigh = apply_filter(slide_pn_map, size=args.heatmap_conv_size)
    slide_percentage_map_conv, slide_percentage_qmap_conv = create_percentage_map(neigh,
                                                                                  neighbourhood=args.heatmap_neighbourhood, min_num_cells=args.heatmap_min_num_cells)
    
    # Save heatmap maps
    Image.fromarray((slide_percentage_map * 255).astype(np.uint8)).save(
        osp.join(heatmap_dir, 'slide_percentage_map.png'))
    Image.fromarray(slide_percentage_qmap).save(
        osp.join(heatmapq_dir, 'slide_percentage_qmap.png'))
    Image.fromarray((slide_percentage_map_conv * 255).astype(np.uint8)).save(
        osp.join(heatmap_dir, 'slide_percentage_map_conv.png'))
    Image.fromarray(slide_percentage_qmap_conv).save(
        osp.join(heatmapq_dir, 'slide_percentage_qmap_conv.png'))
    
    t5 = time.time()
    print(f"Merging results completed in {t5 - t4:.2f} seconds.")

    # Remove tmp file (inference results)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Complete WSI analysis completed in {t5 - t0:.2f} seconds ({(t5 - t0)/60:.2f} minutes).")


def get_tile_preproc_pipe(tile_size):
    """
    Preprocessing pipeline for tiles.
    """
    return T.Pipeline([
        T.AlbumentationsTransform(A.Compose([
            A.Resize(tile_size, tile_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]), image=TILE),
        T.Transpose(axes=(2,0,1), input=TILE),
        T.ToDtype(dtype=np.float32, input=TILE),
        T.ToTensor(input=TILE)
    ])

def get_tile_loader(args, coords, preprocessing, batch_size=4, num_workers=8):
    # Dataset and preprocessing
    dataset = SlideTileDataset(
        slide_path=args.wsi_path, tile_coords=coords,
        tile_size=args.tile_size, tile_mpp=args.tile_mpp,
        transform=preprocessing,
    )
    loader = TileDataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False, drop_last=False)
    return dataset, loader

def apply_filter(pn_map, size=3):
    
    # Compute the constant filter
    filter = np.ones((size,size))
    
    # Apply the filter
    neigh = np.zeros_like(pn_map)
    for i in range(pn_map.shape[-1]):
        neigh[:,:,i] = F.convolve(pn_map[:,:,i], filter, mode='constant')
    return neigh

def create_tile_block_map(lbls: np.ndarray, centroids: np.ndarray,
                          tile_size: int = 1024, tile_overlap: int = 100,
                          blocks_tile: int = 16):

    # Block coordinates
    pixels_block = int((tile_size-tile_overlap) // blocks_tile)
    indices_block = ((centroids - tile_overlap//2) // pixels_block).astype(int)
    indices_block = np.clip(indices_block, 0, blocks_tile-1)

    # Counts block map
    block_map = np.zeros((blocks_tile, blocks_tile, args.num_cls), dtype=int)
    np.add.at(block_map, (indices_block[:,1], indices_block[:,0], lbls[:,1]-1), 1)

    return block_map

def create_percentage_map(block_map: np.ndarray,
                          neighbourhood: int = None, min_num_cells: int = None):
    
    # Compute positivity percentage
    denominator = block_map[:,:,0] + block_map[:,:,1]
    pct = np.divide(block_map[:,:,1], denominator, where=denominator != 0)
    
    # Quantize the percentage (3 levels)
    pct_q = np.zeros_like(pct).astype(np.uint8)
    pct_q[pct < 0.2] = 1
    pct_q[(0.2 <= pct) & (pct < 0.4)] = 2
    pct_q[0.4 <= pct] = 3

    # Make sure that there is at least a certain number of cells
    if neighbourhood is not None and min_num_cells is not None:
        
        # Mask
        mask = copy.deepcopy(block_map)
        if neighbourhood > 1:
            mask = apply_filter(mask, size=neighbourhood)
        mask = mask.sum(axis=2)
        mask = mask > min_num_cells

        # Filter results
        pct   = np.where(mask, pct, 0)
        pct_q = np.where(mask, pct_q, 0)
    
    return pct, pct_q


if __name__ == '__main__':
    args = get_args()
    main(args)
