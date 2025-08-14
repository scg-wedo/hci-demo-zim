import os
from zim_anything import zim_model_registry, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns
import numpy as np
from PIL import Image
import torch
import time
import pickle

# Model setup
backbone = "vit_l"
ckpt_p = "results/zim_vit_l_2092"
model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

# Folder paths
input_path = r"D:\3d-recon\RoomSceneSegmentation\RoomSceneImage\S__20209990_0.jpg"
output_folder = r"D:\3d-recon\hci-demo-zim\output_automask_params"
os.makedirs(output_folder, exist_ok=True)

pred_iou_thresh_list = [0.5, 0.7, 0.9]
points_per_batch_list = [4, 8, 12]
stability_score_thresh_list = [0.5, 0.7, 0.9]


# Loop through all JPG images in the folder
for iou in pred_iou_thresh_list:
    for points in points_per_batch_list:
        for stability in stability_score_thresh_list:
            

            # Mask generator setup
            mask_generator = ZimAutomaticMaskGenerator(
                model, 
                pred_iou_thresh=iou,  # A filtering threshold in [0,1], using the model's predicted mask quality.
                points_per_batch=points, # Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
                stability_score_thresh=stability, # A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
            )

            print(f"Processing: {input_path}")
            start_time = time.time()
            image = np.array(Image.open(input_path).convert("RGB"))

            masks = mask_generator.generate(image)
            masks_vis = show_mat_anns(image, masks)

            print("len(masks)", len(masks))

            filename = f"params_iou_{iou}_points_{points}_stability_{stability}.jpg"
            output_path = os.path.join(output_folder, filename)

            output_image = Image.fromarray(masks_vis)
            output_image.save(output_path)

            with open(f"params_iou_{iou}_points_{points}_stability_{stability}.pkl", 'wb') as f:
                pickle.dump(masks, f)

            elapsed_time = time.time() - start_time
            print(f"Saved: {output_path} (Time: {elapsed_time:.2f} seconds)")

# masks:
# list : dict(str, any)
# A list over records for masks. Each record is a dict containing the following keys:
    # segmentation (dict(str, any) or np.ndarray): The mask. If
    # output_mode='binary_mask', is an array of shape HW. Otherwise,
    # is a dictionary containing the RLE.
    # bbox (list(float)): The box around the mask, in XYWH format.
    # area (int): The area in pixels of the mask.
    # predicted_iou (float): The model's own prediction of the mask's
    # quality. This is filtered by the pred_iou_thresh parameter.
    # point_coords (list(list(float))): The point coordinates input
    # to the model to generate this mask.
    # stability_score (float): A measure of the mask's quality. This
    # is filtered on using the stability_score_thresh parameter.
    # crop_box (list(float)): The crop of the image used to generate
    # the mask, given in XYWH format.


# class ZimAutomaticMaskGenerator:
#     def __init__(
#         self,
#         model: Zim,
#         points_per_side: Optional[int] = 32,
#         points_per_batch: int = 64,
#         pred_iou_thresh: float = 0.88,
#         stability_score_thresh: float = 0.9,
#         stability_score_offset: float = 0.1,
#         box_nms_thresh: float = 0.7,
#         crop_n_layers: int = 0,
#         crop_nms_thresh: float = 0.7,
#         crop_overlap_ratio: float = 512 / 1500,
#         crop_n_points_downscale_factor: int = 1,
#         point_grids: Optional[List[np.ndarray]] = None,
#         min_mask_region_area: int = 0,
#         output_mode: str = "binary_mask",
#     ) -> None:
#         """
#         Using a SAM model, generates masks for the entire image.
#         Generates a grid of point prompts over the image, then filters
#         low quality and duplicate masks. The default settings are chosen
#         for SAM with a ViT-H backbone.

#         Arguments:
#           model (Sam): The SAM model to use for mask prediction.
#           points_per_side (int or None): The number of points to be sampled
#             along one side of the image. The total number of points is
#             points_per_side**2. If None, 'point_grids' must provide explicit
#             point sampling.
#           points_per_batch (int): Sets the number of points run simultaneously
#             by the model. Higher numbers may be faster but use more GPU memory.
#           pred_iou_thresh (float): A filtering threshold in [0,1], using the
#             model's predicted mask quality.
#           stability_score_thresh (float): A filtering threshold in [0,1], using
#             the stability of the mask under changes to the cutoff used to binarize
#             the model's mask predictions.
#           stability_score_offset (float): The amount to shift the cutoff when
#             calculated the stability score.
#           box_nms_thresh (float): The box IoU cutoff used by non-maximal
#             suppression to filter duplicate masks.
#           crop_n_layers (int): If >0, mask prediction will be run again on
#             crops of the image. Sets the number of layers to run, where each
#             layer has 2**i_layer number of image crops.
#           crop_nms_thresh (float): The box IoU cutoff used by non-maximal
#             suppression to filter duplicate masks between different crops.
#           crop_overlap_ratio (float): Sets the degree to which crops overlap.
#             In the first crop layer, crops will overlap by this fraction of
#             the image length. Later layers with more crops scale down this overlap.
#           crop_n_points_downscale_factor (int): The number of points-per-side
#             sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
#           point_grids (list(np.ndarray) or None): A list over explicit grids
#             of points used for sampling, normalized to [0,1]. The nth grid in the
#             list is used in the nth crop layer. Exclusive with points_per_side.
#           min_mask_region_area (int): If >0, postprocessing will be applied
#             to remove disconnected regions and holes in masks with area smaller
#             than min_mask_region_area. Requires opencv.
#           output_mode (str): The form masks are returned in. Can be 'binary_mask',
#             'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
#             For large resolutions, 'binary_mask' may consume large amounts of
#             memory.
#         """