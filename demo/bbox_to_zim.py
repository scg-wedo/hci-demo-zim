import os, sys
sys.path.append(os.getcwd())

import json
import torch
import numpy as np
import cv2
from PIL import Image
from zim_anything import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator

image_path = r"D:\3d-recon\WESHAPExZIM\506b85bb-2b19-400b-93fc-ea661aa35db5\input.jpg"
auto_detect_path =  r"D:\3d-recon\WESHAPExZIM\506b85bb-2b19-400b-93fc-ea661aa35db5\auto_detect.json"
folder_path = r"D:\3d-recon\WESHAPExZIM\506b85bb-2b19-400b-93fc-ea661aa35db5"

# READ AUTO DETECT JSON
with open(auto_detect_path, "r", encoding="utf-8") as f:
    auto_detect = json.load(f)

wall_bboxes = np.array(auto_detect["wall_bboxes"])
floor_bboxes = np.array(auto_detect["floor_bboxes"])
print("wall_bboxes", wall_bboxes)
print("floor_bboxes", floor_bboxes)

# LOAD ZIM MODEL
backbone = "vit_l"
ckpt_mat = "./results/zim_vit_l_2092"
zim = zim_model_registry[backbone](checkpoint=ckpt_mat)
if torch.cuda.is_available():
    zim.cuda()
zim_predictor = ZimPredictor(zim)
zim_mask_generator = ZimAutomaticMaskGenerator(
    zim, 
    pred_iou_thresh=0.7, 
    points_per_batch=8,
    stability_score_thresh=0.9, 
)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
zim_predictor.set_image(image)

# GENERATE ZIM MASKS
wall_zim_masks_path = []
for idx, bbox in enumerate(wall_bboxes):
    zim_mask, _, _ = zim_predictor.predict(
    box          = bbox,
    multimask_output = False,
    )
    zim_mask = np.squeeze(zim_mask, axis=0)
    zim_mask = np.uint8(zim_mask * 255)

    zim_mask_path = os.path.join(folder_path, f"wall_zim_mask_{idx}.jpg")
    cv2.imwrite(zim_mask_path, zim_mask)

    wall_zim_masks_path.append(zim_mask_path)

floor_zim_masks_path = []
for idx, bbox in enumerate(floor_bboxes):
    zim_mask, _, _ = zim_predictor.predict(
    box          = bbox,
    multimask_output = False,
    )
    zim_mask = np.squeeze(zim_mask, axis=0)
    zim_mask = np.uint8(zim_mask * 255)

    zim_mask_path = os.path.join(folder_path, f"floor_zim_mask_{idx}.jpg")
    cv2.imwrite(zim_mask_path, zim_mask)

    floor_zim_masks_path.append(zim_mask_path)