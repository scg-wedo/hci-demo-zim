from zim_anything import zim_model_registry, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns
import numpy as np
from PIL import Image
import torch

backbone = "vit_l"
ckpt_p = "results/zim_vit_l_2092"

model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

image_path = r"D:\3d-recon\RoomSceneSegmentation\RoomSceneImage-40\21.jpg"
image = np.array(Image.open(image_path).convert("RGB"))

mask_generator = ZimAutomaticMaskGenerator(
    model, 
    pred_iou_thresh=0.7, 
    points_per_batch=8,
    stability_score_thresh=0.9, 
)
    
masks = mask_generator.generate(image)  # Automatically generated masks
masks_vis = show_mat_anns(image, masks)  # Visualize masks
output_image = Image.fromarray(masks_vis)
output_image.save("automask_output.jpg")