import os
from zim_anything import zim_model_registry, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns
import numpy as np
from PIL import Image
import torch
import time

# Model setup
backbone = "vit_l"
ckpt_p = "results/zim_vit_l_2092"
model = zim_model_registry[backbone](checkpoint=ckpt_p)
# if torch.cuda.is_available():
model.cpu()

# Folder paths
input_folder = r"D:\3d-recon\RoomSceneSegmentation\RoomSceneImage-40"
output_folder = r"D:\3d-recon\hci-demo-zim\output_automask"
os.makedirs(output_folder, exist_ok=True)

# Mask generator setup
mask_generator = ZimAutomaticMaskGenerator(
    model, 
    pred_iou_thresh=0.7, 
    points_per_batch=8,
    stability_score_thresh=0.9, 
)

all_imgs = os.listdir(input_folder)[15:]

# Loop through all JPG images in the folder
for filename in all_imgs:
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # use same name

        print(f"Processing: {filename}")
        start_time = time.time()
        image = np.array(Image.open(input_path).convert("RGB"))

        masks = mask_generator.generate(image)
        masks_vis = show_mat_anns(image, masks)

        output_image = Image.fromarray(masks_vis)
        output_image.save(output_path)

        elapsed_time = time.time() - start_time
        print(f"Saved: {output_path} (Time: {elapsed_time:.2f} seconds)")
