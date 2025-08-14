import os
from zim_anything import zim_model_registry, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns
import numpy as np
from PIL import Image
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import time
import pickle




input_folder = r"D:\3d-recon\RoomSceneSegmentation\input_image"
image_name   = "GP_40x40_CANAVARO_NAVY_R2.jpg"
input_path = os.path.join(input_folder, image_name)
image = np.array(Image.open(input_path).convert("RGB"))


output_folder = os.path.join(r"D:\3d-recon\RoomSceneSegmentation\output_automask_nonms", image_name.split('.')[0])
os.makedirs(output_folder, exist_ok=True)

output_img_path = os.path.join(output_folder, f"masks.jpg")
output_pkl_path = os.path.join(output_folder, f"masks.pkl")

# Model setup
backbone = "vit_l"
ckpt_p = "./results/zim_vit_l_2092"
model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

mask_generator = ZimAutomaticMaskGenerator(
                model, 
                pred_iou_thresh        = 0.7,  # A filtering threshold in [0,1], using the model's predicted mask quality.
                points_per_batch       = 8, # Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.
                stability_score_thresh = 0.9, # A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.
                points_per_side        = 32, # A filtering threshold in [0,1], using the model's predicted bounding boxes.                

            )

print(f"Processing: {input_path}")
start_time = time.time()
image = np.array(Image.open(input_path).convert("RGB"))

masks = mask_generator.generate(image)
masks_vis = show_mat_anns(image, masks)

output_image = Image.fromarray(masks_vis)
output_image.save(output_img_path)

with open(output_pkl_path, 'wb') as f:
    pickle.dump(masks, f)

elapsed_time = time.time() - start_time
print(f"Saved: {output_img_path} (Time: {elapsed_time:.2f} seconds)")

def save_separate_masks(image, anns, output_folder):
    if len(anns) == 0:
        return np.zeros_like(image) + 128
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    image = image.astype(np.float32)
    
    for idx, ann in enumerate(sorted_anns):
        colorized_mat = np.zeros_like(image)

        color = (np.random.random(3) * 255).astype(np.float32)
        if 'logit' in ann:
            mat = ann['logit'].astype(np.float32) / 255.
            print("ann['logit']", mat.shape)
            print(ann['logit'])

        else:
            mat = ann['segmentation'].astype(np.float32)
            print("ann['segmentation']", mat.shape)

        color_mat = np.zeros_like(image) + color[None, None]
        colorized_mat = color_mat * mat[:, :, None] + colorized_mat * (1. - mat[:, :, None])

        colorized_mat = np.uint8(colorized_mat)

        output_image = Image.fromarray(colorized_mat)
        output_image.save(os.path.join(output_folder, f"{idx}_mask.png"))
        print(f"Saved mask {idx} to {output_folder}")

masks_vis = save_separate_masks(image, masks, output_folder)