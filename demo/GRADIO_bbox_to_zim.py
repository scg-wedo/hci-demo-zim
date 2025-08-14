import os, sys
import time
import shutil
sys.path.append(os.getcwd())

import json
import torch
import numpy as np
import cv2
import gradio as gr
from PIL import Image

from zim_anything import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator

# Function to generate masks (ZIM prediction only once, threshold applied dynamically)
def run_zim(image_path, json_path):
    base_dir = r"D:\3d-recon\WESHAPExZIM\output"
    folder_name = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    image_copy_path = os.path.join(output_dir, os.path.basename(image_path))
    json_copy_path = os.path.join(output_dir, os.path.basename(json_path))
    shutil.copy(image_path, image_copy_path)
    shutil.copy(json_path, json_copy_path)

    try:
        with open(json_copy_path, "r", encoding="utf-8") as f:
            auto_detect = json.load(f)
    except Exception as e:
        return f"Error reading JSON: {e}", None, None, None, None, None, None

    wall_bboxes = np.array(auto_detect["wall_bboxes"])
    floor_bboxes = np.array(auto_detect["floor_bboxes"])

    backbone = "vit_l"
    ckpt_mat = "./results/zim_vit_l_2092"
    zim = zim_model_registry[backbone](checkpoint=ckpt_mat)
    if torch.cuda.is_available():
        zim.cuda()
    zim_predictor = ZimPredictor(zim)

    image = cv2.imread(image_copy_path)
    if image is None:
        return f"Error reading image at {image_copy_path}", None, None, None, None, None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    zim_predictor.set_image(image_rgb)

    wall_masks = []
    floor_masks = []

    # Process wall masks
    for idx, bbox in enumerate(wall_bboxes):
        zim_mask, _, _ = zim_predictor.predict(box=bbox, multimask_output=False)
        zim_mask = np.squeeze(zim_mask, axis=0)
        zim_mask = np.uint8(zim_mask * 255)
        wall_masks.append(zim_mask)

        zim_mask_path = os.path.join(output_dir, f"wall_zimmask_{idx+1}.jpg")
        cv2.imwrite(zim_mask_path, zim_mask)

    # Process floor masks
    for idx, bbox in enumerate(floor_bboxes):
        zim_mask, _, _ = zim_predictor.predict(box=bbox, multimask_output=False)
        zim_mask = np.squeeze(zim_mask, axis=0)
        zim_mask = np.uint8(zim_mask * 255)
        floor_masks.append(zim_mask)

        zim_mask_path = os.path.join(output_dir, f"floor_zimmask_{idx+1}.jpg")
        cv2.imwrite(zim_mask_path, zim_mask)

    return output_dir, image_rgb, wall_masks, floor_masks, output_dir

# Function to apply threshold dynamically and show side-by-side
def show_mask_with_threshold(image_rgb, masks, index, threshold):
    if masks and 0 <= index < len(masks):
        mask = np.where(masks[index] >= threshold, 255, 0).astype(np.uint8)
        mask_img = Image.fromarray(mask).convert("RGB")

        # Draw overlay
        image_float = np.float32(image_rgb)
        mask_float = np.float32(mask)/255
        blended_image = np.zeros_like(image_float, dtype=np.float32)
        blended_image[:, :, :] = [255, 0, 0]
        blended_image = (image_float * 0.5) + (blended_image * 0.5)
        img_with_mask = np.uint8(mask_float[:, :, None] * blended_image + (1 - mask_float[:, :, None]) * image_float)
        img_with_mask_pil = Image.fromarray(img_with_mask)

        combined = Image.new('RGB', (mask_img.width + img_with_mask_pil.width, mask_img.height))
        combined.paste(mask_img, (0,0))
        combined.paste(img_with_mask_pil, (mask_img.width,0))
        return combined
    return None

# Functions to move index forward/backward
def next_image(index, masks):
    if masks and index + 1 < len(masks):
        return index + 1
    return index

def prev_image(index, masks):
    if index > 0:
        return index - 1
    return index

# Function to save current binary mask
def save_current_mask(output_dir, masks, index, threshold, mask_type):
    if masks and 0 <= index < len(masks):
        mask = np.where(masks[index] >= threshold, 255, 0).astype(np.uint8)
        filename = f"{mask_type}_mask_{index+1}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, mask)
        return f"Saved {save_path}"
    return "No mask to save"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## ZIM Mask Generator — Upload Image & JSON")

    with gr.Row():
        image_input = gr.File(label="Upload Image", type="filepath", file_types=[".jpg", ".png"])
        json_input = gr.File(label="Upload auto_detect.json", type="filepath", file_types=[".json"])

    threshold_slider = gr.Slider(0, 255, value=128, step=1, label="Mask Threshold")
    run_btn = gr.Button("Run")
    output_text = gr.Textbox(label="Status")

    wall_index_state = gr.State(0)
    floor_index_state = gr.State(0)
    wall_masks_state = gr.State([])
    floor_masks_state = gr.State([])
    image_rgb_state = gr.State(None)
    output_dir_state = gr.State("")

    with gr.Accordion("Wall Masks", open=True):
        wall_mask_display = gr.Image(label="Wall Mask Preview")
        with gr.Row():
            wall_back_btn = gr.Button("Previous Wall")
            wall_save_btn = gr.Button("Save Wall Mask")
            wall_next_btn = gr.Button("Next Wall")

    with gr.Accordion("Floor Masks", open=True):
        floor_mask_display = gr.Image(label="Floor Mask Preview")
        with gr.Row():
            floor_back_btn = gr.Button("Previous Floor")
            floor_save_btn = gr.Button("Save Floor Mask")
            floor_next_btn = gr.Button("Next Floor")

    def run_and_store_separated(image_path, json_path):
        output_dir, image_rgb, wall_masks, floor_masks, out_dir = run_zim(image_path, json_path)
        return f"Done! Masks saved in {output_dir}", 0, 0, wall_masks, floor_masks, image_rgb, output_dir

    run_btn.click(
        fn=run_and_store_separated,
        inputs=[image_input, json_input],
        outputs=[output_text, wall_index_state, floor_index_state, wall_masks_state, floor_masks_state, image_rgb_state, output_dir_state]
    )

    wall_next_btn.click(fn=next_image, inputs=[wall_index_state, wall_masks_state], outputs=[wall_index_state])
    wall_back_btn.click(fn=prev_image, inputs=[wall_index_state, wall_masks_state], outputs=[wall_index_state])
    floor_next_btn.click(fn=next_image, inputs=[floor_index_state, floor_masks_state], outputs=[floor_index_state])
    floor_back_btn.click(fn=prev_image, inputs=[floor_index_state, floor_masks_state], outputs=[floor_index_state])

    wall_save_btn.click(fn=save_current_mask, inputs=[output_dir_state, wall_masks_state, wall_index_state, threshold_slider, gr.State("wall")], outputs=[output_text])
    floor_save_btn.click(fn=save_current_mask, inputs=[output_dir_state, floor_masks_state, floor_index_state, threshold_slider, gr.State("floor")], outputs=[output_text])

    def update_wall_display(index, threshold, masks, image_rgb):
        return show_mask_with_threshold(image_rgb, masks, index, threshold)

    def update_floor_display(index, threshold, masks, image_rgb):
        return show_mask_with_threshold(image_rgb, masks, index, threshold)

    wall_index_state.change(fn=update_wall_display, inputs=[wall_index_state, threshold_slider, wall_masks_state, image_rgb_state], outputs=[wall_mask_display])
    floor_index_state.change(fn=update_floor_display, inputs=[floor_index_state, threshold_slider, floor_masks_state, image_rgb_state], outputs=[floor_mask_display])

    threshold_slider.change(fn=update_wall_display, inputs=[wall_index_state, threshold_slider, wall_masks_state, image_rgb_state], outputs=[wall_mask_display])
    threshold_slider.change(fn=update_floor_display, inputs=[floor_index_state, threshold_slider, floor_masks_state, image_rgb_state], outputs=[floor_mask_display])


demo.launch(server_port=8888)
