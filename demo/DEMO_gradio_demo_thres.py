"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os, sys
sys.path.append(os.getcwd())

# Gradio demo, comparison SAM vs ZIM
import json
import datetime
import uuid
import torch
import gradio as gr
from gradio_image_prompter import ImagePrompter
import numpy as np
import cv2
from PIL import Image
from zim_anything import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator

def get_shortest_axis(image):
    h, w, _ = image.shape
    return h if h < w else w

def reset_image(image, prompts):
    if image is None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        output_dir = None
    else:
        image = image['image']

        # create directory for this upload
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"output/{timestamp}_{uuid.uuid4().hex[:8]}"
        os.makedirs(directory_name, exist_ok=True)

        # cv2.imwrite(os.path.join(directory_name, "input.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # save at 300 dpi
        image_pil = Image.fromarray(image)
        image_pil.save(os.path.join(directory_name, "input.jpg"), dpi=(300, 300))

        # Store directory in prompts or a state
        prompts = {"output_dir": directory_name}

    zim_predictor.set_image(image)

    black = np.zeros(image.shape[:2], dtype=np.uint8)

    return (image, image, image, black, prompts)

    
def run_model(image, prompts):
    if not prompts:
        raise gr.Error(f'Please input any point or BBox')
    point_coords = None
    point_labels = None
    boxes = None

    if "point" in prompts:
        point_coords, point_labels = [], []

        for type, pts in prompts["point"]:
            point_coords.append(pts)
            point_labels.append(type)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    if "bbox" in prompts:
        boxes = prompts['bbox']
        boxes = np.array(boxes)

    if "scribble" in prompts:
        point_coords, point_labels = [], []

        for pts in prompts["scribble"]:
            point_coords.append(np.flip(pts))
            point_labels.append(1)
        if len(point_coords) == 0:
            raise gr.Error("Please input any scribbles.")
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

    # run ZIM
    zim_mask, _, _ = zim_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        multimask_output=False,
    )
    zim_mask = np.squeeze(zim_mask, axis=0)
    zim_mask = np.uint8(zim_mask * 255)

    return zim_mask

def reset_scribble(image, scribble, prompts):
    # scribble = dict()
    for k in prompts.keys():
        prompts[k] = []

    for k, v in scribble.items():
        scribble[k] = None

    black = np.zeros(image.shape[:1], dtype=np.uint8)

    return scribble, black, black

def update_scribble(image, scribble, prompts):
    if "point" in prompts:
        del prompts["point"]

    if "bbox" in prompts:
        del prompts["bbox"]
    
    directory = prompts.get("output_dir") if prompts else None

    prompts = dict() # reset prompt
    scribble_mask = scribble["layers"][0][..., -1] > 0

    scribble_coords = np.argwhere(scribble_mask)
    n_points = min(len(scribble_coords), 24)
    indices = np.linspace(0, len(scribble_coords)-1, n_points, dtype=int)
    scribble_sampled = scribble_coords[indices]

    prompts["scribble"] = scribble_sampled
    
    zim_mask = run_model(image, prompts)
    img_with_zim_mask = draw_images(image, zim_mask, prompts)

    if directory is None:
        directory = "output/backup"
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    zim_mask_file = os.path.join(directory, f"{timestamp}_zim_scribble_mask.jpg")
    # cv2.imwrite(zim_mask_file, zim_mask)
    zim_mask_pil = Image.fromarray(zim_mask)
    zim_mask_pil.save(zim_mask_file, dpi=(300, 300))

    # img_with_zim_mask_file = os.path.join(directory, f"{timestamp}_zim_scribble_img_with_mask.jpg")
    # cv2.imwrite(img_with_zim_mask_file, cv2.cvtColor(img_with_zim_mask, cv2.COLOR_RGB2BGR))

    prompts_save = prompts.copy()
    prompts_save["scribble"] = scribble_sampled.tolist()  # Convert to list for JSON serialization
    prompts_save["output_dir"] = directory  # Save output directory in prompts
    with open(os.path.join(directory, f"{timestamp}_zim_scribble_prompt.json"), "w") as f:
        json.dump(prompts_save, f, indent=4)

    prompts['output_dir'] = directory  # Update prompts with output directory

    return zim_mask, img_with_zim_mask, prompts


def draw_point(img, pt, size, color):
    # draw circle with white boundary region
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 1.3), (255, 255, 255), -1)
    cv2.circle(img, (int(pt[0]), int(pt[1])), int(size * 0.9), color, -1)


def draw_images(image, mask, prompts):
    if len(prompts) == 0 or mask.shape[1] == 1:
        return image, image, image

    minor = get_shortest_axis(image)
    size = int(minor / 80)

    image = np.float32(image)

    def blending(image, mask):
        mask = np.float32(mask) / 255
        blended_image = np.zeros_like(image, dtype=np.float32)
        blended_image[:, :, :] = [255, 0, 0]
        blended_image = (image * 0.5) + (blended_image * 0.5)
    
        img_with_mask = mask[:, :, None] * blended_image + (1 - mask[:, :, None]) * image
        img_with_mask = np.uint8(img_with_mask)

        return img_with_mask

    img_with_mask = blending(image, mask)
    img_with_point = img_with_mask.copy()

    if "point" in prompts:
        for type, pts in prompts["point"]:
            if type == "Positive":
                color = (0, 0, 255)
                draw_point(img_with_point, pts, size, color)
            elif type == "Negative":
                color = (255, 0, 0)
                draw_point(img_with_point, pts, size, color)

    size = int(minor / 200)

    return img_with_mask

# def draw_images_black_background(image, mask, prompts):
#     if len(prompts) == 0 or mask.shape[1] == 1:
#         return image, image, image

#     minor = get_shortest_axis(image)
#     size = int(minor / 80)

#     image = np.float32(image)

#     def blending(image, mask):
#         mask = np.float32(mask) / 255
#         img_with_mask = np.uint8(image)
#         img_with_mask[mask==0] = 0
#         return img_with_mask

#     img_with_mask = blending(image, mask)
#     img_with_point = img_with_mask.copy()

#     if "point" in prompts:
#         for type, pts in prompts["point"]:
#             if type == "Positive":
#                 color = (0, 0, 255)
#                 draw_point(img_with_point, pts, size, color)
#             elif type == "Negative":
#                 color = (255, 0, 0)
#                 draw_point(img_with_point, pts, size, color)

#     size = int(minor / 200)

#     return (
#         img,
#         img_with_mask,
#     )


def get_point_or_box_prompts(img, prompts):
    image, img_prompts = img['image'], img['points']
    point_prompts = []
    box_prompts = []
    for prompt in img_prompts:
        for p in range(len(prompt)):
            prompt[p] = int(prompt[p])
        if prompt[2] == 2 and prompt[5] == 3:  # box prompt
            if len(box_prompts) != 0:
                raise gr.Error("Please input only one BBox.", duration=5)
            box_prompts.append([prompt[0], prompt[1], prompt[3], prompt[4]])
        elif prompt[2] == 1 and prompt[5] == 4:  # Positive point prompt
            point_prompts.append((1, (prompt[0], prompt[1])))
        elif prompt[2] == 0 and prompt[5] == 4:  # Negative point prompt
            point_prompts.append((0, (prompt[0], prompt[1])))

    if "scribble" in prompts:
        del prompts["scribble"]

    if len(point_prompts) > 0:
        prompts['point'] = point_prompts
    elif 'point' in prompts:
        del prompts['point']

    if len(box_prompts) > 0:
        prompts['bbox'] = box_prompts
    elif 'bbox' in prompts:
        del prompts['bbox']

    zim_mask = run_model(image, prompts)
    img_with_zim_mask = draw_images(image, zim_mask, prompts)

    directory = prompts.get("output_dir") if prompts else None
    if directory is None:
        directory = "output/backup"
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    zim_mask_file = os.path.join(directory, f"{timestamp}_zim_mask.jpg")
    # cv2.imwrite(zim_mask_file, zim_mask)
    zim_mask_pil = Image.fromarray(zim_mask)
    zim_mask_pil.save(zim_mask_file, dpi=(300, 300))


    # img_with_zim_mask_file = os.path.join(directory, f"{timestamp}_zim_img_with_mask.jpg")
    # cv2.imwrite(img_with_zim_mask_file, cv2.cvtColor(img_with_zim_mask, cv2.COLOR_RGB2BGR))

    with open(os.path.join(directory, f"{timestamp}_zim_prompt.json"), "w") as f:
        json.dump(prompts, f, indent=4)

    return image, zim_mask, img_with_zim_mask, prompts

def get_examples():
    assets_dir = os.path.join(os.path.dirname(__file__), 'examples')
    images = os.listdir(assets_dir)
    return [os.path.join(assets_dir, img) for img in images]

def get_mask_area(img, binary):
    # Make sure binary is 0 and 1 only
    binary_mask = (binary == 255).astype(np.uint8)

    # Convert to 3 channels if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        binary_mask = np.repeat(binary_mask[:, :, None], 3, axis=2)

    binary_mask[img==1] = 1  # Set all 1s to 0 in the image

    # Apply mask
    masked_img = img * binary_mask

    return masked_img

def apply_threshold(img, zim_mask_array, threshold, prompts=None):
    """
    Apply threshold and save results to directory.
    """
    directory = prompts.get("output_dir") if prompts else None
    
    if directory is None:
        directory = "output/backup"

    os.makedirs(directory, exist_ok=True)

    # Threshold mask
    binary = (zim_mask_array > threshold).astype(np.uint8) * 255
    masked_img = get_mask_area(img, binary)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    mask_file = os.path.join(directory, f"{timestamp}_thres_mask_{threshold}.jpg")
    # masked_file = os.path.join(directory, f"{timestamp}_thres_masked_img.jpg")
    # mask_with_img_file = os.path.join(directory, f"{timestamp}_thres_mask_with_img.jpg")

    # Save files
    # cv2.imwrite(mask_file, binary)
    binary_pil = Image.fromarray(binary)
    binary_pil.save(mask_file, dpi=(300, 300))

    
    # cv2.imwrite(masked_file, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    # mask_with_img = img.copy()
    # mask_with_img[binary == 255] = 255
    img_with_zim_mask = draw_images(img, binary, prompts)

    # mask_with_img_save = cv2.cvtColor(mask_with_img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(mask_with_img_file, mask_with_img_save)

    return binary, masked_img, img_with_zim_mask


if __name__ == "__main__":
    backbone = "vit_l"
    
    # load ZIM
    ckpt_mat = "results/zim_vit_l_2092"
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
    
    with gr.Blocks() as demo:
        gr.Markdown("# <center> [Demo] Image Segmentation with Threshold")

        gr.Markdown("## =========================================================================")
        gr.Markdown("## Instructions")
        gr.Markdown("### 1. Upload Image then wait for the image to process until the initial black image show up")
        gr.Markdown("### 2. Draw Prompts on the input image")
        gr.Markdown("There are 3 input methods, ***only 1 method can be used at a time***")
        gr.Markdown("Eraser or Undo button at the top right corner")
        gr.Markdown("- ***Point Prompt :***  Left Click for Positive Points (area that you want in the mask) and Middle/Right Click for Negative Points (area that you do NOT want in the mask) *you can draw Multiple Positive and Negative Points*")
        gr.Markdown("- ***Box Prompt :***  Left Click then drag to draw a bounding box ***please draw only one box!***")
        gr.Markdown("- ***Scribble :***  Go to 'Scribble' tab then click the Pen button then start drawing Scribbles on the area that you want in the mask")
        gr.Markdown("### 3. Click 'Run' button to process the prompts")
        gr.Markdown("- See the initial Image and Mask in the tabs")
        gr.Markdown("### 4. Use the slider bar to adjust the Threshold for the mask. See output in different tabs")
        gr.Markdown("## =========================================================================")


        prompts = gr.State(dict())
        img = gr.Image(visible=False)
        example_image = gr.Image(visible=False)
        
        gr.Markdown("## Upload Image & Draw Prompts")
        with gr.Row(): 
            # Point and Bbox prompt
            with gr.Tab(label="Point or Box"):
                img_with_point_or_box = ImagePrompter(
                    label="query image", 
                    sources="upload"
                )
                interactions = "[ Left Click (Positive Points) | Middle/Right Click (Negative Points) ]  -OR-  [ Click and Drag to Draw Bounding Box ]"
                gr.Markdown("<h3 style='text-align: center'> 🌟 {} 🌟 </h3>".format(interactions))
                run_bttn = gr.Button("Run")

            # Scribble prompt
            with gr.Tab(label="Scribble"):
                img_with_scribble = gr.ImageEditor(
                    label="Scribble", 
                    brush=gr.Brush(colors=["#00FF00"], default_size=40),
                    sources="upload", 
                    transforms=None, 
                    layers=False
                )
                interactions = "Press Move (Scribble)"
                gr.Markdown("<h3 style='text-align: center'> Step 1. Select Draw button </h3>")
                gr.Markdown("<h3 style='text-align: center'> Step 2. 🌟 {} 🌟 </h3>".format(interactions))
                scribble_bttn = gr.Button("Run")
                scribble_reset_bttn = gr.Button("Reset Scribbles")
            

        gr.Markdown("## Output Mask and Image")
        gr.Markdown("* We can see that the mask here has different confidence value (lighter and darker areas) which allow us to adjust the threshold")

        with gr.Row():
            with gr.Tab(label="ZIM Mask Only"):
                zim_mask = gr.Image(
                    label="ZIM Mask", 
                    image_mode="L", 
                    interactive=False
                )
            with gr.Tab(label="ZIM Image"):
                img_with_zim_mask = gr.Image(
                    label="ZIM Image", 
                    interactive=False
                )
            # with gr.Tab(label="ZIM Image Black"):
            #     img_with_zim_mask_black = gr.Image(
            #         label="ZIM Mask", 
            #         image_mode="L", 
            #         interactive=False
            #     )

        gr.Markdown("## Adjust Mask Threshold")
        gr.Markdown("Lower threshold = More mask Area | Higher threshold = Less mask area")

        with gr.Row():
            threshold_slider = gr.Slider(minimum=0, maximum=255, value=128, step=1, label="ZIM Threshold")
        # with gr.Row():
        #     apply_thresh_btn = gr.Button("Apply Threshold")  
        with gr.Row():
            with gr.Tab(label="ZIM Image with Mask"):
                mask_with_img = gr.Image()
            with gr.Tab(label="ZIM Mask Only"):
                zim_mask_thresh = gr.Image()
            with gr.Tab(label="ZIM Masked Image"):
                masked_img = gr.Image()

        img_with_point_or_box.upload(
            reset_image,
            [img_with_point_or_box, prompts],
            [
                img, 
                img_with_scribble, 
                img_with_zim_mask, 
                zim_mask, 
                prompts,
            ],
        )
        
        run_bttn.click(
            get_point_or_box_prompts,
            [img_with_point_or_box, prompts],
            [img, zim_mask, img_with_zim_mask, prompts]
        )

        scribble_reset_bttn.click(
            reset_scribble,
            [img, img_with_scribble, prompts],
            [img_with_scribble, zim_mask],
        )
        scribble_bttn.click(
            update_scribble,
            [img, img_with_scribble, prompts],
            [zim_mask, img_with_zim_mask, prompts],
        )

        threshold_slider.change(
            fn=apply_threshold,
            inputs=[img, zim_mask, threshold_slider, prompts],
            outputs=[zim_mask_thresh, masked_img, mask_with_img]
        )

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=11928,
    )