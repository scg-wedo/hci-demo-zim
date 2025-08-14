import os
from PIL import Image

def downsample_image(image: Image.Image, max_side: int = 4000) -> Image.Image:
    """
    Resize the image so the largest side is `max_side` pixels, keeping aspect ratio.
    """
    width, height = image.size
    if width > height:
        new_width = max_side
        new_height = int((max_side / width) * height)
    else:
        new_height = max_side
        new_width = int((max_side / height) * width)
    return image.resize((new_width, new_height), Image.LANCZOS)

def process_folder(input_dir, output_dir, max_side=4000):
    os.makedirs(output_dir, exist_ok=True)
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    resized_img = downsample_image(img, max_side)
                    resized_img.save(output_path)
                    print(f"Saved resized image: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_folder = r"D:\3d-recon\datasets\Lab Image"
output_folder = r"D:\3d-recon\datasets\Lab Image\downsampled"
process_folder(input_folder, output_folder, max_side=3000)
