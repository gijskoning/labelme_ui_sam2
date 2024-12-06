from PIL import Image
import os

img_dir = "/code_projects/labelme-with-segment-anything/video_imgs"
smaller_img_dir = "/code_projects/labelme-with-segment-anything/video_imgs_cropped"
os.makedirs(smaller_img_dir, exist_ok=True)

for i, filename in enumerate(sorted(os.listdir(img_dir))):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Open the image file
        left, upper, right, lower = 10, 20, 128 - 10, 128 - 16
        crop_area = (left, upper, right, lower)
        png_image = Image.open(os.path.join(img_dir, filename)).crop(crop_area)
        print(filename)
        # Convert the image to RGB mode
        rgb_image = png_image.convert("L")

        new_filename = f"{i:05d}.jpg"

        # Save the image in JPG format
        rgb_image.save(os.path.join(smaller_img_dir, new_filename), "JPEG")
