import os
import numpy as np
from PIL import Image

# Set your image folder path
folder_path = 'test_img'
output_npz = 'pred.npz'

# Initialize dictionary to hold image arrays
images_dict = {}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(folder_path, filename)

        # Load image and convert to RGB
        image = Image.open(file_path).convert('RGB')
        img_array = np.array(image)

        # Rearrange to (3, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add to dictionary
        images_dict[filename] = img_array

# Save to .npz file
np.savez(output_npz, **images_dict)

print(f"Saved {len(images_dict)} images to {output_npz}")
