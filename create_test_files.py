import yaml
import os
import cv2
from tqdm import tqdm

# === Please CHANGE these according to your setup ===
test_yaml_path = '/mnt/d/Traffic Dataset/dataset_test_rgb/test.yaml'
test_images_root = '/mnt/d/Traffic Dataset/dataset_test_rgb/rgb/test'
output_folder = '/mnt/d/Traffic Dataset/processed_test_lights'

# Create output folders for test
os.makedirs(os.path.join(output_folder, 'Red'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Yellow'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Green'), exist_ok=True)

# Load YAML
with open(test_yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Process images
for entry in tqdm(data, desc="Processing Test Images"):
    relative_path = entry['path']

    # Extract only the filename (remove 'rgb/test/' prefix if present)
    filename = os.path.basename(relative_path)

    # Full correct path to image
    image_path = os.path.join(test_images_root, filename)

    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: could not load image {image_path}")
        continue
    
    for box in entry.get('boxes', []):
        label = box['label']
        if label not in ['Red', 'Yellow', 'Green']:
            continue
        
        x_min = int(box['x_min'])
        x_max = int(box['x_max'])
        y_min = int(box['y_min'])
        y_max = int(box['y_max'])
        
        cropped = img[y_min:y_max, x_min:x_max]
        
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (32, 32))
            save_folder = os.path.join(output_folder, label)
            new_filename = filename.replace('.png', f'_{x_min}_{y_min}.png')
            save_path = os.path.join(save_folder, new_filename)
            cv2.imwrite(save_path, resized)

def count_processed_images(base_folder):
    red_count = len(os.listdir(os.path.join(base_folder, 'Red')))
    yellow_count = len(os.listdir(os.path.join(base_folder, 'Yellow')))
    green_count = len(os.listdir(os.path.join(base_folder, 'Green')))
    
    print(f"\nProcessed images count:")
    print(f"Red: {red_count}")
    print(f"Yellow: {yellow_count}")
    print(f"Green: {green_count}")
    print(f"Total: {red_count + yellow_count + green_count}")

count_processed_images(output_folder)


print("Finished preprocessing testing images (using correct path)!")
