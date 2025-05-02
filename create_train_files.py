import yaml
import os
import cv2
from tqdm import tqdm

# === Please CHANGE these according to your setup ===
train_yaml_path = '/mnt/d/Traffic Dataset/dataset_train_rgb/train.yaml'
train_images_root = '/mnt/d/Traffic Dataset/dataset_train_rgb'
output_folder = '/mnt/d/Traffic Dataset/processed_train_lights'

# Create output folders for trainning
os.makedirs(os.path.join(output_folder, 'Red'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Yellow'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'Green'), exist_ok=True)

# Load YAML
with open(train_yaml_path, 'r') as file:
    data = yaml.safe_load(file)

# Process each image
for entry in tqdm(data, desc="Processing Training Images"):
    image_path = os.path.join(train_images_root, entry['path'])
    img = cv2.imread(image_path)
    
    if img is None:
        print(f" Warning: could not load image {image_path}")
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
            filename = os.path.basename(entry['path']).replace('.png', f'_{x_min}_{y_min}.png')
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, resized)

print("Finished preprocessing training images!")

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
