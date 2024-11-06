import os 
import random 
import shutil 
from pathlib import Path

#paths 

random.seed(42)

image_dir = Path("..") / "output_chunks" / "filled_chunks"
mask_dir = Path("..") / "masks"
train_dir = Path("training_images")
val_dir = Path("validation_images")
train_mask_dir = Path("training_masks")
val_mask_dir = Path("validation_masks")


os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

if image_dir.exists():
    images = os.listdir(image_dir)
else:
    print(f"Directory does not exist: {image_dir}")
random.shuffle(images)

train_images = images[:int(0.8*len(images))]
val_images= images[int(0.8*len(images)):]

train_images_set = set(train_images)
val_images_set = set(val_images)

# Ensure no overlap by removing common elements from the validation set
val_images_set -= train_images_set  # Remove any training images from validation set
train_images_set = set(train_images) 

for img in train_images:
    shutil.copy(os.path.join(image_dir, img), train_dir)
    shutil.copy(os.path.join(mask_dir, f"{img[:-3]}bmp"), train_mask_dir)

for img in val_images:

    shutil.copy(os.path.join(image_dir, img), val_dir)
    shutil.copy(os.path.join(mask_dir, f"{img[:-3]}bmp"), val_mask_dir)

