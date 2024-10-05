import os 
import random 
import shutil 

#paths 

random.seed(42)

image_dir = '..\\output_chunks\\filled_chunks'
mask_dir = '..\\masks'
train_dir = 'training_images'
val_dir = 'validation_images'
train_mask_dir = 'training_masks'
val_mask_dir ='validation_masks'


os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

images = os.listdir(image_dir)
random.shuffle(images)

train_images = images[:int(0.8*len(images))]
val_images= images[int(0.8*len(images)):]

for img in train_images:
    shutil.copy(os.path.join(image_dir, img), train_dir)
    shutil.copy(os.path.join(mask_dir, img), train_mask_dir)

for img in val_images:

    shutil.copy(os.path.join(image_dir, img), val_dir)
    shutil.copy(os.path.join(mask_dir, img), val_mask_dir)

