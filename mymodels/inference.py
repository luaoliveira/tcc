import sys
import torch
import os
sys.path.append('../U-2-Net/')
from model.u2net import U2NET
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image 
from pathlib import Path
import utils

def run_inference():

    path_weights = 'u2net_fine_tuned_weights.pth'
    images_dir='validation_images'
    masks_dir='validation_masks'
    result_masks_dir='result_masks'
    overlay_dir = 'overlay_masks'
    os.makedirs(result_masks_dir, exist_ok=True)

    model = U2NET()

    inference_dataset= SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=4,
        shuffle=False
    )
    model.load_state_dict(torch.load(path_weights))
    model.eval()

    with torch.no_grad():

        for idx, (images, _) in enumerate(inference_loader):
            outputs = model(images)
            masks = (outputs[0] > 0.5).cpu().numpy()

            for i in range(masks.shape[0]):
                mask = masks[i,0]
                mask_image = (mask*255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_image)
                mask_resized=utils.remove_padding(mask_pil)

                image_name = inference_dataset.images[idx*4+i]
                mask_name = os.path.splitext(image_name)[0]

                print(mask_name)
                file_name= os.path.join(result_masks_dir, mask_name)
                print(file_name)
                mask_pil.save(f"{file_name}.JPEG", format="JPEG")


if __name__ == "__main__":

    run_inference()
