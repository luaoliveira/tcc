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
from segmentation_models_pytorch import Unet  # Using segmentation_models_pytorch for U-Net
import utils
from metrics import calc_all_metrics


def run_inference(model_name):

    images_dir='validation_images'
    masks_dir='validation_masks'
    result_masks_dir='result_masks'
    overlay_dir = Path('overlay_masks')
    os.makedirs(result_masks_dir, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
   

    if model_name.upper() == "U2NET":

        model = U2NET()
        path_weights = 'u2net_fine_tuned_weights.pth'

    elif model_name.upper() == "UNET":

        model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
        )
    elif model_name== 'UNET-NOFT':

        path_weights=''

        path_weights = 'u2net-original-weights.pth'

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
            if "U2NET" in model_name.upper():
                outputs = outputs[0]
            masks = (outputs > 0.5).cpu().numpy()

            for i in range(masks.shape[0]):
                mask = masks[i,0]
                mask_image = (mask*255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_image)
                mask_resized=utils.remove_padding(mask_pil)

                image_name = inference_dataset.images[idx*4+i]
                mask_name = os.path.splitext(image_name)[0]

                print(mask_name)
                result_masks_dir=f'{model_name}_{result_masks_dir}'
                file_name= os.path.join(result_masks_dir, mask_name)
                print(file_name)
                mask_pil.save(f"{file_name}.jpg", format="jpeg")
                # blended = utils.overlay_mask(Path(images_dir) / image_name, Path(f"{file_name}.jpg"))
                # blended.save(overlay_dir / Path(image_name).with_suffix(".jpg"))
                mask_resized.save(f"{file_name}.jpg", format="jpeg")

    calc_all_metrics(model_name)


if __name__ == "__main__":

    run_inference('u2net-noft')
