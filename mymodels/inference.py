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

def overlay_mask(img_path, mask, color=(0, 0, 255), pad=(8,8), opacity=0.3):
    img = Image.open(img_path).convert("RGB")
    # img.convert("RGBA")

    color_mask = Image.new("RGB", img.size, color)

    box = (pad[0], pad[1], mask.size[0] - pad[0], mask.size[1] - pad[1])
    color_mask = Image.composite(color_mask, img, mask.crop(box))

    # print(color_mask.size, img.size, mask_pil.crop((8, 8, mask_pil.size[0] - 8, mask_pil.size[1] - 8)).size)

    img = img.convert(color_mask.mode)

    return Image.blend(img, color_mask, opacity)

def run_inference():

    path_weights = 'u2net_fine_tuned_weights.pth'
    images_dir='validation_images'
    masks_dir='validation_masks'
    result_masks_dir='result_masks'
    overlay_dir = Path('overlay_masks')
    os.makedirs(result_masks_dir, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

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

                image_name = inference_dataset.images[idx*4+i]
                mask_name = os.path.splitext(image_name)[0]

                blended = overlay_mask(Path(images_dir) / image_name, mask_pil)

                print(mask_name)
                file_name= os.path.join(result_masks_dir, mask_name)
                print(file_name)
                mask_pil.save(f"{file_name}.JPEG", format="JPEG")
                blended.save(overlay_dir / Path(image_name).with_suffix(".png"))


if __name__ == "__main__":

    run_inference()
