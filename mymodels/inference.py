import sys
import torch
import os
sys.path.append('..\\U-2-Net\\')
from model.u2net import U2NET
from dataset import SegmentationDataset
from torch.utils.data import DataLoader

def run_inference():

    path_weights = 'u2net_fine_tuned_weights.pth'
    image_dir='validation_images'
    result_masks_dir='result_masks'
    os.makedirs(result_masks_dir, exist_ok=True)

    model = U2NET()

    inference_dataset= SegmentationDataset(
        image_dir=image_dir
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=4,
        shuffle=False
    )
    model.load_state_dict(torch.load(path_weights))
    model.eval()

    with torch.no_grad():

        for idx, (images, masks) in enumerate(inference_loader):
            outputs = model(images)
            masks = (outputs[0] > 0.5).cpu().numpy()

            for i in range(masks.shape[0]):
                mask = masks[i,0]
                mask_image = (mask*255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_image)

                image_name = inference_dataset.images[idx]
                mask_name = os.path.splitext(image_name)[0]
                mask_pil.save(os.path.join(result_masks_dir, mask_name))



