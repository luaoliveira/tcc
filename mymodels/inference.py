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
from utils import parse_args

def run_inference(args):

    images_dir=args['validation_images']
    masks_dir=args['validation_masks']
    output_path = Path("output") / (args["fold_name"] if args.get("fold_name") else args["name"])
    
    result_masks_dir=output_path / 'result_masks'
    result_masks_dir.mkdir(parents=True, exist_ok=True)
    
    overlay_dir = output_path / 'overlay_masks'
    overlay_dir.mkdir(parents=True, exist_ok=True)


    if args["name"].upper() == "U2NET":

        model = U2NET()
        path_weights = output_path / args["weights_output_path"]
        if not path_weights.exists():
            path_weights = Path(args["weights_output_path"])

    elif args["name"].upper() == "UNET":

        model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
        )
        path_weights = output_path / args["weights_output_path"]
        if not path_weights.exists():
            path_weights = Path(args["weights_output_path"])

    elif args["name"].upper() == 'U2NET-NOFT':

        model = U2NET()
        path_weights = 'u2net-original-weights.pth'

    else:
        raise ValueError(f"Unexpected Model Name {args['name'].upper()}. Use one of the options [U2NET, UNET, U2NET_NOFT]")

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
            if "U2NET" in args["name"].upper():
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
                file_name= result_masks_dir / mask_name
                print(file_name)
                mask_pil.save(file_name.with_suffix('.bmp'), format="bmp")
                blended = utils.overlay_mask(Path(images_dir) / image_name, file_name.with_suffix(".bmp"))
                blended.save(overlay_dir / Path(image_name).with_suffix(".jpg"))
                mask_resized.save(file_name.with_suffix(".bmp"), format="bmp")

    calc_all_metrics(args)


if __name__ == "__main__":
    run_inference(parse_args())
