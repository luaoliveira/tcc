import yaml
import numpy as np

# Append U-2-Net to path for import resolution
import sys
sys.path.append('../U-2-Net/')
from model.u2net import U2NET

from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from dataset import SegmentationDataset
from segmentation_models_pytorch import Unet  # Using segmentation_models_pytorch for U-Net

import torch
import torch.optim as optim 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from train import train_model, eval_model
from inference import run_inference
from utils import plot_error_curves, parse_args
from metrics import calc_all_metrics


def main(args):

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses=np.zeros(args["epochs"])
    eval_losses=np.zeros(args["epochs"])
    output_path = args["output"] if args.get("output") else Path("output") / args["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    if(args["name"] == "U2NET"):
        model = U2NET().to(device)
    elif(args["name"] == "UNET"):
        model = Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)
    else:
        raise ValueError(f"Unknown Model {args["name"]}. Use one of the following options: [UNET, U2NET]")

    if(args["pretrained"]):
        print("Initializing weights...")
        model.load_state_dict(torch.load(args["weights_path"]))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= args["lr"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args["T_0"], T_mult=args["T_mult"])
    dataset_training= SegmentationDataset(
        images_dir=args["images_dir"],
        masks_dir=args["masks_dir"],
    )
    dataloader_training=DataLoader(
        dataset_training,
        batch_size=args["batch_size"],
        shuffle=True)

    dataset_eval= SegmentationDataset(
        images_dir=args["validation_images"],
        masks_dir=args["validation_masks"]
    )
    dataloader_eval=DataLoader(
        dataset_eval,
        batch_size=args["batch_size"],
        shuffle=False)

    model_name = type(model).__name__
    print(f'Training model {model_name}...')
    

    for epoch in range(args["epochs"]):
        
        running_loss_train=train_model(
            dataloader_training, 
            model=model, 
            criterion=criterion, 
            optimizer=optimizer,
            device=device
        )
        running_loss_eval=eval_model(
            dataloader_eval, 
            model=model, 
            criterion=criterion, 
            device=device
        )

        #Calculate the average loss and evaluation for each batch
        loss_train = running_loss_train / len(dataloader_training)
        eval_train = running_loss_eval / len(dataloader_eval)

        train_losses[epoch]=loss_train
        eval_losses[epoch] = eval_train

        # average loss per batch during training
        print(f"Epoch {epoch +1},\
            Loss_train: {loss_train},\
            Loss_eval: {eval_train}") 

        # if loss_train < best_val_loss - args["delta"]:

        #     best_val_loss = eval_train
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve +=1

        # if epochs_no_improve >= args["patience"]:
        #     print(f"Early stopping on epoch {epoch+1}")
        #     break 

        scheduler.step()


    output_path.mkdir(parents=True, exist_ok=True)

    plot_error_curves(train_losses, eval_losses, output_path / f"{args['name']}_error_curves", model_name)

    with open(output_path / f"{args['name']}_errors.txt", 'w') as f:
        f.writelines([f'{i} {train} {val}\n' for i, (train, val) in enumerate(zip(train_losses, eval_losses))])
    
    print("Saving the Fine Tuned Weights")
    torch.save(model.state_dict(), output_path / args["weights_output_path"])

    print("Runnin inference...")
    run_inference(args)
    
    # Already called in run_inference
    # calc_all_metrics(model_name)

if __name__ == '__main__':
    main(parse_args())

