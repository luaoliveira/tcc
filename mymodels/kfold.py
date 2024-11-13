import yaml
import random
import shutil
import numpy as np
import pandas as pd

# Append U-2-Net to path for import resolution
import sys
sys.path.append('../U-2-Net/')
from model.u2net import U2NET

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from tempfile import TemporaryDirectory
from dataset import SegmentationDataset
from segmentation_models_pytorch import Unet  # Using segmentation_models_pytorch for U-Net

import torch
import torch.optim as optim 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from train import train_model, eval_model
from inference import run_inference
from utils import plot_error_curves
from metrics import calc_all_metrics

random.seed(42)

def parse_args() -> dict :
    
    parser = ArgumentParser()

    parser.add_argument("config", type=Path, help="Path to YAML file containing the training parameters")
    parser.add_argument("--folds", "-f", type=int, help="Number of Folds to split the dataset", default=4)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f, yaml.SafeLoader)
    
    configs.update({"folds": args.folds})
    return configs

def build_folds(args) -> list[tuple[Path]]:
    print("Building Temporary Directories to train K-Folds")
    paths = list()

    all_img = list(Path(args["images_dir"]).rglob("*")) + list(Path(args["validation_images"]).rglob("*"))
    all_masks = list(Path(args["masks_dir"]).rglob("*")) + list(Path(args["validation_masks"]).rglob("*"))

    # Dictionary with all pairs (image/mask)
    data = defaultdict(dict)

    for f in all_img:
        data[f.stem]["img"] = f
    for f in all_masks:
        data[f.stem]["mask"] = f

    keys = list(data.keys())
    random.shuffle(keys)
    chunk_size = len(keys) // args["folds"]
    folds = [keys[k*chunk_size:(k+1)*chunk_size] for k in range(args["folds"])]

    # Builds the Temporary Directories with Original Data split in Folds
    for i in range(args["folds"]):
        with TemporaryDirectory(delete=False) as d:
            for k, fold in enumerate(tqdm(folds)):
                if i != k:
                    images_path = Path(d) / "images_dir"
                    masks_dir = Path(d) / "masks_dir"

                else:
                    images_path = Path(d) / "validation_images"
                    masks_dir = Path(d) / "validation_masks"
                
                images_path.mkdir(parents=True, exist_ok=True)
                masks_dir.mkdir(parents=True, exist_ok=True)
                for key in fold:
                    # print(data[key])
                    shutil.copy(data[key]["img"], images_path / Path(key).with_suffix(".jpg"))
                    shutil.copy(data[key]["mask"], masks_dir / Path(key).with_suffix(".bmp"))
            paths.append(tuple(Path(d) / end for end in ["images_dir","masks_dir","validation_images","validation_masks"]))


    return paths

def main(args):

    best_val_loss = float('inf')
    epochs_no_improve = 0

    fold_train_loss = list()
    fold_eval_loss = list()
    output_paths = list()

    kfolds_paths = build_folds(args)

    for k in range(args["folds"]):
        print(f"Running Fold {k}")
        train_im, train_mask, val_im, val_mask = kfolds_paths[k]
        train_losses=np.zeros(args["epochs"])
        eval_losses=np.zeros(args["epochs"])
        output_path = args["output"] if args.get("output") else Path("output") / f'kfold_{args["name"]}_{k}'
        output_paths.append(output_path)
        
        args.update({"validation_images": val_im, "validation_masks": val_mask})

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
            images_dir=train_im,
            masks_dir=train_mask,
        )
        dataloader_training=DataLoader(
            dataset_training,
            batch_size=args["batch_size"],
            shuffle=True)

        dataset_eval= SegmentationDataset(
            images_dir=val_im,
            masks_dir=val_mask
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

        fold_train_loss.append(loss_train)
        fold_eval_loss.append(eval_train)

        output_path.mkdir(parents=True, exist_ok=True)

        plot_error_curves(train_losses, eval_losses, output_path / f"{args['name']}_error_curves", model_name)

        with open(output_path / f"{args['name']}_errors.txt", 'w') as f:
            f.writelines([f'{i} {train} {val}\n' for i, (train, val) in enumerate(zip(train_losses, eval_losses))])
        
        print("Saving the Fine Tuned Weights")
        torch.save(model.state_dict(), output_path / args["weights_output_path"])

        print("Runnin inference...")

        args
        args.update({"fold_name": output_path.stem})
        run_inference(args)

        del(model)
    
    # Already called in run_inference
    # calc_all_metrics(model_name)
    print(f"KFold Train Loss: {np.mean(fold_train_loss)} +- {np.std(fold_train_loss)}")
    print(f"KFold Eval Loss: {np.mean(fold_eval_loss)} +- {np.std(fold_eval_loss)}")

    # Delete TempDir with Fold
    for (p,_,_,_) in kfolds_paths:
        shutil.rmtree(p.parent)

    dfs = list()
    for op in output_paths:
        dfs.append(pd.read_csv(op / f'{args["name"]}_test_metrics.txt', sep=','))
    
    df = pd.concat(dfs)
    df.to_csv(output_paths[0] / "kfold_data.csv", index=False)

    print(df.describe().loc[['mean', 'std']])

if __name__ == '__main__':
    main(parse_args())