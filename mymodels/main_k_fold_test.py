import torch.optim as optim 
import torch.nn as nn 
import sys
sys.path.append('../U-2-Net/')
from model.u2net import U2NET
from dataset import SegmentationDataset
from torch.utils.data import DataLoader, Subset
import torch
from train import train_model, eval_model
from PIL import Image
from inference import run_inference
from pathlib import Path
from sklearn.model_selection import KFold
from metrics import calc_all_metrics
from utils import plot_error_curves
import numpy as np

images_dir='training_images'
masks_dir = 'training_masks'
epochs=25
batch_size=4
k_folds = 5
model_weights_file=Path('../U-2-Net/saved_models/u2net.pth')
all_fold_val_losses = []
training_losses=[]
validation_losses=[]
delta = 0.001
patience = 5
error_file = Path("Erros.txt")


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    dataset_training= SegmentationDataset(
    images_dir=images_dir,
    masks_dir=masks_dir,
    )

    kfold=KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_training)):

        train_losses=[]
        eval_losses=[]

        best_val_loss = float('inf')
        epochs_no_improve = 0

        model = U2NET().to(device)
        print("Initializing weights...")
        model.load_state_dict(torch.load(model_weights_file))

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)


        train_subset = Subset(dataset_training, train_idx)
        val_subset= Subset(dataset_training, val_idx)

        ###################################################
        #Data loaders anteriores
        ###################################################

        # dataloader_training=DataLoader(dataset_training, batch_size=4, shuffle=True)

        # dataset_eval= SegmentationDataset(
        #     images_dir="validation_images",
        #     masks_dir="validation_masks",
        # )

        ###################################################

        dataloader_training=DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=False
        )

        dataloader_eval=DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False
        )

        print('Training model...')
        for epoch in range(epochs):

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

            avg_loss_train = running_loss_train / len(dataloader_training)
            avg_loss_eval = running_loss_eval /len (dataloader_eval)

            if avg_loss_eval < best_val_loss - delta:
                best_val_loss = avg_loss_eval
                epochs_no_improve = 0
            else:
                epochs_no_improve +=1

            if epochs_no_improve >= patience:
                print(f"Early stopping on fold {fold+1}, {epoch+1}")
                break

            train_losses.append(avg_loss_train)
            eval_losses.append(avg_loss_eval)

            # average loss per batch during training
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, "
                f"Loss_train: {avg_loss_train}, "
                f"Loss_eval: {avg_loss_eval}") 

        all_fold_val_losses.append(min(eval_losses))

        # print("Saving the fine-tuned-weights")
        # torch.save(model.state_dict(), 'u2net_fine_tuned_weights.pth')
        plot_error_curves(train_losses, eval_losses,f'{fold}_fold_learning_curve')

        error_file = Path(f"Erros_fold_{fold + 1}.txt")

        with error_file.open('w') as f:
            f.writelines([f'{i} {train} {val}\n' for i, (train, val) in enumerate(zip(train_losses, eval_losses))])

        print("Runnin inference...")
        run_inference()

    avg_k_fold_eval = np.mean(all_fold_val_losses)
    std_k_fold_eval = np.std(all_fold_val_losses)

    path_eval_u2net = Path(f"u2net_evaluation.txt")

    with path_eval_u2net.open('w') as f:
        f.write(f'avg: {avg_k_fold_eval}, std: {std_k_fold_eval} ')


if __name__ == '__main__':
    main()
