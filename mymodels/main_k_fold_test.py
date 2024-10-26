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

images_dir='training_images'
masks_dir = 'training_masks'
epochs=15
batch_size=4
k_folds = 5
model_weights_file=Path('../U-2-Net/saved_models/u2net.pth')
all_fold_val_losses = []
training_losses=[]
validation_losses=[]
delta = 0.001
patience = 5



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    dataset_training= SegmentationDataset(
    images_dir=images_dir,
    masks_dir=masks_dir,
    )

    kfold=KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_training)):

        fold_train_losses = []
        fold_val_losses = []

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



            # average loss per batch during training
            print(f"Fold {fold + 1}, Epoch {epoch + 1}, "
                f"Loss_train: {running_loss_train / len(dataloader_training)}, "
                f"Loss_eval: {running_loss_eval / len(dataloader_eval)}") 


        # print("Saving the fine-tuned-weights")
        # torch.save(model.state_dict(), 'u2net_fine_tuned_weights.pth')

    
    
        print("Runnin inference...")
        run_inference()

if __name__ == '__main_k_fold_test__':
    main()
