import torch.optim as optim 
import torch.nn as nn 
import sys
sys.path.append('../U-2-Net/')
sys.path.append('../U-2-Net/')
from model.u2net import U2NET
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import torch
from train import train_model, eval_model
from PIL import Image
from inference import run_inference
import numpy as np
from utils import plot_error_curves
from metrics import calc_all_metrics
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

images_dir='training_images'
masks_dir = 'training_masks'
epochs=30
train_losses=np.zeros(epochs)
eval_losses=np.zeros(epochs)
delta = 0.0001
patience = 8
batch_size=3

def main():

    best_val_loss = float('inf')
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    model = U2NET().to(device)
    print("Initializing weights...")
    model.load_state_dict(torch.load('../U-2-Net/saved_models/u2net.pth'))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= 2e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    dataset_training= SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    dataloader_training=DataLoader(
        dataset_training,
        batch_size=batch_size,
        shuffle=True)

    dataset_eval= SegmentationDataset(
        images_dir="validation_images",
        masks_dir="validation_masks"
    )
    dataloader_eval=DataLoader(
        dataset_eval,
        batch_size=4,
        shuffle=False)

    model_name = type(model).__name__
    print(f'Training model {model_name}...')
    

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

        #Calculate the average loss and evaluation for each batch
        loss_train = running_loss_train / len(dataloader_training)
        eval_train = running_loss_eval / len(dataloader_eval)

        train_losses[epoch]=loss_train
        eval_losses[epoch] = eval_train

        # average loss per batch during training
        print(f"Epoch {epoch +1},\
            Loss_train: {loss_train},\
            Loss_eval: {eval_train}") 

        # if loss_train < best_val_loss - delta:

        #     best_val_loss = eval_train
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve +=1

        # if epochs_no_improve >= patience:
        #     print(f"Early stopping on epoch {epoch+1}")
        #     break 

        scheduler.step()

    plot_error_curves(train_losses, eval_losses, 'error-curves-2unet', model_name)

    with open('Erros_u2net.txt', 'w') as f:
        f.writelines([f'{i} {train} {val}\n' for i, (train, val) in enumerate(zip(train_losses, eval_losses))])
    
    print("Saving the fine-tuned-weights-2unet")
    torch.save(model.state_dict(), 'u2net_fine_tuned_weights.pth')

    print("Runnin inference...")
    run_inference(model_name)
    
    calc_all_metrics(model_name)

if __name__ == '__main__':
    main()

