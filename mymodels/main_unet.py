import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet  # Using segmentation_models_pytorch for U-Net
from dataset import SegmentationDataset
from train import train_model, eval_model
from inference import run_inference
from utils import plot_error_curves
from metrics import calc_all_metrics
from pathlib import Path

# Directory paths
images_dir = 'training_images'
masks_dir = 'training_masks'
epochs = 30
train_losses = np.zeros(epochs)
eval_losses = np.zeros(epochs)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize U-Net model with pre-trained encoder weights
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    print("Initializing weights...")

    # Define loss function and optimizer
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Prepare training and validation datasets and dataloaders
    dataset_training = SegmentationDataset(images_dir=images_dir, masks_dir=masks_dir)
    dataloader_training = DataLoader(dataset_training, batch_size=4, shuffle=True)

    dataset_eval = SegmentationDataset(images_dir="validation_images", masks_dir="validation_masks")
    dataloader_eval = DataLoader(dataset_eval, batch_size=4, shuffle=False)

    print('Training model...')

    for epoch in range(epochs):
        # Training and evaluation steps
        running_loss_train = train_model(
            dataloader_training,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        running_loss_eval = eval_model(
            dataloader_eval,
            model=model,
            criterion=criterion,
            device=device
        )

        # Calculate average losses per epoch
        loss_train = running_loss_train / len(dataloader_training)
        eval_loss = running_loss_eval / len(dataloader_eval)

        # Store losses for plotting
        train_losses[epoch] = loss_train
        eval_losses[epoch] = eval_loss

        print(f"Epoch {epoch + 1}, Loss_train: {loss_train}, Loss_eval: {eval_loss}")

    # Plot error curves for visualization
    plot_error_curves(train_losses, eval_losses, 'error-curves-unet')

    # Save losses to a file for record-keeping
    with Path('Errors_unet.txt').open('w') as f:
        f.writelines([f'{i} {train} {val}\n' for i, (train, val) in enumerate(zip(train_losses, eval_losses))])

    # Save the fine-tuned model weights
    torch.save(model.state_dict(), 'unet_fine_tuned_weights.pth')
    print("Model saved successfully.")

    # Run inference and calculate metrics
    print("Running inference...")
    model_name = type(model).__name__
    run_inference(model_name)

    calc_all_metrics()

if __name__ == '__main__':
    main()
