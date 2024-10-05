import torch.optim as optim 
import torch.nn as nn 
import sys
sys.path.append('..\\U-2-Net\\')
from model.u2net import U2NET
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import torch
from train import train_model, eval_model
from PIL import Image
from inference import run_inference

images_dir='training_images'
masks_dir = 'training_masks'
epochs=15

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)

    model = U2NET().to(device)
    print("Initializing weights...")
    model.load_state_dict(torch.load('..\\U-2-Net\\saved_models\\u2net.pth'))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset_training= SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    dataloader_training=DataLoader(dataset_training, batch_size=4, shuffle=True)

    dataset_eval= SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
    )
    dataloader_eval=DataLoader(dataset_eval, batch_size=4, shuffle=True)

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
            dataloader_training, 
            model=model, 
            criterion=criterion, 
            device=device
        )

        # average loss per batch during training
        print(f"Epoch {epoch +1},\
            Loss_train: {running_loss_train / len(dataloader_training)},\
            Loss_eval: {running_loss_eval / len(dataloader_eval)}") 


    print("Saving the fine-tuned-weights")
    torch.save(model.state_dict(), 'u2net_fine_tuned_weights.pth')

    
    
    print("Runnin inference...")
    run_inference()

if __name__ == '__main__':
    main()