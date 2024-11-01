import torch
import torch.nn.functional as F
# from PIL import Image
from tqdm import tqdm

def eval_model(dataloader_train, model, criterion, device):

    #Carrega o modelo
    model.eval()

    running_loss = 0.0 
    with torch.no_grad():
        for images, masks in tqdm(dataloader_train):
            images = images.to(device)
            outputs = model(images)
            if (type(model).__name__).upper() == "U2NET":
                outputs = outputs[0].cpu()
            elif (type(model).__name__).upper() == "UNET":
                outputs = F.sigmoid(outputs).cpu()
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss


def train_model(dataloader_train, model, criterion, optimizer, device):
    
    model.train()

    running_loss = 0.0 
    for images, masks in tqdm(dataloader_train):
        images, masks = images.to(device), masks.to(device)
        # reset the gradients
        optimizer.zero_grad()

        outputs = model(images)
        # print(f"Model is {type(model).__name__}")
        if (type(model).__name__).upper() == "U2NET":
            outputs = outputs[0]
        elif (type(model).__name__).upper() == "UNET":
            # print(outputs.shape)
            outputs = F.sigmoid(outputs)


        # print("Sample outputs:", outputs[0, 0, :5, :5])
        #calculate the loss between the model predictions and the ground truths
        loss = criterion(outputs, masks)
        #perform backpropagation > compute the gradients of loss
        loss.backward()
        # update the model parameters using the gradients in loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss



