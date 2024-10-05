# from PIL import Image
from tqdm import tqdm

def eval_model(dataloader_train, model, criterion, device):

    model.eval()

    running_loss = 0.0 
    for images, masks in tqdm(dataloader_train):
        images = images.to(device)
        outputs = model(images)
        outputs=outputs[0].cpu()
        loss = criterion(outputs, masks)
        running_loss += loss.item()

    return running_loss


def train_model(dataloader_train, model, criterion, optimizer, device):
    
    model.train()

    running_loss = 0.0 
    for images, masks in tqdm(dataloader_train):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        outputs=outputs[0]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss



