import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from dataset import FacialKeyPoint
from tqdm import tqdm
from utils import *

TRAINING = False


DEVICE = 'mps'
LR = 3e-4
BATCH_SIZE=16
EPOCHS = 10

model = resnet18(weights='ResNet18_Weights.DEFAULT')
model.fc = nn.Linear(512, 2)
model = model.to(device=DEVICE)
criterion = nn.MSELoss(reduction="sum")
optimizer = optim.Adam(lr=LR, params=model.parameters())

dataset = FacialKeyPoint()
trainset, testset = random_split(dataset, lengths=[0.8, 0.2])

trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train():

    for epoch in range(EPOCHS):
        current_loss = 0.0
        for x, y in tqdm(trainloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss+=loss.item()
        print(f"for epoch {epoch} , current loss is {current_loss/len(trainloader)}")

        save_checkpoint(model, f"saved_checkpoints/epoch_{epoch}")

        with torch.no_grad():
            img, gt = testset[0]
            img = img.to(DEVICE)
            pred = model(img.unsqueeze(0))
            save_example(img.cpu(), gt, pred, f"saved_images/example_epoch_{epoch}")

def test(weigths_file):
    load_checkpoint(model, path=weigths_file)
    model.eval()
    with torch.no_grad():
        gts, preds = [], []
        for x, y in tqdm(testset):
            pred = model(x.to(DEVICE).unsqueeze(0))
            gts.append([y[0].item(), y[1].item()])
            preds.append([pred[0][0].item(), pred[0][1].item()])
        print(f"for this model, loss = {criterion(torch.Tensor(gts), torch.Tensor(preds))}")


if __name__=='__main__':

    if TRAINING:
        train()
    else:
        test("saved_checkpoints/epoch_10")

    

