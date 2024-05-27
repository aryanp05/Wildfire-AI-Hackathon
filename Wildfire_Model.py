import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

# Paths for data and fires
data_path = "./"
train_path = data_path + "train/"
test_path = data_path + "test/"
tr_fnums = ["fire1209", "fire1298", "fire1386", "fire2034", "fire2210", "fire2211", "fire2212"]
te_fnums = ["fire2214"]

# Util variables
device = 'cpu' # Use CUDA IF AVAILIBLE
target_shape = (528, 720)

# Util functions
def pad_to_fit(d, shape):
    h, w = d.shape
    pad_h = shape[0] - h
    pad_w = shape[1] - w
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        d = np.pad(d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return d

def normalize(d):
    m = np.mean(d)
    s = np.std(d)
    return (d - m)/s

def tif2np(tif):
    with rio.open(tif) as src:
        data = src.read(1)  # Read the first band
    return pad_to_fit(np.nan_to_num(data, nan=0.0), target_shape)


def load_day(path, day):
    # fire_weather (Made of BUI & ISI)
    fwi = path+'/fire_weather/fire_weather_index_day{}.tif'.format(day)
    fwi = normalize(tif2np(fwi))

    # sub components of BUI
    # drought code
    dc = path+'/fire_weather/drought_code_day{}.tif'.format(day)
    dc = normalize(tif2np(dc))
    # duff moisture code
    dmc = path+'/fire_weather/duff_moisture_code_day{}.tif'.format(day)
    dmc = normalize(tif2np(dmc))
    
    # sub components of ISI
    # fine fuel moisture code
    ffmc = path+'/fire_weather/fine_fuel_moisture_code_day{}.tif'.format(day)
    ffmc = normalize(tif2np(ffmc))
    # NOON weather wind speed
    wws = path+'/weather/noon_wind_speed_day{}.tif'.format(day)
    wws = normalize(tif2np(wws))

    # NOON featurs to help accuary of NOOn wind
    # NOON weather wind direction
    wdi = path+'/weather/noon_wind_direction_day{}.tif'.format(day)
    wdi = normalize(tif2np(wdi))
    # NOON weather relative humidity
    wrh = path+'/weather/noon_relative_humidity_day{}.tif'.format(day)
    wrh = normalize(tif2np(wrh))
    # NOON temperature
    ntp = path+'/weather/noon_temperature_day{}.tif'.format(day)
    ntp = normalize(tif2np(ntp))
    return [fwi, dc, dmc, ffmc, wws, wdi, wrh, ntp]

def load_fire(fire_num, split = "Train"):
    path = train_path + fire_num
    if split == "Test":
        path = test_path + fire_num
    
    ftif = path + "/fire/{}.tif".format(fire_num)
    if split == "Test":
        ftif = path + "/fire/{}_train.tif".format(fire_num)
    fdata = tif2np(ftif)

    minjd, maxjd = int(np.min(fdata[np.nonzero(fdata)])), int(np.max(fdata))
    lastjd = maxjd
    if split == "Test":
        maxjd += 21
    
    elev = normalize(tif2np(path+'/topography/dem.tif'))
    slope = normalize(tif2np(path+'/topography/slope.tif'))
    fuels = tif2np(path+'/fuels/fbp_fuels.tif')
    ignition = tif2np(path+'/fire/ignitions.tif')

    dataset = []
    gt = ignition
    cfire = ignition
    for d in range(minjd, maxjd):
        data = {}

        fuels[cfire != 0] = 0
        ft = [fuels]
        ft.extend([cfire, gt, slope, elev])
        ft.extend(load_day(path, d))
        ft = np.stack(ft)
        data['ft'] = ft

        if d < lastjd:
            gt = fdata == float(d)
            data['gt'] = gt

        cfire = np.logical_or(cfire ,gt)
        
        dataset.append(data)
    return dataset

class FireDataset(Dataset):
    def __init__(self, split="Train"):
        fnums = tr_fnums if split=="Train" else te_fnums
        self.dataset = []
        for fnum in fnums:
            self.dataset.extend(load_fire(fnum, split=split))
        print(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
trainset = FireDataset(split="Train")
trainset, valset = torch.utils.data.random_split(trainset, [0.9,0.1])
testset = FireDataset(split="Test")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

import torch
import torch.nn as nn

class FuelEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(FuelEmbeddings, self).__init__()

        unique_values = [0, 1, 2, 3, 4, 7, 13, 31, 101, 425, 635, 650, 665]
        self.unique_values = torch.tensor(unique_values).to(device)  # Unique values in the categorical feature
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=len(unique_values), embedding_dim=embedding_dim)

    def forward(self, categorical_feature):
        # (B,H,W) -> (B,H,W,U) wher U is unique values count
        mask = categorical_feature.unsqueeze(-1) == self.unique_values
        matching_indices = torch.argmax(mask.int(), dim=-1)

        # Apply embedding and reshape
        # (B,H,W,U) -> (B,H,W,6) -> (B,6,H,W) in default setting
        embedded_fuel = self.embedding(matching_indices)
        embedded_reshaped_fuel = embedded_fuel.permute(0, 3, 1, 2)

        return embedded_reshaped_fuel

class CNN1(nn.Module):
    def __init__(self, embedding_dim=6, num_features=8):
        super(CNN1, self).__init__()

        self.fuelembedding = FuelEmbeddings(embedding_dim)

        # (266, 433)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=(embedding_dim+num_features-1), out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        categorical_feature = x[:, 0, :, :]  # Extract the categorical feature
        embedded_fuel = self.fuelembedding(categorical_feature)  # Transform the categorical feature

        # Replace the original categorical feature with the embedded feature
        x = torch.cat((embedded_fuel, x[:, 1:, :, :]), dim=1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        out = self.sigmoid(x)

        return out
    
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, labels):
        # threshold condition is not differentiable so just use softmaxed data
        # Flatten the tensors
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        # Compute the intersection
        intersection = (outputs * labels).sum()

        # Compute the union
        union = outputs.sum() + labels.sum() - intersection
        iou = intersection / (union + 1e-6)  # Add a small epsilon for numerical stability
        loss = 1 - iou
        return loss

# Train
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0
    total_steps = 0
    for i, batch in enumerate(dataloader):
        ft = batch['ft'].to(device).float()
        gt = batch['gt'].to(device).float()

        optimizer.zero_grad()
        output = model(ft).squeeze()

        loss = criterion(output, gt)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_steps += 1
    return running_loss/total_steps

def eval(model, dataloader):
    model.eval()
    acc = []
    iou = []
    f1 = []
    total_steps = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            ft = batch['ft'].to(device)
            gt = torch.flatten(batch['gt'])

            output = torch.flatten(model(ft)).squeeze().cpu()
            output = (output > 0.5)

            acc.append(accuracy_score(gt, output))
            iou.append(jaccard_score(gt, output))
            f1.append(f1_score(gt, output))
            total_steps += 1
    return sum(acc)/total_steps, sum(iou)/total_steps, sum(f1)/total_steps

def inference(model, dataloader):
    model.eval()
    with torch.no_grad():
        cfire = torch.zeros(target_shape)
        for i, day in enumerate(dataloader):
            ft = day['ft'].to(device)

            # Create the submission file after 10 days
            if i > 9:
                cfire = torch.logical_or(output, cfire) # define the cumulative fire
                ft[0][1] = cfire # set the cumulative fire for the next input
                ft[0][2] = output # set the next step fire for the next input
            else:
                cfire = ft[0][1]

            output = model(ft)
            output = (output > 0.5)
    
    # Save the cumulative fire
    pred = cfire.cpu().squeeze().numpy()
    save_df = pd.DataFrame(pred)  # convert img data to df
    save_df.to_csv("./submission.csv", index_label='row')
    return pred

model = CNN1(num_features=13)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = IoULoss()
epochs = 1
best_miou = 0
for e in range(epochs):
    loss = train(model, trainloader, optimizer, criterion)
    aa, miou, mf1 = eval(model,valloader)

    if miou > best_miou:
        best_miou = miou
        inference(model, testloader)
        e = str(e)+"*"
    print(e, " avg iou loss:{:.3f} avg acc: {:.3f} avg f1: {:.3f} avg iou: {:.3f}".format(loss, aa, mf1, miou))