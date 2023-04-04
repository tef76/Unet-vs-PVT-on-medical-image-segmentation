from functools import partial
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import SegmentationDataset
from utils import train
from PVT import PyramidVisionTransformer
from UNET import UNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices = ['UNET', 'PVT'], required = True)
parser.add_argument('--dataPath', required = True)

args = parser.parse_args()

if args.model == 'UNET':    
    model = UNet(3,4)
    
if args.model == 'PVT':
    model = PyramidVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, num_classes = 4)

loss_fn = nn.CrossEntropyLoss()
loss_fn.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
t = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.PILToTensor()])
trainDS = SegmentationDataset(transform, t, args.dataPath)

trainLoader = DataLoader(trainDS,batch_size=32)

train(150, model, trainLoader, optimizer, loss)