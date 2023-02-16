import json
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import io
import requests
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
import pip

def install(package):
    pip.main(['install', package])
     

def Net():
    install('smdebug')
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False 
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def model_fn(model_dir):
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Model is loading...")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('Locked and loaded')
    model.eval()
    return model
