#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout)) 


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing the model on the whole dataset...")
    model.eval()
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred == labels.data).item()
    
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    running_loss / (len(test_loader.dataset)), 
                    running_corrects, 
                    len(test_loader.dataset), 
                    100.0 * running_corrects / len(test_loader.dataset)
        ))

    pass

def train(model, train_loader, criterion, optimizer, device, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Training the dataset...")
    epochs = 2
    loss_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"Now in Epoch {epoch}")
        model.train()
        running_loss = 0
        running_corrects = 0

        for part, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1, keepdim= True)           
            running_corrects += torch.sum(pred == labels.data).item()
            
            
            if part % 100 == 0:
                logger.info("Train Epoch : {} [{}/{}({:.0f}%)] loss:{:.6f} accuracy:{:.6f}%".format(
                    epoch,
                    part * len(inputs),
                    len(train_loader.dataset),
                    100* part/ len(train_loader),
                    loss.item(),
                    100*(running_corrects/len(train_loader.dataset))
                ))
        
    return model

    pass

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

    pass

def create_data_loaders(data_dir, batch_size, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get {} data loader from s3 path {}".format(mode, data_dir))
    
    transformers = {
                    "training": transforms.Compose([transforms.Resize((224, 224)), 
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomPerspective(distortion_scale = 0.5, p = 0.5),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ]),
                    "testing": transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    "validation": transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                }
    data = ImageFolder(data_dir, transform = transformers[mode])
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)
    return data_loader

    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device {device}")
    model=net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    train_loader = create_data_loaders(args.data_dir_training , args.batch_size , "training")
    model=train(model, train_loader, criterion, optimizer, device, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.data_dir_testing , args.test_batch_size , "testing")
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    logger.info(f"Saving the model in {path}.")
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type = int ,
        default = 32, 
        metavar = "N",
        help = "input batch size for training (default : 32)"
    )
    parser.add_argument(
        "--test-batch-size",
        type = int ,
        default = 200, 
        metavar = "N",
        help = "input test batch size for training (default : 200)"
    )
    parser.add_argument(
        "--lr",
        type = float ,
        default = 0.1, 
        metavar = "LR",
        help = "learning rate (default : 0.1)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir-testing", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--data-dir-validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)