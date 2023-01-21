#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug
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

import smdebug.pytorch as smd
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

def test(model, test_loader):
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
            _, pred = torch.max(outputs, 1, keepdim = True)
            running_corrects += torch.sum(pred == labels.data).item()
            running_loss += loss.item() + inputs.size(0)
    
    total_loss = running_loss / len(test_loader.dataset)
    total_corrects = running_corrects / len(test_loader / dataset)
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    total_loss, 
                    total_corrects, 
                    len(test_loader.dataset), 
                    100.0 * total_corrects / len(test_loader.dataset)
        ))

    pass

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Training the dataset...")
    epochs = 2
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Now in Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else: 
                model.eval()
        
        running_loss = 0
        running_corrects = 0
        running_samples = 0

        for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            _, pred = torch.max(outputs, 1, keepdim= True)
            running_loss += loss.item() * input.size()            
            running_corrects += torch.sum(pred == labels.data).item()
            running_samples += len(inputs)
            
            if running_samples % 100 == 0:
                logger.info("Train Epoch : {} [{}/{}({:.0f}%)] loss:{:.6f} accuracy:{:.6f}%".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100* batch_idx/ len(train_loader),
                    loss.item(),
                    100*(running_corrects/len(train_loader.dataset))
                ))

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1

        if loss_counter == 1:
            break
        
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
    model.fc = nn.Sequential(nn.Linear(num_features, 512))
    return model
    pass

def create_data_loaders(data_dir, batch_size, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get {} data loader from s3 path {}".format(mode, data_dir))
    
    transformers = {
                    "training": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(256),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomPerspective(distortion_scale = 0.5, p = 0.5),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ]),
                    "testing": transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    "validation": transforms.Compose([transforms.Resize(255),
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

    train_loader = create_data_loaders(args.data_dir_train , args.batch_size , "training")
    model=train(model, train_loader, criterion, optimizer, device, args)
    
    validation_loader = create_data_loaders(args.data_dir_test , args.test_batch_size , "validation")
    valid(model, validation_loader, criterion, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.data_dir_test , args.test_batch_size , "testing")
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
        default = 64, 
        metavar = "N",
        help = "input batch size for training (default : 64)"
    )
    parser.add_argument(
        "--test-batch-size",
        type = int ,
        default = 1000, 
        metavar = "N",
        help = "input test batch size for training (default : 1000)"
    )
    parser.add_argument(
        "--lr",
        type = float ,
        default = 0.001, 
        metavar = "LR",
        help = "learning rate (default : 0.001)"
    )

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--data-dir-test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--data-dir-valid", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    
    args=parser.parse_args()
    
    main(args)

