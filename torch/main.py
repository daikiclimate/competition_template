import torch
from Dataset import train_datasets, test_datasets
import torch.optim as optim
from torch import nn
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.dataset import Subset

from sklearn.metrics import accuracy_score as acc

import Model

import os
import numpy as np
import random

import argparse
from addict import Dict
import yaml
# import wandb

def set_seed():
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Grid anchor based image cropping')
    parser.add_argument("config", type=str, help="path of a config file")
    return parser.parse_args()

def get_config():
    args = get_arguments()
    config = Dict(yaml.safe_load(open(args.config)))
    return config

def main():
    config = get_config()
    config.save_folder = config.save_folder + config.base + "/"  
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    cuda = True if torch.cuda.is_available() else False

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    #Creating transform for training set
    train_transforms = transforms.Compose(
    [transforms.Resize(255), 
    # transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.RandomHorizontalFlip(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #Creating transform for test set
    test_transforms = transforms.Compose(
    [transforms.Resize(255),
    # transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # data_loader_train = data.DataLoader()
    train_datasets(transform = train_transforms)

    # data_loader_test = data.DataLoader(
    test_datasets(transform = test_transforms)
                                  # )
    model = Model.build_model(base = config.base, num_classes = confing.num_classes)
    # print(model)

    if not config.no_wandb:
            wandb.init(
                config=config, project="sample", job_type="training",
            )

            # Magic
            wandb.watch(net, log="all")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

def train_test_split_model(model, trainval_datasets, optim, criterion, config,test_size = 0.8):
    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * test_size) 
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    
    data_loader_train = data.DataLoader(batch_size = config.batch_size, shuffle = True)
    data_loader_test = data.DataLoader(batch_size = config.batch_size, shuffle = False)

    for epoch in config.epochs:
        model.train()
        total_loss = 0
        for id, data in enumerate(data_loader_train):
            X, y = data
            X = X.cuda()
            y = y.cuda()
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            avg_loss = total_loss / (id+1)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grads():
            labels = []
            outputs = []
            for id, data in enumerate(data_loader_test):
                X, y = data
                X = X.cuda()
                output = model(X)
                outputs.extend(output.cpu().detach.numpy().reshape(-1))
                labels.extend(y)
            accuracy = acc(labels, outputs)
        if accuracy > config.acc_th:
            torch.save(model.state_dict(), config.save_path + "epoch_"+str(int(accuracy*100)) + ".pth")


                






if __name__=="__main__":
    main()

