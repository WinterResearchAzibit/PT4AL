'''Train CIFAR10 with PyTorch.'''
from utils import progress_bar, makeDeterministic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import random

from models import *
from loader import Loader, RotationLoader


# Extra imports
from datetime import datetime
import pandas as pd
import Config as Config
import numpy as np

random_seeds = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
DATASET_NAME = 'CIFAR10'
TOTAL_EPOCHS = Config.pretraining_epochs
file_name = datetime.now()

# Run experiments with different random seeds
for random_seed in random_seeds:

    # Initialize with new random seed
    makeDeterministic(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def seed_worker(worker_id):
        worker_seed = random_seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(random_seed)

    trainset = RotationLoader(is_train=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.no_of_workers)

    testset = RotationLoader(is_train=False,  transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.no_of_workers, worker_init_fn=seed_worker,generator=g)

    # Model
    print('==> Building model..')
    net = ResNet18()
    net.linear = nn.Linear(512, 4)
    # net.linear = nn.Linear(25088, 4) # For 224 * 224 Input sized image
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    # Training hyperparameters
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(trainloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            optimizer.zero_grad()
            outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)

            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            loss.mean().backward()
            optimizer.step()

            train_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)*4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        global best_acc

        net.eval()

        test_loss = 0
        correct = 0
        total = 0

        correct_per_file = [] # Correct prediction for each image file
        loss_per_file = []

        file_names = []

        state = torch.get_rng_state()
        with torch.no_grad():
            for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
                inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)

                outputs = net(inputs)
                outputs1 = net(inputs1)
                outputs2 = net(inputs2)
                outputs3 = net(inputs3)

                loss1 = criterion(outputs, targets)
                loss2 = criterion(outputs1, targets1)
                loss3 = criterion(outputs2, targets2)
                loss4 = criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.

                test_loss += loss.mean().item()
                loss_per_file.extend(loss.tolist())

                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                # Compute the correct predictions for each file
                file_preds = predicted.eq(targets)*1 + predicted1.eq(targets1)*1 + predicted2.eq(targets2)*1 + predicted3.eq(targets3)*1

                # Update list of correct predictions
                correct_per_file.extend(file_preds.tolist())

                # Collect the file paths
                file_names.extend(path)

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        torch.set_rng_state(state)

        # Save checkpoint.
        acc = 100.*correct/total
        with open('./best_rotation.txt','a') as f:
            f.write(str(acc)+':'+str(epoch)+'\n')
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # save rotation weights
            torch.save(state, './checkpoint/rotation_rotate.pth')
            best_acc = acc

        return test_loss/(batch_idx+1), 100.*correct/total, correct_per_file, loss_per_file, file_names

    # Store the confusion prediction after each epoch
    dataframe = pd.DataFrame({})
    dataframe_loss = pd.DataFrame({})
    dataframe_filenames = pd.DataFrame({})

    # Name to save dataframe
    filename_conf = f'{DATASET_NAME}_{file_name}_conf_preds_{TOTAL_EPOCHS}_exp{random_seed}.csv'
    filename_loss = f'{DATASET_NAME}_{file_name}_loss_preds_{TOTAL_EPOCHS}_exp{random_seed}.csv'
    filenames = f'{DATASET_NAME}_{file_name}_filenames_{TOTAL_EPOCHS}_exp{random_seed}.csv'


    for epoch in range(start_epoch, TOTAL_EPOCHS):
        train(epoch)
        scheduler.step()
        test_loss, test_acc, correct_per_file, loss_per_file, file_names = test(epoch)

        # Insert and save after each epoch
        dataframe[str(epoch)] = correct_per_file
        dataframe.to_csv(filename_conf, index=False)

        dataframe_loss[str(epoch)] = loss_per_file
        dataframe_loss.to_csv(filename_loss, index=False)

        # Save the filenames
        dataframe_filenames[str(epoch)] = file_names
        dataframe_filenames.to_csv(filenames, index=False)
