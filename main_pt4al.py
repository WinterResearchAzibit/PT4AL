import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse
import random

from models import *
from utils import progress_bar, makeDeterministic, get_selected_items
from loader import Loader, Loader2, Loader_Cold
import pandas as pd
from datetime import datetime
import Config as Config
import numpy as np

active_confusion_true_or_false = False #epoch = 0
file_extra_info = 'conf' if active_confusion_true_or_false else 'loss'
random_seeds = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
result_df = pd.DataFrame({}, columns = ['random_seed', 'number_of_samples_to_select', 'saved_train_acc', 'saved_train_loss', 'best_acc', 'best_test_loss'])
file_name_with_date = datetime.now()

for random_seed in random_seeds:

    # Make experiment deterministic
    makeDeterministic(random_seed)

    # Choose specific file to run
    result_file = f'CIFAR10_2023-01-08 20:23:12.648987_{file_extra_info}_preds_15_exp{random_seed}.csv'

    # Try all these samples
    list_of_no_of_samples = [100, 200, 500, 1000, 5000]

    # Select number of samples for cold-start problem
    for number_of_samples_to_select in list_of_no_of_samples:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Add extra variables to track training loss and accuracy
        best_test_loss = 0
        best_train_acc = 0
        best_train_loss = 0
        saved_train_acc = 0
        saved_train_loss = 0

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        epoch_to_use = 0 if active_confusion_true_or_false else Config.pretraining_epochs - 1
        all_samples = get_selected_items(result_file, epoch_to_use)

        # Select the number of samples needed and uniform selection ratio
        range_to_check = 50000 if active_confusion_true_or_false else 5000
        uniform_selection_ratio = int(range_to_check / number_of_samples_to_select)
        selected_samples = all_samples[:range_to_check][::uniform_selection_ratio]

        # Collect the Distribution of the selected data
        class_dist = {}
        for item in selected_samples:
            class_name = item.split('/')[-2]
            class_dist[class_name] = class_dist.get(class_name, 0) + 1

        # Print the Distribution
        print("Class distributions for selected samples !!!")
        print(f"There are {len(class_dist)} classes and their distributions are: {(class_dist)}")

        def seed_worker(worker_id):
            worker_seed = random_seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(random_seed)

        trainset = Loader_Cold(is_train=True, transform=transform_train, train_list = selected_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.no_of_workers)

        testset = Loader_Cold(is_train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.no_of_workers, worker_init_fn=seed_worker,generator=g)

        # Model
        print('==> Building model..')
        net = ResNet18()
        net = net.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            # cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        # Training
        def train(epoch):
            global best_train_acc
            global best_train_loss

            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Calculate the global loss and accuracy for training
            best_train_acc = 100.*correct/total
            best_train_loss = train_loss/(batch_idx+1)

        def test(epoch):
            global best_acc
            global best_test_loss

            global best_train_acc
            global best_train_loss

            global saved_train_acc
            global saved_train_loss

            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            state = torch.get_rng_state()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            torch.set_rng_state(state)
            # Save checkpoint.
            acc = 100.*correct/total
            test_loss_result = test_loss/(batch_idx+1)
            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                best_acc = acc

                # Save train and test loss and accuracy
                best_test_loss = test_loss_result
                saved_train_acc = best_train_acc
                saved_train_loss = best_train_loss


        print(f"Now experimenting with random_seed {random_seed} with {number_of_samples_to_select} selected samples on Active Confusion: {active_confusion_true_or_false}")
        for epoch in range(start_epoch, start_epoch+Config.downstream_epochs):
            train(epoch)
            test(epoch)
            scheduler.step()

        # Save result in dataframe
        res = [random_seed, number_of_samples_to_select, saved_train_acc, saved_train_loss, best_acc, best_test_loss]
        result_df.loc[len(result_df)] = res
        result_df.to_csv(f'{file_name_with_date}_downstream_result.csv', index=False)
