import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from models import *
from utils import progress_bar, makeDeterministic, get_selected_items
from loader import Loader, Loader2, Loader_Cold

random_seeds = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
for random_seed in random_seeds:

    # Make experiment deterministic
    makeDeterministic(random_seed)

    # Choose specific file to run
    result_file = f'CIFAR10_2023-01-02 18:13:18.437326_conf_preds_15_exp{random_seed}.csv'

    # Try all these samples
    list_of_no_of_samples = [100, 200] #[100, 200, 500, 1000, 5000]

    # Select number of samples for cold-start problem
    for number_of_samples_to_select in list_of_no_of_samples:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

        active_confusion_true_or_false = True #epoch = 0
        epoch_to_use = 0 if active_confusion_true_or_false else 14
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


        trainset = Loader_Cold(is_train=True, transform=transform_train, train_list = selected_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)

        testset = Loader_Cold(is_train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

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


        def test(epoch):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
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

            # Save checkpoint.
            acc = 100.*correct/total
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


        print(f"Now experimenting with random_seed {random_seed} with {number_of_samples_to_select} selected samples on Active Confusion: {active_confusion_true_or_false}")
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
            scheduler.step()
