'''
    Cold-start experiment for randomly selecting the initial subset.
    Random items are selected based on the size of the initial subset
'''
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
from utils import progress_bar, makeDeterministic
from loader import Loader, Loader2
import numpy as np
import pandas as pd
import Config as Config
from datetime import datetime

# To ensure reproducibility, we used specific random seeds for each experiment
random_seeds = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# Dataframe to store the results of each experiment
result_df = pd.DataFrame({}, columns = ['random_seed', 'number_of_samples_to_select', 'saved_train_acc', 'saved_train_loss', 'best_acc', 'best_test_loss'])

# Adding a time stamp for each result
file_name_with_date = datetime.now()

# Perform each experiment using the random seed
for random_seed in random_seeds:

    # Make experiment deterministic
    makeDeterministic(random_seed)

    # Try all these initial subset sizes during each experiment
    list_of_no_of_samples = [100, 200, 500, 1000, 5000]

    # Select number of samples for random case
    for number_of_samples_to_select in list_of_no_of_samples:
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        args = parser.parse_args()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([

            # Depending on the dataset used and the size of images, we
            # use 28*28 for the medical images and 32 for CIFAR10 and CIFAR100
            transforms.Resize((28, 28)),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            # Applying size 28*28 for medical image scenario
            # The resize is removed if not medical dataset
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Depending on the dataset, the number of training images
        # from which the random selection is to be made
        dataset_training_size = 11959
        indices = list(range(dataset_training_size))

        # Shuffle the indices
        random.shuffle(indices)

        # Select the specified number of random samples
        labeled_set = indices[:number_of_samples_to_select]

        # Introducing this method to ensure that the testloader is deterministic
        def seed_worker(worker_id):
            worker_seed = random_seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # The following two lines are needed to make testloader deterministic
        g = torch.Generator()
        g.manual_seed(random_seed)

        trainset = Loader(is_train=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=Config.batch_size, num_workers=Config.no_of_workers,
        sampler=SubsetRandomSampler(labeled_set))

        testset = Loader(is_train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset,
        batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.no_of_workers, worker_init_fn=seed_worker,
        generator=g)

        # Model
        print('==> Building model..')
        net = ResNet18()

        # Depending on the number of classes in the dataset.
        number_of_classes_in_dataset = 8
        net.linear = nn.Linear(512, number_of_classes_in_dataset)
        net = net.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            # cudnn.benchmark = True

        # Use parameters from PT4AL
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

            # Track accuracy and loss for training and test
            global best_acc
            global best_test_loss

            # These values are updated based on the best_acc
            global best_train_acc
            global best_train_loss

            global saved_train_acc
            global saved_train_loss

            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            state = torch.get_rng_state() # Added for reproducibility
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

            torch.set_rng_state(state) # Added for reproducibility
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


        print(f"Now experimenting with random_seed {random_seed} with {number_of_samples_to_select} selected samples for random selection")
        for epoch in range(start_epoch, start_epoch+Config.downstream_epochs):
            train(epoch)
            test(epoch)
            scheduler.step()

        # Save result in dataframe
        res = [random_seed, number_of_samples_to_select, saved_train_acc, saved_train_loss, best_acc, best_test_loss]
        result_df.loc[len(result_df)] = res
        result_df.to_csv(f'{file_name_with_date}_random_result.csv', index=False)
