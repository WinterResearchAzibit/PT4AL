'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os, random
import sys
import time
import math
import numpy as np
import pandas as pd
import glob as glob

import torch
import torch.nn as nn
import torch.nn.init as init

def get_selected_items(result_file, epoch):
    '''
    result_file - csv file containing the loss or confusion per epoch
    '''

    # Get the file list
    file_list = glob.glob('./DATA/train/*/*')

    # Read the confusion or loss file
    df = pd.read_csv(result_file)

    #Sum over all epochs for each file
    score_list = df[[str(epoch)]].sum(axis = 1).tolist() #df.sum(axis = 1).tolist()

    # Consider result as string
    new_score_list = []
    for item in score_list:
        new_score_list.append(str(item))
    score_list = new_score_list

    # Setup code as done in PT4AL
    s = np.array(score_list)
    sort_index = np.argsort(s)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x)
    # print(f"Sorted index: {sort_index[:10]}")
    sorted_list_result = []

    # Sort files based on sort_index
    for item in sort_index:
        sorted_list_result.append(file_list[item])

    # New dataframe to hold files and their scores
    new_df = pd.DataFrame({})
    new_df['0'] = score_list
    new_df['1'] = file_list

    # Sort in descending order and return the file names
    first_sample = new_df.sort_values('0', ascending=False)['1']
    # print(f"Sorted list: {sorted_list_result[:10]}")
    # print(f"First sample: {first_sample[:10]}")
    # print(f"Top 10 in second sorted: {first_sample[:10]}")
    # # Collect the Distribution of the selected data
    # class_dist = {}
    # for item in first_sample:
    #     class_name = item.split('/')[-2]
    #     class_dist[class_name] = class_dist.get(class_name, 0) + 1
    #
    # # Print the Distribution
    # print(f"There are {len(class_dist)} classes and their distributions are: {(class_dist)}")
    #
    # # Return the selected files
    # return first_sample.tolist()
    return sorted_list_result

def makeDeterministic(random_seed):
    # random_seed=42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
