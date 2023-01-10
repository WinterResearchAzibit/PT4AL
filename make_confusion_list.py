import pandas as pd
import glob as glob
from collections import Counter
import random

# ALWAYS CHECK
#1. Confusion file Name
#2. Folder of dataset

# Name of the confusion file
confusion_file = 'CIFAR10_2023-01-03 04:24:21.661858_conf_preds_15_exp100.csv'
# Get the file list
file_list = glob.glob('./DATA/train/*/*')

# Read the confusion file
df = pd.read_csv(confusion_file)

#Sum over all epochs for each file
confusion_score_list = df[['14']].mean(axis = 1).tolist() #df.sum(axis = 1).tolist()

new_df = pd.DataFrame({})
new_df['0'] = confusion_score_list
new_df['1'] = file_list

first_sample = new_df.sort_values('0', ascending=False)['1'][:5000][::5]
print("Understanding class Distribution")
class_dist = {}
for item in first_sample:
    class_name = item.split('/')[-2]
    class_dist[class_name] = class_dist.get(class_name, 0) + 1

print((class_dist))
print(len(class_dist))

with open('loss/batch_0_compare.txt', 'w') as f:
    for line in first_sample:
        f.write(f"{line}\n")
