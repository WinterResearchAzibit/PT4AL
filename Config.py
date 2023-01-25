import torchvision.transforms as transforms

batch_size = 256 # Depending on dataset used, batch size changes
no_of_workers = 1 # For reproducibility, used 1 worker

downstream_epochs = 200 # Number of epochs for downstream classification
pretraining_epochs = 15 # Number of epochs for pretraining as in PT4AL

# To enable reproducibility, selected random seeds are used.
random_seeds = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# Depending on the dataset used and the size of images, we
# use 28*28 for the medical images and 32 for CIFAR10 and CIFAR100
# transforms.Resize((28, 28)), OR transforms.RandomCrop(32, padding=4),
image_size = 32 # 28
images_in_dataset = len(glob.glob('./DATA/train/*/*'))

downstream_classification_classes = 10 # Number of classes in the classification dataset

# Initials in the files saved from running rotation.py
rotation_saved_file = 'CIFAR10_2023-01-08 20:23:12.648987'

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if image_size == 32:
    transform_train.transforms.insert(0, transforms.RandomCrop(32, padding=4))
else:
    transform_train.transforms.insert(0, transforms.Resize(28))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if image_size == 28:
    transform_test.transforms.insert(0, transforms.Resize(28))
