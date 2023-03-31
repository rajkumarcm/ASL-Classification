import numpy as np
import torch
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


# Define function to store labels and images into dictionary

def get_labels_images(path):
    directories = []
    labels = []
    images = []

    for directory in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir, directory)
        if not os.path.isdir(dir_path):
            continue

        for label in os.listdir(dir_path):
            label_path = os.path.join(dir_path, label)
            if not os.path.isdir(label_path):
                continue

            for image in os.listdir(label_path):
                directories.append(directory)
                labels.append(label)
                images.append(os.path.join(directory, label, image))

    print(len(images), len(labels), len(directories))
    return pd.DataFrame({'directories': directories, 'labels': labels, 'images': images})

dataset_dir = os.path.abspath('/Users/daqian.dang/Desktop/DATS 6303/Project/dataset5/')
data = get_labels_images(dataset_dir)


# Show random image samples (mixed color images and non-color images)

def random_image_samples(data: pd.DataFrame, num_rows: int=4, num_columns: int=4):
    num_images = num_rows * num_columns
    random_indices = random.sample(range(len(data)), num_images)
    random_images = data.iloc[random_indices]

    plt.figure(figsize=(15, 15))

    for i, (index, row) in enumerate(random_images.iterrows()):
        image_path = os.path.join(dataset_dir, row['images'])
        image = imread(image_path)
        label = row['labels']

        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

random_image_samples(data, num_rows=4, num_columns=4)


# Show random image samples

color_data = data[data['images'].map(
    lambda x: True if x.find('color')!=-1 else False)].reset_index(drop=True)
deep_data = data[data['images'].map(
    lambda x: True if x.find('color')==-1 else False)].reset_index(drop=True)
data.shape[0]==(color_data.shape[0] + deep_data.shape[0])

random_image_samples(color_data, num_rows=4, num_columns=4) # color images only
random_image_samples(deep_data, num_rows=4, num_columns=4) # non-color images only


# Image size
def select_random_image(data: pd.DataFrame, dataset_dir: str):
    random_row = data.sample()
    image_path = os.path.join(dataset_dir, random_row['images'].values[0])
    image = Image.open(image_path)

    return image

random_image = select_random_image(data, dataset_dir)
shape = random_image.size + (len(random_image.getbands()),)
print(shape)


# Check the image label values if they are balanced or imbalanced
label_counts = data['labels'].value_counts(normalize=True)
print(label_counts)

plt.figure(figsize=(10, 5))
plt.bar(label_counts.index, label_counts.values)

plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Balance Check')
plt.show()

# After checking and plotting the balance of the image label values, the dataset is fairly balanced.


# Rename data labels to target
target = data['labels']

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(target)
data['encoded_labels'] = label_encoder.transform(target)


# Split dataset into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


# Data augmentation
#Define the data augmentation transforms for the training set
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #resize 224*224
    transforms.RandomHorizontalFlip(), #random horizontal flip
    transforms.RandomVerticalFlip(), #random vertical flip
    transforms.RandomRotation(15), #random rotation by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Color jitter
    transforms.ToTensor(), # transform to corvent the PIL image to a PyTorch tensor
    transforms.Lambda(lambda x: x / 255.0) # scale image values by dividing them by 255
])

#Define the data augmentation transforms for the validation and test sets
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0)
])

class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, dataset_dir: str, transform=None):
        self.data = data
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_name = os.path.join(self.dataset_dir, self.data.iloc[i]['images'])
        image = Image.open(image_name).convert('RGB')
        label = self.data.iloc[i]['encoded_labels']

        if self.transform:
            image = self.transform(image)

        return image, label

# Create CustomDataset objects for train, validation, and test sets
train_dataset = CustomDataset(train_data, dataset_dir, transform=train_transforms)
val_dataset = CustomDataset(val_data, dataset_dir, transform=val_test_transforms)
test_dataset = CustomDataset(test_data, dataset_dir, transform=val_test_transforms)


# Employ DataLoader as prefetch for training
BATCH_SIZE = 32
NUM_WORKERS = 4  # this is parameter that determines the number of worker processes to use for loading the data
PREFETCH_FACTOR = 2  # this determines how many samples will be loaded per worker process during the prefetching

new_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
new_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
new_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)

