#%%
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchmetrics import Accuracy
import re
import pandas as pd
import os
from os import chdir, listdir
from os.path import abspath, join

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
INTERPOLATION_MODE = transforms.InterpolationMode.NEAREST

#%%
# dataset_dir = os.path.abspath('/Users/daqian.dang/Desktop/DATS 6303/Project/dataset5/')
DATA_DIR = r"C:\Users\Rajkumar\Downloads\ASL\dataset5\collated"

#%%
# Define function to store labels and images into dictionary
# def get_labels_images(path):
#     directories = []
#     labels = []
#     images = []
#
#     for directory in os.listdir(dataset_dir):
#         dir_path = os.path.join(dataset_dir, directory)
#
#         for label in os.listdir(dir_path):
#             label_path = os.path.join(dir_path, label)
#
#             for image in os.listdir(label_path):
#                 directories.append(directory)
#                 labels.append(label)
#                 images.append(os.path.join(directory, label, image))
#
#     print(len(images), len(labels), len(directories))
#     return pd.DataFrame({'directories': directories, 'labels': labels, 'images': images})

#%%
# Data augmentation
#Define the data augmentation transforms for the training set
def normalize(X):
    return X * 1./255

train_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=INTERPOLATION_MODE), #resize 224*224
    transforms.RandomHorizontalFlip(), #random horizontal flip
    # Vertical flip not required
    # transforms.RandomVerticalFlip(), #random vertical flip
    transforms.RandomRotation(15, interpolation=INTERPOLATION_MODE), #random rotation by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Color jitter
    transforms.ToTensor(), # transform to corvent the PIL image to a PyTorch tensor
    transforms.Lambda(normalize) # scale image values by dividing them by 255
])

#Define the data augmentation transforms for the validation and test sets
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=INTERPOLATION_MODE),
    transforms.ToTensor(),
    transforms.Lambda(normalize)
])


#%%
class ImageDataset(Dataset):
    def __init__(self, data_dir, transforms, training, split_indices=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # self.is_color = lambda x: len(re.findall("color", x)) > 0
        self.le = LabelEncoder()
        self.training = training

        n_imgs = 0
        # join_path = lambda x: join(self.data_dir, join(x[0], x[1]))

        imgs = []
        labels = []
        for char_ in listdir(self.data_dir):
            fnames = listdir(join(self.data_dir, char_))
            # fnames = list(filter(self.is_color, fnames))
            color_filter_mask = [len(re.findall("color", fname)) > 0 for fname in fnames]
            fnames = np.array(fnames)[color_filter_mask]
            fnames = [join(self.data_dir, join(char_, fname)) for fname in fnames]
            # fnames = list(map(join_path, zip([char_]*len(fnames), fnames)))
            imgs = np.r_[imgs, fnames]
            labels = np.r_[labels, [char_]*len(fnames)]
            n_imgs += len(fnames)
        self.n_imgs = n_imgs
        labels = self.le.fit_transform(labels)
        indices = list(range(n_imgs))
        labels = labels[indices]
        imgs = imgs[indices]
        self.df = pd.DataFrame({"img": imgs, "label": labels})
        if not split_indices:
            split_indices = list(range(self.df.shape[0]))
            tr_indices, vl_indices, tr_labels, vl_labels = train_test_split(split_indices, labels,
                                                                            test_size=0.3, stratify=labels)
            self.tr_indices = tr_indices
            self.vl_indices = vl_indices
        else:
            tr_indices, vl_indices = split_indices

        self.tr_data, self.vl_data = self.df.iloc[tr_indices].reset_index(), \
                                     self.df.iloc[vl_indices].reset_index()

    def __len__(self):
        if self.training:
            return self.tr_data.shape[0]
        else:
            return self.vl_data.shape[0]

    def __getitem__(self, idx):
        if self.training:
            img_fname, label = self.tr_data.loc[idx, "img"], self.df.loc[idx, "label"]
        else:
            img_fname, label = self.vl_data.loc[idx, "img"], self.df.loc[idx, "label"]
        image = Image.open(img_fname)
        image = torchvision.transforms.PILToTensor()(image)
        image = torchvision.transforms.ConvertImageDtype(dtype=torch.float32)(image)
        image = torchvision.transforms.ToPILImage()(image)
        image = self.transforms(image)
        return image, label

#%%
def random_image_samples(data: pd.DataFrame, num_rows: int=4, num_columns: int=4):
    num_images = num_rows * num_columns
    random_indices = random.sample(range(len(data)), num_images)
    random_images = data.iloc[random_indices]

    plt.figure(figsize=(15, 15))

    for i, (index, row) in enumerate(random_images.iterrows()):
        image_path = os.path.join(DATA_DIR, row['img'])
        image = imread(image_path)
        label = row['label']

        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {tr_data.le.inverse_transform([label])[0]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#%%
# random_image_samples(data, num_rows=4, num_columns=4)

tr_data = ImageDataset(DATA_DIR, train_transforms, training=True)
split_indices = (tr_data.tr_indices, tr_data.vl_indices)
vl_data = ImageDataset(DATA_DIR, val_test_transforms, training=False,
                       split_indices=split_indices)

#%%
random_image_samples(tr_data.tr_data, num_rows=4, num_columns=4) # color images only

# Check the image label values if they are balanced or imbalanced
label_counts = tr_data.tr_data.label.value_counts(normalize=False)
# print(label_counts)

plt.figure(figsize=(10, 5))
plt.bar(label_counts.index, label_counts.values)

plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Balance Check')
plt.show()


# Employ DataLoader as prefetch for training
BATCH_SIZE = 32
NUM_WORKERS = 4  # this is parameter that determines the number of worker processes to use for loading the data
PREFETCH_FACTOR = 2  # this determines how many samples will be loaded per worker process during the prefetching

tr_data.__getitem__(0)
tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
vl_loader = DataLoader(vl_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
# new_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
N_CLASSES = len(tr_data.tr_data.label.unique())

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, 1, 1, 'replicate'), # 222x222x16
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 1), # 111x111x16
    torch.nn.Conv2d(16, 32, 3, 1, 1, 'replicate'), # 109x109x32
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 1), # 54x54x32
    torch.nn.Conv2d(32, 64, 3, 1, 1, 'replicate'), # 52x52x64
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 1), # 26x26x64
    torch.nn.Flatten(),
    torch.nn.Linear(26*26*64, N_CLASSES)
    # Ignored Softmax since the loss has this integrated.
)

LR = 1e-3
N_EPOCHS = 10

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()
acc = Accuracy(task='multiclass', num_classes=N_CLASSES)
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train()
    for i, (X_train, y_train) in enumerate(tr_loader):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        loss_val = 0
        count = 0
        for i, (X_val, y_val) in enumerate(vl_loader):
            y_val_pred = model(X_val)
            loss = criterion(y_val_pred, y_val)
            loss_val += loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train/BATCH_SIZE, acc(torch.nn.Softmax(logits), y_train),
        loss_val, acc(X_val, y_val)))


















