#%%
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchsummary import summary
from torchmetrics import Accuracy
import re
import pandas as pd
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

class ImageDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data.loc[:, ["img", "label"]].reset_index()
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_fname, label = self.data.iloc[idx]["img"], self.data.iloc[idx]["label"]
        image = Image.open(img_fname)
        image = torchvision.transforms.PILToTensor()(image)
        image = torchvision.transforms.ConvertImageDtype(dtype=torch.float32)(image)
        image = torchvision.transforms.ToPILImage()(image)
        image = self.transforms(image)
        return image, label

class ASLRecognition:
    # Data augmentation
    #Define the data augmentation transforms for the training set
    def normalize(self, X):
        return X * 1./255

    def load_filenames_df(self, data_dir, le):
        total_n_imgs = 0
        # join_path = lambda x: join(self.data_dir, join(x[0], x[1]))

        total_imgs = []
        total_labels = []
        for char_ in listdir(data_dir):
            fnames = listdir(join(data_dir, char_))
            color_filter_mask = [len(re.findall("color", fname)) > 0 for fname in fnames]
            fnames = np.array(fnames)[color_filter_mask]
            fnames = [join(data_dir, join(char_, fname)) for fname in fnames]
            total_imgs = np.r_[total_imgs, fnames]
            total_labels = np.r_[total_labels, [char_] * len(fnames)]
            total_n_imgs += len(fnames)
        total_n_imgs = total_n_imgs
        total_labels = le.fit_transform(total_labels)
        indices = list(range(total_n_imgs))
        total_labels = total_labels[indices]
        total_imgs = total_imgs[indices]
        df = pd.DataFrame({"img": total_imgs, "label": total_labels})
        tr_data, ts_data, tr_labels, ts_labels = train_test_split(df, df.label, test_size=0.3, stratify=total_labels)
        tr_data, vl_data, tr_labels, vl_labels = train_test_split(tr_data, tr_data.label, test_size=0.3, stratify=tr_labels)
        return tr_data, vl_data, ts_data, tr_labels, vl_labels, ts_labels

    def __init__(self):
        # dataset_dir = os.path.abspath('/Users/daqian.dang/Desktop/DATS 6303/Project/dataset5/')
        DATA_DIR = r"C:\Users\Rajkumar\Downloads\ASL\dataset5\collated"
        # DATA_DIR = r"/home/ubuntu/ASL_Data/dataset5/collated"
        NUM_WORKERS = 8
        PREFETCH_FACTOR = 10
        self.BATCH_SIZE = 200
        self.LR = 1e-3
        self.N_EPOCHS = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.le = LabelEncoder()

        train_transforms = transforms.Compose([
            transforms.Resize((56, 56), interpolation=INTERPOLATION_MODE), #resize 224*224
            transforms.RandomHorizontalFlip(), #random horizontal flip
            # Vertical flip not required
            # transforms.RandomVerticalFlip(), #random vertical flip
            transforms.RandomRotation(15, interpolation=INTERPOLATION_MODE), #random rotation by 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Color jitter
            transforms.ToTensor(), # transform to corvent the PIL image to a PyTorch tensor
            transforms.ConvertImageDtype(torch.float32) # scale image values by dividing them by 255
        ])

        #Define the data augmentation transforms for the validation and test sets
        val_test_transforms = transforms.Compose([
            transforms.Resize((56, 56), interpolation=INTERPOLATION_MODE),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        tr_data, vl_data, ts_data, tr_labels, vl_labels, ts_labels = self.load_filenames_df(DATA_DIR, self.le)
        tr_dset = ImageDataset(tr_data, train_transforms)
        vl_dset = ImageDataset(vl_data, val_test_transforms)
        ts_dset = ImageDataset(ts_data, val_test_transforms)

        self.N_CLASSES = len(tr_data.label.unique())
        self.tr_loader = DataLoader(tr_dset, batch_size=self.BATCH_SIZE, num_workers=NUM_WORKERS,
                                    prefetch_factor=PREFETCH_FACTOR)
        self.vl_loader = DataLoader(vl_dset, batch_size=self.BATCH_SIZE, num_workers=NUM_WORKERS,
                                    prefetch_factor=PREFETCH_FACTOR)
        # for i, (X_tmp, y_tmp) in enumerate(vl_dset):
        #     print(i)
        self.ts_loader = DataLoader(ts_dset, batch_size=self.BATCH_SIZE, num_workers=NUM_WORKERS,
                                    prefetch_factor=PREFETCH_FACTOR)

    #%%
    def random_image_samples(self, data, num_rows=4, num_columns=4):
        num_images = num_rows * num_columns
        random_indices = random.sample(range(len(data)), num_images)
        random_images = data.iloc[random_indices]

        plt.figure(figsize=(15, 15))
        for i, (index, row) in enumerate(random_images.iterrows()):
            image_path = os.path.join(self.DATA_DIR, row['img'])
            image = imread(image_path)
            label = row['label']

            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image)
            plt.title(f"Label: {self.le.inverse_transform([label])[0]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    #%%

    def model_def(self):

        # new_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (3, 3), 1, 1, padding_mode='replicate'), # 56x56x16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 28x28x16
            torch.nn.Conv2d(16, 32, (3, 3), 1, 1, padding_mode='replicate'), # 28x28x32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 14x14x32
            torch.nn.Conv2d(32, 64, (3, 3), 1, 1, padding_mode='replicate'), # 14x14x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 7x7x64
            torch.nn.Flatten(),
            torch.nn.Linear(7*7*64, self.N_CLASSES)
            # Ignored Softmax since the loss has this integrated.
        ).cuda()
        return self.model

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        acc = Accuracy(task='multiclass', num_classes=self.N_CLASSES).to(self.device)
        print("Starting training loop...")
        for epoch in range(self.N_EPOCHS):
            loss_train = 0
            self.model.train()
            # summary(self.model, torch.cuda.FloatTensor([3, 224, 224]))
            for i, (X_train, y_train) in enumerate(self.tr_loader):
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)
                optimizer.zero_grad()
                tr_logits = self.model(X_train)
                loss = criterion(tr_logits.to(self.device), y_train.to(self.device))
                loss.backward()
                optimizer.step()
                loss_train += loss.item()

            self.model.eval()
            with torch.no_grad():
                loss_val = 0
                count = 0
                for i, (X_val, y_val) in enumerate(self.vl_loader):
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)
                    vl_logits = self.model(X_val)
                    loss = criterion(vl_logits.to(self.device), y_val.to(self.device))
                    loss_val += loss.item()

            print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
                epoch, loss_train / self.BATCH_SIZE, acc(torch.argmax(torch.nn.Softmax()(tr_logits), axis=1), y_train),
                loss_val, acc(torch.argmax(torch.nn.Softmax()(vl_logits), axis=1), y_val)))


if __name__ == "__main__":
    # random_image_samples(data, num_rows=4, num_columns=4)
    # random_image_samples(tr_data.tr_data, num_rows=4, num_columns=4)  # color images only
    # # Check the image label values if they are balanced or imbalanced
    # label_counts = tr_data.tr_data.label.value_counts(normalize=False)
    # # print(label_counts)
    #
    # plt.figure(figsize=(10, 5))
    # plt.bar(label_counts.index, label_counts.values)
    # plt.xlabel('Labels')
    # plt.ylabel('Frequency')
    # plt.title('Label Balance Check')
    # plt.show()
    asl = ASLRecognition()
    asl.model_def()
    asl.fit()










