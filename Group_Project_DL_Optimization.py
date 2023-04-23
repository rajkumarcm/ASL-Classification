# %% Import libraries
import numpy as np
import os
from os import chdir, listdir
from os.path import abspath, join
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchmetrics import Accuracy
from sklearn.metrics import f1_score
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import copy
INTERPOLATION_MODE = transforms.InterpolationMode.NEAREST


# %% This custom dataset class is to load and preprocess the images for training, validation, and testing.
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
        image = torchvision.transforms.ConvertImageDtype(dtype=torch.float32)(image) # convert the image tensor's datatype to torch.float32
        image = torchvision.transforms.ToPILImage()(image)
        image = self.transforms(image)
        return image, label

# %%
class ASLRecognition:
    # Data augmentation
    # Define the data augmentation transforms for the training set
    def normalize(self, X):
        return X * 1 /255

    def load_filenames_df(self, data_dir, le):
        total_n_imgs = 0


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
        total_imgs = np.array(total_imgs)[indices].tolist()
        df = pd.DataFrame({"img": total_imgs, "label": total_labels})
        tr_data, ts_data, tr_labels, ts_labels = train_test_split(df, df.label, test_size=0.3, stratify=total_labels)
        tr_data, vl_data, tr_labels, vl_labels = train_test_split(tr_data, tr_data.label, test_size=0.3, stratify=tr_labels)
        return tr_data, vl_data, ts_data, tr_labels, vl_labels, ts_labels

    def __init__(self):
        DATA_DIR = r"/home/ubuntu/Project/dataset5/collated"
        NUM_WORKERS = 8
        PREFETCH_FACTOR = 30
        self.BATCH_SIZE = 512
        self.LR = 1e-3
        self.N_EPOCHS = 30
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.le = LabelEncoder()
        self.IMG_SIZE = 100

        train_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE), interpolation=INTERPOLATION_MODE),  # resize 224*224
            transforms.RandomHorizontalFlip(),  # random horizontal flip
            # Vertical flip not required
            # transforms.RandomVerticalFlip(), #random vertical flip
            transforms.RandomRotation(15, interpolation=INTERPOLATION_MODE),  # random rotation by 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Color jitter
            transforms.ToTensor(), # transform to corvent the PIL image to a PyTorch tensor
            transforms.ConvertImageDtype(torch.float32) # scale image values by dividing them by 255
        ])

        # Define the data augmentation transforms for the validation and test sets
        val_test_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE), interpolation=INTERPOLATION_MODE),
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

    def model_def(self) -> torch.nn.Module:
        final_layer_img_size = int(((self.IMG_SIZE / 2) / 2) / 2)
        last_input_shape = final_layer_img_size * final_layer_img_size * 64

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding='same', padding_mode='replicate'),  # 100x100x16
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 50x50x16
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(16, 32, 3, padding='same', padding_mode='replicate', groups=2),  # 50x50x32
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 25x25x32
            torch.nn.Dropout(0.3),

            torch.nn.Conv2d(32, 64, 3, padding='same', padding_mode='replicate', groups=2),  # 25x25x64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),  # 12x12x64
            torch.nn.Dropout(0.3),

            torch.nn.Flatten(),
            torch.nn.Linear(last_input_shape, self.N_CLASSES)
            # Ignored Softmax since the loss has this integrated.
        )
        return model

    def fit(self, model):

        best_val_f1 = 0
        best_model = None
        best_results = None

        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=0.001)  # Add L1
        criterion = torch.nn.CrossEntropyLoss().cuda()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
        acc = Accuracy(task='multiclass', num_classes=self.N_CLASSES).to(self.device)
        print("Starting training loop...")
        model.cuda()
        model.train()

        patience = 5  # Number of epochs to wait before stopping if there's no improvement
        min_val_loss = np.inf  # Initialize minimum validation loss as infinity
        counter = 0  # Initialize early stopping counter

        for epoch in range(self.N_EPOCHS):
            loss_train = 0

            for i, (X_train, y_train) in enumerate(self.tr_loader):
                imgs = Variable(X_train).to(device="cuda", memory_format=torch.channels_last, dtype=torch.float32)
                tr_labels = Variable(y_train).to(device="cuda", dtype=torch.int64)
                optimizer.zero_grad()
                tr_logits = model(imgs)
                loss = criterion(tr_logits, tr_labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()


            model.eval()
            with torch.no_grad():
                loss_val = 0
                for i, (X_val, y_val) in enumerate(self.vl_loader):
                    imgs = Variable(X_val).to(device="cuda", memory_format=torch.channels_last, dtype=torch.float32)
                    vl_labels = Variable(y_val).to(device="cuda", dtype=torch.int64)
                    vl_logits = model(imgs)
                    loss = criterion(vl_logits, vl_labels)
                    loss_val += loss.item()

            train_pred_labels = torch.argmax(torch.nn.Softmax(dim=1)(tr_logits), axis=1)
            train_f1 = f1_score(tr_labels.cpu().numpy(), train_pred_labels.cpu().numpy(), average='macro')

            val_pred_labels = torch.argmax(torch.nn.Softmax(dim=1)(vl_logits), axis=1)
            val_f1 = f1_score(vl_labels.cpu().numpy(), val_pred_labels.cpu().numpy(), average='macro')

            print("Epoch {} | Train Loss {:.5f}, Train Acc {:.5f}, Train F1 {:.5f} - Val Loss {:.5f}, Val Acc {:.5f}".format(
                    epoch + 1,
                    loss_train / len(self.tr_loader), acc(train_pred_labels, tr_labels), train_f1,
                    loss_val / len(self.vl_loader), acc(val_pred_labels, vl_labels)))

            scheduler.step(loss_val)

            # Save the best model and results
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = copy.deepcopy(model.state_dict())
                best_results = {
                    'epoch': epoch + 1,
                    'train_loss': '{:.5f}'.format(loss_train / len(self.tr_loader)),
                    'train_acc': '{:.5f}'.format(acc(train_pred_labels, tr_labels)),
                    'train_f1': '{:.5f}'.format(train_f1),
                    'val_loss': '{:.5f}'.format(loss_val / len(self.vl_loader)),
                    'val_acc': '{:.5f}'.format(acc(val_pred_labels, vl_labels)),
                    'val_f1': '{:.5f}'.format(val_f1),
                }

            # Early stopping
            if loss_val < min_val_loss:
                min_val_loss = loss_val
                counter = 0
            else:
                counter += 1
                print(f'Early stopping counter:{counter} out of {patience}')
                if counter >= patience:
                    print("Early stopping")
                    break

        # Save the best model to a file
        torch.save(best_model, "best_model.pt")

        # print the best results
        print('Best results:')
        for key, value in best_results.items():
                print(f"{key}: {value}")

        with torch.no_grad():
            ts_pred = []
            ts_labels = []
            model.eval()
            for i, (X_test, y_test) in enumerate(self.ts_loader):
                pred = torch.nn.Softmax(dim=1)(model(X_test.to("cuda")))
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                ts_pred = np.r_[ts_pred, np.ravel(pred)]
                ts_labels = np.r_[ts_labels, np.ravel(y_test.cpu())]

            # Calculate F1 score for the test set
            test_f1 = f1_score(ts_labels, ts_pred, average='macro')
            print(f"Test F1 score: {test_f1: .5f}")



if __name__ == "__main__":
    asl = ASLRecognition()
    model = asl.model_def()
    asl.fit(model)

#####################################################################################################################################

# Below the best results obtained during the training process of a deep learning - convolutional neural network model.
# Best results:
# epoch: 29
# train_loss: 0.13692
# train_acc: 0.95041
# train_f1: 0.95235
# val_loss: 0.14474
# val_acc: 0.96607

# Based on the provided results, the model performed well in recognizing American Sign Language (ASL) gestures.
# The model achieved the best performance after 29 epochs of training. The training loss at this point was 0.13692, and
# the training accuracy and F1 score were 0.95041 and 0.95235, respectively, indicating that the model learned to classify
# the training data quite effectively.

# In terms of validation performance, the model achieved a validation loss of 0.14474, a validation accuracy of 0.96607,
# and a validation F1 score of 0.96669. These results demonstrate that the model generalizes well to unseen data and is
# not overfitting the training set.

# Finally, when tested on a completely separate test dataset, the model achieved an F1 score of 0.96467. This score suggests
# that the model is reliable in recognizing ASL fingerspelling and can be considered a good fit for the given task.
# Overall, the results demonstrate the success of the chosen architecture and training strategy in solving the ASL recognition problem.

#####################################################################################################################################
