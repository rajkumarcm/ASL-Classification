#%% Load the data
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
from os import chdir, listdir
from os.path import abspath, join
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
from torchsummary import summary
from torchmetrics import Accuracy, F1Score
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from time import time
INTERPOLATION_MODE = transforms.InterpolationMode.NEAREST


#%%
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

#%%
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
        
        DATA_DIR = r"D:/GWU/DATS-6303/project/archive/dataset5/collated"
        # DATA_DIR = r"/home/ubuntu/ASL_Data/dataset5/collated"
        NUM_WORKERS = 8
        PREFETCH_FACTOR = 30
        self.BATCH_SIZE = 1000 # *****
        self.LR = 1e-4 # *****
        self.N_EPOCHS = 300 # *****
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.le = LabelEncoder()
        self.IMG_SIZE = 100
        self.filename = "model_optimization_2.pt"
        self.SAVE_DIR = f"D:/GWU/DATS-6303/project/saved_models/{self.filename}"
        
        train_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE), interpolation=INTERPOLATION_MODE), #resize 224*224
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
        # new_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
        model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 5, padding='same', padding_mode='replicate'), # 56x56x16
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),  # 28x28x16
                
                # *****
                torch.nn.Dropout(0.3),
                # *****
                
                torch.nn.Conv2d(16, 32, 5, padding='same', padding_mode='replicate',
                                groups=2), # 28x28x32
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),  # 14x14x32
                
                # *****
                torch.nn.Dropout(0.3),
                # *****
                
                torch.nn.Conv2d(32, 64, 5, padding='same', padding_mode='replicate',
                                groups=2), # 14x14x64
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2)),  # 7x7x64
                
                # *****
                torch.nn.Dropout(0.3),
                # *****
                
                torch.nn.Flatten(),
                
                # *****
                torch.nn.Linear(last_input_shape, 128),
                #torch.nn.Linear(128, 64),
                torch.nn.Dropout(0.3),
                # *****
                
                torch.nn.Linear(128, self.N_CLASSES)
                # Ignored Softmax since the loss has this integrated.
        )
        return model

    def fit(self, model):
        start_time = time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=0.001) # *****
        criterion = torch.nn.CrossEntropyLoss().cuda()
        acc = Accuracy(task='multiclass', num_classes=self.N_CLASSES).to(self.device)
        # f1_macro = F1Score(task='multiclass', num_classes=self.N_CLASSES, average='macro').to(self.device)
        
        # Initialize test loss threshold for early stopping and parameters to track *****
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) # I think mode should be min for loss?
        early_stopping_patience = 30          
        current_patience = early_stopping_patience
        early_stoping_sensitivity = 4
        best_validation_loss = round(1000000., early_stoping_sensitivity) # made this arbitrarily large so that the first round of training should always succeed
        # current_learning_rate = self.LR
        best_validation_score = 0.0000000001
        #***** 
        
        print("Starting training loop...")
        model.cuda()
        model.train()
        for epoch in range(self.N_EPOCHS):
            loss_train = 0
            # summary(self.model, torch.cuda.FloatTensor([3, 224, 224]))
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
            
            # Get evaluation metrics for printing
            tr_preds = torch.argmax(torch.nn.Softmax(dim=1)(tr_logits), axis=1)
            val_preds = torch.argmax(torch.nn.Softmax(dim=1)(vl_logits), axis=1)
            acc_tr = acc(tr_preds, tr_labels)
            acc_val = acc(val_preds, vl_labels)
            train_loss_batch = round(loss_train / self.BATCH_SIZE, 5)
            val_loss_batch = round(loss_val / self.BATCH_SIZE, 5)
            
            print(f"Epoch {epoch+1} | Train Loss {train_loss_batch}, Train Acc {acc_tr} - Validation Loss {val_loss_batch}, Validation Acc {acc_val}")
        
            # EARLY STOPPING and LEARNING RATE SCHEDULING *****
            # Check if the loss is going down
            if round(loss_val, early_stoping_sensitivity) >= best_validation_loss:
                # If loss not decerasing, remove one level of patience and drop learning rate
                current_patience -=1
                print(f"early stopping validation loss check activated, early stopping patience remaing: {current_patience}")
            else:
                # If loss is decreasing, update lowest loss and reset ES patience
                best_validation_loss = loss_val
                current_patience = early_stopping_patience
            
            # If ES patience exhausted, stop training and print ES message
            if current_patience == 0:
                print(f"early stopping implemented at Epoch {epoch}")
                break
            #*****
            
            # Include LR scheduler *****
            scheduler.step(loss_val)
            #*****

            # Saving best performing model *****
            if acc_val > best_validation_score or (acc_val == 1.0000 and acc_tr == 1.0000):
                best_validation_score = acc_val
                torch.save(model.state_dict(), self.SAVE_DIR)
                print("**Model Saved**")
            # *****
                
        print(f"Total time taken: {time()-start_time}")
        
    def test(self, model):
        print("Starting testing...")
        model.cuda()
        model.eval()
        with torch.no_grad():
            ts_pred = []
            ts_labels = []
            model.eval()
            for i, (X_test, y_test) in enumerate(self.ts_loader):
                pred = torch.nn.Softmax(dim=1)(model(X_test.to("cuda")))
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                ts_pred = np.r_[ts_pred, np.ravel(pred)]
                ts_labels = np.r_[ts_labels, np.ravel(y_test.cpu())]
            f1 = f1_score(y_true=ts_labels, y_pred=ts_pred, labels=np.unique(ts_labels), average='macro')
            # f1 = F1Score(task="multiclass", num_classes=24)
            # f1_score = f1(ts_pred, ts_labels)
            print(f'Test Set F1: {f1}')
                
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
    model = asl.model_def()
    asl.fit(model)
    asl.test(model)

#%%

####################
## CODE GRAVEYARD ##
####################

# # Get evaluation metrics for printing
# tr_preds_array = []
# tr_labels_array = []
# val_preds_array = []
# val_labels_array = []
# # Get torch of predictions
# tr_preds = torch.argmax(torch.nn.Softmax(dim=1)(tr_logits), axis=1)
# val_preds = torch.argmax(torch.nn.Softmax(dim=1)(vl_logits), axis=1)
# # Send prediction torches to numpy
# tr_preds_np = np.argmax(tr_preds.cpu().detach().numpy())
# val_preds_np = np.argmax(val_preds.cpu().detach().numpy())
# # Create prediction and target arrays
# tr_preds_array = np.r_[tr_preds_array, np.ravel(tr_preds_np)]
# tr_labels_array = np.r_[tr_labels_array, np.ravel(y_train.cpu())]
# val_preds_array = np.r_[val_preds_array, np.ravel(val_preds_np)]
# val_labels_array = np.r_[val_labels_array, np.ravel(y_val.cpu())]
# # Use prediciont and target array to calculate macro F1
# f1_tr = round(f1_score(y_true=tr_labels_array, y_pred=tr_preds_array, labels=np.unique(tr_labels_array), average='macro'), 4)
# f1_val = round(f1_score(y_true=val_labels_array, y_pred=val_preds_array, labels=np.unique(val_labels_array), average='macro'), 4)
# # f1_tr = round(f1_macro(tr_preds_np, tr_labels), 4)
# # f1_val = round(f1_macro(val_preds_np, vl_labels), 4)
# acc_tr = torch.round(acc(tr_preds, tr_labels), 4)
# acc_val = torch.round(acc(val_preds, vl_labels), 4)
# train_loss_batch = round(loss_train / self.BATCH_SIZE, 5)
# val_loss_batch = round(loss_val / self.BATCH_SIZE, 5)

# %%
