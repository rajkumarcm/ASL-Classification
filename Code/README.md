# Instructions for Downloading Data and Running the Code

## 1. Download data set from Kaggle
###     a. Access the Kaggle page for [ASL Fingerspelling Images (RGB & Depth)](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out) 
###     b. Under the dataset section of the page, use the download button to download *dataset5* to your computer
###     c. Unzip the downloaded files and move them to a desired location on your computer

<br>

## 2. Reformat the data to be used by the model
###    a. Open up the *rearrange_files.py* code file
###    b. On line 9, change the variable, *DATA_DIR*, to the location of where you saved the Kaggle data set on your pc
###    c. For example, it may look like: *f"C:/my_data/archive/dataset5"*
###    c. Run *rearrange_files.py*, it may take several minutes to complete
###    d. After the script completes, you should see a new folder, *collated*, inside the *dataset5* folder

<br>

## 3. Train the Model
### a. Open the *Group_Project_DL* code file
### b. On line, 109, change the variable, *DATA_DIR*, to the location of the newly created *collated* folder
### c. For example, it may look like: *f"C:/my_data/archive/dataset5/collated"*
### d. On line, 118, change the variable, *self.filename*, to what you would like to name the produced model
### d. On line, 121, change the variable, *self.SAVE_DIR*, to the location you would like to store the produced model
### e. For example, it may look like *f"C:/my_data/saved_models/{self.filename}"*

<br>

## 4. Train the model
### a. Open the *Group_Project_DL* code file and examine the code at lines 323-327
### b. Ensure that lines 323, 324, and 325 are uncommented
### c. This would look like: 
    asl = ASLRecognition()
    model = asl.model_def()
    asl.fit(model)
    # asl.test(model, used_saved=False)
    # asl.test(model, used_saved=True)
### d. Run the file

<br>

## 5. Test the model
### a. Open the *Group_Project_DL* code file and examine the code at lines 323-327
### b. Ensure that lines 323, 324, and 327 are uncommented and that line 325 is commented out
### c. This would look like
    asl = ASLRecognition()
    model = asl.model_def()
    # asl.fit(model)
    # asl.test(model, used_saved=False)
    asl.test(model, used_saved=True)

<br>

## 6. Making personal changes to the model
-If you desire to make changes to and experiment with the model, you can find most of the **hyperparamters** from lines *107-122*  
-**Augmentation** transformations are at lines *138-145*  
-**Model architecture** is at lines *190-199*  
-The **optimization** type is on line *208*, along with **L2 regularization**  
-The **performance index** is on line *209*  
-**Early stopping** and **LR scheduling** details are from lines *214-220*  
