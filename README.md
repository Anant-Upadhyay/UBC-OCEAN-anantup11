# Ovarian Cancer Sub-Type Identification
This repository contains code for identifying the type of ovarian cancer from images of ovarian cells, as part of the UBC-OCEAN competition on Kaggle. The solution involves separate pipelines for TMA (Tissue Microarray) and non-TMA images, which are processed through a common ResNet-101 convolutional neural network.
<img src="https://storage.googleapis.com/kaggle-media/competitions/UBC/OCEAN-Optional-Figure.png"></img>
- Competition Link: https://www.kaggle.com/competitions/UBC-OCEAN
## Installation
To set up the environment and install the necessary dependencies, run the following command:
```pip install torch albumentations opencv-python pillow pandas numpy matplotlib scikit-learn timm```
## Dataset
The dataset used in this competition consists of high-resolution images of ovarian cells. The data is split into training and test sets, with labels indicating the type of ovarian cancer. The training set contains both TMA and non-TMA images, which are handled differently in the preprocessing step.
- Training images: /kaggle/input/UBC-OCEAN/train_images/
- Training thumbnails: /kaggle/input/UBC-OCEAN/train_thumbnails/
- Test images: /kaggle/input/UBC-OCEAN/test_images/
- Test thumbnails: /kaggle/input/UBC-OCEAN/test_thumbnails/
- Training CSV: /kaggle/input/UBC-OCEAN/train.csv
- Test CSV: /kaggle/input/UBC-OCEAN/test.csv
## Preprocessing
Images are preprocessed using the Albumentations library. Different transformations are applied to TMA and non-TMA images:
- TMA images are resized to 4000x4000
- Non-TMA images are resized to 3000x3000 and then padded to 4000x4000
Common augmentations applied include horizontal flip, random rotation, and color jitter.
## Model Architecture
The model architecture consists of:
1. ResNet-101: A pre-trained ResNet-101 model from the timm library, used as the base model
2. TMA-specific layer: Convolutional layers to process TMA images
3. Non-TMA-specific layer: Convolutional layers to process non-TMA images.
The final layer consists of a linear layer with a softmax activation to classify the images into one of six types of ovarian cancer.
## Training
The training script trains the model on the processed dataset using the Adam optimizer and a step learning rate scheduler. The loss function used is CrossEntropyLoss. The training loop includes both training and validation steps.
## Inference
The inference script processes the test set images and predicts the type of ovarian cancer for each image. The predictions are saved in a CSV file for submission.
## Results
The model is evaluated using the balanced accuracy score. The final model achieved a balanced accuracy score of 0.2180 on the validation set.
