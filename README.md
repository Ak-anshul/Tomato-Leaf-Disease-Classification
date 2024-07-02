# Tomato-Leaf-Disease-Classification

This repository contains a machine learning project that classifies tomato leaf diseases using images of the leaves. The model is built using TensorFlow and Keras.

### Overview
This project aims to identify and classify different diseases in tomato leaves through image classification. The model uses convolutional neural networks (CNN) to achieve high accuracy in detecting various diseases.

### Dataset

The dataset used in this project consists of images of tomato leaves, each labeled with the corresponding disease. The images are 256x256 pixels and in color.

### Preprocessing

- Images are normalized by dividing pixel values by 256.
- Data is split into training, validation, and test sets with an 80-10-10 ratio.

### Data Augmentation

To improve the model's robustness, data augmentation is applied to the training images, including:

- Rotation
- Zoom
- Horizontal and vertical flip

### Model Architecture

- **Convolutional Layers:**
  - 1st Layer: 32 filters, 3x3 kernel, ReLU activation
  - 2nd Layer: 64 filters, 3x3 kernel, ReLU activation
  - 3rd Layer: 64 filters, 3x3 kernel, ReLU activation
  - 4th Layer: 64 filters, 3x3 kernel, ReLU activation
  - 5th Layer: 64 filters, 3x3 kernel, ReLU activation

- **Pooling Layers:**
  - MaxPooling2D after each convolutional layer with a 2x2 pool size

- **Fully Connected Layers:**
  - Flatten layer to convert 2D data to 1D
  - Dense layer with 64 neurons, ReLU activation
  - Output layer with 10 neurons, softmax activation for classification


### Training

The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss. It is trained for 20 epochs with a batch size of 64.

### Results

The training and validation accuracy and loss are plotted to visualize the model's performance.

### Requirements

- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn



Feel free to contribute and open issues if you encounter any problems!
