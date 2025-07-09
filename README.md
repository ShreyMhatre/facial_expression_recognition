# Facial Expression Recognition - Emotion Classification using CNN

This repository contains my implementation of a Convolutional Neural Network (CNN) for facial expression recognition. The goal is to classify facial emotions (e.g., happy, sad, angry) from grayscale face images using deep learning techniques.

---

## Learning Reference

This project was based on various deep learning resources including TensorFlow documentation and CNN-based emotion classification tutorials.  
I used these references to understand how to build and train CNNs from scratch for image classification tasks.  

---

## Objective

Build a multi-class classification model using CNN to recognize human emotions from facial expressions in images.

---

## Tools Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- scikit-learn

---

## Data Preprocessing

- **Dataset**: FER2013 (or similar publicly available facial emotion dataset)
- **Steps Involved**:
  - Loading grayscale facial images and associated emotion labels
  - Normalizing image pixel values
  - One-hot encoding emotion labels
  - Splitting data into training, validation, and test sets
  - Reshaping image data into (48x48x1) format for CNN input

---

## CNN Architecture

- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for spatial dimensionality reduction
- **Batch Normalization**: Applied to stabilize and accelerate training
- **Dropout**: Added for regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Softmax activation for multi-class emotion prediction

### Training Strategy:

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Callbacks: 
  - ReduceLROnPlateau for dynamic learning rate adjustment
  - EarlyStopping to prevent overfitting
- Evaluation Metric: Accuracy

---

## Evaluation Metric

- **Accuracy** — Multi-class classification accuracy on validation and test datasets

---

## Data Usage

Due to dataset licensing and size, the dataset is not included in this repository.  
You can download the FER2013 dataset from [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) and place it in a local `data/` folder for usage.

---

## Credits

- FER2013 dataset by Kaggle community
- TensorFlow and Keras documentation
- StackOverflow discussions for debugging training performance
- Inspiration from public deep learning facial emotion classification notebooks

---

## Status & Learning Outcome

 **In Progress**

- Data preprocessing and augmentation
- CNN model definition and tuning
- Training with callbacks and regularization
- Evaluation and prediction on test images
- Visualization of confusion matrix and class distribution

This project helped me understand:
- How to construct CNNs for emotion classification
- Best practices for training deep models on image data
- Using learning rate schedulers and callbacks
- Model evaluation using real-world facial data

---

*This notebook is a part of my deep learning self-study path and reflects my efforts to implement practical models using TensorFlow and Keras for computer vision tasks.*

