# Human vs. Horse Image Classifier

## Overview

This repository contains code for a Human vs. Horse Image Classifier using a Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras and aims to classify images as either humans or horses. The dataset is divided into training and validation sets, and data augmentation is applied to improve model performance.

## Dataset

The dataset used for training and validation consists of images of humans and horses. The images are organized into the following directories:

- `training/horses`: Contains training images of horses.
- `training/humans`: Contains training images of humans.
- `validation/horses`: Contains validation images of horses.
- `validation/humans`: Contains validation images of humans.

## Data Preparation

The dataset is divided into training and validation sets using the `create_generator` function. The images are resized to 300x300 pixels and normalized to [0, 1]. Data augmentation is applied to the training set to enhance the model's ability to generalize.

## Model Architecture

The Human vs. Horse Image Classifier model consists of a series of Convolutional and MaxPooling layers followed by Dense layers. The final layer uses a sigmoid activation function to predict binary output (human or horse).

## Transfer Learning

An InceptionV3 pre-trained model with weights from ImageNet is used as the base model. The top layer of the base model is replaced with custom Dense layers to suit the binary classification task of humans vs. horses.

## Training

The model is trained using the training data and validated using the validation data. The training process involves minimizing the binary cross-entropy loss using the Adam optimizer with a learning rate of 1e-4. The training will stop early if the training accuracy reaches 99%.

## Training Performance

The training and validation accuracy and loss are plotted to visualize the model's performance during training.

## Usage

To use the Human vs. Horse Image Classifier, you can load the model and call its `predict` method on new images to classify them as either humans or horses.

Feel free to experiment with different hyperparameters, add more layers, or adjust the pre-trained model for better classification performance.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy classifying! 🐴👨‍🦰
