# CNN for Image Classification on FashionMNIST Dataset

This Jupyter notebook implements Convolutional Neural Networks (CNNs) using PyTorch for image classification on the FashionMNIST dataset. The notebook demonstrates training of multiple CNN architectures with variations like dropout and batch normalization, and evaluates the model's performance on both training and testing datasets.
We used some of the code from the Udacity course repository: "Intro to deep learning with Pytorch"

https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/cifar-cnn/cifar10_cnn_solution.ipynb


## Table of Contents

1. [Importing Libraries](#1-importing-libraries)
2. [Check CUDA Availability](#2-check-cuda-availability)
3. [Data Loading and Preprocessing](#3-data-loading-and-preprocessing)
4. [Data Visualization](#4-data-visualization)
5. [Defining CNN Architectures](#5-defining-cnn-architectures)
6. [Training Various CNN Models](#6-training-various-cnn-models)
7. [Accuracy Evaluation of Each Model](#7-accuracy-evaluation-of-each-model)
8. [Plotting Model Accuracy](#8-plotting-model-accuracy)
9. [Visualizing Model Predictions](#9-visualizing-model-predictions)

## 1. Importing Libraries

The necessary libraries for building and training the CNN are imported. This includes:
- `torch` for neural network operations.
- `numpy` for numerical computations.
- `matplotlib` for visualizations.
- `torchvision` for accessing datasets and transformations.

## 2. Check CUDA Availability

We check if CUDA (GPU acceleration) is available for faster training. If not, the model will be trained on the CPU.

## 3. Data Loading and Preprocessing

- The `FashionMNIST` dataset is loaded and preprocessed.
- We apply necessary transformations like converting images to tensors and normalizing.
- A subset of training data is randomly selected for training using `SubsetRandomSampler`.
  
## 4. Data Visualization

A batch of training images is displayed along with their corresponding class labels, allowing us to visually inspect the dataset.

## 5. Defining CNN Architectures

Three variations of CNN models are defined based on the base architecture `BaseNet`:
- **BaseNet**: Basic CNN architecture with two convolutional layers, pooling, and fully connected layers.
- **Net_dropout**: Adds dropout before the fully connected layers to reduce overfitting.
- **Net_BatchNorm**: Introduces batch normalization after each convolutional layer to improve training speed and stability.

## 6. Training Various CNN Models

Each model is trained for 15 epochs using the following steps:
- Define the model and optimizer (Adam optimizer is used).
- The model is trained on the training dataset, and the loss is tracked for each epoch.
- The model weights are saved after each epoch for evaluation.

To train a model:
1. Choose the model from the defined list: `model_origin`, `model_dropout`, `model_batchnorm`, or `model_weightdecay`.
2. Set the optimizer (Adam with weight decay for `model_weightdecay`).
3. Specify the number of epochs (15 epochs are used).
4. Use the command `torch.save(model.state_dict(), f'{model_name}_{epoch}.pt')` to save weights after each epoch.

## 7. Accuracy Evaluation of Each Model

After training the models, the notebook evaluates each model's performance by:
- Loading the saved model weights from each epoch.
- Evaluating both training and test datasets.
- Calculating class-wise and overall accuracy.

To evaluate the model:
1. Load the saved weights for the desired epoch using `model.load_state_dict(torch.load('model_name_epoch.pt'))`.
2. Switch the model to evaluation mode using `model.eval()`.
3. Compute the accuracy for each dataset (training and testing).

## 8. Plotting Model Accuracy

The notebook visualizes the accuracy of each model across all epochs. Training and testing accuracy are plotted to show the model's performance over time.

## 9. Visualizing Model Predictions

The notebook also visualizes model predictions on test images, showing the true and predicted labels for each image, and provides insight into how well the model generalizes.

## How to Train a Model

1. **Choose a Model**:
   You can choose one of the following models for training:
   - `Net()` (Basic CNN)
   - `Net_dropout()` (CNN with dropout)
   - `Net_BatchNorm()` (CNN with batch normalization)
   
2. **Train the Model**:
   - Run the training loop, which will automatically save the model weights after each epoch.
   - You can specify the number of epochs (15 in the notebook).
   
3. **Monitor Training Progress**:
   - During training, the loss will be displayed for each epoch.
   - The model weights will be saved in the format: `model_name_epoch.pt`.

## How to Test the Model with Saved Weights

1. **Load the Saved Weights**:
   - To load the saved weights of a model, use `model.load_state_dict(torch.load('model_name_epoch.pt'))`, where `model_name_epoch.pt` is the specific saved model weight file.
   
2. **Evaluate the Model**:
   - Once the model weights are loaded, set the model to evaluation mode using `model.eval()`.
   - Use the test dataset (`test_loader`) to compute the model's accuracy on the test set.
   - The notebook automatically computes the accuracy and prints the results.

## Requirements

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

Install the required libraries using:

```bash
pip install torch torchvision numpy matplotlib
