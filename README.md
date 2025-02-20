# Medical Image Classification Project

This project aims to develop and compare several neural network models for the classification of medical images, specifically for the detection of pneumonia from chest X-rays.

## Project Structure

The project is organized into several Jupyter notebooks, each exploring and implementing different neural network models. Here is an overview of the included notebooks:

### 1. `CNNs.ipynb`

This notebook implements several Convolutional Neural Network (CNN) models for image classification. The main steps include:
- Loading and preprocessing data.
- Building basic CNN models.
- Training the models and evaluating performance.
- Visualizing model architecture and results.

### 2. `VGG16_CNN_model.ipynb`

This notebook focuses on using the pre-trained VGG16 model for image classification. The main steps include:
- Loading data and data augmentation.
- Using the VGG16 model with custom layers for classification.
- Training the model with callbacks to save weights.
- Evaluating model performance and visualizing results.

### 3. `ViT_model.ipynb`

This notebook explores the use of the Vision Transformer (ViT) model for image classification. The main steps include:
- Preparing data and defining transformations.
- Loading and configuring the pre-trained ViT model.
- Training the model and saving checkpoints.
- Evaluating model performance and visualizing results.

### 4. `GCN_model.ipynb`

This notebook implements a Graph Convolutional Network (GCN) model for image classification. The main steps include:
- Preparing data and constructing the graph.
- Defining and training the GCN model.
- Evaluating model performance.

### 5. `image_generation_model.ipynb`

This notebook explores the use of generative models for generating medical images. The main steps include:
- Preparing data for image generation.
- Building and training generative models.
- Evaluating the quality of generated images.

## Problem to Solve

The main problem addressed in this project is the automatic detection of pneumonia from chest X-rays. Pneumonia is a lung infection that can be diagnosed using X-rays. The goal is to develop machine learning models capable of automatically classifying X-rays as normal or showing signs of pneumonia.

## Installation

To run the notebooks, you will need the following libraries:
- TensorFlow
- Keras
- PyTorch
- Transformers
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Pandas

You can install these libraries using pip:

```sh
pip install tensorflow keras torch transformers scikit-learn matplotlib seaborn numpy pandas
