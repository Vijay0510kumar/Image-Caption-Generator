# Image-Caption-Generator
Image Caption generator using CNN and LSTM

## Project Overview

This project is an Image Caption Generator that combines Computer Vision (CNN) and Natural Language Processing (LSTM) to generate meaningful captions for images. We used a pretrained VGG16 model to extract image features and an LSTM-based language model to generate captions.

## Dataset

We used the Flickr8K dataset, which consists of 8,000 images with five captions per image. The dataset is sourced from Kaggle.

## Tech Stack & Libraries

Python (Primary Language)

Deep Learning Frameworks: TensorFlow, Keras

Pretrained Model: VGG16 (Feature Extraction)

Natural Language Processing: LSTM, Tokenization

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

## Project Architecture

### Data Preprocessing

Load and clean image captions

Tokenize and create vocabulary

Convert text to sequences

### Feature Extraction (CNN - VGG16)

Use VGG16 (pretrained on ImageNet) to extract deep image features

Remove the fully connected layer and keep convolutional layers

### Caption Generation (LSTM)

Build an LSTM-based language model

Train it to generate captions from extracted features

Use Beam Search Decoding for better caption generation

### Model Training & Optimization

Use Categorical Cross-Entropy Loss

Optimizer: Adam

Implement Early Stopping to avoid overfitting

## Performance Metrics

BLEU Score: Used to evaluate the quality of generated captions



