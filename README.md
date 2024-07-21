# Breast-Cancer-Histopathology-Image-Classification

## Overview

This project implements an advanced ensemble model for classifying breast cancer histopathology images. It combines deep learning and machine learning techniques to achieve high accuracy in identifying different types of breast cancer tissues.

## Features

- Utilizes both standard histopathology images and Whole Slide Images (WSI)
- Implements an ensemble of deep learning models (EfficientNetB0, ResNet50V2, DenseNet121)
- Incorporates machine learning classifiers (Random Forest, Gradient Boosting, SVM)
- Handles data augmentation for improved model generalization
- Processes and analyzes Whole Slide Images (WSI)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn
- OpenSlide
- Pillow

## Installation

1. Clone this repository
2. Install the required packages
3. Download the BACH (BreAst Cancer Histology) dataset and place it in the appropriate directories as specified in the code.
datasetlink:https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images

## Usage

1. Ensure your data is organized
2. Update the data paths in the script to match your directory structure.

3. Run the main script
   4. The script will train the models, evaluate performance, and generate predictions for test images.

## Algorithm Details

This project employs a sophisticated ensemble approach combining deep learning and traditional machine learning:

1. **Deep Learning Models**:
- EfficientNetB0
- ResNet50V2
- DenseNet121

These pre-trained models are fine-tuned on the breast cancer histopathology dataset.

2. **Machine Learning Classifiers**:
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

These classifiers are trained on features extracted from the deep learning models.

3. **Ensemble Technique**:
The final prediction is an average of predictions from both deep learning and machine learning models.

4. **Data Processing**:
- Standard image preprocessing and augmentation
- Extraction and processing of patches from Whole Slide Images

5. **Training Process**:
- Transfer learning with pre-trained weights
- Fine-tuning with breast cancer histopathology images
- Feature extraction for machine learning models

## Impact and Contribution

This project has the potential to significantly impact breast cancer diagnosis and research:

1. **Improved Accuracy**: By combining multiple models and techniques, it aims to achieve higher accuracy in classifying breast cancer tissues.

2. **Enhanced Efficiency**: Automating the classification process can save time for pathologists and allow them to focus on complex cases.

3. **Standardization**: This approach can help standardize breast cancer tissue classification across different healthcare institutions.

4. **Research Tool**: The model can be used as a research tool to analyze large datasets of histopathology images, potentially uncovering new patterns or insights.

5. **Accessibility**: By open-sourcing this project, it contributes to the democratization of advanced medical image analysis techniques.

6. **Interdisciplinary Collaboration**: This project encourages collaboration between computer scientists, data scientists, and medical professionals.

## Future Work

- Incorporate more diverse datasets for improved generalization
- Experiment with additional deep learning architectures
- Implement explainable AI techniques for better interpretability
- Develop a user-friendly interface for clinical use

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- BACH (BreAst Cancer Histology) Dataset
- TensorFlow and Keras teams
- scikit-learn contributors
- OpenSlide developers
   
