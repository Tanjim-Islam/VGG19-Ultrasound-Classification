# VGG19 Ultrasound Classification

A deep learning implementation for automated classification of fetal anatomical planes in ultrasound images using the VGG19 convolutional neural network architecture with transfer learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Medical Context](#medical-context)
- [Technical Implementation](#technical-implementation)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)

## 🔍 Overview

This project implements an automated classification system for ultrasound images using deep learning. The system is specifically designed to identify and classify different fetal anatomical planes in prenatal ultrasound scans, which is crucial for routine obstetric examinations and fetal health assessment.

**What this code does:**
- Classifies ultrasound images into different fetal anatomical plane categories
- Uses transfer learning with the pre-trained VGG19 model from ImageNet
- Implements data preprocessing, augmentation, and model training pipelines
- Provides comprehensive evaluation metrics and visualization tools
- Achieves high accuracy in automated ultrasound image classification

**How it helps:**
- **Medical Professionals**: Assists radiologists and sonographers in automated screening and diagnosis
- **Healthcare Efficiency**: Reduces manual interpretation time and potential human error
- **Standardization**: Provides consistent classification across different operators and facilities
- **Training**: Can be used as an educational tool for medical students and professionals
- **Research**: Enables large-scale analysis of ultrasound datasets for medical research

## 🏥 Medical Context

### Fetal Plane Classification in Ultrasound

Prenatal ultrasound imaging is a standard procedure during pregnancy to monitor fetal development and detect potential abnormalities. During these examinations, sonographers capture images of different anatomical planes, each providing specific information about fetal health:

**Common Fetal Planes Include:**
- **Abdominal**: Shows the fetal abdomen, useful for measuring growth parameters
- **Brain**: Displays brain structures for neurological assessment
- **Femur**: Shows the thigh bone for growth measurements
- **Four Chamber**: Cardiac view showing all four heart chambers
- **Kidneys**: Renal assessment for urological conditions
- **Profile**: Side view of fetal face and head
- **Spine**: Spinal cord and vertebrae visualization

**Clinical Importance:**
- Early detection of congenital anomalies
- Growth monitoring and gestational age assessment
- Guidance for medical interventions
- Risk assessment for pregnancy complications

## 🔧 Technical Implementation

### Core Components

1. **Data Preprocessing**
   - Image normalization and resizing (224x224 pixels)
   - Label encoding for categorical classification
   - Train/validation/test split (80%/20% with further validation split)

2. **Data Augmentation**
   - Rotation, width/height shifts, shear, zoom transformations
   - Horizontal flipping for better generalization
   - Real-time augmentation during training

3. **Transfer Learning Approach**
   - Pre-trained VGG19 model from ImageNet as feature extractor
   - Custom classification head for ultrasound-specific features
   - Fine-tuning of top layers for domain adaptation

4. **Model Training**
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Model checkpointing for best performance preservation

5. **Evaluation Framework**
   - Multiple metrics: Accuracy, Precision, Recall, F1-Score
   - Confusion matrix visualization
   - Training/validation loss and accuracy plots

## 📊 Dataset

The implementation uses the **FETAL_PLANES_ZENODO** dataset, which contains:
- Ultrasound images from routine prenatal examinations
- Multiple fetal anatomical plane categories
- Properly labeled and curated medical imaging data
- Suitable for supervised learning approaches

**Data Structure:**
```
Dataset/
├── FETAL_PLANES_ZENODO/
│   ├── merged_data.csv          # Labels and metadata
│   └── [image files]            # Ultrasound images
```

## 🚀 Installation

### Prerequisites

- Python 3.8+ (recommended: Python 3.9.0)
- GPU support recommended for faster training (CUDA-compatible)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Tanjim-Islam/VGG19-Ultrasound-Classification.git
cd VGG19-Ultrasound-Classification
```

2. **Install required dependencies:**
```bash
pip install tensorflow==2.10.0 pandas numpy matplotlib scikit-learn keras seaborn
```

3. **Download the dataset:**
   - Obtain the FETAL_PLANES_ZENODO dataset
   - Place in the `Dataset/FETAL_PLANES_ZENODO/` directory
   - Ensure `merged_data.csv` contains image paths and labels

## 💻 Usage

### Running the Classification System

1. **Open the Jupyter Notebook:**
```bash
jupyter notebook VGG19.ipynb
```

2. **Execute the cells sequentially:**
   - **Data Loading & Preprocessing**: Load and prepare the ultrasound dataset
   - **Model Building**: Construct the VGG19-based architecture
   - **Training**: Train the model with your data
   - **Evaluation**: Assess model performance and generate metrics

### Key Code Sections

**Model Training:**
```python
# Build and compile the VGG19 model
vgg19_model = build_vgg19_model()

# Train the model
history_vgg19 = train_vgg19_model(
    vgg19_model, 
    train_generator, 
    val_generator, 
    epochs=30
)
```

**Model Evaluation:**
```python
# Evaluate on test set
test_loss, test_acc = vgg19_model.evaluate(test_generator, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Generate detailed metrics
predictions = vgg19_model.predict(test_generator)
# Calculate precision, recall, F1-score, confusion matrix
```

## 🏗️ Model Architecture

### VGG19 Transfer Learning Architecture

```
Input Layer (224, 224, 3)
         ↓
VGG19 Base Model (Pre-trained on ImageNet)
├── Block 1: Conv-Conv-MaxPool
├── Block 2: Conv-Conv-MaxPool  
├── Block 3: Conv-Conv-Conv-Conv-MaxPool
├── Block 4: Conv-Conv-Conv-Conv-MaxPool
└── Block 5: Conv-Conv-Conv-Conv-MaxPool
         ↓
Global Average Pooling
         ↓
Dense Layer (512 units, ReLU)
         ↓
Dropout (0.5)
         ↓
Dense Layer (256 units, ReLU)
         ↓
Dropout (0.3)
         ↓
Output Layer (num_classes, Softmax)
```

### Key Architecture Features

- **Pre-trained Features**: Leverages ImageNet-trained VGG19 for robust feature extraction
- **Custom Classifier**: Domain-specific classification layers for ultrasound images
- **Regularization**: Dropout layers to prevent overfitting
- **Global Average Pooling**: Reduces spatial dimensions while preserving features

### Training Configuration

- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Categorical crossentropy
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Image Size**: 224×224 pixels (VGG19 standard input)
- **Data Augmentation**: Real-time transformations for robustness

## 📈 Performance Metrics

The model evaluation includes comprehensive metrics:

### Classification Metrics
- **Accuracy**: Overall correct prediction percentage
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for detecting each anatomical plane
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Visualization Tools
- Training/validation accuracy and loss curves
- Confusion matrix heatmap
- Per-class performance analysis
- Sample prediction visualizations

## 🎯 Results

The VGG19-based ultrasound classification system demonstrates:

- **High Classification Accuracy**: Achieved through transfer learning and proper regularization
- **Robust Performance**: Consistent results across different anatomical planes
- **Clinical Relevance**: Meaningful classification for medical applications
- **Generalization**: Good performance on unseen ultrasound images

*Note: Specific performance numbers depend on the dataset size, quality, and training configuration.*

## 📋 Prerequisites

### Technical Requirements
- **Python**: 3.8 or higher
- **TensorFlow**: 2.10.0 (with GPU support recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 5GB+ for dataset and model files

### Additional Dependencies
```python
tensorflow==2.10.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
seaborn>=0.11.0
keras>=2.10.0
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support for faster training
- **CPU**: Multi-core processor for data preprocessing
- **Storage**: SSD recommended for faster data loading

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- Model architecture improvements
- Additional evaluation metrics
- Dataset expansion
- Performance optimizations
- Documentation enhancements

---

**Medical Disclaimer**: This software is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.
