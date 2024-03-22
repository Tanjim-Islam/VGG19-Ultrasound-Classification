# VGG19 Ultrasound Classification

Implementation of the VGG19 model for ultrasound image classification using TensorFlow and Keras.

## Prerequisites

- Python 3.9.0
- TensorFlow 2.10.0

## Installation

Clone this repository and navigate into the project directory:

```
git clone https://github.com/Tanjim-Islam/VGG19-Ultrasound-Classification.git
cd VGG19-Ultrasound-Classification
```

Install the required Python dependencies:

```
pip install tensorflow==2.10.0 pandas numpy matplotlib scikit-learn keras
```

## Model Training

To train the model, adjust the hyperparameters as needed and run the training function. Training and validation accuracy will be printed, along with the loss for each epoch.

```
vgg19_model = build_vgg19_model()
history_vgg19 = train_vgg19_model(vgg19_model, train_generator, val_generator, epochs=30)
```

## Evaluation

After training, the model's performance can be evaluated on the test set:

```
# Evaluate on test set
test_loss, test_acc = vgg19_model.evaluate(test_generator, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")
```
