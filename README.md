# Handwritten Digit Recognition using Neural Networks

## Project Overview
This project recognizes handwritten digits from 0 to 9 using a Multilayer Perceptron (MLP) neural network and the MNIST dataset.

## Objectives
- Load and preprocess handwritten digit images.
- Train a neural network classifier.
- Evaluate predictions using accuracy and classification metrics.
- Visualize model performance.
- Save the trained model for reuse.

## Dataset
MNIST contains 70,000 grayscale images of handwritten digits. Each image is 28 × 28 pixels.

This implementation uses 20,000 training samples and 5,000 testing samples.

## Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Methodology
1. Download the MNIST dataset.
2. Normalize pixel values to the range 0–1.
3. Split the data into training and testing sets.
4. Train an MLP neural network with hidden layers of 128 and 64 neurons.
5. Evaluate the model on test data.
6. Save the trained model and result plots.

## Project Structure

```text
MNIST-Digit-Recognition/
├── main.py
├── mnist_recognition.py
├── requirements.txt
├── README.md
├── mnist_model.joblib
└── results/
    ├── metrics.txt
    ├── mnist_loss.png
    ├── mnist_confusion_matrix.png
    └── mnist_predictions.png
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

The first run requires an internet connection to download MNIST.

## Results
The project generates a measured test accuracy, a classification report, a training-loss graph, a confusion matrix, and sample predictions.

Check `results/metrics.txt` for the actual accuracy. Do not assume a result without checking the program output.

## Future Improvements
- Compare MLP with a Convolutional Neural Network (CNN).
- Tune model parameters.
- Build a user interface for handwritten digit predictions.

## Conclusion
This project demonstrates dataset preprocessing, neural-network training, model evaluation, visualization, and saving a trained machine-learning model.

## GitHub Repository
https://github.com/tusharsaini-del/MNIST-Digit-Recognition
