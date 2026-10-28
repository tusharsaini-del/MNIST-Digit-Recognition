
import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Create output folder
os.makedirs("results", exist_ok=True)
print("Downloading the MNIST dataset...")

# 2. Load actual MNIST data (70,000 images, each 28x28 pixels)
try:
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto"
    )
except TypeError:
    # Compatibility fallback for older scikit-learn versions
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

print("Dataset loaded:", X.shape)
print("Number of classes:", len(np.unique(y)))

# 3. Use a reproducible subset to reduce training time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=20000,
    test_size=5000,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Build the neural network
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=15,
    random_state=42,
    verbose=True
)

# 5. Train the model
print("\nTraining neural network...")
start_time = time.time()
model.fit(X_train, y_train)
print("Training time:", round(time.time() - start_time, 2), "seconds")

# 6. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 7. Save trained model
joblib.dump(model, "mnist_model.joblib")
print("Model saved: mnist_model.joblib")

# 8. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MNIST Neural Network Training Loss")
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mnist_loss.png", dpi=150)
plt.close()

# 9. Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues",
    values_format="d"
)
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("results/mnist_confusion_matrix.png", dpi=150)
plt.close()

# 10. Save sample predictions
rng = np.random.default_rng(42)
sample_indices = rng.choice(len(X_test), size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for ax, idx in zip(axes.flat, sample_indices):
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax.set_title(
        f"Actual: {y_test[idx]}\nPredicted: {predictions[idx]}"
    )
    ax.axis("off")

plt.suptitle("MNIST Handwritten Digit Predictions")
plt.tight_layout()
plt.savefig("results/mnist_predictions.png", dpi=150)
plt.close()

# 11. Save actual measured accuracy
with open("results/metrics.txt", "w") as file:
    file.write(f"Test accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Check the results folder for plots and metrics.")
