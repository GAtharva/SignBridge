
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Paths
DATASET_PATH = Path("datasets/processed/RealSign/Alphabet")
MODEL_PATH = Path("models/isl_alphabet_cnn.keras")
REPORT_PATH = Path("reports")

REPORT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

print("=" * 55)
print("       SIGNBRIDGE - MODEL EVALUATION")
print("=" * 55)

# Load test dataset
test_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH / "Testing",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names

# Load trained model
print("\nLoading trained CNN model...")
model = keras.models.load_model(MODEL_PATH)

# Actual labels
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Predictions
print("\nGenerating predictions...")
predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nCLASSIFICATION REPORT")
print("=" * 55)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# Save classification report
report_file = REPORT_PATH / "classification_report.txt"
report_file.write_text(report, encoding="utf-8")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion matrix shape:", cm.shape)

# Plot confusion matrix
plt.figure(figsize=(16, 14))

plt.imshow(cm, interpolation="nearest")
plt.title("SignBridge - ISL Alphabet Confusion Matrix")
plt.colorbar()

positions = np.arange(len(class_names))

plt.xticks(positions, class_names)
plt.yticks(positions, class_names)

plt.xlabel("Predicted Alphabet")
plt.ylabel("Actual Alphabet")

plt.tight_layout()

matrix_path = REPORT_PATH / "confusion_matrix.png"
plt.savefig(matrix_path, dpi=200)
plt.close()

# Save confusion matrix values
np.savetxt(
    REPORT_PATH / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

print("\n" + "=" * 55)
print("EVALUATION COMPLETED")
print("=" * 55)

print("Classification report:", report_file)
print("Confusion matrix:", matrix_path)
print("Confusion matrix CSV:", REPORT_PATH / "confusion_matrix.csv")

print("=" * 55)
