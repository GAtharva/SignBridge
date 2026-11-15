
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------------
# 1. Configuration
# -----------------------------------

DATASET_PATH = Path("datasets/processed/RealSign/Alphabet")
MODEL_PATH = Path("models/isl_alphabet_cnn.keras")
HISTORY_PATH = Path("models/training_history.csv")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

print("=" * 55)
print("       SIGNBRIDGE - ISL ALPHABET CNN")
print("=" * 55)

# -----------------------------------
# 2. Load datasets
# -----------------------------------

print("\nLoading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH / "Training",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH / "Validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH / "Testing",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nAlphabet classes:", class_names)
print("Number of classes:", num_classes)

# Verify consistent class mappings
if (
    class_names != val_ds.class_names
    or class_names != test_ds.class_names
):
    raise ValueError("Class mappings do not match!")

if num_classes != 26:
    raise ValueError(f"Expected 26 classes, found {num_classes}")

# Improve data pipeline performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------------
# 3. Build CNN architecture
# -----------------------------------

print("\nBuilding CNN model...")

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# 4. Training callbacks
# -----------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),

    keras.callbacks.CSVLogger(
        str(HISTORY_PATH)
    )
]

# -----------------------------------
# 5. Train model
# -----------------------------------

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------------
# 6. Evaluate on test dataset
# -----------------------------------

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n" + "=" * 55)
print("          FINAL MODEL RESULTS")
print("=" * 55)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nBest model saved at:", MODEL_PATH)
print("Training history saved at:", HISTORY_PATH)

print("=" * 55)
