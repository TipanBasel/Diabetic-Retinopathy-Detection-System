import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Dataset Paths 
TRAIN_DIR = "datasets/training"
VALID_DIR = "datasets/testing"

# Model Save Path
MODEL_NAME = "densenet_optimized.h5"

IMG_SIZE = 380  # DenseNet121 default input size
BATCH_SIZE = 32
EPOCHS = 50  
LEARNING_RATE = 1e-4  

os.makedirs("models", exist_ok=True)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    channel_shift_range=0.2, 
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

def create_densenet_model():
    """DenseNet121 with fine-tuning and improved regularization."""
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(train_generator.num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Training with Callbacks
def train_densenet():
    if os.path.exists(f"models/{MODEL_NAME}"):
        print(f"Loading existing DenseNet model from {MODEL_NAME}")
        model = load_model(f"models/{MODEL_NAME}")
    else:
        print("Training new DenseNet model...")
        model = create_densenet_model()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            class_weight=class_weights_dict,  # Apply class weights
            callbacks=callbacks
        )

        # Save the trained model
        model.save(f"models/{MODEL_NAME}")
        print(f"Model saved as models/{MODEL_NAME}")

        plot_training_curves(history)

    return model

# Generate Classification Report
def generate_classification_report(model):
    val_generator.reset()
    y_true = val_generator.classes
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_labels = list(val_generator.class_indices.keys())

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot Training Curves
def plot_training_curves(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_vs_validation_curves.png")
    plt.show()

if __name__ == "__main__":
    trained_model = train_densenet()
    generate_classification_report(trained_model)
