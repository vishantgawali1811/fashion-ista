# ============================================================
# VGG16 - FASHION-MNIST CLASSIFICATION
# ============================================================
# Dataset:
#   Fashion-MNIST from local CSV files
#
# Important memory optimization:
#   Images remain 28x28x3 in RAM.
#   Resizing to 224x224 happens INSIDE the model.
#
# Training:
#   Phase 1 -> Transfer Learning
#   Phase 2 -> Fine Tuning
#
# Outputs:
#   results/VGG16/
#       best_model.keras
#       final_model.keras
#       model_summary.txt
#       classification_report.csv
#       confusion_matrix.csv
#       normalized_confusion_matrix.csv
#       class_wise_metrics.csv
#       predictions.csv
#       metrics.json
#       training_history.csv
#       training_validation_accuracy.png
#       training_validation_loss.png
#       confusion_matrix.png
#       normalized_confusion_matrix.png
#       prediction_confidence.png
# ============================================================


import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# Import common functions
from common import (
    load_fashion_mnist,
    make_augmentation,
    compile_model,
    get_callbacks,
    save_model_summary,
    evaluate_model
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "VGG16"

# VGG16 expects 224x224 input
IMAGE_SIZE = 224

# IMPORTANT:
# Dataset remains 28x28 in memory
DATASET_IMAGE_SIZE = 28

NUM_CLASSES = 10

BATCH_SIZE = 32

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

VALIDATION_SIZE = 5000


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("VGG16 - FASHION-MNIST CLASSIFICATION")
    print("=" * 75)

    print("\nConfiguration:")
    print(f"Model                  : {MODEL_NAME}")
    print(f"Dataset image size     : {DATASET_IMAGE_SIZE}x{DATASET_IMAGE_SIZE}")
    print(f"VGG16 image size       : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Initial epochs         : {INITIAL_EPOCHS}")
    print(f"Fine-tuning epochs     : {FINE_TUNE_EPOCHS}")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\n")
    print("=" * 75)
    print("LOADING DATASET")
    print("=" * 75)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_fashion_mnist(
        image_size=DATASET_IMAGE_SIZE,
        validation_size=VALIDATION_SIZE
    )

    print("\nDataset successfully loaded.")

    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Testing    : {X_test.shape}")

    # ========================================================
    # CREATE DATA AUGMENTATION
    # ========================================================

    augmentation = make_augmentation()

    # ========================================================
    # BUILD VGG16 BASE MODEL
    # ========================================================

    print("\n")
    print("=" * 75)
    print("BUILDING VGG16")
    print("=" * 75)

    # --------------------------------------------------------
    # IMPORTANT:
    # Input remains 28x28x3.
    # This prevents huge 224x224 dataset allocation.
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(28, 28, 3),
        name="input_image"
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    x = augmentation(inputs)

    # --------------------------------------------------------
    # RESIZE INSIDE MODEL
    # --------------------------------------------------------

    x = layers.Resizing(
        IMAGE_SIZE,
        IMAGE_SIZE,
        interpolation="bilinear",
        name="resize_to_224"
    )(x)

    # --------------------------------------------------------
    # CONVERT 0-1 PIXELS BACK TO 0-255
    # --------------------------------------------------------

    x = layers.Rescaling(
        255.0,
        name="rescale_to_255"
    )(x)

    # --------------------------------------------------------
    # VGG16 PREPROCESSING
    # --------------------------------------------------------

    x = preprocess_input(x)

    # ========================================================
    # VGG16 BASE
    # ========================================================

    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Freeze VGG16 during initial training
    base_model.trainable = False

    # Forward pass
    x = base_model(
        x,
        training=False
    )

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = layers.Dropout(
        0.40,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="VGG16_FashionMNIST"
    )

    # ========================================================
    # PRINT MODEL SUMMARY
    # ========================================================

    print("\nVGG16 model created successfully.")

    model.summary()

    # ========================================================
    # SAVE MODEL SUMMARY
    # ========================================================

    save_model_summary(
        model,
        MODEL_NAME
    )

    # ========================================================
    # PHASE 1 - TRANSFER LEARNING
    # ========================================================

    print("\n")
    print("=" * 75)
    print("PHASE 1 - TRANSFER LEARNING")
    print("=" * 75)

    print("\nVGG16 base layers are frozen.")

    compile_model(
        model,
        learning_rate=INITIAL_LEARNING_RATE
    )

    callbacks_phase1 = get_callbacks(
        MODEL_NAME
    )

    print("\nStarting Phase 1 training...\n")

    history_phase1 = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=INITIAL_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks_phase1,

        verbose=1,

        shuffle=True
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        "results",
        MODEL_NAME,
        "best_model.keras"
    )

    if os.path.exists(checkpoint_path):

        print("\nLoading best Phase 1 checkpoint...")

        model = tf.keras.models.load_model(
            checkpoint_path
        )

        print("Best checkpoint loaded.")

    # ========================================================
    # PHASE 2 - FINE TUNING
    # ========================================================

    print("\n")
    print("=" * 75)
    print("PHASE 2 - FINE TUNING")
    print("=" * 75)

    # --------------------------------------------------------
    # Unfreeze VGG16
    # --------------------------------------------------------

    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "vgg16" in layer.name.lower():

                base_model = layer

                break

    if base_model is None:

        print(
            "\nWARNING: VGG16 base model was not found."
        )

    else:

        base_model.trainable = True

        # ----------------------------------------------------
        # Freeze earlier layers
        # ----------------------------------------------------

        # Only last 8 layers are trainable
        for layer in base_model.layers[:-8]:

            layer.trainable = False

        # Keep BatchNormalization layers frozen
        for layer in base_model.layers:

            if isinstance(
                layer,
                layers.BatchNormalization
            ):

                layer.trainable = False

        print(
            "\nVGG16 last 8 layers are now trainable."
        )

    # ========================================================
    # RECOMPILE WITH SMALL LEARNING RATE
    # ========================================================

    compile_model(
        model,
        learning_rate=FINE_TUNE_LEARNING_RATE
    )

    callbacks_phase2 = get_callbacks(
        MODEL_NAME
    )

    print("\nStarting Phase 2 fine-tuning...\n")

    history_phase2 = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=FINE_TUNE_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks_phase2,

        verbose=1,

        shuffle=True
    )

    # ========================================================
    # COMBINE TRAINING HISTORY
    # ========================================================

    print("\nCombining training histories...")

    combined_history = {}

    all_keys = set(
        list(history_phase1.history.keys())
        +
        list(history_phase2.history.keys())
    )

    for key in all_keys:

        combined_history[key] = (
            history_phase1.history.get(key, [])
            +
            history_phase2.history.get(key, [])
        )

    # ========================================================
    # SAVE TRAINING HISTORY AND GRAPHS
    # ========================================================

    results_dir = os.path.join(
        "results",
        MODEL_NAME
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    # Save CSV
    history_csv = os.path.join(
        results_dir,
        "training_history.csv"
    )

    import pandas as pd

    history_df = pd.DataFrame(
        combined_history
    )

    history_df.to_csv(
        history_csv,
        index=False
    )

    print(
        f"\nTraining history saved to:\n{history_csv}"
    )

    # ========================================================
    # TRAINING ACCURACY GRAPH
    # ========================================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    if "accuracy" in combined_history:

        plt.plot(
            combined_history["accuracy"],
            label="Training Accuracy"
        )

    if "val_accuracy" in combined_history:

        plt.plot(
            combined_history["val_accuracy"],
            label="Validation Accuracy"
        )

    plt.title(
        "VGG16 Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "training_validation_accuracy.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # TRAINING LOSS GRAPH
    # ========================================================

    plt.figure(figsize=(10, 6))

    if "loss" in combined_history:

        plt.plot(
            combined_history["loss"],
            label="Training Loss"
        )

    if "val_loss" in combined_history:

        plt.plot(
            combined_history["val_loss"],
            label="Validation Loss"
        )

    plt.title(
        "VGG16 Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "training_validation_loss.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = os.path.join(
        results_dir,
        "final_model.keras"
    )

    model.save(
        final_model_path
    )

    print(
        f"\nFinal model saved to:\n{final_model_path}"
    )

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL MODEL EVALUATION")
    print("=" * 75)

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=MODEL_NAME
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 75)
    print("VGG16 TRAINING COMPLETE")
    print("=" * 75)

    print("\nFinal Results:")

    if "accuracy" in metrics:

        print(
            f"Accuracy           : "
            f"{metrics['accuracy']:.4f}"
        )

    if "precision" in metrics:

        print(
            f"Precision          : "
            f"{metrics['precision']:.4f}"
        )

    if "recall" in metrics:

        print(
            f"Recall             : "
            f"{metrics['recall']:.4f}"
        )

    if "f1_score" in metrics:

        print(
            f"F1-Score           : "
            f"{metrics['f1_score']:.4f}"
        )

    print("\nResults saved in:")

    print(
        os.path.abspath(
            results_dir
        )
    )

    print("\n")
    print("=" * 75)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()