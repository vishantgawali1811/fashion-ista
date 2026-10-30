# ============================================================
# COMMON FUNCTIONS
# FASHION-MNIST DEEP LEARNING PROJECT
# ============================================================
#
# Dataset:
#   data/fashion-mnist_train.csv
#   data/fashion-mnist_test.csv
#
# Models:
#   CNN
#   VGG16
#   ResNet50
#   MobileNetV2
#   EfficientNetB0
#   InceptionV3
#
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DATASET PATHS
# ============================================================

DATA_DIR = "data"

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "fashion-mnist_train.csv"
)

TEST_FILE = os.path.join(
    DATA_DIR,
    "fashion-mnist_test.csv"
)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# FASHION-MNIST CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

NUM_CLASSES = 10


# ============================================================
# LOAD FASHION-MNIST FROM LOCAL CSV
# ============================================================

def load_fashion_mnist(image_size=28, validation_size=5000):
    """
    Load Fashion-MNIST from local CSV files.

    IMPORTANT:
    Images are kept at 28x28 in memory.
    Resizing for transfer-learning models is performed
    inside the model instead of resizing the entire dataset.
    """

    print("\n" + "=" * 70)
    print("LOADING FASHION-MNIST FROM LOCAL CSV")
    print("=" * 70)

    # ---------------------------------------------------------
    # CHECK FILES
    # ---------------------------------------------------------
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_FILE}"
        )

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_FILE}"
        )

    # ---------------------------------------------------------
    # READ CSV
    # ---------------------------------------------------------
    print("\nReading training CSV...")
    train_df = pd.read_csv(TRAIN_FILE)

    print("Reading test CSV...")
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training CSV shape: {train_df.shape}")
    print(f"Test CSV shape:     {test_df.shape}")

    # ---------------------------------------------------------
    # FIND LABEL COLUMN
    # ---------------------------------------------------------
    possible_label_columns = [
        "label",
        "Label",
        "target",
        "Target",
        "class",
        "Class"
    ]

    label_column = None

    for col in possible_label_columns:
        if col in train_df.columns:
            label_column = col
            break

    # If no known label name, assume first column
    if label_column is None:
        label_column = train_df.columns[0]

    print(f"\nLabel column: {label_column}")

    # ---------------------------------------------------------
    # SEPARATE LABELS AND PIXELS
    # ---------------------------------------------------------
    y_train_full = train_df[label_column].values.astype(np.int64)
    X_train_full = train_df.drop(columns=[label_column]).values

    y_test = test_df[label_column].values.astype(np.int64)
    X_test = test_df.drop(columns=[label_column]).values

    # ---------------------------------------------------------
    # CHECK PIXEL COUNT
    # ---------------------------------------------------------
    expected_pixels = 28 * 28

    if X_train_full.shape[1] != expected_pixels:
        raise ValueError(
            f"Expected {expected_pixels} pixel columns, "
            f"but found {X_train_full.shape[1]}"
        )

    if X_test.shape[1] != expected_pixels:
        raise ValueError(
            f"Expected {expected_pixels} pixel columns, "
            f"but found {X_test.shape[1]}"
        )

    # ---------------------------------------------------------
    # CONVERT TO FLOAT32 AND NORMALIZE
    # ---------------------------------------------------------
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # ---------------------------------------------------------
    # VALIDATION SPLIT
    # ---------------------------------------------------------
    if validation_size >= len(X_train_full):
        raise ValueError(
            "validation_size must be smaller than training dataset"
        )

    X_train = X_train_full[:-validation_size]
    y_train = y_train_full[:-validation_size]

    X_val = X_train_full[-validation_size:]
    y_val = y_train_full[-validation_size:]

    # ---------------------------------------------------------
    # RESHAPE TO 28x28x1
    # ---------------------------------------------------------
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # ---------------------------------------------------------
    # CONVERT GRAYSCALE → RGB
    # ---------------------------------------------------------
    X_train = np.repeat(X_train, 3, axis=-1)
    X_val = np.repeat(X_val, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

    # ---------------------------------------------------------
    # DO NOT RESIZE HERE
    # ---------------------------------------------------------
    # IMPORTANT:
    # We intentionally keep images at 28x28.
    #
    # VGG16 / ResNet50 / MobileNetV2 / EfficientNetB0
    # will resize images inside the neural network.
    #
    # InceptionV3 will resize to 299x299 inside the network.
    # ---------------------------------------------------------

    print("\nDataset loaded successfully!")

    print("\nDataset shapes:")
    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"y_val   : {y_val.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_test  : {y_test.shape}")

    print(f"\nPixel range:")
    print(f"Min: {X_train.min():.4f}")
    print(f"Max: {X_train.max():.4f}")

    print("\nClass distribution:")
    for class_id in range(NUM_CLASSES):
        count = np.sum(y_train == class_id)
        print(
            f"{class_id}: "
            f"{CLASS_NAMES[class_id]:15s} -> {count}"
        )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

# ============================================================
# DATA AUGMENTATION
# ============================================================

def make_augmentation():

    augmentation = tf.keras.Sequential(

        [

            tf.keras.layers.RandomRotation(
                0.05
            ),

            tf.keras.layers.RandomTranslation(
                0.05,
                0.05
            ),

            tf.keras.layers.RandomZoom(
                0.08
            )

        ],

        name="data_augmentation"

    )


    return augmentation


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(

    model,

    learning_rate=0.001

):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=learning_rate

        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(
    model_name
):

    model_dir = os.path.join(

        RESULTS_DIR,

        model_name

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    callbacks = [

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True,

            verbose=1

        ),


        # ----------------------------------------------------
        # REDUCE LEARNING RATE
        # ----------------------------------------------------

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.3,

            patience=2,

            min_lr=1e-7,

            verbose=1

        ),


        # ----------------------------------------------------
        # MODEL CHECKPOINT
        # ----------------------------------------------------

        tf.keras.callbacks.ModelCheckpoint(

            filepath=os.path.join(

                model_dir,

                "best_model.keras"

            ),

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1

        )

    ]


    return callbacks


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

def save_model_summary(

    model,

    model_name

):

    model_dir = os.path.join(

        RESULTS_DIR,

        model_name

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    summary_file = os.path.join(

        model_dir,

        "model_summary.txt"

    )


    with open(

        summary_file,

        "w",

        encoding="utf-8"

    ) as file:

        model.summary(

            print_fn=lambda line:

            file.write(
                line + "\n"
            )

        )


# ============================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================

def plot_training_history(

    history,

    model_name

):

    model_dir = os.path.join(

        RESULTS_DIR,

        model_name

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    # ========================================================
    # ACCURACY GRAPH
    # ========================================================

    plt.figure(

        figsize=(10, 6)

    )


    plt.plot(

        history.history["accuracy"],

        label="Training Accuracy",

        linewidth=2

    )


    plt.plot(

        history.history["val_accuracy"],

        label="Validation Accuracy",

        linewidth=2

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(

        f"{model_name} - "
        "Training vs Validation Accuracy"

    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "training_validation_accuracy.png"

        ),

        dpi=200

    )


    plt.close()


    # ========================================================
    # LOSS GRAPH
    # ========================================================

    plt.figure(

        figsize=(10, 6)

    )


    plt.plot(

        history.history["loss"],

        label="Training Loss",

        linewidth=2

    )


    plt.plot(

        history.history["val_loss"],

        label="Validation Loss",

        linewidth=2

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(

        f"{model_name} - "
        "Training vs Validation Loss"

    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "training_validation_loss.png"

        ),

        dpi=200

    )


    plt.close()


    # ========================================================
    # SAVE TRAINING HISTORY CSV
    # ========================================================

    history_df = pd.DataFrame(

        history.history

    )


    history_df.to_csv(

        os.path.join(

            model_dir,

            "training_history.csv"

        ),

        index=False

    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(

    model,

    X_test,

    y_test,

    model_name

):

    print("\n")
    print("=" * 70)
    print(
        f"EVALUATION - {model_name}"
    )
    print("=" * 70)


    model_dir = os.path.join(

        RESULTS_DIR,

        model_name

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nGenerating predictions..."
    )


    probabilities = model.predict(

        X_test,

        batch_size=128,

        verbose=1

    )


    predictions = np.argmax(

        probabilities,

        axis=1

    )


    confidence = np.max(

        probabilities,

        axis=1

    )


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    accuracy = accuracy_score(

        y_test,

        predictions

    )


    weighted_precision = precision_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0

    )


    weighted_recall = recall_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0

    )


    weighted_f1 = f1_score(

        y_test,

        predictions,

        average="weighted",

        zero_division=0

    )


    macro_precision = precision_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    macro_recall = recall_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0

    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\n")
    print("-" * 50)
    print("MODEL PERFORMANCE")
    print("-" * 50)


    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )


    print(
        f"Weighted Precision : "
        f"{weighted_precision:.4f}"
    )


    print(
        f"Weighted Recall    : "
        f"{weighted_recall:.4f}"
    )


    print(
        f"Weighted F1 Score  : "
        f"{weighted_f1:.4f}"
    )


    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )


    print(
        f"Macro Recall       : "
        f"{macro_recall:.4f}"
    )


    print(
        f"Macro F1 Score     : "
        f"{macro_f1:.4f}"
    )


    print(
        f"Mean Confidence    : "
        f"{np.mean(confidence):.4f}"
    )


    print(
        f"Median Confidence  : "
        f"{np.median(confidence):.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nGenerating classification report..."
    )


    report = classification_report(

        y_test,

        predictions,

        target_names=CLASS_NAMES,

        output_dict=True,

        zero_division=0

    )


    report_df = pd.DataFrame(
        report
    )


    report_df.to_csv(

        os.path.join(

            model_dir,

            "classification_report.csv"

        )

    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "Generating confusion matrix..."
    )


    cm = confusion_matrix(

        y_test,

        predictions

    )


    np.savetxt(

        os.path.join(

            model_dir,

            "confusion_matrix.csv"

        ),

        cm,

        delimiter=",",

        fmt="%d"

    )


    # ========================================================
    # CONFUSION MATRIX GRAPH
    # ========================================================

    plt.figure(

        figsize=(11, 9)

    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=CLASS_NAMES,

        yticklabels=CLASS_NAMES

    )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "True Label"
    )


    plt.title(

        f"{model_name} - "
        "Confusion Matrix"

    )


    plt.xticks(

        rotation=45,

        ha="right"

    )


    plt.yticks(
        rotation=0
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "confusion_matrix.png"

        ),

        dpi=220

    )


    plt.close()


    # ========================================================
    # NORMALIZED CONFUSION MATRIX
    # ========================================================

    cm_normalized = confusion_matrix(

        y_test,

        predictions,

        normalize="true"

    )


    np.savetxt(

        os.path.join(

            model_dir,

            "normalized_confusion_matrix.csv"

        ),

        cm_normalized,

        delimiter=",",

        fmt="%.6f"

    )


    plt.figure(

        figsize=(11, 9)

    )


    sns.heatmap(

        cm_normalized,

        annot=True,

        fmt=".2f",

        cmap="Blues",

        vmin=0,

        vmax=1,

        xticklabels=CLASS_NAMES,

        yticklabels=CLASS_NAMES

    )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "True Label"
    )


    plt.title(

        f"{model_name} - "
        "Normalized Confusion Matrix"

    )


    plt.xticks(

        rotation=45,

        ha="right"

    )


    plt.yticks(
        rotation=0
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "normalized_confusion_matrix.png"

        ),

        dpi=220

    )


    plt.close()


    # ========================================================
    # CLASS-WISE METRICS
    # ========================================================

    class_metrics = report_df.loc[

        CLASS_NAMES,

        [

            "precision",

            "recall",

            "f1-score"

        ]

    ].copy()


    # ========================================================
    # CLASS-WISE ACCURACY
    # ========================================================

    class_accuracy = []


    for i in range(
        NUM_CLASSES
    ):

        total_samples = cm[i].sum()


        if total_samples > 0:

            class_acc = (

                cm[i, i] /

                total_samples

            )

        else:

            class_acc = 0.0


        class_accuracy.append(
            class_acc
        )


    class_metrics[
        "accuracy"
    ] = class_accuracy


    # ========================================================
    # SAVE CLASS-WISE METRICS
    # ========================================================

    class_metrics.to_csv(

        os.path.join(

            model_dir,

            "class_wise_metrics.csv"

        )

    )


    # ========================================================
    # CLASS-WISE GRAPH
    # ========================================================

    plt.figure(

        figsize=(14, 7)

    )


    class_metrics[

        [

            "accuracy",

            "precision",

            "recall",

            "f1-score"

        ]

    ].plot(

        kind="bar",

        figsize=(14, 7)

    )


    plt.xlabel(
        "Fashion-MNIST Class"
    )


    plt.ylabel(
        "Score"
    )


    plt.title(

        f"{model_name} - "
        "Class-wise Performance"

    )


    plt.ylim(
        0,
        1.05
    )


    plt.xticks(

        rotation=45,

        ha="right"

    )


    plt.grid(

        axis="y",

        alpha=0.25

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "class_wise_metrics.png"

        ),

        dpi=220

    )


    plt.close()


    # ========================================================
    # PREDICTION CONFIDENCE GRAPH
    # ========================================================

    plt.figure(

        figsize=(10, 6)

    )


    plt.hist(

        confidence,

        bins=20,

        edgecolor="black"

    )


    plt.xlabel(
        "Prediction Confidence"
    )


    plt.ylabel(
        "Number of Images"
    )


    plt.title(

        f"{model_name} - "
        "Prediction Confidence Distribution"

    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            model_dir,

            "prediction_confidence.png"

        ),

        dpi=220

    )


    plt.close()


    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics = {

        "model": model_name,

        "accuracy": float(
            accuracy
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "mean_prediction_confidence": float(
            np.mean(confidence)
        ),

        "median_prediction_confidence": float(
            np.median(confidence)
        )

    }


    with open(

        os.path.join(

            model_dir,

            "metrics.json"

        ),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({

        "actual_label": y_test,

        "predicted_label": predictions,

        "confidence": confidence

    })


    predictions_df[
        "actual_class"
    ] = predictions_df[
        "actual_label"
    ].apply(

        lambda x:
        CLASS_NAMES[int(x)]

    )


    predictions_df[
        "predicted_class"
    ] = predictions_df[
        "predicted_label"
    ].apply(

        lambda x:
        CLASS_NAMES[int(x)]

    )


    predictions_df.to_csv(

        os.path.join(

            model_dir,

            "predictions.csv"

        ),

        index=False

    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        f"{model_name} EVALUATION COMPLETED"
    )
    print("=" * 70)


    print(
        "\nResults saved in:"
    )


    print(
        model_dir
    )


    return metrics