# ============================================================
# CNN MODEL
# FASHION-MNIST IMAGE CLASSIFICATION
# ============================================================
#
# Model:
#   Custom Convolutional Neural Network
#
# Dataset:
#   Fashion-MNIST
#
# Input:
#   28 x 28 x 3
#
# Output:
#   10 Fashion-MNIST classes
#
# ============================================================


import os
import sys


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT COMMON FUNCTIONS
# ============================================================

from common import (
    load_fashion_mnist,
    make_augmentation,
    compile_model,
    get_callbacks,
    save_model_summary,
    plot_training_history,
    evaluate_model,
    RESULTS_DIR
)


import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "CNN"

IMAGE_SIZE = 28

BATCH_SIZE = 128

EPOCHS = 20

LEARNING_RATE = 0.001


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("FASHION-MNIST CNN MODEL")
    print("=" * 70)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")


    (

        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test

    ) = load_fashion_mnist(

        image_size=IMAGE_SIZE,

        validation_size=5000

    )


    # ========================================================
    # DISPLAY DATASET SHAPES
    # ========================================================

    print("\n")
    print("-" * 70)
    print("DATASET SHAPES")
    print("-" * 70)


    print(
        "X_train:",
        X_train.shape
    )


    print(
        "y_train:",
        y_train.shape
    )


    print(
        "X_validation:",
        X_val.shape
    )


    print(
        "y_validation:",
        y_val.shape
    )


    print(
        "X_test:",
        X_test.shape
    )


    print(
        "y_test:",
        y_test.shape
    )


    # ========================================================
    # BUILD CNN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BUILDING CNN MODEL")
    print("=" * 70)


    model = tf.keras.Sequential(

        [

            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            tf.keras.layers.Input(

                shape=(

                    IMAGE_SIZE,

                    IMAGE_SIZE,

                    3

                )

            ),


            # ------------------------------------------------
            # DATA AUGMENTATION
            # ------------------------------------------------

            make_augmentation(),


            # ------------------------------------------------
            # CONVOLUTION BLOCK 1
            # ------------------------------------------------

            tf.keras.layers.Conv2D(

                filters=32,

                kernel_size=(3, 3),

                padding="same",

                activation="relu"

            ),


            tf.keras.layers.BatchNormalization(),


            tf.keras.layers.MaxPooling2D(

                pool_size=(2, 2)

            ),


            # ------------------------------------------------
            # CONVOLUTION BLOCK 2
            # ------------------------------------------------

            tf.keras.layers.Conv2D(

                filters=64,

                kernel_size=(3, 3),

                padding="same",

                activation="relu"

            ),


            tf.keras.layers.BatchNormalization(),


            tf.keras.layers.MaxPooling2D(

                pool_size=(2, 2)

            ),


            # ------------------------------------------------
            # CONVOLUTION BLOCK 3
            # ------------------------------------------------

            tf.keras.layers.Conv2D(

                filters=128,

                kernel_size=(3, 3),

                padding="same",

                activation="relu"

            ),


            tf.keras.layers.BatchNormalization(),


            tf.keras.layers.MaxPooling2D(

                pool_size=(2, 2)

            ),


            # ------------------------------------------------
            # GLOBAL AVERAGE POOLING
            # ------------------------------------------------

            tf.keras.layers.GlobalAveragePooling2D(),


            # ------------------------------------------------
            # FULLY CONNECTED LAYER
            # ------------------------------------------------

            tf.keras.layers.Dense(

                128,

                activation="relu"

            ),


            # ------------------------------------------------
            # DROPOUT
            # ------------------------------------------------

            tf.keras.layers.Dropout(

                0.35

            ),


            # ------------------------------------------------
            # OUTPUT LAYER
            # ------------------------------------------------

            tf.keras.layers.Dense(

                10,

                activation="softmax"

            )

        ],

        name="FashionMNIST_CNN"

    )


    # ========================================================
    # COMPILE MODEL
    # ========================================================

    print("\nCompiling model...")


    model = compile_model(

        model,

        learning_rate=LEARNING_RATE

    )


    # ========================================================
    # DISPLAY MODEL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)


    model.summary()


    # ========================================================
    # SAVE MODEL SUMMARY
    # ========================================================

    save_model_summary(

        model,

        MODEL_NAME

    )


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = get_callbacks(

        MODEL_NAME

    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING CNN TRAINING")
    print("=" * 70)


    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1

    )


    # ========================================================
    # TRAINING GRAPHS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAVING TRAINING GRAPHS")
    print("=" * 70)


    plot_training_history(

        history,

        MODEL_NAME

    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model_dir = os.path.join(

        RESULTS_DIR,

        MODEL_NAME

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    final_model_path = os.path.join(

        model_dir,

        "final_model.keras"

    )


    model.save(

        final_model_path

    )


    print(
        "\nFinal model saved:"
    )


    print(
        final_model_path
    )


    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    metrics = evaluate_model(

        model,

        X_test,

        y_test,

        MODEL_NAME

    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CNN FINAL RESULTS")
    print("=" * 70)


    print(
        f"\nAccuracy          : "
        f"{metrics['accuracy']:.4f}"
    )


    print(
        f"Weighted Precision: "
        f"{metrics['weighted_precision']:.4f}"
    )


    print(
        f"Weighted Recall   : "
        f"{metrics['weighted_recall']:.4f}"
    )


    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1']:.4f}"
    )


    print(
        f"Macro F1          : "
        f"{metrics['macro_f1']:.4f}"
    )


    print(
        f"Mean Confidence   : "
        f"{metrics['mean_prediction_confidence']:.4f}"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CNN TRAINING AND EVALUATION COMPLETED! ✅")
    print("=" * 70)


    print("\nResults are available in:")

    print(
        os.path.abspath(model_dir)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()