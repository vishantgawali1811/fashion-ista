# ============================================================
# EFFICIENTNETB0 - FASHION-MNIST
# ============================================================

import os
import sys
import tensorflow as tf


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


MODEL_NAME = "EfficientNetB0"

IMAGE_SIZE = 224

BATCH_SIZE = 64

INITIAL_EPOCHS = 10

FINE_TUNE_EPOCHS = 10


def main():

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 - FASHION-MNIST")
    print("=" * 70)


    # ========================================================
    # DATA
    # ========================================================

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
    # EFFICIENTNETB0
    # ========================================================

    print(
        "\nLoading ImageNet EfficientNetB0..."
    )


    base_model = tf.keras.applications.EfficientNetB0(

        include_top=False,

        weights="imagenet",

        input_shape=(

            IMAGE_SIZE,

            IMAGE_SIZE,

            3

        )

    )


    base_model.trainable = False


    # ========================================================
    # INPUT
    # ========================================================

    inputs = tf.keras.Input(

        shape=(

            IMAGE_SIZE,

            IMAGE_SIZE,

            3

        )

    )


    # ========================================================
    # AUGMENTATION
    # ========================================================

    x = make_augmentation()(inputs)


    # ========================================================
    # EFFICIENTNET PREPROCESSING
    #
    # Keras EfficientNet expects pixel values in the
    # 0-255 range.
    # ========================================================

    x = tf.keras.layers.Rescaling(
        255.0
    )(x)


    # ========================================================
    # BACKBONE
    # ========================================================

    x = base_model(

        x,

        training=False

    )


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = tf.keras.layers.GlobalAveragePooling2D()(x)


    x = tf.keras.layers.BatchNormalization()(x)


    x = tf.keras.layers.Dense(

        256,

        activation="relu"

    )(x)


    x = tf.keras.layers.Dropout(

        0.4

    )(x)


    outputs = tf.keras.layers.Dense(

        10,

        activation="softmax"

    )(x)


    model = tf.keras.Model(

        inputs,

        outputs,

        name="FashionMNIST_EfficientNetB0"

    )


    # ========================================================
    # COMPILE
    # ========================================================

    compile_model(

        model,

        learning_rate=1e-3

    )


    save_model_summary(

        model,

        MODEL_NAME

    )


    # ========================================================
    # INITIAL TRAINING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 CLASSIFICATION HEAD TRAINING")
    print("=" * 70)


    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=INITIAL_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=get_callbacks(

            MODEL_NAME

        ),

        verbose=1

    )


    # ========================================================
    # FINE-TUNING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 FINE-TUNING")
    print("=" * 70)


    base_model.trainable = True


    for layer in base_model.layers[:-20]:

        layer.trainable = False


    for layer in base_model.layers:

        if isinstance(

            layer,

            tf.keras.layers.BatchNormalization

        ):

            layer.trainable = False


    compile_model(

        model,

        learning_rate=1e-5

    )


    fine_history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=FINE_TUNE_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=get_callbacks(

            MODEL_NAME

        ),

        verbose=1

    )


    # ========================================================
    # MERGE HISTORY
    # ========================================================

    for key in history.history:

        history.history[key].extend(

            fine_history.history[key]

        )


    # ========================================================
    # SAVE GRAPHS
    # ========================================================

    plot_training_history(

        history,

        MODEL_NAME

    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_dir = os.path.join(

        RESULTS_DIR,

        MODEL_NAME

    )


    os.makedirs(

        model_dir,

        exist_ok=True

    )


    model.save(

        os.path.join(

            model_dir,

            "final_model.keras"

        )

    )


    # ========================================================
    # EVALUATION
    # ========================================================

    metrics = evaluate_model(

        model,

        X_test,

        y_test,

        MODEL_NAME

    )


    print("\n")
    print("=" * 70)
    print("EFFICIENTNETB0 FINAL RESULTS")
    print("=" * 70)


    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['weighted_precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['weighted_recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['weighted_f1']:.4f}"
    )


    print(
        "\nEfficientNetB0 completed successfully! ✅"
    )


if __name__ == "__main__":
    main()