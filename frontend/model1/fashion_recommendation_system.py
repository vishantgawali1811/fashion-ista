# ============================================================
# CNN BASED FASHION PREDICTION AND RECOMMENDATION SYSTEM
# Dataset: Fashion MNIST CSV
# Framework: TensorFlow / Keras
# Classes: 10 clothing categories
# Special Focus: T-Shirt and Shirt Recommendations
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from google.colab import files
from PIL import Image
import io

print("TensorFlow Version:", tf.__version__)

# 2. UPLOAD DATASET
print("Upload fashion-mnist_train.csv")
uploaded_train = files.upload()
train_file = list(uploaded_train.keys())[0]

print("Upload fashion-mnist_test.csv")
uploaded_test = files.upload()
test_file = list(uploaded_test.keys())[0]

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Training Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

# 3. DEFINE FASHION CLASSES
class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# 4. SEPARATE LABELS AND PIXEL VALUES
X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 6. SPLIT TRAINING DATA INTO TRAIN AND VALIDATION
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

# 7. BUILD CNN MODEL
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax")
])

# 8. COMPILE CNN MODEL
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. TRAIN CNN MODEL
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 14. FASHION RECOMMENDATION DATABASE
recommendation_database = {
    "T-shirt/Top": [
        {
            "name": "Classic Cotton T-Shirt",
            "description": "Comfortable cotton T-shirt for everyday wear.",
            "style": "Casual",
            "color": "White / Black"
        },
        {
            "name": "Oversized T-Shirt",
            "description": "Relaxed fit T-shirt for a modern look.",
            "style": "Streetwear",
            "color": "Black / Beige"
        }
    ],
    "Shirt": [
        {
            "name": "Formal Cotton Shirt",
            "description": "Classic shirt suitable for office and meetings.",
            "style": "Formal",
            "color": "White / Light Blue"
        }
    ]
}

# 15. CREATE RECOMMENDATION FUNCTION
def recommend_products(predicted_class, confidence):
    category = class_names[predicted_class]
    print("\nPredicted Category:", category)
    if category in recommendation_database:
        products = recommendation_database[category]
        for i, product in enumerate(products, start=1):
            print(f"Rec {i}: {product['name']}")
    else:
        print("No recommendations for this category.")

# 16. CREATE IMAGE PREDICTION FUNCTION
def predict_fashion(image_array):
    image_array = np.array(image_array).astype("float32")
    if image_array.max() > 1:
        image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    recommend_products(predicted_class, confidence)
