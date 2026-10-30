# ============================================================
# TEST LOCAL FASHION-MNIST DATASET
# ============================================================

from common import (
    load_fashion_mnist,
    CLASS_NAMES
)


print("\n")
print("=" * 70)
print("FASHION-MNIST DATASET TEST")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist(
    image_size=28,
    validation_size=5000
)


# ============================================================
# DATASET SHAPES
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET SHAPES")
print("=" * 70)


print(
    "\nX_train       :",
    X_train.shape
)

print(
    "y_train       :",
    y_train.shape
)


print(
    "\nX_validation  :",
    X_val.shape
)

print(
    "y_validation  :",
    y_val.shape
)


print(
    "\nX_test        :",
    X_test.shape
)

print(
    "y_test        :",
    y_test.shape
)


# ============================================================
# PIXEL VALUE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("PIXEL VALUE CHECK")
print("=" * 70)


print(
    "\nMinimum pixel value:",
    X_train.min()
)


print(
    "Maximum pixel value:",
    X_train.max()
)


# ============================================================
# LABEL CHECK
# ============================================================

print("\n")
print("=" * 70)
print("LABEL CHECK")
print("=" * 70)


print(
    "\nUnique labels:"
)


print(
    sorted(
        set(y_train.tolist())
    )
)


# ============================================================
# CLASS NAMES
# ============================================================

print("\nClass mapping:")


for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i} -> {class_name}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DATASET TEST COMPLETED SUCCESSFULLY! ✅")
print("=" * 70)