
# ============================================================
# MANUSCRIPT WORD IMAGE CLASSIFICATION
# Mixed Feature Extraction: HOG + Zoning + LBP
# Classifier: Random Forest
# Dataset: DATASET_Manuscript
# ============================================================

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from skimage.feature import hog, local_binary_pattern
from skimage.filters import threshold_otsu

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (128, 128)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

ZONING_GRID = (8, 8)

LBP_RADIUS = 1
LBP_POINTS = 8
LBP_METHOD = "uniform"

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
)

MODEL_PATH = "manuscript_hog_zoning_lbp_rf.pkl"
OUTPUT_DIR = "Manuscript_Results"


# ============================================================
# 1. IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    image = Image.open(image_path).convert("L")

    # Resize while keeping a consistent input dimension
    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS
    )

    image = np.asarray(image, dtype=np.float32) / 255.0

    return image


# ============================================================
# 2. HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(image):

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)


# ============================================================
# 3. ZONING FEATURE EXTRACTION
# ============================================================

def extract_zoning_features(image):

    # Convert grayscale image into foreground/background
    threshold = threshold_otsu(image)

    binary = (image < threshold).astype(np.float32)

    height, width = binary.shape

    rows, cols = ZONING_GRID

    zoning_features = []

    for row in range(rows):

        for col in range(cols):

            y1 = int(row * height / rows)
            y2 = int((row + 1) * height / rows)

            x1 = int(col * width / cols)
            x2 = int((col + 1) * width / cols)

            zone = binary[y1:y2, x1:x2]

            density = np.mean(zone)

            zoning_features.append(density)

    return np.asarray(zoning_features, dtype=np.float32)


# ============================================================
# 4. LBP FEATURE EXTRACTION
# ============================================================

def extract_lbp_features(image):

    image_uint8 = (image * 255).astype(np.uint8)

    lbp = local_binary_pattern(
        image_uint8,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    bins = LBP_POINTS + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, bins + 1),
        range=(0, bins)
    )

    histogram = histogram.astype(np.float32)

    histogram /= (histogram.sum() + 1e-7)

    return histogram


# ============================================================
# 5. MIXED FEATURE EXTRACTION
# ============================================================

def extract_mixed_features(image_path):

    image = load_image(image_path)

    hog_features = extract_hog_features(image)

    zoning_features = extract_zoning_features(image)

    lbp_features = extract_lbp_features(image)

    # Feature fusion
    mixed_features = np.concatenate([
        hog_features,
        zoning_features,
        lbp_features
    ])

    return mixed_features


# ============================================================
# 6. LOAD DATASET AUTOMATICALLY
# ============================================================

def load_dataset(dataset_path):

    features = []
    labels = []
    paths = []

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    class_folders = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    if len(class_folders) < 2:
        raise ValueError(
            "At least two class folders are required."
        )

    print("\nDATASET LOADING")
    print("=" * 55)

    for class_name in class_folders:

        class_path = os.path.join(dataset_path, class_name)

        image_files = sorted([
            file for file in os.listdir(class_path)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ])

        print(
            f"{class_name}: {len(image_files)} images"
        )

        for filename in image_files:

            image_path = os.path.join(class_path, filename)

            try:

                feature = extract_mixed_features(image_path)

                features.append(feature)
                labels.append(class_name)
                paths.append(image_path)

            except Exception as error:

                print(
                    f"Could not process {image_path}: {error}"
                )

    if len(features) == 0:
        raise ValueError("No valid images found.")

    X = np.asarray(features, dtype=np.float32)

    y = np.asarray(labels)

    print("\nFeature extraction completed.")
    print("Total images:", len(X))
    print("Feature dimensions:", X.shape[1])
    print("Number of classes:", len(np.unique(y)))

    return X, y, paths


# ============================================================
# 7. RANDOM FOREST CLASSIFIER
# ============================================================

def train_classifier(dataset_path):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, paths = load_dataset(dataset_path)

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    # Check whether a stratified split is possible
    class_counts = np.bincount(y_encoded)

    if np.min(class_counts) < 2:
        raise ValueError(
            "Each class must contain at least 2 valid images."
        )

    # Test set must contain at least one image per class
    n_classes = len(encoder.classes_)
    n_test = int(np.ceil(0.20 * len(y_encoded)))
    n_train = len(y_encoded) - n_test

    if n_test < n_classes or n_train < n_classes:
        raise ValueError(
            "Not enough images for a stratified 80:20 split. "
            "Add more images to each class."
        )

    # Train-test split
    X_train, X_test, y_train, y_test, paths_train, paths_test = (
        train_test_split(
            X,
            y_encoded,
            paths,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )
    )

    print("\nTRAINING RANDOM FOREST")
    print("=" * 55)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    classifier = RandomForestClassifier(
        n_estimators=300,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    print("\nMODEL PERFORMANCE")
    print("=" * 55)

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )

    print(report)

    # Save classification report
    with open(
        os.path.join(OUTPUT_DIR, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # Save metrics
    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-score"
        ],
        "Value (%)": [
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100
        ]
    })

    metrics.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"),
        index=False
    )

    # Save individual predictions
    predictions = pd.DataFrame({
        "Image": paths_test,
        "Actual Class": encoder.inverse_transform(y_test),
        "Predicted Class": encoder.inverse_transform(y_pred)
    })

    predictions.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_))
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d"
    )

    plt.title("HOG + Zoning + LBP with Random Forest")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Feature importance
    importances = classifier.feature_importances_

    plt.figure(figsize=(12, 5))

    plt.plot(importances)

    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=300
    )

    plt.close()

    # Save complete model
    model_bundle = {
        "classifier": classifier,
        "encoder": encoder,
        "image_size": IMAGE_SIZE,
        "hog_orientations": HOG_ORIENTATIONS,
        "hog_pixels_per_cell": HOG_PIXELS_PER_CELL,
        "hog_cells_per_block": HOG_CELLS_PER_BLOCK,
        "zoning_grid": ZONING_GRID,
        "lbp_radius": LBP_RADIUS,
        "lbp_points": LBP_POINTS,
        "lbp_method": LBP_METHOD
    }

    joblib.dump(
        model_bundle,
        os.path.join(OUTPUT_DIR, MODEL_PATH)
    )

    print("\nALL RESULTS SAVED IN:", OUTPUT_DIR)


# ============================================================
# 8. PREDICT A NEW MANUSCRIPT WORD IMAGE
# ============================================================

def predict_new_image(image_path):

    model_file = os.path.join(OUTPUT_DIR, MODEL_PATH)

    if not os.path.exists(model_file):
        raise FileNotFoundError(
            "Trained model not found. Train the classifier first."
        )

    model_bundle = joblib.load(model_file)

    classifier = model_bundle["classifier"]

    encoder = model_bundle["encoder"]

    feature = extract_mixed_features(image_path)

    feature = feature.reshape(1, -1)

    prediction = classifier.predict(feature)[0]

    predicted_label = encoder.inverse_transform(
        [prediction]
    )[0]

    print("\nNEW IMAGE CLASSIFICATION")
    print("=" * 55)

    print("Image:", image_path)

    print("Predicted class:", predicted_label)

    probabilities = classifier.predict_proba(feature)[0]

    print("\nCLASS-WISE MODEL SCORES")

    for index in np.argsort(probabilities)[::-1]:

        print(
            f"{encoder.classes_[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manuscript classification using HOG + Zoning + LBP"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="DATASET_Manuscript",
        help="Path to manuscript dataset"
    )

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to a new manuscript word image"
    )

    args = parser.parse_args()

    if args.predict:

        predict_new_image(args.predict)

    else:

        train_classifier(args.data_dir)
