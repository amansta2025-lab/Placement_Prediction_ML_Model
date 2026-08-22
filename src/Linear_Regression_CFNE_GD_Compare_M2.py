# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Images are stored in ONE separate folder
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = (
    "C:/Users/AMAN/Documents/placement_prediction/"
    "dataset/final_preprocess_M2.csv"
)

data = pd.read_csv(DATASET_PATH)


print("=" * 60)
print("LINEAR REGRESSION")
print("Closed-Form Normal Equation vs Gradient Descent")
print("=" * 60)


print("\nOriginal Dataset Shape:")
print(data.shape)


print("\nDataset Columns:")

for column in data.columns:
    print(column)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

# The original program used:
#
# X = all columns except last
# y = last column
#
# We keep that logic, but make sure the feature columns
# are numeric before using Linear Regression.

feature_columns = list(
    data.columns[:-1]
)

target_column = data.columns[-1]


print("\nFeature Columns:")
print(feature_columns)


print("\nTarget Column:")
print(target_column)


# ============================================================
# 3. CREATE MODEL DATA
# ============================================================

model_data = data[
    feature_columns + [target_column]
].copy()


# ============================================================
# 4. CONVERT FEATURE COLUMNS TO NUMERIC
# ============================================================

print("\nConverting feature columns to numeric...")


for column in feature_columns:

    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="coerce"
    )


# ============================================================
# 5. CONVERT TARGET COLUMN TO NUMERIC
# ============================================================

print("\nOriginal target values:")

print(
    model_data[target_column]
    .value_counts(dropna=False)
)


# If target is already numeric, keep it.
# Otherwise convert common placement labels.

if not pd.api.types.is_numeric_dtype(
    model_data[target_column]
):

    model_data[target_column] = (
        model_data[target_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    placement_mapping = {

        "placed": 1,
        "not placed": 0,
        "notplaced": 0,

        "yes": 1,
        "no": 0,

        "true": 1,
        "false": 0,

        "1": 1,
        "0": 0
    }

    model_data[target_column] = (
        model_data[target_column]
        .map(placement_mapping)
    )


print("\nTarget values after conversion:")

print(
    model_data[target_column]
    .value_counts(dropna=False)
)


# ============================================================
# 6. HANDLE MISSING FEATURE VALUES
# ============================================================

print("\nMissing values before handling:")

print(
    model_data.isnull().sum()
)


# Do NOT use dropna() for feature columns.
#
# Filling with median keeps the number of rows unchanged.

for column in feature_columns:

    if model_data[column].isnull().any():

        median_value = (
            model_data[column].median()
        )

        model_data[column] = (
            model_data[column]
            .fillna(median_value)
        )


# ============================================================
# 7. HANDLE MISSING TARGET VALUES
# ============================================================

# A model cannot calculate a target for a row if
# the actual target value is missing.
#
# Normally PlacementStatus should have no missing values.

missing_target_count = (
    model_data[target_column]
    .isnull()
    .sum()
)


if missing_target_count > 0:

    print(
        "\nWARNING:"
    )

    print(
        "Missing target values:",
        missing_target_count
    )

    print(
        "Rows with missing target values "
        "will be removed."
    )

    model_data = model_data.dropna(
        subset=[target_column]
    )


# ============================================================
# 8. CREATE X AND Y
# ============================================================

X = model_data[
    feature_columns
].values


y = model_data[
    target_column
].values


# Force everything to float
X = X.astype(float)
y = y.astype(float)


print("\nFinal Model Dataset Shape:")
print(model_data.shape)


print("\nX Shape:")
print(X.shape)


print("\ny Shape:")
print(y.shape)


# ============================================================
# 9. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    "C:/Users/AMAN/Documents/placement_prediction/"
    "outputs/Linear_Regression_CFNE_GD_Compare_M2"
)


os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:")
print(len(X_train))


print("Testing samples:")
print(len(X_test))


# ============================================================
# 11. FEATURE SCALING
#     Important for Gradient Descent
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 12. CLOSED FORM SOLUTION
#     NORMAL EQUATION
# ============================================================

# Add bias column

X_train_bias = np.c_[
    np.ones(
        (X_train.shape[0], 1)
    ),
    X_train
]


X_test_bias = np.c_[
    np.ones(
        (X_test.shape[0], 1)
    ),
    X_test
]


# ============================================================
# NORMAL EQUATION
#
# theta = (X^T X)^(-1) X^T y
#
# We use pseudo-inverse instead of inverse.
#
# pinv() prevents "Singular Matrix" errors.
# ============================================================

theta = np.linalg.pinv(
    X_train_bias.T.dot(
        X_train_bias
    )
).dot(
    X_train_bias.T
).dot(
    y_train
)


# ============================================================
# 13. NORMAL EQUATION PREDICTION
# ============================================================

pred_normal = (
    X_test_bias.dot(theta)
)


# ============================================================
# 14. NORMAL EQUATION METRICS
# ============================================================

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)


r2_normal = r2_score(
    y_test,
    pred_normal
)


print(
    "\n------ Closed Form Normal Equation ------"
)


print("\nCoefficients:")

print(theta)


print("\nMSE:")

print(mse_normal)


print("\nR2 Score:")

print(r2_normal)


# ============================================================
# 15. GRADIENT DESCENT
# ============================================================

X_train_gd = np.c_[
    np.ones(
        (X_train_scaled.shape[0], 1)
    ),
    X_train_scaled
]


X_test_gd = np.c_[
    np.ones(
        (X_test_scaled.shape[0], 1)
    ),
    X_test_scaled
]


m = len(y_train)


theta_gd = np.zeros(
    X_train_gd.shape[1]
)


learning_rate = 0.01


epochs = 1000


# ============================================================
# 16. STORE LOSS FOR EACH EPOCH
# ============================================================

loss_history = []


# ============================================================
# 17. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = (
        X_train_gd.dot(theta_gd)
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = (
        predictions - y_train
    )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (
        (2 / m)
        * X_train_gd.T.dot(errors)
    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (
        learning_rate * gradients
    )


    # --------------------------------------------------------
    # Calculate training MSE
    # --------------------------------------------------------

    current_predictions = (
        X_train_gd.dot(theta_gd)
    )


    loss = np.mean(
        (current_predictions - y_train) ** 2
    )


    loss_history.append(
        loss
    )


# ============================================================
# 18. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = (
    X_test_gd.dot(theta_gd)
)


# ============================================================
# 19. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)


r2_gd = r2_score(
    y_test,
    pred_gd
)


print(
    "\n------ Gradient Descent ------"
)


print("\nCoefficients:")

print(theta_gd)


print("\nMSE:")

print(mse_gd)


print("\nR2 Score:")

print(r2_gd)


# ============================================================
# 20. COMPARISON
# ============================================================

print(
    "\n=========== Comparison ==========="
)


print(
    "\nNormal Equation"
)


print(
    "MSE =",
    mse_normal
)


print(
    "R2 =",
    r2_normal
)


print(
    "\nGradient Descent"
)


print(
    "MSE =",
    mse_gd
)


print(
    "R2 =",
    r2_gd
)


# ============================================================
# 21. IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


# ============================================================
# Perfect prediction line
# ============================================================

minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)


maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)
#this line is very important

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Values"
)


plt.ylabel(
    "Predicted Values"
)


plt.title(
    "Actual vs Predicted Values"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nImage saved:"
)

print(
    image1
)


# ============================================================
# 22. IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

# Residual = actual - predicted

normal_residuals = (
    y_test - pred_normal
)


gd_residuals = (
    y_test - pred_gd
)


plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)


plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Comparison"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)


plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)

print(
    image2
)


# ============================================================
# 23. IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    range(
        1,
        epochs + 1
    ),
    loss_history
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Mean Squared Error"
)


plt.title(
    "Gradient Descent Convergence"
)


plt.grid(
    True
)


plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)


plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)

print(
    image3
)


# ============================================================
# 24. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [

        "actual_vs_predicted.png",

        "residual_comparison.png",

        "gradient_descent_loss.png"
    ],

    "Description": [

        "Actual values versus predictions "
        "from both methods",

        "Residual comparison between "
        "Normal Equation and Gradient Descent",

        "MSE loss across Gradient "
        "Descent epochs"
    ]
})


image_info.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "image_information.csv"
    ),
    index=False
)


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)


print(
    "PROCESS COMPLETED SUCCESSFULLY"
)


print(
    "=========================================="
)


print(
    "\nAll images are stored in ONE folder:"
)


print(
    IMAGE_FOLDER
)


print(
    "\nGenerated images:"
)


print(
    "1. actual_vs_predicted.png"
)


print(
    "2. residual_comparison.png"
)


print(
    "3. gradient_descent_loss.png"
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


print(
    "\nOriginal dataset was NOT modified."
)