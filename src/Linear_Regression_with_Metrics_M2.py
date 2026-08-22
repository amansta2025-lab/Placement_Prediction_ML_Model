# ============================================================
# LINEAR REGRESSION - PLACEMENT PREDICTION DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = (
    "C:/Users/AMAN/Documents/placement_prediction/"
    "dataset/final_preprocess_M2.csv"
)

OUTPUT_FOLDER = (
    "C:/Users/AMAN/Documents/placement_prediction/"
    "outputs/Linear_Regression_with_Metrics_M2"
)

IMAGE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "images"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)


print("=" * 60)
print("LINEAR REGRESSION - PLACEMENT PREDICTION")
print("=" * 60)


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. DISPLAY COLUMN NAMES
# ============================================================

print("\nDataset Columns:")

for column in df.columns:
    print(column)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

# Multiple Linear Regression
#
# x1 = CGPA
# x2 = AptitudeTestScore
# x3 = CodingTestScore
# x4 = MockInterviewScore

feature_columns = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]


# Target variable
target_column = "PlacementStatus"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    feature_columns +
    [target_column]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR!")

    print(
        "The following columns were not found:"
    )

    print(
        missing_columns
    )

    print("\nAvailable columns are:")

    print(
        list(df.columns)
    )

    raise ValueError(
        "Please change feature_columns and "
        "target_column according to your dataset."
    )


# ============================================================
# 6. CREATE MODEL DATA
# ============================================================

model_df = df[
    required_columns
].copy()


print("\nModel data shape:")
print(model_df.shape)


# ============================================================
# 7. CONVERT FEATURE COLUMNS TO NUMERIC
# ============================================================

print(
    "\nConverting feature columns to numeric..."
)


for column in feature_columns:

    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


# ============================================================
# 8. CONVERT TARGET COLUMN TO NUMERIC
# ============================================================

print("\nOriginal PlacementStatus values:")

print(
    model_df[target_column]
    .value_counts(dropna=False)
)


# If target is not already numeric
if not pd.api.types.is_numeric_dtype(
    model_df[target_column]
):

    model_df[target_column] = (
        model_df[target_column]
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

    model_df[target_column] = (
        model_df[target_column]
        .map(placement_mapping)
    )


print(
    "\nPlacementStatus after conversion:"
)

print(
    model_df[target_column]
    .value_counts(dropna=False)
)


# ============================================================
# 9. HANDLE MISSING FEATURE VALUES
# ============================================================

print(
    "\nMissing values before handling:"
)

print(
    model_df.isnull().sum()
)


# IMPORTANT:
# Do NOT use dropna().
#
# Instead, fill missing feature values with
# the median so that the number of rows remains
# unchanged.


for column in feature_columns:

    if model_df[column].isnull().any():

        median_value = model_df[column].median()

        model_df[column] = (
            model_df[column]
            .fillna(median_value)
        )


# ============================================================
# 10. HANDLE MISSING TARGET VALUES
# ============================================================

# A model cannot train without a target value.
#
# Normally PlacementStatus should not contain
# missing values.
#
# If it does, we cannot create a valid target for
# those rows without changing the actual target data.


if model_df[target_column].isnull().any():

    missing_target_count = (
        model_df[target_column]
        .isnull()
        .sum()
    )

    print(
        "\nWARNING:"
    )

    print(
        f"{missing_target_count} rows have "
        "missing PlacementStatus values."
    )

    print(
        "These rows cannot be used for training."
    )

    model_df = model_df.dropna(
        subset=[target_column]
    )


# ============================================================
# 11. CHECK FINAL DATASET SIZE
# ============================================================

print(
    "\nFinal model data shape:"
)

print(
    model_df.shape
)


# ============================================================
# 12. DEFINE X AND Y
# ============================================================

X = model_df[
    feature_columns
]


y = model_df[
    target_column
]


print(
    "\nFeature data types:"
)

print(
    X.dtypes
)


print(
    "\nTarget data type:"
)

print(
    y.dtype
)


# ============================================================
# 13. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 14. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 15. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print(
    "\nModel training completed."
)


# ============================================================
# 16. MODEL COEFFICIENTS
# ============================================================

print(
    "\nIntercept is b0:"
)

print(
    model.intercept_
)


print(
    "\nCoefficients (b1,b2,b3,b4) values:"
)


coefficient_df = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": model.coef_
})


print(
    coefficient_df
)


# ============================================================
# 17. LINEAR REGRESSION EQUATION
# ============================================================

# y = b0 + b1x1 + b2x2 + b3x3 + b4x4

equation = (
    f"{target_column} = "
    f"{model.intercept_:.4f}"
)


for feature, coefficient in zip(
    feature_columns,
    model.coef_
):

    equation += (
        f" + ({coefficient:.4f} × {feature})"
    )


print(
    "\nLinear Regression Equation:"
)

print(
    equation
)


# ============================================================
# 18. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 19. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


mse = mean_squared_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mse
)


r2 = r2_score(
    y_test,
    y_pred
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL EVALUATION"
)

print(
    "=" * 60
)


print(
    f"MAE  : {mae:.4f}"
)


print(
    f"MSE  : {mse:.4f}"
)


print(
    f"RMSE : {rmse:.4f}"
)


print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 20. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()


results["Actual"] = (
    y_test.values
)


results["Predicted"] = (
    y_pred
)


results["Residual"] = (
    results["Actual"] -
    results["Predicted"]
)


results["Absolute_Error"] = (
    results["Residual"].abs()
)


# ============================================================
# 21. SAVE PREDICTION RESULTS
# ============================================================

prediction_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_predictions.csv"
)


results.to_csv(
    prediction_file,
    index=False
)


print(
    "\nPrediction results saved to:"
)

print(
    prediction_file
)


# ============================================================
# 22. SAVE MODEL COEFFICIENTS
# ============================================================

coefficient_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_coefficients.csv"
)


coefficient_df.to_csv(
    coefficient_file,
    index=False
)


print(
    "\nCoefficient file saved to:"
)

print(
    coefficient_file
)


# ============================================================
# 23. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],

    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})


metrics_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


print(
    "\nMetrics file saved to:"
)

print(
    metrics_file
)


# ============================================================
# 24. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)


maximum = max(
    y_test.max(),
    y_pred.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.xlabel(
    "Actual placement"
)


plt.ylabel(
    "Predicted placement"
)


plt.title(
    "Linear Regression: "
    "Actual vs Predicted placement"
)


plt.grid(
    True
)


actual_predicted_image = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    actual_predicted_image,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 25. RESIDUAL GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_pred,
    results["Residual"],
    alpha=0.6
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Placement"
)


plt.ylabel(
    "Residual"
)


plt.title(
    "Residual Plot - Linear Regression"
)


plt.grid(
    True
)


residual_image = os.path.join(
    IMAGE_FOLDER,
    "residual_plot.png"
)


plt.savefig(
    residual_image,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 26. COEFFICIENT GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(
    coefficient_df["Feature"],
    coefficient_df["Coefficient"]
)


plt.xlabel(
    "Features"
)


plt.ylabel(
    "Coefficient"
)


plt.title(
    "Linear Regression Feature Coefficients"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.grid(
    axis="y"
)


coefficient_image = os.path.join(
    IMAGE_FOLDER,
    "feature_coefficients.png"
)


plt.savefig(
    coefficient_image,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 27. SAVE EQUATION
# ============================================================

equation_file = os.path.join(
    OUTPUT_FOLDER,
    "linear_regression_equation.txt"
)


with open(
    equation_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Linear Regression Equation\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(
        equation
    )


# ============================================================
# 28. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PROCESS COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)


print(
    "\nOutput Folder:"
)

print(
    OUTPUT_FOLDER
)


print(
    "\nGenerated Files:"
)


print(
    "- linear_regression_predictions.csv"
)


print(
    "- linear_regression_coefficients.csv"
)


print(
    "- linear_regression_metrics.csv"
)


print(
    "- linear_regression_equation.txt"
)


print(
    "\nGenerated Images:"
)


print(
    "- actual_vs_predicted.png"
)


print(
    "- residual_plot.png"
)


print(
    "- feature_coefficients.png"
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
    "\n" + "=" * 60
)

print(
    "ALL FILES GENERATED SUCCESSFULLY"
)

print(
    "=" * 60
)