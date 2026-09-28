import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==================================================
# 2. FEATURES AND TARGET
# ==================================================

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})


# Remove customer ID
X = X.drop("customerID", axis=1)


# ==================================================
# 3. IDENTIFY COLUMNS
# ==================================================

categorical_columns = X.select_dtypes(
    include="str"
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==================================================
# 4. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# ==================================================
# 5. PREPROCESSING
# ==================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)


# ==================================================
# 6. MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
}


# ==================================================
# 7. TRAIN AND EVALUATE MODELS
# ==================================================

results = []

for model_name, model in models.items():

    print("\n========================================")
    print(model_name.upper())
    print("========================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ==================================================
# 8. MODEL COMPARISON
# ==================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results_df)


# ==================================================
# 9. SAVE RESULTS
# ==================================================

results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# ==================================================
# 10. MODEL COMPARISON CHART
# ==================================================

results_df.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Customer Churn Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visual/model_comparison.png"
)

plt.show()

print("\nModel comparison chart saved successfully!")